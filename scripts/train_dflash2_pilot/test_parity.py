#!/usr/bin/env python3
"""CPU parity, gradient, and checkpoint tests for the DFlash2 pilot."""

import argparse
import contextlib
import importlib.util
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

if __package__:
    from .model import TOP_K, DFlash2DraftModel, load_target_head_and_embed
    from . import collect_rollouts, train
    from .train import Anchors, Sequence, configure_stage, loss_rows
else:
    from model import TOP_K, DFlash2DraftModel, load_target_head_and_embed
    import collect_rollouts
    import train
    from train import Anchors, Sequence, configure_stage, loss_rows

oracle_path = Path(__file__).resolve().parents[2] / "engine/tools/arch06a/reference.py"
spec = importlib.util.spec_from_file_location("arch06a_reference", oracle_path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

EXPECTED = {
    "fc.weight": (2304, 9216),
    "hidden_norm.weight": (2304,),
    "norm.weight": (2304,),
    "candidate_selector.hidden_projection.weight": (256, 2304),
    "candidate_selector.predecessor_codebook": (201024, 256),
    "candidate_selector.successor_codebook": (201024, 256),
}
for i in range(5):
    for suffix, shape in {
        "input_layernorm.weight": (2304,),
        "attention_conv.base_kernel": (2, 2, 2304),
        "attention_conv.kernel_projection.weight": (576, 2304),
        "self_attn.q_proj.weight": (4096, 2304),
        "self_attn.k_proj.weight": (1024, 2304),
        "self_attn.v_proj.weight": (1024, 2304),
        "self_attn.o_proj.weight": (2304, 4096),
        "self_attn.q_norm.weight": (128,),
        "self_attn.k_norm.weight": (128,),
        "post_attention_layernorm.weight": (2304,),
        "mlp_conv.base_kernel": (2, 2, 2304),
        "mlp_conv.kernel_projection.weight": (576, 2304),
        "mlp.gate_proj.weight": (6912, 2304),
        "mlp.up_proj.weight": (6912, 2304),
        "mlp.down_proj.weight": (2304, 6912),
    }.items():
        EXPECTED[f"layers.{i}.{suffix}"] = shape


def synthetic_weights():
    rng = torch.Generator().manual_seed(141)
    weights = {}
    for name, shape in EXPECTED.items():
        tensor = torch.randn(shape, generator=rng, dtype=torch.bfloat16)
        if "norm.weight" in name:
            tensor.mul_(0.02).add_(1)
        elif name.endswith(".base_kernel"):
            tensor.mul_(0.15)
            tensor[:, 0].add_(0.6)
        elif name.endswith("_codebook"):
            tensor.mul_(0.2)
        else:
            tensor.mul_(0.35 / shape[-1] ** 0.5)
        weights[name] = tensor
    return weights


def model_with(weights):
    model = DFlash2DraftModel(device="meta")
    model.load_state_dict(weights, assign=True)
    return model


class DFlash2Parity(unittest.TestCase):
    real_dir = None
    target_dir = None
    stages = ("selector", "full")

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.weights = synthetic_weights()

    @torch.no_grad()
    def compare_oracle(self, model, weights, head=None):
        rng = torch.Generator().manual_seed(708)
        lengths = (0, 2, 7, 11)
        ctx = torch.randn(4, max(lengths), 4, 2304, generator=rng, dtype=torch.bfloat16)
        block = torch.randn(4, 8, 2304, generator=rng, dtype=torch.bfloat16)
        mask = torch.arange(ctx.shape[1]) < torch.tensor(lengths)[:, None]
        ctx_pos = torch.arange(ctx.shape[1]).expand(4, -1).clone()
        block_pos = torch.arange(8).expand(4, -1).clone()
        block_pos += torch.tensor([0, 15, 40, 2048])[:, None]
        ctx_pos[1, :2] += 13
        ctx_pos[2, :7] += 33
        # The last example straddles the strict lower bound at query position 2048.
        ctx_pos[3, 3:] += 2037
        if head is None:
            head = torch.randn(128, 2304, generator=rng, dtype=torch.bfloat16) * 0.05
        anchors = torch.tensor([1, 2, 3, 4])
        hidden_batch = model.forward_backbone(ctx, mask, ctx_pos, block, block_pos)
        for i, length in enumerate(lengths):
            inputs = {"taps": ctx[i, :length], "block_embeds": block[i], "head": head,
                      "anchor": anchors[i:i + 1], "positions": torch.cat((ctx_pos[i, :length], block_pos[i]))}
            out = reference.draft(weights, inputs)
            hidden_single = model.forward_backbone(
                ctx[i:i + 1, :length], mask[i:i + 1, :length], ctx_pos[i:i + 1, :length],
                block[i:i + 1], block_pos[i:i + 1],
            )[0]
            self.assertLess((hidden_single.float() - out["hidden"].float()).abs().max().item(), 1e-4)
            self.assertLess((hidden_batch[i].float() - out["hidden"].float()).abs().max().item(), 2e-2)

            unary = F.linear(hidden_single, head).float()
            values, ids = unary.topk(16, dim=-1)
            self.assertTrue(torch.equal(ids, out["candidates"]))
            predecessors = torch.cat((anchors[i].expand(1, 16), ids[:-1]))
            lattice = model.candidate_selector(hidden_single[None], values[None], ids[None], predecessors[None])[0]
            self.assertLess((lattice - out["lattice"]).abs().max().item(), 5e-2)
            idx = 0
            walked = []
            for edge in range(7):
                idx = int(lattice[edge, idx].argmax())
                walked.append(ids[edge, idx])
            self.assertTrue(torch.equal(torch.stack(walked), out["tokens"]))
            path = torch.cat((anchors[i:i + 1], out["tokens"][:-1]))
            reached_ids, scores = model.score_selector_reached(hidden_batch[i:i + 1], unary[None], path[None])
            rows = torch.cat((torch.zeros(1, dtype=torch.long), (ids[:-1] == path[1:, None]).long().argmax(-1)))
            self.assertTrue(torch.equal(reached_ids[0], ids))
            torch.testing.assert_close(scores[0], out["lattice"][torch.arange(7), rows], rtol=0, atol=5e-2)

    def test_oracle(self):
        self.compare_oracle(model_with(self.weights), self.weights)

    def test_real_oracle(self):
        if self.real_dir is None:
            self.skipTest("provide --dflash2-dir to exercise real weights")
        model = DFlash2DraftModel.from_checkpoint(self.real_dir)
        head = load_target_head_and_embed(self.target_dir, "cpu")[2] if self.target_dir else None
        self.compare_oracle(model, model.state_dict(), head)

    def test_gradients(self):
        rng = torch.Generator().manual_seed(17)
        ctx = torch.randn(1, 2, 4, 2304, generator=rng, dtype=torch.bfloat16)
        mask = torch.ones(1, 2, dtype=torch.bool)
        pos = torch.arange(2)[None]
        ids = torch.arange(8)[None]
        block_pos = torch.arange(2, 10)[None]
        pred = torch.arange(7)[None]
        teacher = torch.randn(1, 7, 2304, generator=rng, dtype=torch.bfloat16)
        embed = torch.randn(128, 2304, generator=rng, dtype=torch.bfloat16)
        head = torch.randn(128, 2304, generator=rng, dtype=torch.bfloat16) * 0.05
        for stage in self.stages:
            for freeze in (False, True):
                with self.subTest(stage=stage, freeze_codebooks=freeze):
                    model = model_with(self.weights)
                    groups = configure_stage(model, stage, freeze_codebooks=freeze)
                    rows = loss_rows(
                        model, (ctx, mask, pos, ids, block_pos, pred, teacher),
                        embed, torch.ones(2304), head, head.float(), stage == "full", 1.0,
                        loss_mode="tv_accept", grad_checkpoint=stage == "full",
                    )[0]
                    self.assertTrue(torch.isfinite(rows).all().item())
                    rows.mean().backward()
                    group_for = {id(parameter): (group["name"], group["weight_decay"])
                                 for group in groups for parameter in group["params"]}
                    self.assertEqual(len(group_for), sum(len(group["params"]) for group in groups))
                    for name, parameter in model.named_parameters():
                        selector = name.startswith("candidate_selector.")
                        codebook = name.endswith("_codebook")
                        if (stage == "selector" and not selector) or (freeze and codebook):
                            self.assertFalse(parameter.requires_grad, name)
                            self.assertIsNone(parameter.grad, name)
                            self.assertNotIn(id(parameter), group_for)
                        else:
                            component = "selector" if selector else "backbone"
                            decay = 0.0 if parameter.ndim == 1 or codebook else 0.01
                            self.assertEqual(group_for[id(parameter)], (component, decay), name)
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertEqual(parameter.grad.dtype, torch.float32)
                            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                            self.assertGreater(parameter.grad.abs().max().item(), 0, name)
                    del model, groups, rows, parameter

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source", Path(directory) / "dest"
            source.mkdir()
            (source / "config.json").write_text("{}\n")
            (source / "README.md").write_text("DFlash2 test\n")
            model_with(self.weights).save_checkpoint(dest, source)
            with safe_open(dest / "model.safetensors", framework="pt") as saved:
                self.assertEqual(len(saved.keys()), 81)
                self.assertEqual(set(saved.keys()), set(EXPECTED))
                for name, shape in EXPECTED.items():
                    view = saved.get_slice(name)
                    self.assertEqual(view.get_shape(), list(shape), name)
                    self.assertEqual(view.get_dtype(), "BF16", name)
            loaded = DFlash2DraftModel.from_checkpoint(dest).state_dict()
            for name in EXPECTED:
                self.assertTrue(torch.equal(loaded[name], self.weights[name]), name)
            self.assertEqual((dest / "config.json").read_text(), "{}\n")
            self.assertEqual((dest / "README.md").read_text(), "DFlash2 test\n")


class AnchorAlignment(unittest.TestCase):
    def test_boundaries_and_reached_prefix(self):
        tokens = np.arange(14, dtype=np.int32)
        taps = torch.arange(14, dtype=torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
        taps = np.broadcast_to(taps[:, None, None], (14, 4, 2304))
        anchors = Anchors([Sequence(tokens, taps, 2), Sequence(tokens + 40, taps, 6)])
        self.assertEqual(len(anchors), 6)
        ctx, mask, pos, ids, block, pred, teacher = anchors.batch([0, 4, 5], 3, "cpu")
        self.assertEqual(mask.tolist(), [[True, True, False], [True] * 3, [True] * 3])
        self.assertEqual(pos.tolist(), [[0, 1, 0], [3, 4, 5], [3, 4, 5]])
        self.assertEqual(ctx[:, :, 0, 0].tolist(), [[0, 1, 0], [3, 4, 5], [3, 4, 5]])
        self.assertEqual(ids[:, 0].tolist(), [2, 6, 46])
        self.assertTrue((ids[:, 1:] == 200056).all())
        self.assertEqual(block[:, 0].tolist(), [2, 6, 6])
        self.assertEqual(pred.tolist(), [list(range(2, 9)), list(range(6, 13)), list(range(46, 53))])
        self.assertEqual(teacher[:, :, 0].tolist(), [list(range(2, 9)), list(range(6, 13)), list(range(6, 13))])


    def test_memmap_sequence_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = 24
            np.arange(rows, dtype="<i4").tofile(root / "shard_0000.tokens.bin")
            np.zeros((rows, 4, 2304), dtype="<u2").tofile(root / "shard_0000.taps.bin")
            meta = {
                "shard": 0, "num_tokens": rows, "num_seqs": 2, "hidden": 2304, "num_taps": 4,
                "seqs": [{"offset": 0, "length": 12, "prompt_len": 2},
                         {"offset": 12, "length": 12, "prompt_len": 4}],
            }
            (root / "shard_0000.meta.json").write_text(json.dumps(meta))
            sequences = train.load_sequences(root)
            self.assertTrue(all(isinstance(sequence.taps, np.memmap) for sequence in sequences))
            training, held_out = train.split_sequences(sequences, 0.05, 42)
            self.assertIsNot(training[0], held_out[0])
            self.assertIn((training[0].tokens[0], held_out[0].tokens[0]), ((0, 12), (12, 0)))
            again = train.split_sequences(sequences, 0.05, 42)
            self.assertIs(again[1][0], held_out[0])


class CalibrationMetrics(unittest.TestCase):
    def test_selector_kl_and_optional_cross_entropy(self):
        ids = torch.zeros(1, 8, dtype=torch.long)
        head = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
        teacher = torch.tensor([1.0, 2.0], dtype=torch.bfloat16).expand(1, 7, 2)
        top_ids = torch.arange(2).expand(1, 7, 2)
        scores = torch.tensor([0.25, -0.5]).expand(1, 7, 2)
        model = SimpleNamespace(forward_backbone=lambda *_: torch.zeros_like(teacher),
                                score_selector_reached=lambda *_, top_k: (top_ids, scores))
        batch = (None, None, None, ids, None, None, teacher)
        weights = (head, torch.ones(2), head, head.float())
        loss, target, *_ = loss_rows(model, batch, *weights, False, 1.0, loss_mode="kl", temps=(0.25,))
        expected = torch.zeros_like(loss)
        for temperature, coefficient in ((0.8, 0.5 * 0.8 ** 2), (1.0, 0.5)):
            expected += coefficient * F.kl_div(
                (scores / temperature).log_softmax(-1),
                (target / temperature).softmax(-1), reduction="none",
            ).sum(-1)
        torch.testing.assert_close(loss, expected)
        with_ce, *_ = loss_rows(model, batch, *weights, False, 1.0, loss_mode="kl", ce_weight=0.2)
        ce = F.cross_entropy(scores.transpose(1, 2), target.argmax(-1), reduction="none")
        torch.testing.assert_close(with_ce, expected + 0.2 * ce)

    def test_tv_accept_improves_overlap_and_loss(self):
        head = torch.zeros(40, 2, dtype=torch.bfloat16)
        head[0, 0] = 5.0
        teacher = torch.tensor([1.0, 0.0], dtype=torch.bfloat16).expand(1, 7, 2)
        top_ids = torch.arange(TOP_K).expand(1, 7, TOP_K)
        logits = torch.nn.Parameter(torch.zeros(TOP_K))

        def score(*_, top_k):
            self.assertEqual(top_k, TOP_K)
            return top_ids, logits.expand_as(top_ids)

        model = SimpleNamespace(forward_backbone=lambda *_: torch.zeros_like(teacher), score_selector_reached=score)
        batch = (None, None, None, torch.zeros(1, 8, dtype=torch.long), None, None, teacher)
        weights = (head, torch.ones(2), head, head.float())
        optimizer = torch.optim.SGD([logits], lr=1.0)
        previous_loss = previous_overlap = None
        for step in range(5):
            optimizer.zero_grad()
            rows, target, _, scores = loss_rows(model, batch, *weights, False, 1.0)
            expected = []
            overlaps = []
            for temperature in (0.25, 0.8, 1.0):
                p = (target / temperature).softmax(-1)
                p_top = p.gather(-1, top_ids)
                overlap = torch.minimum(p_top, (scores / temperature).softmax(-1)).sum(-1)
                overlaps.append(overlap.detach())
                survival = torch.stack([overlap[:, :depth].prod(-1) for depth in range(1, 8)], dim=-1)
                expected.append(p_top.sum(-1) - overlap - survival.mean(-1, keepdim=True))
            torch.testing.assert_close(rows, torch.stack(expected).mean(0))
            if step == 0:
                full_rows = loss_rows(model, batch, *weights, True, 0.7, temps=(0.25,), ce_weight=0.2)[0]
                p = (target / 0.8).softmax(-1)
                unary_tv = 0.5 * (p - 1 / p.shape[-1]).abs().sum(-1)
                ce = F.cross_entropy(scores.transpose(1, 2), target.gather(-1, top_ids).argmax(-1), reduction="none")
                torch.testing.assert_close(full_rows, expected[0] + 0.7 * unary_tv + 0.2 * ce)
                reference_grad = torch.autograd.grad(torch.stack(expected).mean(), logits, retain_graph=True)[0]
            loss = rows.mean()
            overlap = torch.stack(overlaps)
            if previous_loss is not None:
                self.assertLess(loss.item(), previous_loss)
                self.assertTrue((overlap > previous_overlap).all().item())
            previous_loss, previous_overlap = loss.item(), overlap
            loss.backward()
            if step == 0:
                torch.testing.assert_close(logits.grad, reference_grad)
                self.assertGreater(logits.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_teacher_bf16_is_opt_in(self):
        rng = torch.Generator().manual_seed(83)
        head = torch.randn(32, 8, generator=rng, dtype=torch.bfloat16)
        teacher = torch.randn(1, 7, 8, generator=rng, dtype=torch.bfloat16)
        model = SimpleNamespace(
            forward_backbone=lambda *_: torch.zeros_like(teacher),
            score_selector_reached=lambda _hidden, unary, _pred, top_k: tuple(reversed(unary.topk(top_k))),
        )
        batch = (None, None, None, torch.zeros(1, 8, dtype=torch.long), None, None, teacher)
        target = loss_rows(model, batch, head, torch.ones(8), head, head.float(), False, 1.0)[1]
        fast_target = loss_rows(model, batch, head, torch.ones(8), head, None, False, 1.0, teacher_bf16=True)[1]
        torch.testing.assert_close(target, train.softcap(F.linear(teacher.float(), head.float())), rtol=0, atol=0)
        torch.testing.assert_close(fast_target, train.softcap(F.linear(teacher, head).float()), rtol=0, atol=0)
        self.assertFalse(torch.equal(target, fast_target))

    def test_survival_and_greedy_miss_denominator(self):
        # Uniform teacher over twice the proposal's support, so overlap is 0.5.
        target = torch.zeros(1, 7, 2 * TOP_K)
        top_ids = torch.arange(TOP_K).repeat(1, 7, 1)
        top_ids[0, 1, :2] = torch.tensor([1, 0])
        top_ids[0, 2:] = torch.arange(2, TOP_K + 2)
        scores = torch.zeros(1, 7, TOP_K)
        loss = torch.arange(7, dtype=torch.float32)[None]
        args = SimpleNamespace(batch_size=1, ctx_len=0, stage="selector", unary_weight=1.0,
                               loss_mode="tv_accept", temps=(0.25, 0.8, 1.0), ce_weight=0.0, teacher_bf16=True)
        anchors = SimpleNamespace(batch=lambda *_: None)
        with patch.object(train, "loss_rows", return_value=(loss, target, top_ids, scores)) as compute_loss:
            metrics = train.validate(torch.nn.Module(), anchors, [0], args, "cpu", (), torch.ones(7))
        self.assertEqual(compute_loss.call_args.kwargs["temps"], args.temps)
        self.assertTrue(compute_loss.call_args.kwargs["teacher_bf16"])
        self.assertEqual(metrics["val_loss"], 3.0)
        self.assertEqual(metrics["top16_mass_t08"], 0.5)
        self.assertEqual(metrics["overlap_t08"], 0.5)
        self.assertEqual(metrics["calib_gap_t08"], 0.0)
        self.assertEqual(metrics["overlap_t10"], 0.5)
        self.assertEqual(metrics["calib_gap_t10"], 0.0)
        self.assertEqual(metrics["greedy_acc_p0"], 1.0)
        self.assertAlmostEqual(metrics["greedy_acc_mean"], 1 / 7)
        self.assertAlmostEqual(metrics["greedy_miss_in_top16_pct"], 100 / 6)
        self.assertEqual(metrics["pred_alpha_p1_t0"], 2.0)
        expected_alpha = 1 + sum(0.5 ** k for k in range(1, 8))
        self.assertAlmostEqual(metrics["pred_alpha_p1_t08"], expected_alpha)
        self.assertAlmostEqual(metrics["pred_alpha_p1_t10"], expected_alpha)


class RolloutExclusions(unittest.TestCase):
    def test_speedbench_tools_do_not_exclude_training_prompts(self):
        original = collect_rollouts.build_bank([], 0.72, random.Random(42))
        speedbench_prompt, jsonl_prompt, bare_prompt, tool_output = (prompt for _, prompt in original[:4])
        speedbench = {"items": [{
            "name": "tool benchmark",
            "messages": [
                {"role": "system", "content": "Use available tools when appropriate."},
                {"role": "user", "content": [{"type": "text", "text": speedbench_prompt}]},
                {"role": "tool", "content": tool_output},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "open_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "enum": ["a", "the", "de", "un"]}},
                        "required": ["path"],
                        "examples": [{"prompt": tool_output}],
                    },
                },
            }],
            "meta": {"tags": ["a", "string"], "text": tool_output},
        }]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark, audit = root / "speedbench-v1.json", root / "prompts.jsonl"
            benchmark.write_text(json.dumps(speedbench), encoding="utf-8")
            audit.write_text("\n".join(map(json.dumps, (
                {"name": "event audit", "prompt": "Please " + jsonl_prompt},
                bare_prompt.upper(),
                "a",
            ))), encoding="utf-8")
            exclusions = collect_rollouts.load_exclusions([benchmark, audit])
        normalize = collect_rollouts.normalize
        self.assertEqual(set(exclusions), {
            normalize(speedbench_prompt), normalize("Please " + jsonl_prompt), normalize(bare_prompt),
            "use available tools when appropriate", "a",
        })
        bank = {prompt for _, prompt in collect_rollouts.build_bank(exclusions, 0.72, random.Random(42))}
        self.assertGreaterEqual(len(bank), 432)
        self.assertTrue({speedbench_prompt, jsonl_prompt, bare_prompt}.isdisjoint(bank))
        self.assertIn(tool_output, bank)

    def test_short_prompts_match_only_exactly(self):
        domains = {"code": (("a", "b"), ("Open file {topic}.",))}
        with patch.object(collect_rollouts, "DOMAINS", domains), patch.object(collect_rollouts, "MULTILINGUAL", ()):
            bank = collect_rollouts.build_bank(["a", "open file a"], 0.5, random.Random(42))
        self.assertEqual(bank, [("code", "Open file b.")])


class RolloutCollection(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.output = self.root / "rollouts.jsonl"
        exclusions = self.root / "excluded.json"
        exclusions.write_text(json.dumps([
            "Repair café inventory display",
            "Summarize the tidal marsh survey",
            "Calculate annual garden yield from monthly notes",
        ]))
        self.argv = ["--output", str(self.output), "--exclude-files", str(exclusions)]
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    @staticmethod
    def rollout(plan, _args):
        domain, prompt, *_ = plan
        return {"prompt": prompt, "domain": domain}, None

    def rows(self):
        return [json.loads(line) for line in self.output.read_text().splitlines()]

    def test_external_files_share_leakage_filter_and_precede_builtin(self):
        source = self.root / "train.json"
        source.write_text(json.dumps([
            {"prompt": "REPAIR CAFÉ inventory display!"},
            "Please summarize the tidal marsh survey accurately.",
            {"instruction": "From monthly notes calculate garden yield minus annual waste."},
            {"prompt": "Explain chromatic aberration.", "domain": "optics"},
            "Create a recipe for lentil bread.",
        ]))
        turns = self.root / "train.jsonl"
        turns.write_text("\n".join(map(json.dumps, (
            {"instruction": "Paint a thistle using watercolors."},
            {"prompt": "Explain chromatic aberration"},
            {"messages": [
                {"role": "system", "content": "Act as a youth sports coach."},
                {"role": "user", "content": "Plan a bicycle safety lesson."},
                {"role": "assistant", "content": "Start with a helmet."},
            ]},
        ))))
        argv = [*self.argv, "--prompt-files", str(source), "--prompt-files", str(turns),
                "--jaccard", str(7 / 9)]
        external = [
            ("optics", "Explain chromatic aberration."),
            ("external", "Create a recipe for lentil bread."),
            ("external", "Paint a thistle using watercolors."),
            ("external", "Plan a bicycle safety lesson."),
        ]
        with patch.object(collect_rollouts, "request_rollout", side_effect=self.rollout) as request:
            self.assertEqual(collect_rollouts.main([*argv, "--no-builtin-bank", "--count", "4"]), 0)
            self.assertEqual(request.call_count, 4)
            self.assertEqual([(row["domain"], row["prompt"]) for row in self.rows()], external)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                collect_rollouts.main([*argv, "--no-builtin-bank", "--count", "5"])
            self.assertEqual(request.call_count, 4)
            self.assertEqual(collect_rollouts.main([*argv, "--count", "5"]), 0)
        rows = self.rows()
        self.assertEqual([(row["domain"], row["prompt"]) for row in rows[:4]], external)
        self.assertIn(rows[4]["domain"], collect_rollouts.DOMAINS)

    def test_resume_skips_normalized_prompts_and_backfills_failures(self):
        source = self.root / "train.json"
        prompts = [
            "Check the water pressure.",
            "Write a failed completion.",
            "Describe a brass telescope.",
            "Sort rocks by density.",
            "Place a spare prompt here.",
        ]
        source.write_text(json.dumps(prompts))
        original = json.dumps({"prompt": "CHECK THE WATER PRESSURE!", "response": "Done."})
        self.output.write_text(original)
        argv = [*self.argv, "--prompt-files", str(source), "--resume", "--no-builtin-bank", "--count", "3"]

        def request(plan, args):
            if plan[1] == prompts[1]:
                return None, "test failure"
            return self.rollout(plan, args)

        with (
            patch.object(collect_rollouts, "request_rollout", side_effect=request) as collect,
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(collect_rollouts.main(argv), 0)
            self.assertEqual(collect.call_count, 3)
            self.assertIn("test failure", errors.getvalue())
            self.assertTrue(self.output.read_text().startswith(original + "\n"))
            self.assertEqual([row["prompt"] for row in self.rows()], ["CHECK THE WATER PRESSURE!", *prompts[2:4]])
            before = self.output.read_bytes()
            self.assertEqual(collect_rollouts.main(argv), 0)
            self.assertEqual(collect.call_count, 3)
            self.assertEqual(self.output.read_bytes(), before)

    def test_mix_parsing_and_payload(self):
        _, defaults = collect_rollouts.parse_args(self.argv)
        self.assertEqual(defaults.temp_mix, ((0.0, 0.25), (0.8, 0.50), (1.0, 0.25)))
        self.assertEqual(defaults.effort_mix, (("off", 0.20), ("low", 0.40), ("high", 0.40)))
        for flag, values in (
            ("--temp-mix", ("", "nan:1", "-0.1:1", "2.1:1", "0.8:inf", "0.8:-1", "0.8:0", "0.8:1,0.80:1")),
            ("--effort-mix", ("", "medium:1", "off:nan", "off:0,low:0", "low:1,low:2")),
        ):
            for value in values:
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        collect_rollouts.parse_args([*self.argv, f"{flag}={value}"])
                    self.assertEqual(error.exception.code, 2)
        source = self.root / "train.json"
        source.write_text(json.dumps(["Invent a name for a constellation."]))
        for temperature, effort, wire_effort in ((0.0, "off", "none"), (0.8, "low", "low"), (1.0, "high", "high")):
            with self.subTest(temperature=temperature, effort=effort):
                body = json.dumps({"choices": [{"message": {"content": "The Crane"}, "finish_reason": "stop"}]})
                with patch.object(collect_rollouts, "urlopen", return_value=io.BytesIO(body.encode())) as open_url:
                    self.assertEqual(collect_rollouts.main([
                        *self.argv, "--prompt-files", str(source), "--no-builtin-bank", "--count", "1",
                        "--temp-mix", f"{temperature}:2", "--effort-mix", f"{effort}:3",
                    ]), 0)
                payload = json.loads(open_url.call_args.args[0].data)
                self.assertEqual(payload["temperature"], temperature)
                self.assertEqual(payload["enable_thinking"], effort != "off")
                self.assertEqual(payload["chat_template_kwargs"], {"reasoning_effort": wire_effort})
                self.assertEqual((payload["top_k"], payload["top_p"], payload["max_tokens"]), (0, 1.0, 1024))
                self.assertIsInstance(payload["seed"], int)
                row, = self.rows()
                self.assertEqual(row["reasoning_effort"], wire_effort)
                self.assertEqual(row["thinking"], effort != "off")
                self.assertEqual(row["temperature"], temperature)


class TrainingArguments(unittest.TestCase):
    argv = ["--stage", "selector", "--data-dir", "unused", "--model-dir", "unused",
            "--dflash2-dir", "draft", "--out-dir", "draft/stage/.."]

    def test_warmup_and_cosine_edges(self):
        self.assertEqual([train.lr_scale(step, 3, 0) for step in (1, 3)], [1.0, 0.1])
        self.assertAlmostEqual(train.lr_scale(2, 3, 0), 0.55)
        self.assertEqual([train.lr_scale(step, 4, 4) for step in range(1, 5)], [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(train.lr_scale(1, 1, 0), 1.0)
        self.assertEqual(train.lr_scale(1, 1, 1), 1.0)
        self.assertEqual(train.lr_scale(2, 4, 2), 1.0)
        self.assertEqual(train.lr_scale(4, 4, 2), 0.1)

    def test_freeze_codebooks(self):
        for stage in ("selector", "full"):
            with self.subTest(stage=stage):
                model = DFlash2DraftModel(device="meta")
                groups = configure_stage(model, stage, freeze_codebooks=True)
                optimized = {id(p) for group in groups for p in group["params"]}
                for codebook in (model.candidate_selector.predecessor_codebook,
                                 model.candidate_selector.successor_codebook):
                    self.assertFalse(codebook.requires_grad)
                    self.assertNotIn(id(codebook), optimized)
                self.assertIn(id(model.candidate_selector.hidden_projection.weight), optimized)

    def test_defaults_and_checkpoint_path_guard(self):
        with patch.object(sys, "argv", ["train.py", *self.argv]):
            args = train.parse_args()
            self.assertEqual(args.ctx_len, 2048)
            self.assertEqual(args.lr_selector, 5e-5)
            self.assertEqual(args.lr_backbone, 2e-5)
            self.assertEqual(args.warmup_frac, 0.10)
            self.assertEqual(args.ce_weight, 0.0)
            self.assertEqual(args.loss_mode, "tv_accept")
            self.assertEqual(args.temps, (0.25, 0.8, 1.0))
            self.assertIsNone(args.lr_codebook)
            self.assertFalse(args.teacher_bf16)
            self.assertFalse(args.freeze_codebooks)
            self.assertFalse(args.grad_checkpoint)
            with self.assertRaisesRegex(ValueError, "--out-dir must differ from --dflash2-dir"):
                train.main()
        args = train.parse_args([*self.argv, "--stage", "full", "--loss-mode", "kl"])
        self.assertEqual(args.lr_selector, 2e-5)
        self.assertEqual(args.loss_mode, "kl")

    def test_temperatures_and_teacher_flag(self):
        args = train.parse_args([*self.argv, "--temps", "0.25, 0.8,1.0", "--teacher-bf16"])
        self.assertEqual(args.temps, (0.25, 0.8, 1.0))
        self.assertTrue(args.teacher_bf16)
        self.assertEqual(train.parse_args([*self.argv, "--temps", "0.5"]).temps, (0.5,))
        for invalid in ("", "0", "-0.1", "inf", "nan", "1.0,,0.8", "0.25,"):
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as error:
                    train.parse_args([*self.argv, f"--temps={invalid}"])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--temps", errors.getvalue())

    def test_codebook_learning_rate(self):
        for stage in ("selector", "full"):
            with self.subTest(stage=stage):
                args = train.parse_args([*self.argv, "--stage", stage, "--lr-codebook", "3e-6"])
                model = DFlash2DraftModel(device="meta")
                groups = configure_stage(model, args.stage, lr_codebook=args.lr_codebook)
                optimizer = torch.optim.AdamW(groups, lr=args.lr_selector)
                codebooks, = [group for group in optimizer.param_groups if group["name"] == "codebook"]
                self.assertEqual(codebooks["lr"], 3e-6)
                self.assertEqual(codebooks["weight_decay"], 0.0)
                self.assertEqual({id(parameter) for parameter in codebooks["params"]},
                                 {id(parameter) for name, parameter in model.named_parameters()
                                  if name.endswith("_codebook")})
                for group in optimizer.param_groups:
                    if group["name"] == "selector":
                        self.assertEqual(group["lr"], args.lr_selector)
        for options in (("--freeze-codebooks", "--lr-codebook", "1e-5"),
                        *((f"--lr-codebook={value}",) for value in ("0", "-0.1", "nan", "inf"))):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as error:
                    train.parse_args([*self.argv, *options])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--lr-codebook", errors.getvalue())

    def test_best_and_latest_checkpoints(self):
        class TinyDraft(torch.nn.Module):
            save_checkpoint = DFlash2DraftModel.save_checkpoint

            def __init__(self):
                super().__init__()
                self.candidate_selector = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.candidate_selector.weight.fill_(0.5)

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text("{}\n")
            seq = Sequence(np.arange(8), np.zeros((8, 4, 2304), dtype=np.uint16), 0)
            baseline = {"pred_alpha_p1_t0": 3.0, "pred_alpha_p1_t08": 3.0, "pred_alpha_p1_t10": 3.0}
            trained = {"pred_alpha_p1_t0": 3.1, "pred_alpha_p1_t08": 3.1, "pred_alpha_p1_t10": 2.7}
            for steps in (0, 1):
                with self.subTest(steps=steps):
                    model = TinyDraft()
                    output = root / f"out-{steps}"
                    argv = ["train.py", *self.argv, "--dflash2-dir", str(source), "--out-dir", str(output),
                            "--steps", str(steps), "--device", "cpu", "--lr-selector", "0.1", "--teacher-bf16"]
                    head = torch.zeros(1, dtype=torch.bfloat16)

                    def loss(draft, *_args, **kwargs):
                        self.assertTrue(kwargs["teacher_bf16"])
                        return (draft.candidate_selector.weight.square(),)

                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(train.DFlash2DraftModel, "from_checkpoint", return_value=model),
                        patch.object(train, "load_sequences", return_value=[seq, seq]),
                        patch.object(train, "load_target_head_and_embed", return_value=(None, None, head)),
                        patch.object(train, "loss_rows", side_effect=loss),
                        patch.object(train, "validate", side_effect=[baseline, trained]) as validate,
                    ):
                        train.main()
                    self.assertIsNone(validate.call_args.args[5][3])
                    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual(rows[0]["score"], 9.0)
                    self.assertTrue(rows[0]["best"])
                    latest = output / "latest"
                    self.assertEqual(latest.is_dir(), steps > 0)
                    if steps > 0:
                        self.assertAlmostEqual(rows[1]["score"], 8.9)
                        self.assertFalse(rows[1]["best"])
                        with (
                            safe_open(output / "model.safetensors", framework="pt") as best_checkpoint,
                            safe_open(latest / "model.safetensors", framework="pt") as latest_checkpoint,
                        ):
                            name = "candidate_selector.weight"
                            self.assertLess(latest_checkpoint.get_tensor(name).item(),
                                            best_checkpoint.get_tensor(name).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dflash2-dir", type=Path, help="Optional real draft checkpoint")
    parser.add_argument("--model-dir", type=Path, help="Optional real target head for the real draft test")
    parser.add_argument("--stage", choices=("selector", "full"), help="Gradient stage (default tests both)")
    args, remaining = parser.parse_known_args()
    DFlash2Parity.real_dir = args.dflash2_dir
    DFlash2Parity.target_dir = args.model_dir
    if args.stage:
        DFlash2Parity.stages = (args.stage,)
    unittest.main(argv=[sys.argv[0], *remaining])
