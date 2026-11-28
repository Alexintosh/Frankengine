#!/usr/bin/env python3
"""CPU parity, gradient, and checkpoint tests for the DFlash2 pilot."""

import argparse
import importlib.util
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
    from .model import DFlash2DraftModel, load_target_head_and_embed
    from . import collect_rollouts, train
    from .train import Anchors, Sequence, configure_stage, loss_rows
else:
    from model import DFlash2DraftModel, load_target_head_and_embed
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
            with self.subTest(stage=stage):
                model = model_with(self.weights)
                groups = configure_stage(model, stage)
                rows, *_ = loss_rows(model, (ctx, mask, pos, ids, block_pos, pred, teacher),
                                     embed, torch.ones(2304), head, head.float(), stage == "full", 1.0,
                                     grad_checkpoint=stage == "full")
                rows.mean().backward()
                group_for = {id(parameter): (group["name"], group["weight_decay"])
                             for group in groups for parameter in group["params"]}
                self.assertEqual(len(group_for), sum(len(group["params"]) for group in groups))
                for name, parameter in model.named_parameters():
                    if stage == "selector" and not name.startswith("candidate_selector."):
                        self.assertFalse(parameter.requires_grad)
                        self.assertIsNone(parameter.grad)
                        self.assertNotIn(id(parameter), group_for)
                    else:
                        component = "selector" if name.startswith("candidate_selector.") else "backbone"
                        decay = 0.0 if parameter.ndim == 1 or name.endswith("_codebook") else 0.01
                        self.assertEqual(group_for[id(parameter)], (component, decay), name)
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertEqual(parameter.grad.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                        self.assertGreater(parameter.grad.abs().max().item(), 0, name)
                del model, groups, rows, parameter, _

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
                                score_selector_reached=lambda *_: (top_ids, scores))
        batch = (None, None, None, ids, None, None, teacher)
        weights = (head, torch.ones(2), head, head.float())
        loss, target, *_ = loss_rows(model, batch, *weights, False, 1.0)
        expected = torch.zeros_like(loss)
        for temperature, coefficient in ((0.8, 0.5 * 0.8 ** 2), (1.0, 0.5)):
            expected += coefficient * F.kl_div(
                (scores / temperature).log_softmax(-1),
                (target / temperature).softmax(-1), reduction="none",
            ).sum(-1)
        torch.testing.assert_close(loss, expected)
        with_ce, *_ = loss_rows(model, batch, *weights, False, 1.0, ce_weight=0.2)
        ce = F.cross_entropy(scores.transpose(1, 2), target.argmax(-1), reduction="none")
        torch.testing.assert_close(with_ce, expected + 0.2 * ce)

    def test_survival_and_greedy_miss_denominator(self):
        # The teacher is uniform on four tokens; the proposal has two candidates.
        # At T=0.8 the proposal is uniform there, giving an overlap of 0.5 on every edge.
        target = torch.zeros(1, 7, 4)
        top_ids = torch.tensor([[[0, 1], [1, 0]] + [[2, 3]] * 5])
        scores = torch.zeros(1, 7, 2)
        loss = torch.arange(7, dtype=torch.float32)[None]
        args = SimpleNamespace(batch_size=1, ctx_len=0, stage="selector", unary_weight=1.0, ce_weight=0.0)
        anchors = SimpleNamespace(batch=lambda *_: None)
        with patch.object(train, "loss_rows", return_value=(loss, target, top_ids, scores)):
            metrics = train.validate(torch.nn.Module(), anchors, [0], args, "cpu", (), torch.ones(7))
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
        self.assertAlmostEqual(metrics["pred_alpha_p1_t08"], 1 + sum(0.5 ** k for k in range(1, 8)))


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


class TrainingArguments(unittest.TestCase):
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
        argv = ["train.py", "--stage", "selector", "--data-dir", "unused", "--model-dir", "unused",
                "--dflash2-dir", "draft", "--out-dir", "draft/stage/.."]
        with patch.object(sys, "argv", argv):
            args = train.parse_args()
            self.assertEqual(args.ctx_len, 2048)
            self.assertEqual(args.lr_selector, 5e-5)
            self.assertEqual(args.lr_backbone, 2e-5)
            self.assertEqual(args.warmup_frac, 0.10)
            self.assertEqual(args.ce_weight, 0.0)
            self.assertFalse(args.freeze_codebooks)
            self.assertFalse(args.grad_checkpoint)
            with self.assertRaisesRegex(ValueError, "--out-dir must differ from --dflash2-dir"):
                train.main()
        with patch.object(sys, "argv", [*argv, "--stage", "full"]):
            self.assertEqual(train.parse_args().lr_selector, 2e-5)


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
