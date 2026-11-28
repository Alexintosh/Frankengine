#!/usr/bin/env python3
"""Distill a DFlash2 checkpoint on generated sequences and their native target taps."""

import argparse
import json
import math
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

if __package__:
    from .model import (
        BLOCK, GAMMA, HIDDEN, MASK_TOKEN_ID, NUM_TAPS, TOP_K, WINDOW,
        DFlash2DraftModel, embed_tokens, load_target_head_and_embed, softcap,
    )
else:
    from model import (
        BLOCK, GAMMA, HIDDEN, MASK_TOKEN_ID, NUM_TAPS, TOP_K, WINDOW,
        DFlash2DraftModel, embed_tokens, load_target_head_and_embed, softcap,
    )


@dataclass(frozen=True)
class Sequence:
    tokens: np.ndarray
    taps: np.ndarray
    prompt_len: int

    @property
    def anchors(self):
        return len(self.tokens) - self.prompt_len - BLOCK + 1


def load_sequences(data_dir):
    sequences = []
    for path in sorted(Path(data_dir).glob("shard_*.meta.json")):
        meta = json.loads(path.read_text())
        rows = meta["num_tokens"]
        if (meta["hidden"], meta["num_taps"]) != (HIDDEN, NUM_TAPS) or rows < 1:
            raise ValueError(f"Invalid tap dimensions in {path}")
        if meta["num_seqs"] != len(meta["seqs"]):
            raise ValueError(f"Invalid sequence count in {path}")
        stem = path.name.removesuffix(".meta.json")
        token_file, tap_file = (path.with_name(stem + suffix) for suffix in (".tokens.bin", ".taps.bin"))
        if token_file.stat().st_size != rows * 4 or tap_file.stat().st_size != rows * NUM_TAPS * HIDDEN * 2:
            raise ValueError(f"Binary size disagrees with {path}")
        tokens = np.memmap(token_file, mode="r", dtype="<i4", shape=(rows,))
        taps = np.memmap(tap_file, mode="r", dtype="<u2", shape=(rows, NUM_TAPS, HIDDEN))
        end = 0
        for seq in meta["seqs"]:
            offset, length, prompt_len = (seq[k] for k in ("offset", "length", "prompt_len"))
            if offset != end or not 0 <= prompt_len <= length - BLOCK or offset + length > rows:
                raise ValueError(f"Invalid sequence range in {path}: {seq}")
            end = offset + length
            sequences.append(Sequence(tokens[offset:end], taps[offset:end], prompt_len))
        if end != rows:
            raise ValueError(f"Incomplete sequence ranges in {path}")
    if len(sequences) < 2:
        raise ValueError("Training and held-out validation require at least two sequences")
    return sequences


def split_sequences(sequences, fraction, seed):
    order = np.random.default_rng(seed).permutation(len(sequences))
    count = max(1, min(len(sequences) - 1, round(len(sequences) * fraction)))
    return [sequences[i] for i in order[count:]], [sequences[i] for i in order[:count]]


class Anchors:
    def __init__(self, sequences):
        self.sequences = sequences
        self.ends = np.cumsum([seq.anchors for seq in sequences]).tolist()

    def __len__(self):
        return self.ends[-1]

    def batch(self, indices, ctx_len, device):
        rows = []
        for index in indices:
            seq_id = bisect_right(self.ends, index)
            seq = self.sequences[seq_id]
            anchor = seq.prompt_len + index - (self.ends[seq_id - 1] if seq_id else 0)
            rows.append((seq, anchor, max(0, anchor - ctx_len)))
        width = max(anchor - start for _, anchor, start in rows)
        size = len(rows)
        ctx = np.zeros((size, width, NUM_TAPS, HIDDEN), dtype=np.uint16)
        ctx_mask = np.zeros((size, width), dtype=bool)
        ctx_pos = np.zeros((size, width), dtype=np.int64)
        block_pos = np.empty((size, BLOCK), dtype=np.int64)
        block_ids = np.full((size, BLOCK), MASK_TOKEN_ID, dtype=np.int64)
        predecessors = np.empty((size, GAMMA), dtype=np.int64)
        teacher = np.empty((size, GAMMA, HIDDEN), dtype=np.uint16)
        for i, (seq, anchor, start) in enumerate(rows):
            length = anchor - start
            ctx[i, :length] = seq.taps[start:anchor]
            ctx_mask[i, :length] = True
            ctx_pos[i, :length] = np.arange(start, anchor)
            block_pos[i] = np.arange(anchor, anchor + BLOCK)
            block_ids[i, 0] = seq.tokens[anchor]
            predecessors[i] = seq.tokens[anchor:anchor + GAMMA]
            teacher[i] = seq.taps[anchor:anchor + GAMMA, -1]

        def tensor(array):
            return torch.from_numpy(array).to(device)

        return (
            tensor(ctx.view(np.int16)).view(torch.bfloat16), tensor(ctx_mask), tensor(ctx_pos),
            tensor(block_ids), tensor(block_pos), tensor(predecessors),
            tensor(teacher.view(np.int16)).view(torch.bfloat16),
        )


def configure_stage(model, stage, freeze_codebooks=False):
    full = stage == "full"
    model.requires_grad_(full)
    # Keep FP32 master parameters and Adam moments while the model casts
    # operands to BF16 at the reference model's rounding points.
    if full:
        model.float()
    else:
        model.candidate_selector.float().requires_grad_(True)
    if freeze_codebooks:
        model.candidate_selector.predecessor_codebook.requires_grad_(False)
        model.candidate_selector.successor_codebook.requires_grad_(False)

    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = "selector" if name.startswith("candidate_selector.") else "backbone"
        # Decaying a codebook would update even the vocabulary rows with no gradient.
        decay = 0.0 if parameter.ndim == 1 or name.endswith("_codebook") else 0.01
        groups.setdefault((component, decay), []).append(parameter)
    return [{"name": component, "weight_decay": decay, "params": parameters}
            for (component, decay), parameters in groups.items()]


def kl_rows(target, student, temperature):
    p = F.log_softmax(target / temperature, dim=-1)
    q = F.log_softmax(student / temperature, dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1)


def support_overlap(target, top_ids, scores, temperature):
    scaled = target / temperature
    p = (scaled.gather(-1, top_ids) - scaled.logsumexp(-1, keepdim=True)).exp()
    q = (scores / temperature).softmax(-1)
    return p.sum(-1), torch.minimum(p, q).sum(-1)


def tv_accept_rows(target, top_ids, scores, temperature):
    mass, overlap = support_overlap(target, top_ids, scores, temperature)
    accept_loss = -overlap.clamp_min(1e-6).cumprod(-1).sum(-1) / GAMMA
    return mass - overlap + accept_loss.unsqueeze(-1)


def tv_rows(target, student, temperature):
    p = (target / temperature).softmax(-1)
    q = (student / temperature).softmax(-1)
    return 0.5 * (p - q).abs().sum(-1)


def loss_rows(model, batch, embed, embed_norm, head, head_fp32, full, unary_weight,
              *, loss_mode="tv_accept", ce_weight=0.0, grad_checkpoint=False):
    ctx, mask, positions, ids, block_positions, predecessors, teacher = batch
    with torch.no_grad():
        embeds = embed_tokens(ids, embed, embed_norm)
        target = softcap(F.linear(teacher.float(), head_fp32))
    with torch.set_grad_enabled(full and torch.is_grad_enabled()):
        backbone_inputs = (ctx, mask, positions, embeds, block_positions)
        if grad_checkpoint and torch.is_grad_enabled():
            hidden = checkpoint(model.forward_backbone, *backbone_inputs, use_reentrant=False)
        else:
            hidden = model.forward_backbone(*backbone_inputs)
        unary = F.linear(hidden, head).float()
    tv_accept = loss_mode == "tv_accept"
    top_ids, scores = model.score_selector_reached(hidden, unary, predecessors, top_k=TOP_K)
    t_top = target.gather(-1, top_ids)
    if tv_accept:
        selector_loss = 0.5 * (tv_accept_rows(target, top_ids, scores, 0.8)
                               + tv_accept_rows(target, top_ids, scores, 1.0))
    else:
        selector_loss = 0.5 * (0.8 ** 2) * kl_rows(t_top, scores, 0.8) + 0.5 * kl_rows(t_top, scores, 1.0)
    if ce_weight > 0:
        selector_loss = selector_loss + ce_weight * F.cross_entropy(
            scores.transpose(1, 2), t_top.argmax(-1), reduction="none",
        )
    if full:
        u_soft = softcap(unary)
        if tv_accept:
            unary_loss = 0.5 * (tv_rows(target, u_soft, 0.8) + tv_rows(target, u_soft, 1.0))
        else:
            unary_loss = 0.5 * 0.8 ** 2 * kl_rows(target, u_soft, 0.8) + 0.5 * kl_rows(target, u_soft, 1.0)
        selector_loss = selector_loss + unary_weight * unary_loss
    return selector_loss, target, top_ids, scores


@torch.no_grad()
def validate(model, anchors, indices, args, device, target_weights, weights):
    model.eval()
    sums = {}
    for start in range(0, len(indices), args.batch_size):
        batch = anchors.batch(indices[start:start + args.batch_size], args.ctx_len, device)
        loss, target, top_ids, scores = loss_rows(
            model, batch, *target_weights, args.stage == "full", args.unary_weight,
            loss_mode=args.loss_mode, ce_weight=args.ce_weight,
        )
        tgt_greedy = target.argmax(-1)
        drafted = top_ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        correct = drafted.eq(tgt_greedy)
        in_support = top_ids.eq(tgt_greedy.unsqueeze(-1)).any(-1)
        row_metrics = {
            "val_loss": (loss * weights).mean(-1),
            "greedy_acc_p0": correct[:, 0],
            "greedy_acc_mean": correct.float().mean(-1),
            "greedy_misses": (~correct).sum(-1),
            "greedy_misses_in_top16": (~correct & in_support).sum(-1),
            "pred_alpha_p1_t0": 1 + correct.float().cumprod(-1).sum(-1),
        }
        for label, temperature in (("t08", 0.8), ("t10", 1.0)):
            mass, overlap = support_overlap(target, top_ids, scores, temperature)
            row_metrics["overlap_" + label] = overlap.mean(-1)
            row_metrics["calib_gap_" + label] = (mass - overlap).mean(-1)
            row_metrics["pred_alpha_p1_" + label] = 1 + overlap.cumprod(-1).sum(-1)
            if label == "t08":
                row_metrics["top16_mass_t08"] = mass.mean(-1)
        for key, value in row_metrics.items():
            sums[key] = sums.get(key, 0) + value.sum(dtype=torch.float64)

    totals = dict(zip(sums, torch.stack(list(sums.values())).tolist()))
    misses = totals.pop("greedy_misses")
    in_top16 = totals.pop("greedy_misses_in_top16")
    metrics = {key: value / len(indices) for key, value in totals.items()}
    metrics["greedy_miss_in_top16_pct"] = 100 * in_top16 / misses if misses else 0.0
    model.train()
    return metrics


def lr_scale(step, steps, warmup):
    """Linear warmup and cosine decay for one-indexed optimizer steps."""
    if warmup and step <= warmup:
        return step / warmup
    # With no warmup, start the cosine at the first optimizer step.
    start = max(1, warmup)
    progress = (step - start) / max(1, steps - start)
    min_lr_scale = 0.1
    return min_lr_scale + (1 - min_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Frozen target model")
    parser.add_argument("--dflash2-dir", type=Path, required=True, help="Initial draft (use selector output for full stage)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("selector", "full"), required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr-selector", type=float, help="Defaults to 5e-5 for selector and 2e-5 for full")
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.10, help="Fraction of training steps for LR warmup in either stage")
    parser.add_argument("--loss-mode", choices=("tv_accept", "kl"), default="tv_accept")
    parser.add_argument("--ce-weight", type=float, default=0.0, help="Weight of optional hard selector cross-entropy")
    parser.add_argument("--freeze-codebooks", action="store_true", help="Train with fixed selector codebooks")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Recompute backbone activations during full-stage backward")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctx-len", type=int, default=WINDOW, help="Context taps retained per anchor")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-anchors", type=int, default=0, help="Fixed held-out anchors (0 evaluates all)")
    parser.add_argument("--pos-decay", type=float, default=0.85)
    parser.add_argument("--unary-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.lr_selector is None:
        args.lr_selector = 5e-5 if args.stage == "selector" else 2e-5
    if (min(args.batch_size, args.grad_accum, args.eval_every) < 1
            or min(args.steps, args.val_anchors, args.ctx_len, args.seed) < 0 or args.ctx_len > WINDOW):
        parser.error("Invalid step, batch, evaluation, context or seed argument")
    floats = (args.val_frac, args.pos_decay, args.lr_selector, args.lr_backbone, args.unary_weight)
    if (not all(map(math.isfinite, floats)) or not 0 < args.val_frac < 1 or not 0 < args.pos_decay <= 1
            or min(args.lr_selector, args.lr_backbone) <= 0 or args.unary_weight < 0):
        parser.error("Invalid validation fraction, learning rate, position decay or unary weight")
    if not math.isfinite(args.ce_weight) or args.ce_weight < 0:
        parser.error("--ce-weight must be finite and nonnegative")
    if not 0 <= args.warmup_frac <= 1:
        parser.error("--warmup-frac must be in [0, 1]")
    return args


def main():
    args = parse_args()
    if args.out_dir.resolve() == args.dflash2_dir.resolve():
        raise ValueError("--out-dir must differ from --dflash2-dir to preserve the initial checkpoint")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    rng = np.random.default_rng(args.seed)
    train_seqs, val_seqs = split_sequences(load_sequences(args.data_dir), args.val_frac, args.seed)
    train, val = Anchors(train_seqs), Anchors(val_seqs)
    if 0 < args.val_anchors < len(val):
        val_indices = sorted(np.random.default_rng(args.seed).choice(len(val), args.val_anchors, replace=False))
    else:
        val_indices = range(len(val))
    print(f"train={len(train_seqs)} seqs, {len(train)} anchors; val={len(val_seqs)} seqs, "
          f"{len(val_indices)}/{len(val)} anchors", flush=True)
    model = DFlash2DraftModel.from_checkpoint(args.dflash2_dir, device)
    embed, norm, head = load_target_head_and_embed(args.model_dir, device)
    target_weights = (embed, norm, head, head.float())
    position_weights = args.pos_decay ** torch.arange(GAMMA, device=device, dtype=torch.float32)
    position_weights /= position_weights.mean()
    full = args.stage == "full"
    groups = configure_stage(model, args.stage, freeze_codebooks=args.freeze_codebooks)
    for group in groups:
        group["lr"] = args.lr_backbone if group["name"] == "backbone" else args.lr_selector
    optimizer = torch.optim.AdamW(groups, foreach=False)
    base_rates = [g["lr"] for g in optimizer.param_groups]
    warmup = round(args.steps * args.warmup_frac)
    params = [p for group in groups for p in group["params"]]
    best = -math.inf
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    window_loss = torch.zeros((), device=device)
    window_seconds = 0.0
    last_progress = 0
    with (args.out_dir / "metrics.jsonl").open("w") as log:
        for step in range(args.steps + 1):
            if step:
                step_start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                scale = lr_scale(step, args.steps, warmup)
                for rate, group in zip(base_rates, optimizer.param_groups):
                    group["lr"] = scale * rate
                for _ in range(args.grad_accum):
                    indices = rng.integers(len(train), size=args.batch_size)
                    batch = train.batch(indices, args.ctx_len, device)
                    loss = loss_rows(
                        model, batch, *target_weights, full, args.unary_weight,
                        loss_mode=args.loss_mode, ce_weight=args.ce_weight, grad_checkpoint=args.grad_checkpoint,
                    )[0]
                    batch_loss = (loss * position_weights).mean() / args.grad_accum
                    batch_loss.backward()
                    window_loss.add_(batch_loss.detach())
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                report_progress = step % 50 == 0 or step == args.steps
                if report_progress:
                    window_steps = step - last_progress
                    train_loss = window_loss.item() / window_steps
                window_seconds += time.monotonic() - step_start
                if report_progress:
                    print(json.dumps({"stage": args.stage, "step": step, "train_loss": train_loss,
                                      "step/s": window_steps / window_seconds}, allow_nan=False), flush=True)
                    window_loss.zero_()
                    window_seconds = 0.0
                    last_progress = step
            if step % args.eval_every == 0 or step == args.steps:
                optimizer.zero_grad(set_to_none=True)
                metrics = validate(model, val, val_indices, args, device, target_weights, position_weights)
                score = metrics["pred_alpha_p1_t08"] + metrics["pred_alpha_p1_t0"]
                if score > best:
                    model.save_checkpoint(args.out_dir, args.dflash2_dir)
                    best = score
                    improved = True
                else:
                    improved = False
                row = {"stage": args.stage, "step": step, "val_anchors": len(val_indices),
                       "score": score, "best": improved, "elapsed_s": time.monotonic() - start, **metrics}
                line = json.dumps(row, allow_nan=False)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()


if __name__ == "__main__":
    main()
