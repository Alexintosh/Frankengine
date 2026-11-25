#!/usr/bin/env python3
"""Compare a rank-zero real-tap dump with the PyTorch backbone and selector."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from reference import layer, rms, select


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()

    def read(name, dtype):
        return torch.frombuffer(bytearray((args.dump / name).read_bytes()), dtype=dtype)

    weights = load_file(args.checkpoint / "model.safetensors")
    taps = read("taps.bf16", torch.bfloat16).view(-1, 4 * 2304)
    start = max(0, len(taps) - 2048)
    positions = torch.arange(start, len(taps) + 8)
    ctx = rms(F.linear(taps[start:], weights["fc.weight"]), weights["hidden_norm.weight"])
    x = read("embeddings.bf16", torch.bfloat16).view(8, 2304)
    print(f"committed position {len(taps)}, comparing {len(taps) - start} live ring rows")
    for i in range(5):
        def compare_ring(keys, values):
            for name, expected in (("keys", keys), ("values", values)):
                filename = f"{name}-{i}.bf16"
                if not (args.dump / filename).exists():
                    continue
                actual = read(filename, torch.bfloat16).view_as(expected)
                delta = (actual.float() - expected.float()).flatten(1)
                rmse = delta.square().mean().sqrt()
                worst_row = delta.square().mean(1).sqrt().max()
                print(f"layer {i} ring {name} RMSE {rmse:.8f}, worst row RMSE {worst_row:.8f}")

        x = layer(x, ctx, positions, weights, f"layers.{i}.", compare_ring)
    hidden = rms(x, weights["norm.weight"])[1:].contiguous()
    actual = read("hidden.bf16", torch.bfloat16).view_as(hidden)
    delta = actual.float() - hidden.float()
    print(f"hidden RMSE {delta.square().mean().sqrt():.8f}, max |delta| {delta.abs().max():.8f}")
    unary = read("unary.f32", torch.float32).view(7, -1)
    reference_unary = unary
    if (args.dump / "head_codes.u8").exists():
        rows = unary.shape[1]
        codes = read("head_codes.u8", torch.uint8).view(rows, -1)
        scales = read("head_scales.u8", torch.float8_e4m3fn).view(rows, -1)
        factor = read("head_scale.f32", torch.float32).item()
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        expected_unary = torch.empty_like(unary)
        reference_unary = torch.empty_like(unary)
        for first in range(0, rows, 2048):
            section = slice(first, first + 2048)
            packed = codes[section]
            indices = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1).long()
            scale = scales[section].float().repeat_interleave(16, dim=1)
            head = (levels[indices] * scale * factor).to(torch.bfloat16)
            expected_unary[:, section] = F.linear(actual, head).float()
            reference_unary[:, section] = F.linear(hidden, head).float()
        head_delta = (unary - expected_unary).abs()
        full_delta = unary - reference_unary
        print(f"NVFP4 head |delta| max {head_delta.max():.8f}, exact {(head_delta == 0).float().mean():.8f}")
        print(f"full unary RMSE {full_delta.square().mean().sqrt():.8f}, max |delta| {full_delta.abs().max():.8f}")

    anchor = read("anchor.i32", torch.int32).long()
    ordered = unary.argsort(descending=True, stable=True)[:, :16]
    native_top = read("top.i32", torch.int32).view_as(ordered)
    torch.testing.assert_close(native_top.long(), ordered)
    ref_top = reference_unary.argsort(descending=True, stable=True)[:, :16]
    overlap = (ref_top[:, :, None] == ordered[:, None, :]).any(-1).sum(-1)
    print("full reference top-16 overlap per slot", overlap.tolist())
    ids, _, expected = select(hidden, reference_unary, anchor, weights)
    actual_tokens = read("tokens.i32", torch.int32).long()
    print("native tokens", actual_tokens.tolist())
    print("reference tokens", expected.tolist())
    if not torch.equal(ids.sort().values, ref_top.sort().values):
        print("PyTorch topk includes different IDs tied at the cutoff")
    torch.testing.assert_close(actual_tokens, expected)


if __name__ == "__main__":
    main()
