#!/usr/bin/env python3
"""CPU oracle for one Inkling DFlash2 block with caller-supplied target tensors.

Input is a safetensors file: taps [P,4,2304], block_embeds [8,2304]
(anchor then seven mask embeddings, with the target embedding transform applied),
head [valid_vocab,2304], anchor [1] and optional positions [P+8].
Outputs include dense unary logits, the lattice, and the walked greedy path.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F


def rms(x, weight):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()).to(x.dtype)


def rope(x, positions):
    dim = x.shape[-1]
    frequencies = 10000 ** (-torch.arange(0, dim, 2).float() / dim)
    angles = (positions.float()[:, None] * frequencies).repeat(1, 2)[:, None]
    left, right = x.float().chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return (x.float() * angles.cos() + rotated * angles.sin()).to(x.dtype)


def convolve(x, coefficients, base):
    groups = x.shape[-1] // 16
    blocks = x.unflatten(-1, (groups, 16))
    shifted = F.pad(blocks[:-1], (0, 0, 0, 0, 1, 0))
    kernel = base.unflatten(-1, (groups, 16)) + coefficients.unsqueeze(-1)
    return (kernel[:, 0] * blocks + kernel[:, 1] * shifted).flatten(-2)


def prepare_conv(x, weights, prefix):
    base = weights[prefix + ".base_kernel"]
    delta = F.linear(x, weights[prefix + ".kernel_projection.weight"])
    delta = delta.reshape(x.shape[0], 2, 2, x.shape[-1] // 16)
    return convolve(x, delta[:, 0], base[0]), delta[:, 1], base[1]


def layer(x, ctx, positions, weights, prefix, inspect_kv=None):
    def linear(value, name):
        return F.linear(value, weights[prefix + name + ".weight"])

    qpos = positions[-8:]
    normed = rms(x, weights[prefix + "input_layernorm.weight"])
    h, delta, base = prepare_conv(normed, weights, prefix + "attention_conv")
    q = linear(h, "self_attn.q_proj").unflatten(-1, (32, 128))
    kv_input = torch.cat((ctx, h))
    k = linear(kv_input, "self_attn.k_proj").unflatten(-1, (8, 128))
    v = linear(kv_input, "self_attn.v_proj").unflatten(-1, (8, 128))
    q = rope(rms(q, weights[prefix + "self_attn.q_norm.weight"]), qpos)
    k = rope(rms(k, weights[prefix + "self_attn.k_norm.weight"]), positions)
    if inspect_kv is not None:
        inspect_kv(k[:-8], v[:-8])
    q = q.transpose(0, 1).float()
    k = k.transpose(0, 1).repeat_interleave(4, dim=0).float()
    v = v.transpose(0, 1).repeat_interleave(4, dim=0).float()
    scores = (q @ k.transpose(-1, -2)) * (128 ** -0.5)
    visible = (positions[None, :] > qpos[:, None] - 2048)
    scores.masked_fill_(~visible, float("-inf"))
    attn = (scores.softmax(-1) @ v).transpose(0, 1).flatten(-2).to(x.dtype)
    x = x + convolve(linear(attn, "self_attn.o_proj"), delta, base)
    normed = rms(x, weights[prefix + "post_attention_layernorm.weight"])
    h, delta, base = prepare_conv(normed, weights, prefix + "mlp_conv")
    activated = F.silu(linear(h, "mlp.gate_proj")) * linear(h, "mlp.up_proj")
    return x + convolve(linear(activated, "mlp.down_proj"), delta, base)


def select(hidden, unary, anchor, weights):
    values, ids = unary.topk(16, dim=-1)
    prefix = "candidate_selector."
    projected = F.linear(hidden, weights[prefix + "hidden_projection.weight"])
    pred_ids = torch.cat((anchor.reshape(1, 1).expand(1, 16), ids[:-1]), dim=0)
    predecessors = weights[prefix + "predecessor_codebook"][pred_ids]
    successors = weights[prefix + "successor_codebook"][ids]
    corrections = torch.einsum("lpr,lcr->lpc", predecessors * projected[:, None], successors)
    lattice = 30 * torch.tanh((values[:, None] + corrections.float()) / 30)
    path = []
    predecessor = 0
    for edge in range(ids.shape[0]):
        predecessor = lattice[edge, predecessor].argmax().item()
        path.append(ids[edge, predecessor])
    return ids, lattice, torch.stack(path)


@torch.inference_mode()
def draft(weights, inputs):
    x = inputs["block_embeds"].to(torch.bfloat16)
    taps = inputs["taps"].to(torch.bfloat16)
    if x.shape != (8, 2304) or taps.ndim != 3 or taps.shape[1:] != (4, 2304):
        raise ValueError("Expected [8,2304] block embeddings and [P,4,2304] taps")
    positions = inputs.get("positions", torch.arange(taps.size(0) + 8))
    ctx = F.linear(taps.flatten(1), weights["fc.weight"])
    ctx = rms(ctx, weights["hidden_norm.weight"])
    for i in range(5):
        x = layer(x, ctx, positions, weights, f"layers.{i}.")
    hidden = rms(x, weights["norm.weight"])[1:].contiguous()
    unary = F.linear(hidden, inputs["head"].to(torch.bfloat16)).float()
    ids, lattice, path = select(hidden, unary, inputs["anchor"], weights)
    return {"context": ctx, "hidden": hidden, "unary": unary,
            "candidates": ids, "lattice": lattice, "tokens": path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("outputs", type=Path)
    args = parser.parse_args()
    weights = load_file(args.model_dir / "model.safetensors")
    outputs = draft(weights, load_file(args.inputs))
    save_file(outputs, args.outputs)


if __name__ == "__main__":
    main()
