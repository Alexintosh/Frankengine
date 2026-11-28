"""Differentiable BF16 DFlash2 block matching engine/tools/arch06a/reference.py."""

import json
import math
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

HIDDEN = 2304
NUM_TAPS = 4
BLOCK = 8
GAMMA = BLOCK - 1
WINDOW = 2048
VOCAB = 201024
# Keep in sync with engine/native/src/model/tp_layout.h.
UNPADDED_VOCAB = 200058
MASK_TOKEN_ID = 200056
SOFTCAP = 30.0
TOP_K = 16
HEAD_DIM = 128
QUERY_HEADS = 32
KV_HEADS = 8
RANK = 256


def load_target_head_and_embed(model_dir, device):
    root = Path(model_dir)
    prefix = "model.llm."
    names = [prefix + suffix for suffix in ("embed.weight", "embed_norm.weight", "unembed.weight")]
    index = root / "model.safetensors.index.json"
    if index.is_file():
        files = json.loads(index.read_text())["weight_map"]
        paths = [root / files[name] for name in names]
    else:
        path = root / "model-00001-of-00001.safetensors"
        if not path.is_file():
            path = root / "model.safetensors"
        paths = [path] * len(names)
    with ExitStack() as stack:
        readers = {path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                   for path in dict.fromkeys(paths)}
        embed = readers[paths[0]].get_tensor(names[0])
        embed_norm = readers[paths[1]].get_tensor(names[1])
        head_slice = readers[paths[2]].get_slice(names[2])
        if (embed.shape != (VOCAB, HIDDEN) or embed.dtype != torch.bfloat16
                or embed_norm.shape != (HIDDEN,) or embed_norm.dtype not in (torch.float32, torch.bfloat16)
                or head_slice.get_shape() != [VOCAB, HIDDEN] or head_slice.get_dtype() != "BF16"):
            raise ValueError("Unexpected target embedding, embedding norm, or unembedding shape/dtype")
        head = head_slice[:UNPADDED_VOCAB]
    return embed.to(device), embed_norm.to(device=device, dtype=torch.float32), head.to(device)


def rms(x, weight):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()).to(x.dtype)


def embed_tokens(ids, embed, embed_norm):
    return rms(embed[ids], embed_norm).to(torch.bfloat16)


def softcap(logits):
    return SOFTCAP * torch.tanh(logits.float() / SOFTCAP)


def linear(x, layer):
    return F.linear(x, layer.weight.to(x.dtype))


class RMSNorm(nn.Module):
    def __init__(self, width, **factory):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, **factory))

    def forward(self, x):
        return rms(x, self.weight.to(x.dtype))


def rope(x, positions):
    dim = x.shape[-1]
    inv_freq = 10000 ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = (positions.float().unsqueeze(-1) * inv_freq).repeat(1, 1, 2).unsqueeze(-2)
    left, right = x.float().chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return (x.float() * angles.cos() + rotated * angles.sin()).to(x.dtype)


class DynamicConv(nn.Module):
    def __init__(self, **factory):
        super().__init__()
        self.base_kernel = nn.Parameter(torch.zeros(2, 2, HIDDEN, **factory))
        self.kernel_projection = nn.Linear(HIDDEN, 4 * (HIDDEN // 16), bias=False, **factory)

    def prepare(self, x):
        delta = linear(x, self.kernel_projection).unflatten(-1, (2, 2, HIDDEN // 16))
        base = self.base_kernel.to(x.dtype)
        return self.apply_kernel(x, delta[:, :, 0], base[0]), delta[:, :, 1], base[1]

    @staticmethod
    def apply_kernel(x, coefficients, base):
        blocks = x.unflatten(-1, (HIDDEN // 16, 16))
        shifted = F.pad(blocks[:, :-1], (0, 0, 0, 0, 1, 0))
        kernel = base.unflatten(-1, (HIDDEN // 16, 16)) + coefficients.unsqueeze(-1)
        return (kernel[:, :, 0] * blocks + kernel[:, :, 1] * shifted).flatten(-2)


class Attention(nn.Module):
    def __init__(self, **factory):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, QUERY_HEADS * HEAD_DIM, bias=False, **factory)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False, **factory)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False, **factory)
        self.o_proj = nn.Linear(QUERY_HEADS * HEAD_DIM, HIDDEN, bias=False, **factory)
        self.q_norm = RMSNorm(HEAD_DIM, **factory)
        self.k_norm = RMSNorm(HEAD_DIM, **factory)

    def forward(self, x, ctx, positions, block_positions, visible, key_valid):
        q = linear(x, self.q_proj).unflatten(-1, (QUERY_HEADS, HEAD_DIM))
        kv_input = torch.cat((ctx, x), dim=1)
        k = linear(kv_input, self.k_proj).unflatten(-1, (KV_HEADS, HEAD_DIM))
        v = linear(kv_input, self.v_proj).unflatten(-1, (KV_HEADS, HEAD_DIM))

        q = rope(self.q_norm(q), block_positions).transpose(1, 2).float()
        group_size = QUERY_HEADS // KV_HEADS
        k = rope(self.k_norm(k), positions).transpose(1, 2).repeat_interleave(group_size, dim=1).float()
        v = v.transpose(1, 2).repeat_interleave(group_size, dim=1).float()
        if x.device.type == "cpu":
            # The CPU oracle reduces over the unpadded key dimension. Keeping that
            # dimension also preserves its FP32 reduction order at BF16 roundoff.
            rows = []
            for qi, ki, vi, allowed, valid in zip(q, k, v, visible, key_valid):
                rows.append(self.attend(qi, ki[:, valid], vi[:, valid], allowed[:, valid].unsqueeze(0)))
            attn = torch.stack(rows)
        else:
            attn = self.attend(q, k, v, visible[:, None])
        return linear(attn.transpose(1, 2).flatten(-2).to(x.dtype), self.o_proj)

    @staticmethod
    def attend(q, k, v, visible):
        scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM ** -0.5)
        return scores.masked_fill(~visible, float("-inf")).softmax(-1) @ v


class MLP(nn.Module):
    def __init__(self, **factory):
        super().__init__()
        width = 6912
        self.gate_proj = nn.Linear(HIDDEN, width, bias=False, **factory)
        self.up_proj = nn.Linear(HIDDEN, width, bias=False, **factory)
        self.down_proj = nn.Linear(width, HIDDEN, bias=False, **factory)

    def forward(self, x):
        activated = F.silu(linear(x, self.gate_proj)) * linear(x, self.up_proj)
        return linear(activated, self.down_proj)


class DraftLayer(nn.Module):
    def __init__(self, **factory):
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN, **factory)
        self.attention_conv = DynamicConv(**factory)
        self.self_attn = Attention(**factory)
        self.post_attention_layernorm = RMSNorm(HIDDEN, **factory)
        self.mlp_conv = DynamicConv(**factory)
        self.mlp = MLP(**factory)

    def forward(self, x, ctx, positions, block_positions, visible, key_valid):
        h, delta, base = self.attention_conv.prepare(self.input_layernorm(x))
        attn = self.self_attn(h, ctx, positions, block_positions, visible, key_valid)
        x = x + self.attention_conv.apply_kernel(attn, delta, base)
        h, delta, base = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return x + self.mlp_conv.apply_kernel(self.mlp(h), delta, base)


class CandidateSelector(nn.Module):
    def __init__(self, **factory):
        super().__init__()
        self.hidden_projection = nn.Linear(HIDDEN, RANK, bias=False, **factory)
        self.predecessor_codebook = nn.Parameter(torch.empty(VOCAB, RANK, **factory))
        self.successor_codebook = nn.Parameter(torch.empty(VOCAB, RANK, **factory))
        for codebook in (self.predecessor_codebook, self.successor_codebook):
            nn.init.normal_(codebook, std=1 / math.sqrt(RANK))

    def forward(self, hidden, values, ids, pred_ids):
        projected = linear(hidden, self.hidden_projection)
        pred = self.predecessor_codebook[pred_ids].to(hidden.dtype)
        succ = self.successor_codebook[ids].to(hidden.dtype)
        # pred_ids [B, 7, P], where P=1 for a reached prefix or 16 for the full lattice.
        corrections = (pred * projected.unsqueeze(-2)) @ succ.transpose(-1, -2)
        return softcap(values.unsqueeze(-2) + corrections.float())


class DFlash2DraftModel(nn.Module):
    def __init__(self, device=None, dtype=torch.bfloat16):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.fc = nn.Linear(NUM_TAPS * HIDDEN, HIDDEN, bias=False, **factory)
        self.hidden_norm = RMSNorm(HIDDEN, **factory)
        self.norm = RMSNorm(HIDDEN, **factory)
        self.layers = nn.ModuleList(DraftLayer(**factory) for _ in range(5))
        self.candidate_selector = CandidateSelector(**factory)

    @classmethod
    def from_checkpoint(cls, model_dir, device="cpu"):
        model = cls(device="meta")
        weights = load_file(Path(model_dir) / "model.safetensors", device=str(device))
        for name, parameter in model.named_parameters():
            if name in weights and (weights[name].dtype != torch.bfloat16 or weights[name].shape != parameter.shape):
                raise ValueError(f"Invalid DFlash2 weight: {name}")
        model.load_state_dict(weights, strict=True, assign=True)
        return model

    def forward_backbone(self, ctx_taps, ctx_mask, ctx_positions, block_embeds, block_positions):
        """Return [B, 7, H] hidden states from an anchor and seven masks."""
        positions = torch.cat((ctx_positions, block_positions), dim=-1)
        key_valid = F.pad(ctx_mask, (0, BLOCK), value=True)
        visible = (positions[:, None] > block_positions[:, :, None] - WINDOW) & key_valid[:, None]
        ctx = self.hidden_norm(linear(ctx_taps.flatten(-2), self.fc))
        x = block_embeds
        for layer in self.layers:
            x = layer(x, ctx, positions, block_positions, visible, key_valid)
        return self.norm(x)[:, 1:].contiguous()

    def score_selector_reached(self, hidden, unary, pred_ids, top_k=TOP_K):
        """Return candidate ids and [B, 7, top_k] scores on the actual predecessor path."""
        values, ids = unary.topk(top_k, dim=-1)
        scores = self.candidate_selector(hidden, values, ids, pred_ids.unsqueeze(-1)).squeeze(-2)
        return ids, scores

    def save_checkpoint(self, out_dir, source_config_dir):
        root, source = Path(out_dir), Path(source_config_dir)
        if not (source / "config.json").is_file():
            raise FileNotFoundError(source / "config.json")
        root.mkdir(parents=True, exist_ok=True)
        tensors = {name: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                   for name, value in self.named_parameters()}
        path = root / "model.safetensors"
        tmp = path.with_suffix(".safetensors.part")
        save_file(tensors, tmp, metadata={"format": "pt"})
        for name in ("config.json", "README.md"):
            original, destination = source / name, root / name
            if original.is_file() and original.resolve() != destination.resolve():
                shutil.copyfile(original, destination)
        tmp.replace(path)
