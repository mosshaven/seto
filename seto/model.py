"""Seto-1B: Decoder-only Transformer with GQA, SwiGLU, RoPE, RMSNorm, SDPA."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: [B, heads, T, head_dim]
    # freqs_cis: [T, head_dim//2] -> [1, 1, T, head_dim//2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    x_rot = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rot.type_as(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        bs, n_kv, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bs, n_kv, self.n_rep, seq_len, head_dim)
            .reshape(bs, self.n_heads, seq_len, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        out = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.o_proj(out))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class SetoLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

        freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len * 2, config.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} > max {self.config.max_seq_len}"

        x = self.dropout(self.tok_embeddings(input_ids))
        freqs_cis = self.freqs_cis[:T]

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = gradient_checkpoint(layer, x, freqs_cis, use_reentrant=False)
            else:
                x = layer(x, freqs_cis)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def resize_embeddings(self, new_vocab_size: int):
        """Resize token embeddings and output head for new vocab size.
        Copies existing weights, reinitializes new tokens.
        Handles tied embeddings correctly.
        """
        old_vocab_size = self.config.vocab_size
        if new_vocab_size == old_vocab_size:
            return

        old_weight = self.tok_embeddings.weight.data
        new_embedding = nn.Embedding(new_vocab_size, self.config.d_model)
        new_embedding.weight.data[:old_vocab_size] = old_weight
        # Reinit new tokens with small values (same std as original init)
        nn.init.normal_(new_embedding.weight.data[old_vocab_size:], mean=0.0, std=0.02)
        self.tok_embeddings = new_embedding

        if self.config.tie_embeddings:
            self.output = nn.Linear(self.config.d_model, new_vocab_size, bias=False)
            self.output.weight = self.tok_embeddings.weight
        else:
            old_out = self.output.weight.data
            new_out = nn.Linear(self.config.d_model, new_vocab_size, bias=False)
            new_out.weight.data[:old_vocab_size] = old_out
            nn.init.normal_(new_out.weight.data[old_vocab_size:], mean=0.0, std=0.02)
            self.output = new_out

        self.config.vocab_size = new_vocab_size
