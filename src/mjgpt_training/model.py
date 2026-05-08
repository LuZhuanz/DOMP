from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised only without training deps
    torch = None
    nn = None
    F = None


@dataclass
class ModelConfig:
    vocab_size: int
    max_position_embeddings: int = 512
    n_layers: int = 6
    n_heads: int = 6
    hidden_size: int = 384
    intermediate_size: int = 1536
    dropout: float = 0.1
    rope_base: float = 10000.0
    scorer_hidden_size: int = 384

    @classmethod
    def preset(cls, name: str, *, vocab_size: int, max_position_embeddings: int = 512) -> ModelConfig:
        presets = {
            "debug": dict(n_layers=1, n_heads=4, hidden_size=64, intermediate_size=128, scorer_hidden_size=64, dropout=0.0),
            "tiny": dict(n_layers=4, n_heads=4, hidden_size=256, intermediate_size=1024, scorer_hidden_size=256),
            "small": dict(n_layers=6, n_heads=6, hidden_size=384, intermediate_size=1536, scorer_hidden_size=384),
            "base": dict(n_layers=12, n_heads=12, hidden_size=768, intermediate_size=3072, scorer_hidden_size=768),
        }
        if name not in presets:
            raise ValueError(f"unknown model preset: {name}")
        return cls(vocab_size=vocab_size, max_position_embeddings=max_position_embeddings, **presets[name])


@dataclass
class PolicyModelOutput:
    logits: "torch.FloatTensor"
    loss: "torch.FloatTensor | None" = None


if torch is not None:

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            return self.weight * x * torch.rsqrt(variance + self.eps)


    class SwiGLU(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
            super().__init__()
            self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


    class RotaryEmbedding(nn.Module):
        def __init__(self, head_dim: int, base: float) -> None:
            super().__init__()
            if head_dim % 2 != 0:
                raise ValueError("RoPE requires an even head_dim")
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device))
            cos = freqs.cos().to(dtype).view(1, 1, seq_len, -1)
            sin = freqs.sin().to(dtype).view(1, 1, seq_len, -1)
            return cos, sin


    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return rotated.flatten(-2)


    class CausalSelfAttention(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            if config.hidden_size % config.n_heads != 0:
                raise ValueError("hidden_size must be divisible by n_heads")
            self.n_heads = config.n_heads
            self.head_dim = config.hidden_size // config.n_heads
            self.rope = RotaryEmbedding(self.head_dim, config.rope_base)
            self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
            self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            batch, seq_len, hidden_size = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.rope(seq_len, device=x.device, dtype=q.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
            key_mask = attention_mask.view(batch, 1, 1, seq_len)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            y = weights @ v
            y = y.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
            return self.dropout(self.out(y))


    class TransformerBlock(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.attn_norm = RMSNorm(config.hidden_size)
            self.attn = CausalSelfAttention(config)
            self.ffn_norm = RMSNorm(config.hidden_size)
            self.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.dropout)

        def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.attn_norm(x), attention_mask)
            x = x + self.ffn(self.ffn_norm(x))
            return x


    class MahjongPolicyModel(nn.Module):
        """Decoder-only Transformer with dynamic legal-action scoring."""

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.dropout = nn.Dropout(config.dropout)
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
            self.norm = RMSNorm(config.hidden_size)
            self.scorer = nn.Sequential(
                nn.Linear(config.hidden_size * 3, config.scorer_hidden_size),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.scorer_hidden_size, 1),
            )

        def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.BoolTensor,
            choice_positions: torch.LongTensor,
            action_positions: torch.LongTensor,
            action_mask: torch.BoolTensor,
            labels: torch.LongTensor | None = None,
        ) -> PolicyModelOutput:
            batch, seq_len = input_ids.shape
            if seq_len > self.config.max_position_embeddings:
                raise ValueError("input sequence exceeds max_position_embeddings")
            hidden = self.token_embedding(input_ids)
            hidden = self.dropout(hidden)
            for block in self.blocks:
                hidden = block(hidden, attention_mask)
            hidden = self.norm(hidden)

            h_state = hidden[torch.arange(batch, device=hidden.device), choice_positions]
            safe_action_positions = action_positions.clamp_min(0)
            gather_index = safe_action_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            h_actions = hidden.gather(dim=1, index=gather_index)
            h_state_expanded = h_state.unsqueeze(1).expand_as(h_actions)
            features = torch.cat([h_state_expanded, h_actions, h_state_expanded * h_actions], dim=-1)
            logits = self.scorer(features).squeeze(-1)
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)

            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
            return PolicyModelOutput(logits=logits, loss=loss)

else:

    class MahjongPolicyModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("MahjongPolicyModel requires torch; install the training extra")
