"""The transformer block that kev's encoders and decoder are built from."""

import torch
import torch.nn.functional as F
from torch import nn

WIDTH = 384
HEADS = 6


class Attention(nn.Module):
    """Each token takes a weighted mix of the values of the tokens it attends to: its own sequence,
    or `context` for cross-attention. `mask` is True for real tokens in whichever is attended to."""

    def __init__(self, width: int = WIDTH, heads: int = HEADS):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, tokens, width) -> (batch, heads, tokens, width // heads)
        batch, tokens, width = x.shape
        return x.view(batch, tokens, self.heads, width // self.heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, context: torch.Tensor | None = None) -> torch.Tensor:
        source = x if context is None else context
        key, value = self.key_value(source).chunk(2, dim=-1)
        attend = None if mask is None else mask[:, None, None, :]
        # softmax(query · key / sqrt(head width)) · value, fused; tests/test_blocks.py spells it out
        mixed = F.scaled_dot_product_attention(self._split(self.query(x)), self._split(key), self._split(value), attn_mask=attend)
        return self.out(mixed.transpose(1, 2).reshape(x.shape))


class Block(nn.Module):
    def __init__(self, width: int = WIDTH, heads: int = HEADS):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask)
        return x + self.feed_forward(self.feed_forward_norm(x))
