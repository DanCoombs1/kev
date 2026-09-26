"""Turns token ids into vectors that carry the meaning of the whole text."""

import torch
from torch import nn

from kev.model.blocks import WIDTH, Block

LAYERS = 4
MAX_TOKENS = 128


class TextEncoder(nn.Module):
    def __init__(self, vocab: int, layers: int = LAYERS, width: int = WIDTH, max_tokens: int = MAX_TOKENS):
        super().__init__()
        self.tokens = nn.Embedding(vocab, width)
        self.positions = nn.Parameter(torch.zeros(max_tokens, width))
        self.blocks = nn.ModuleList(Block(width) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        nn.init.normal_(self.tokens.weight, std=0.02)
        nn.init.normal_(self.positions, std=0.02)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.tokens(ids) + self.positions[: ids.shape[1]]
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """A score for every token in the vocabulary. The embedding table doubles as the output layer."""
        return x @ self.tokens.weight.T
