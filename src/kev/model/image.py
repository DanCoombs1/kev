"""Turns a scan's patches into vectors, each one shaped by the whole bag.

Patches arrive as uint8 pixels (see kev.data.loader.collate) with their row, column and view. Any subset of a scan's
patches can be encoded, in any order, since each carries its own position.
"""

import torch
import torch.nn.functional as F
from torch import nn

from kev.data.prepare import PATCH
from kev.model.blocks import WIDTH, Block

LAYERS = 9
MAX_GRID = 128  # rows or columns in a view
VIEWS = 2
TILE = 4
TILES = (PATCH // TILE) ** 2


def absorption(patches: torch.Tensor) -> torch.Tensor:
    """0 for empty space and padding, towards 1 where the bag absorbed nearly everything."""
    return 1 - patches.float() / 255


class PatchStem(nn.Module):
    """Looks inside each patch on its own, as two convolutions: every 4x4 tile of pixels -> 96 numbers, then the
    4x4 grid of tiles -> one vector. Neither convolution's windows overlap, so each is a matrix multiplication,
    which the Mac's GPU does far faster than a convolution over thousands of tiny images."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        self.tile = nn.Linear(3 * TILE * TILE, 96)
        self.grid = nn.Linear(TILES * 96, width)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        lead = patches.shape[:-1]
        side = PATCH // TILE
        x = absorption(patches).reshape(*lead, 3, side, TILE, side, TILE)  # channel, tile row, y, tile column, x
        x = x.permute(*range(len(lead)), -4, -2, -5, -3, -1).reshape(*lead, TILES, 3 * TILE * TILE)
        return self.grid(F.gelu(self.tile(x)).flatten(-2))


class ImageEncoder(nn.Module):
    def __init__(self, layers: int = LAYERS, width: int = WIDTH, max_grid: int = MAX_GRID):
        super().__init__()
        self.stem = PatchStem(width)
        self.rows = nn.Parameter(torch.zeros(max_grid, width))
        self.cols = nn.Parameter(torch.zeros(max_grid, width))
        self.views = nn.Parameter(torch.zeros(VIEWS, width))
        for table in (self.rows, self.cols, self.views):
            nn.init.normal_(table, std=0.02)
        self.blocks = nn.ModuleList(Block(width) for _ in range(layers))
        self.norm = nn.LayerNorm(width)

    def forward(self, patches: torch.Tensor, row: torch.Tensor, col: torch.Tensor, view: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        x = self.stem(patches) + self.rows[row] + self.cols[col] + self.views[view]
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)
