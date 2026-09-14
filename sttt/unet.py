"""Hierarchical conv U-Net over the canonical 289-float encoding.

uttt.ai's network pools each 3x3 mini-board into one macro cell with a
stride-3 convolution, reasons on the 3x3 macro grid, and upsamples back with a
stride-3 transposed convolution plus skip connections. Our flat MLP has no idea
the board is 3x3-of-3x3. This module keeps the encoding, replay buffer, native
encoder and symmetry augmentation untouched by reshaping the flat row into
(8, 9, 9) planes inside forward().
"""
import numpy as np
import torch
from torch import nn
from .learning import BasePolicyValue


def _cell_to_grid():
    idx = np.arange(81)
    b, c = idx // 9, idx % 9
    return ((b // 3) * 3 + c // 3) * 9 + ((b % 3) * 3 + c % 3)


CELL_TO_GRID = torch.as_tensor(_cell_to_grid(), dtype=torch.long)
GRID_TO_CELL = torch.argsort(CELL_TO_GRID)


def planes_from_flat(x):
    """float[N,289] -> float[N,8,9,9].

    Planes: 0 empty, 1 mine, 2 theirs (cells in grid order); 3 open, 4 mine,
    5 theirs, 6 drawn (boards, each tiled over its 3x3 cells); 7 forced board
    (ones over the forced mini-board, ones everywhere when any board is legal).
    """
    n = x.shape[0]
    cells = x[:, :243].reshape(n, 3, 81)
    grid = torch.empty_like(cells)
    grid[:, :, CELL_TO_GRID.to(x.device)] = cells
    cells9 = grid.reshape(n, 3, 9, 9)
    boards9 = x[:, 243:279].reshape(n, 4, 3, 3).repeat_interleave(3, 2).repeat_interleave(3, 3)
    forced = x[:, 279:289]
    forced3 = forced[:, 1:].reshape(n, 1, 3, 3) + forced[:, :1].reshape(n, 1, 1, 1)
    forced9 = forced3.repeat_interleave(3, 2).repeat_interleave(3, 3)
    return torch.cat([cells9, boards9, forced9], dim=1)
