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


def _conv_block(cin, cout, groups=8):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(groups, cout), nn.ReLU())


class ConvResBlock(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(groups, ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(groups, ch))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class UNet(BasePolicyValue):
    """Micro (9x9) -> macro (3x3) -> micro U-Net with policy, value and Q heads.

    GroupNorm rather than BatchNorm: one module object serves self-play
    inference and training, and GroupNorm has no running statistics to drift
    between the two modes or between batch sizes 1 and 1024.
    """

    def __init__(self, micro=64, macro=128, micro_blocks=2, macro_blocks=3):
        super().__init__()
        self.stem = _conv_block(8, micro)
        self.micro = nn.Sequential(*[ConvResBlock(micro) for _ in range(micro_blocks)])
        # stride-3 3x3 conv: each mini-board becomes one macro cell
        self.pool = nn.Sequential(nn.Conv2d(micro, macro, 3, stride=3, bias=False),
                                  nn.GroupNorm(8, macro), nn.ReLU())
        self.macro = nn.Sequential(*[ConvResBlock(macro) for _ in range(macro_blocks)])
        # stride-3 transposed conv: each macro cell is broadcast back to its 3x3 cells
        self.up = nn.Sequential(nn.ConvTranspose2d(macro, micro, 3, stride=3, bias=False),
                                nn.GroupNorm(8, micro), nn.ReLU())
        self.fuse = _conv_block(micro * 2 + 8, micro)
        self.policy = nn.Conv2d(micro, 1, 1)
        self.q = nn.Conv2d(micro, 1, 1)
        self.value = nn.Sequential(nn.Flatten(), nn.Linear(macro * 9, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward_all(self, x):
        planes = planes_from_flat(x)
        h = self.micro(self.stem(planes))
        m = self.macro(self.pool(h))
        f = self.fuse(torch.cat([h, self.up(m), planes], dim=1))
        order = CELL_TO_GRID.to(x.device)
        logits = self.policy(f).flatten(1)[:, order]
        q = self.q(f).flatten(1)[:, order].tanh()
        value = self.value(m).squeeze(-1).tanh()
        return logits, value, q

    def forward(self, x):
        logits, value, _ = self.forward_all(x)
        return logits, value
