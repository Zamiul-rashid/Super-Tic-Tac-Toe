"""Seeded, per-game opponent diversity for optional population training."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def symmetry_indices():
    """Destination indices for the eight board symmetries (both board levels)."""
    grid = np.arange(9).reshape(3, 3)
    maps = []
    for mirror in (False, True):
        for rotation in range(4):
            order = np.rot90(np.fliplr(grid) if mirror else grid, rotation).ravel()
            local = np.argsort(order)
            cells = (local[:, None] * 9 + local[None, :]).ravel()
            inputs = np.concatenate([cells + 81*k for k in range(3)] +
                                    [local + 243 + 9*k for k in range(4)] +
                                    [np.array([279]), local + 280])
            maps.append((inputs, cells))
    return maps


SYMMETRIES = symmetry_indices()


def augment_batch(x, pi, mask, rng):
    """Transform inputs, forced board, legal mask and policy together."""
    import torch
    out_x, out_pi, out_mask = x.clone(), pi.clone(), mask.clone()
    choices = rng.integers(8, size=len(x))
    for sym, (inputs, cells) in enumerate(SYMMETRIES):
        rows = torch.as_tensor(np.flatnonzero(choices == sym), device=x.device)
        if not len(rows):
            continue
        ii = torch.as_tensor(inputs, device=x.device)
        cc = torch.as_tensor(cells, device=x.device)
        out_x[rows[:, None], ii] = x[rows]
        out_pi[rows[:, None], cc] = pi[rows]
        out_mask[rows[:, None], cc] = mask[rows]
    return out_x, out_pi, out_mask


@dataclass(frozen=True)
class MatchSpec:
    kind: str = 'self'
    learner_side: int = 1
    depth: int = 3
    nodes: int = 3000
    epsilon: float = 0.
    style: str = 'random'
    checkpoint: str = ''
    simulations: int = 128
    opening_moves: int = 0


def sample_match(seed, checkpoint_dir=None):
    """60% self, 15% historical, 10% alpha-beta, 10% tactical, 5% styles.

    Missing history falls back to self-play. Each game gets independent random
    strength, seat, opening length, and move-noise settings. Never use latest.pt.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9127]))
    kind = str(rng.choice(['self', 'history', 'alphabeta', 'tactical', 'style'],
                          p=[.60, .15, .10, .10, .05]))
    checkpoint = ''
    if kind == 'history':
        paths = sorted(Path(checkpoint_dir).glob('model-*.pt')) if checkpoint_dir else []
        if paths:
            checkpoint = str(paths[int(rng.integers(len(paths)))].resolve())
        else:
            kind = 'self'
    return MatchSpec(kind=kind, learner_side=int(rng.choice([-1, 1])),
                     depth=int(rng.integers(1, 5)),
                     nodes=int(rng.choice([500, 1500, 3000, 6000])),
                     epsilon=float(rng.uniform(0., .12)),
                     style=str(rng.choice(['random', 'center', 'corners', 'local-win', 'global-win'])),
                     checkpoint=checkpoint, simulations=int(rng.choice([64, 128, 256])),
                     opening_moves=int(rng.integers(0, 5)))


def make_opponent(spec):
    from .bots import AlphaBetaBot, TacticalBot, StyleBot, CheckpointBot
    if spec.kind == 'self':
        return None
    if spec.kind == 'alphabeta':
        return AlphaBetaBot(depth=spec.depth, node_budget=spec.nodes)
    if spec.kind == 'tactical':
        bot = TacticalBot()
        bot.node_budget = spec.nodes
        return bot
    if spec.kind == 'style':
        return StyleBot(spec.style)
    if spec.kind == 'history':
        return CheckpointBot(spec.checkpoint, simulations=spec.simulations, device='cpu')
    raise ValueError(f'Unknown training opponent: {spec.kind}')
