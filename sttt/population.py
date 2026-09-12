"""Seeded, quota-controlled opponent diversity for population training."""
from dataclasses import dataclass
from pathlib import Path
import shlex
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


# The aggregate buckets requested for the next league are split into concrete
# opponents so that every strong family gets a measurable training signal.
# history + best = 20%, AlphaBeta + tactical + threat = 15%.
POPULATION_WEIGHTS = {
    'self': .30,
    'utttai': .25,
    'alphabeta': .25,
    'history': .10,
    'best': .05,
    'tactical': .02,
    'openspiel': .01,
    'threat': .01,
    'style': .01,
}


def population_quota_counts(games: int, offset: int = 0) -> dict[str, int]:
    """Allocate a slice of a balanced 100-game cycle; offset survives resume."""
    games = int(games)
    if games < 1:
        raise ValueError('At least one population game is required')
    names = list(POPULATION_WEIGHTS)
    # Smooth weighted round-robin gives exact percentages every 100 games,
    # including across small iterations. Reserving one of nine variants per
    # 16 games would overweight tactical bots and reduce self-play to 25%.
    credits = dict.fromkeys(names, 0)
    cycle = []
    for _ in range(100):
        for name in names:
            credits[name] += round(100 * POPULATION_WEIGHTS[name])
        chosen = max(names, key=lambda name: credits[name])
        credits[chosen] -= 100
        cycle.append(chosen)
    counts = {name: 0 for name in names}
    for index in range(offset, offset + games):
        counts[cycle[index % 100]] += 1
    return counts


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
    bot_seed: int = 0
    engine_command: tuple[str, ...] = ()
    engine_protocol: str = 'action_index'
    engine_timeout: float = 30.
    engine_cwd: str = ''
    engine_fallback: str = 'raise'
    engine_name: str = ''


def _engine_fields(engine_registry, seed, simulations):
    """Serialize the configured uttt.ai process settings into MatchSpec."""
    registry = engine_registry or {}
    config = registry.get('utttai') or registry.get('utttai-low') or {}
    command = config.get('command', ())
    if isinstance(command, str):
        command = tuple(shlex.split(command))
    else:
        command = tuple(command)
    return dict(bot_seed=int(seed), engine_command=command,
                engine_protocol=str(config.get('protocol', 'action_index')),
                engine_timeout=float(config.get('timeout', 30.)),
                engine_cwd=str(config.get('cwd') or ''),
                engine_fallback=str(config.get('fallback', 'raise')),
                engine_name=str(config.get('name', 'utttai')))


def sample_match(seed, checkpoint_dir=None, engine_registry=None, kind=None):
    """Sample one deterministic match specification.

    Missing history falls back to self-play. Each game gets independent random
    strength, seat, opening length, and move-noise settings. Never use latest.pt.
    ``kind`` is used by :func:`sample_matches` to enforce exact quotas.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9127]))
    if kind is None:
        names = list(POPULATION_WEIGHTS)
        kind = str(rng.choice(names, p=list(POPULATION_WEIGHTS.values())))
    checkpoint = ''
    if kind == 'history':
        paths = sorted(Path(checkpoint_dir).glob('model-*.pt')) if checkpoint_dir else []
        if not paths and checkpoint_dir and (Path(checkpoint_dir) / 'best.pt').is_file():
            paths = [Path(checkpoint_dir) / 'best.pt']
        if paths:
            checkpoint = str(paths[int(rng.integers(len(paths)))].resolve())
        else:
            kind = 'self'
    if kind == 'best':
        best = Path(checkpoint_dir) / 'best.pt' if checkpoint_dir else None
        if best is not None and best.is_file():
            checkpoint = str(best.resolve())
        elif checkpoint_dir:
            paths = sorted(Path(checkpoint_dir).glob('model-*.pt'))
            if paths:
                kind, checkpoint = 'history', str(paths[int(rng.integers(len(paths)))].resolve())
            else:
                kind = 'self'
        else:
            kind = 'self'

    if kind == 'alphabeta':
        depth = int(rng.choice([3, 4, 5, 6, 7, 8]))
        nodes = int(rng.choice([100000, 250000, 500000]))
        epsilon = float(rng.uniform(0., .02))
    elif kind == 'tactical':
        depth, nodes = 2, int(rng.choice([6000, 10000, 16000]))
        epsilon = float(rng.uniform(0., .12))
    elif kind == 'threat':
        depth, nodes = 2, int(rng.choice([6000, 12000, 24000]))
        epsilon = float(rng.uniform(0., .08))
    elif kind == 'history' or kind == 'best':
        depth, nodes = 3, 0
        epsilon = float(rng.uniform(0., .03))
    elif kind == 'openspiel':
        depth, nodes = 3, 0
        epsilon = float(rng.uniform(0., .03))
    elif kind == 'utttai':
        depth, nodes = 3, 0
        epsilon = float(rng.uniform(0., .02))
    elif kind == 'style':
        depth, nodes = 1, 0
        epsilon = float(rng.uniform(0., .12))
    else:
        depth, nodes, epsilon = 3, 0, 0.

    simulations = int(rng.choice([128, 256, 512]))
    engine_fields = {'bot_seed': int(rng.integers(0, 2**31 - 1))}
    if kind == 'openspiel':
        simulations = int(rng.choice([128, 256, 512, 1024]))
    elif kind == 'utttai':
        # Keep the external reference materially cheaper than the learner's
        # 512 simulations. Its command receives this through {simulations}.
        simulations = int(rng.choice([64, 128, 256]))
        engine_fields = _engine_fields(engine_registry, int(rng.integers(0, 2**31 - 1)), simulations)

    # Legacy incremental engines cannot accept openings or injected moves.
    # The bundled state_json wrapper supports both.
    opening_moves = (0 if kind == 'utttai' and engine_fields.get('engine_protocol') != 'state_json'
                     else int(rng.integers(0, 5)))
    if kind == 'utttai' and engine_fields.get('engine_protocol') != 'state_json':
        epsilon = 0.  # Incremental protocol cannot recover from an injected move.
    return MatchSpec(kind=kind, learner_side=int(rng.choice([-1, 1])),
                     depth=depth, nodes=nodes, epsilon=epsilon,
                     style=str(rng.choice(['random', 'center', 'corners', 'local-win', 'global-win'])),
                     checkpoint=checkpoint, simulations=simulations,
                     opening_moves=opening_moves, **engine_fields)


def sample_matches(seeds, checkpoint_dir=None, engine_registry=None, offset=0):
    """Build one randomized, quota-controlled league schedule for an iteration."""
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError('At least one population game is required')
    counts = population_quota_counts(len(seeds), offset)
    kinds = [kind for kind, count in counts.items() for _ in range(count)]
    slot_rng = np.random.default_rng(np.random.SeedSequence([seeds[0], len(seeds), 41873]))
    slot_rng.shuffle(kinds)
    return [sample_match(seed, checkpoint_dir, engine_registry, kind=kind)
            for seed, kind in zip(seeds, kinds)]


def make_opponent(spec):
    from .bots import (AlphaBetaBot, TacticalBot, ThreatBlockBot, StyleBot,
                       CheckpointBot, OpenSpielBot, ExternalProcessBot)
    if spec.kind == 'self':
        return None
    if spec.kind == 'alphabeta':
        try:
            from .cpp_env import CppAlphaBetaBot, is_cpp_available
            if is_cpp_available():
                return CppAlphaBetaBot(depth=spec.depth, node_budget=spec.nodes)
        except ImportError:
            pass
        return AlphaBetaBot(depth=spec.depth, node_budget=spec.nodes)
    if spec.kind == 'tactical':
        bot = TacticalBot(name=f'tactical-{spec.nodes}')
        bot.node_budget = spec.nodes
        return bot
    if spec.kind == 'threat':
        return ThreatBlockBot(node_budget=spec.nodes)
    if spec.kind == 'style':
        return StyleBot(spec.style)
    if spec.kind in ('history', 'best'):
        return CheckpointBot(spec.checkpoint, simulations=spec.simulations, device='cpu',
                             name='best-checkpoint' if spec.kind == 'best' else None)
    if spec.kind == 'openspiel':
        return OpenSpielBot(algorithm='mcts', simulations=spec.simulations,
                            seed=spec.bot_seed, name=f'openspiel-mcts-{spec.simulations}')
    if spec.kind == 'utttai':
        if not spec.engine_command:
            raise ValueError("uttt.ai training opponent is not configured; provide --engine-config with an 'utttai' entry")
        command = [part.replace('{simulations}', str(spec.simulations)).replace('{seed}', str(spec.bot_seed))
                   for part in spec.engine_command]
        return ExternalProcessBot(command=command, timeout=spec.engine_timeout,
                                  protocol=spec.engine_protocol,
                                  fallback=spec.engine_fallback,
                                  cwd=spec.engine_cwd or None,
                                  name=spec.engine_name or f'utttai-low-{spec.simulations}')
    raise ValueError(f'Unknown training opponent: {spec.kind}')
