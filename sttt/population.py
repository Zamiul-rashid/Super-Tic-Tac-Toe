"""Seeded, quota-controlled opponent diversity for population training."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
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


FAMILIES = tuple(POPULATION_WEIGHTS)

# M6: the per-family budget and noise ranges sample_match used to hold as
# literals. These defaults reproduce the previous sampler exactly; a JSON
# population config may override any of them per family.
DEFAULT_FAMILY_SETTINGS = {
    'self': {},
    'alphabeta': {'depth': [3, 4, 5, 6, 7, 8], 'nodes': [100000, 250000, 500000],
                  'epsilon': [0., .02], 'simulations': [128, 256, 512]},
    'tactical': {'depth': [2], 'nodes': [6000, 10000, 16000], 'epsilon': [0., .12],
                 'simulations': [128, 256, 512]},
    'threat': {'depth': [2], 'nodes': [6000, 12000, 24000], 'epsilon': [0., .08],
               'simulations': [128, 256, 512]},
    'history': {'depth': [3], 'nodes': [0], 'epsilon': [0., .03], 'simulations': [128, 256, 512]},
    'best': {'depth': [3], 'nodes': [0], 'epsilon': [0., .03], 'simulations': [128, 256, 512]},
    'openspiel': {'depth': [3], 'nodes': [0], 'epsilon': [0., .03],
                  'simulations': [128, 256, 512, 1024]},
    # Kept materially cheaper than the learner's 512 simulations.
    'utttai': {'depth': [3], 'nodes': [0], 'epsilon': [0., .02], 'simulations': [64, 128, 256]},
    'style': {'depth': [1], 'nodes': [0], 'epsilon': [0., .12], 'simulations': [128, 256, 512]},
}
_BUDGET_KEYS = ('depth', 'nodes', 'simulations')
POPULATION_CONFIG_VERSION = 1


@dataclass
class PopulationConfig:
    """Versioned curriculum: integer quotas summing to 100 plus per-family ranges."""
    quotas: dict
    families: dict = field(default_factory=dict)
    name: str = 'baseline'
    version: int = POPULATION_CONFIG_VERSION

    def __post_init__(self):
        self.quotas = {k: v for k, v in self.quotas.items()}
        merged = {}
        for family in FAMILIES:
            merged[family] = {**DEFAULT_FAMILY_SETTINGS[family], **(self.families.get(family) or {})}
        unknown = set(self.families) - set(FAMILIES)
        if unknown:
            raise ValueError(f'Unknown population families in settings: {sorted(unknown)}')
        self.families = merged
        self.validate()

    def validate(self):
        if self.version != POPULATION_CONFIG_VERSION:
            raise ValueError(f'Unsupported population config version {self.version!r}; '
                             f'expected {POPULATION_CONFIG_VERSION}')
        unknown = set(self.quotas) - set(FAMILIES)
        if unknown:
            raise ValueError(f'Unknown population families: {sorted(unknown)}')
        for family, quota in self.quotas.items():
            if isinstance(quota, bool) or not isinstance(quota, int):
                raise ValueError(f'Quota for {family!r} must be an integer, got {quota!r}')
            if quota < 0:
                raise ValueError(f'Quota for {family!r} is negative: {quota}')
        total = sum(self.quotas.values())
        if total != 100:
            raise ValueError(f'Population quotas must sum to 100, got {total}')
        for family, settings in self.families.items():
            for key in _BUDGET_KEYS:
                values = settings.get(key)
                if values is None:
                    continue
                if not isinstance(values, list) or not values:
                    raise ValueError(f'{family}.{key} must be a non-empty list, got {values!r}')
                floor = 0 if key == 'nodes' else 1
                for value in values:
                    if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                        raise ValueError(f'{family}.{key} contains an invalid budget {value!r} '
                                         f'(integers >= {floor} required)')
            epsilon = settings.get('epsilon')
            if epsilon is not None:
                ok = (isinstance(epsilon, list) and len(epsilon) == 2
                      and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in epsilon)
                      and 0. <= epsilon[0] <= epsilon[1] <= 1.)
                if not ok:
                    raise ValueError(f'{family}.epsilon must be [low, high] within [0, 1], got {epsilon!r}')

    def weights(self) -> dict:
        return {family: self.quotas.get(family, 0) / 100. for family in FAMILIES}

    def to_dict(self) -> dict:
        return {'version': self.version, 'name': self.name,
                'quotas': {family: int(self.quotas.get(family, 0)) for family in FAMILIES},
                'families': {family: dict(self.families[family]) for family in FAMILIES}}

    @classmethod
    def from_dict(cls, data: dict) -> 'PopulationConfig':
        if not isinstance(data, dict) or 'quotas' not in data:
            raise ValueError("Population config must be an object with a 'quotas' field")
        return cls(quotas=data['quotas'], families=data.get('families') or {},
                   name=str(data.get('name', 'unnamed')),
                   version=data.get('version', POPULATION_CONFIG_VERSION))

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()


def default_population_config() -> PopulationConfig:
    """The legacy hardcoded mix, as configuration. Backward-compatible default."""
    return PopulationConfig(quotas={k: round(100 * v) for k, v in POPULATION_WEIGHTS.items()},
                            name='baseline')


def load_population_config(path) -> PopulationConfig:
    with Path(path).open(encoding='utf-8') as handle:
        data = json.load(handle)
    return PopulationConfig.from_dict(data)


def population_quota_counts(games: int, offset: int = 0, config: PopulationConfig | None = None) -> dict[str, int]:
    """Allocate a slice of a balanced 100-game cycle; offset survives resume."""
    games = int(games)
    if games < 1:
        raise ValueError('At least one population game is required')
    quotas = (config or default_population_config()).quotas
    names = list(FAMILIES)
    # Smooth weighted round-robin gives exact percentages every 100 games,
    # including across small iterations. Reserving one of nine variants per
    # 16 games would overweight tactical bots and reduce self-play to 25%.
    credits = dict.fromkeys(names, 0)
    cycle = []
    for _ in range(100):
        for name in names:
            credits[name] += quotas.get(name, 0)
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
    # M6: the family the quota asked for. Differs from `kind` only for the
    # documented history/best -> self fallback, which used to be invisible in
    # the actual counts.
    requested_kind: str = ''


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


def sample_match(seed, checkpoint_dir=None, engine_registry=None, kind=None, config=None):
    """Sample one deterministic match specification.

    Missing history falls back to self-play and is recorded as such in
    ``requested_kind``. Each game gets independent random strength, seat,
    opening length, and move-noise settings drawn from ``config``. Never use
    latest.pt. ``kind`` is used by :func:`sample_matches` to enforce exact quotas.
    """
    config = config or default_population_config()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9127]))
    if kind is None:
        names = list(FAMILIES)
        kind = str(rng.choice(names, p=list(config.weights().values())))
    requested_kind = kind
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

    settings = config.families[kind]
    if kind == 'self':
        depth, nodes, epsilon = 3, 0, 0.
    else:
        depth = int(rng.choice(settings['depth']))
        nodes = int(rng.choice(settings['nodes']))
        low, high = settings['epsilon']
        epsilon = float(rng.uniform(low, high))

    simulations = int(rng.choice(settings.get('simulations', [128, 256, 512])))
    engine_fields = {'bot_seed': int(rng.integers(0, 2**31 - 1))}
    if kind == 'utttai':
        # Its command receives the budget through {simulations}.
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
                     opening_moves=opening_moves, requested_kind=requested_kind,
                     **engine_fields)


def sample_matches(seeds, checkpoint_dir=None, engine_registry=None, offset=0, config=None):
    """Build one randomized, quota-controlled league schedule for an iteration."""
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError('At least one population game is required')
    config = config or default_population_config()
    counts = population_quota_counts(len(seeds), offset, config=config)
    kinds = [kind for kind, count in counts.items() for _ in range(count)]
    slot_rng = np.random.default_rng(np.random.SeedSequence([seeds[0], len(seeds), 41873]))
    slot_rng.shuffle(kinds)
    return [sample_match(seed, checkpoint_dir, engine_registry, kind=kind, config=config)
            for seed, kind in zip(seeds, kinds)]


def family_report(results):
    """Per-family coverage for one iteration: games, learner positions, wall
    time, score rate, requested-vs-actual family and the opponent budgets used.

    Keyed by the family that actually played. A ``history`` request that fell
    back to ``self`` shows up under ``self`` with ``requested_as: {history: n}``.
    No engine here reports its internal search effort, so the budgets recorded
    are the ones the opponent was constructed with.
    """
    report = {}
    for trajectory, outcome, stats in results:
        match = stats['match']
        kind = match['kind']
        entry = report.setdefault(kind, {'games': 0, 'positions': 0, 'seconds': 0., 'score': 0.,
                                         'requested_as': {}, 'budgets': {k: {} for k in _BUDGET_KEYS}})
        entry['games'] += 1
        entry['positions'] += len(trajectory)
        entry['seconds'] += float(stats.get('seconds', 0.))
        learner_outcome = outcome * match['learner_side']
        entry['score'] += 1. if learner_outcome > 0 else (.5 if learner_outcome == 0 else 0.)
        requested = match.get('requested_kind') or kind
        entry['requested_as'][requested] = entry['requested_as'].get(requested, 0) + 1
        for key in _BUDGET_KEYS:
            bucket = entry['budgets'][key]
            value = str(match.get(key))
            bucket[value] = bucket.get(value, 0) + 1
    for entry in report.values():
        entry['score_rate'] = entry.pop('score') / entry['games']
        entry['seconds'] = round(entry['seconds'], 3)
    return report


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
