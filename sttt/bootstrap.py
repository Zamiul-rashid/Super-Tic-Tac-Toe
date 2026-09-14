"""Two-stage bootstrap: network-free C++ search generates an offline dataset;
`pretrain` fits a fresh network to it and emits a checkpoint `train --resume`
accepts. uttt.ai spent weeks on stage 1 with Python MCTS; the native engine
does it in hours.
"""
from collections import Counter
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from .population import MatchSpec, make_opponent
from .search import SearchConfig

SHARD_FORMAT = 'bootstrap-v1'
DEFAULT_DEPTHS = (4, 5, 6)


def sample_bootstrap_match(seed, stream, alphabeta_share, depths):
    """Self-play (both sides recorded) or heuristic-MCTS vs C++ alpha-beta."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream), 4409]))
    opening = int(rng.integers(0, 5))
    if rng.random() < alphabeta_share:
        return MatchSpec(kind='alphabeta', learner_side=int(rng.choice([1, -1])),
                         depth=int(rng.choice(list(depths))), nodes=500000,
                         epsilon=float(rng.uniform(0., .05)), opening_moves=opening,
                         requested_kind='alphabeta')
    return MatchSpec(kind='self', opening_moves=opening, requested_kind='self')


def generate_rows(seed, games, simulations, leaf_batch, alphabeta_share, depths):
    """Play `games` games with the model-free native tree; return numpy arrays.

    Runs inside a worker process. Imports the native pieces lazily so the
    parent can spawn torch-free workers.
    """
    from .cpp_env import CppTreeSearch, encode_batch, is_cpp_available
    from .selfplay import _play_game
    if not is_cpp_available() or CppTreeSearch is None:
        raise RuntimeError('bootstrap generation requires the native extension; run make -C cpp')
    xs, pis, masks, zs, qs, qms, plies, kinds = [], [], [], [], [], [], [], []
    for g in range(games):
        game_seed = int(np.random.SeedSequence([int(seed), g]).generate_state(1)[0])
        spec = sample_bootstrap_match(seed, g, alphabeta_share, depths)
        rng = np.random.default_rng(game_seed)
        tree = CppTreeSearch(None, rng, SearchConfig())
        opponent = make_opponent(spec)
        try:
            trajectory, outcome, _ = _play_game(tree, rng, simulations, game_seed, leaf_batch,
                                                spec, opponent, use_cpp=True)
        finally:
            if opponent is not None:
                opponent.close()
        if not trajectory:
            continue
        states = [s for s, *_ in trajectory]
        encoded = np.ascontiguousarray(encode_batch(states), dtype=np.float32)
        for i, (state, pi, q, q_mask) in enumerate(trajectory):
            mask = np.zeros(81, dtype=bool)
            mask[state.legal_actions()] = True
            xs.append(encoded[i].copy()); pis.append(pi.astype(np.float32)); masks.append(mask)
            zs.append(float(outcome * state.turn)); qs.append(q); qms.append(q_mask)
            plies.append(81 - int(round(float(encoded[i][:81].sum()))))
        kinds.append(spec.kind)
    return {'x': np.stack(xs), 'pi': np.stack(pis), 'mask': np.stack(masks),
            'z': np.asarray(zs, dtype=np.float64), 'q': np.stack(qs), 'q_mask': np.stack(qms),
            'ply': np.asarray(plies, dtype=np.int16), 'kind': kinds, 'games': games}


def pack_arrays(rows):
    import torch
    return {'format': 'packed-v2', 'count': int(rows['x'].shape[0]),
            'x': torch.from_numpy(rows['x']), 'pi': torch.from_numpy(rows['pi']),
            'mask': torch.from_numpy(rows['mask']), 'z': torch.from_numpy(rows['z']),
            'q': torch.from_numpy(rows['q']), 'q_mask': torch.from_numpy(rows['q_mask'])}


def write_shard(path, rows, provenance):
    import torch
    from .replay_sampling import BIN_LABELS, ply_bin
    histogram = Counter(BIN_LABELS[ply_bin(p)] for p in rows['ply'].tolist())
    payload = {'format': SHARD_FORMAT, 'replay': pack_arrays(rows),
               'ply': torch.from_numpy(rows['ply']), 'kind': list(rows['kind']),
               'provenance': dict(provenance)}
    path = Path(path)
    torch.save(payload, path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)
    return {'positions': int(rows['x'].shape[0]), 'games': int(rows['games']),
            'kinds': dict(Counter(rows['kind'])), 'ply_histogram': dict(histogram)}


def load_shards(dataset_dir):
    import torch
    paths = sorted(Path(dataset_dir).glob('shard-*.pt'))
    if not paths:
        raise FileNotFoundError(f'no shard-*.pt under {dataset_dir}')
    parts = {k: [] for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask', 'ply')}
    for p in paths:
        shard = torch.load(p, map_location='cpu', weights_only=False)
        if shard.get('format') != SHARD_FORMAT:
            raise ValueError(f'{p}: unexpected shard format {shard.get("format")!r}')
        r = shard['replay']
        for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask'):
            parts[k].append(r[k])
        parts['ply'].append(shard['ply'])
    return {k: torch.cat(v) for k, v in parts.items()}
