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
    from .replay_sampling import row_ply
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
            plies.append(row_ply(encoded[i]))
        kinds.append(spec.kind)
    return {'x': np.stack(xs), 'pi': np.stack(pis), 'mask': np.stack(masks),
            'z': np.asarray(zs, dtype=np.float64), 'q': np.stack(qs), 'q_mask': np.stack(qms),
            'ply': np.asarray(plies, dtype=np.int16), 'kind': kinds, 'games': games}


def pack_arrays(rows):
    import torch
    from .ai import REPLAY_PACKED_FORMAT
    return {'format': REPLAY_PACKED_FORMAT, 'count': int(rows['x'].shape[0]),
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


def load_shards(dataset_dir, max_positions=None, seed=0):
    import torch
    paths = sorted(Path(dataset_dir).glob('shard-*.pt'))
    if not paths:
        raise FileNotFoundError(f'no shard-*.pt under {dataset_dir}')

    if max_positions is None:
        selected = None
    else:
        if max_positions < 1:
            raise ValueError('max_positions must be positive when provided')
        manifest_path = Path(dataset_dir) / 'manifest.json'
        total = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            total = int(manifest.get('positions', 0)) or None
        if total is None:
            total = sum(int(torch.load(p, map_location='cpu', weights_only=False)['replay']['count'])
                        for p in paths)
        count = min(int(max_positions), total)
        selected = np.sort(np.random.default_rng(seed).choice(total, count, replace=False))

    parts = {k: [] for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask', 'ply')}
    offset = 0
    for p in paths:
        shard = torch.load(p, map_location='cpu', weights_only=False)
        if shard.get('format') != SHARD_FORMAT:
            raise ValueError(f'{p}: unexpected shard format {shard.get("format")!r}')
        r = shard['replay']
        shard_count = int(r['count'])
        if selected is None:
            indices = None
        else:
            start = int(np.searchsorted(selected, offset, side='left'))
            stop = int(np.searchsorted(selected, offset + shard_count, side='left'))
            indices = selected[start:stop] - offset
        if indices is not None and len(indices) == 0:
            offset += shard_count
            continue
        index_tensor = None if indices is None else torch.as_tensor(indices, dtype=torch.long)
        for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask'):
            parts[k].append(r[k] if index_tensor is None else r[k][index_tensor].clone())
        ply = shard['ply']
        parts['ply'].append(ply if index_tensor is None else ply[index_tensor].clone())
        offset += shard_count
    return {k: torch.cat(v) for k, v in parts.items()}


def _worker_init():
    # The box hangs at 32 x 100 %; workers are polite by construction.
    # os.nice genuinely works here. The env var is belt-and-braces only: by the time this
    # runs, the spawned child has already imported sttt.bootstrap (to unpickle this very
    # function) and therefore numpy/BLAS, so setting OMP_NUM_THREADS here is too late to
    # change BLAS threading -- the real cap is set in the parent before ctx.Pool() spawns.
    os.nice(10)
    os.environ.setdefault('OMP_NUM_THREADS', '1')


def _generate_task(task):
    seed, games, simulations, leaf_batch, share, depths = task
    return generate_rows(seed, games, simulations, leaf_batch, share, depths)


def generate_dataset(args):
    from .cpp_env import is_cpp_available
    from .evaluation import native_build_info
    if not is_cpp_available():
        raise RuntimeError('generate-dataset requires the native extension; run make -C cpp')
    if args.workers > 10:
        raise ValueError('--workers is capped at 10 on this machine (see CPU-load rule)')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # load_shards globs shard-*.pt unconditionally, so re-running into a
    # directory that already holds shards from a different seed/simulation/
    # depth configuration would silently merge them into the training set,
    # and the manifest.json write below would silently overwrite the record
    # of what is actually in the directory -- wrong data plus wrong
    # provenance, with nothing in the output to say so. Refuse up front,
    # before any task is built or any file is written.
    existing = sorted(output.glob('shard-*.pt'))
    if existing:
        raise RuntimeError(
            f'{output} already holds {len(existing)} shard file(s) '
            f'({existing[0].name}..{existing[-1].name}) from a previous generate-dataset '
            f'run; re-running here would silently merge shards from a possibly different '
            f'seed/simulation/depth configuration and overwrite manifest.json so it no '
            f'longer describes the directory. Pick a new --output directory, or remove '
            f'the existing shard-*.pt files first.')
    tasks = []
    remaining, shard = args.games, 0
    while remaining > 0:
        n = min(args.shard_games, remaining)
        tasks.append((int(args.seed) * 100003 + shard, n, args.simulations, args.leaf_batch,
                      args.alphabeta_share, tuple(args.depths)))
        remaining -= n
        shard += 1
    started = time.monotonic()
    totals = Counter(); kinds = Counter(); histogram = Counter(); positions = 0
    provenance = {'simulations': args.simulations, 'leaf_batch': args.leaf_batch,
                  'alphabeta_share': args.alphabeta_share, 'depths': list(args.depths),
                  'seed': args.seed, 'native_build': native_build_info()}
    # A spawned child must import sttt.bootstrap to unpickle _worker_init/_generate_task,
    # which imports numpy at module level (line 13) before _worker_init's body ever runs --
    # so setting OMP_NUM_THREADS there is too late to change BLAS threading (measured:
    # statistically indistinguishable from not setting it at all). Set it here instead, in
    # the parent, immediately before the pool spawns: each child inherits this in its
    # initial OS environment, ahead of any Python -- including the numpy import.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(args.workers, len(tasks)), initializer=_worker_init) as pool:
        for index, rows in enumerate(pool.imap(_generate_task, tasks)):
            summary = write_shard(output / f'shard-{index:04d}.pt', rows, provenance)
            positions += summary['positions']; kinds.update(summary['kinds'])
            histogram.update(summary['ply_histogram']); totals['games'] += summary['games']
            elapsed = time.monotonic() - started
            rate = positions / elapsed if elapsed else 0.0
            print(f'shard {index + 1}/{len(tasks)}: {summary["positions"]} positions, '
                  f'{positions} total, {rate:.0f} positions/s', flush=True)
    elapsed = time.monotonic() - started
    manifest = {'games': int(totals['games']), 'positions': positions, 'kinds': dict(kinds),
                'ply_histogram': dict(histogram), 'workers': args.workers,
                'elapsed_seconds': round(elapsed, 2),
                'positions_per_second': positions / elapsed if elapsed else None,
                'created': time.strftime('%Y-%m-%dT%H:%M:%S'), **provenance}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    return manifest


VALUE_COLLAPSE_STD = 0.02


def pretrain(args):
    # Stage 2: torch and its friends are imported HERE, not at module level --
    # sttt.bootstrap must stay importable (and torch-free) for spawned
    # generate-dataset workers, which unpickle _worker_init/_generate_task from
    # this module without ever needing a network.
    import torch
    from .ai import REPLAY_PACKED_FORMAT, resolve_device, pack_replay, unpack_replay
    from .learning import create_model, policy_value_loss, arch_name
    from .population import augment_batch
    from .replay_sampling import StratifiedSampler, ply_bin
    from .training_schedule import LRSchedule, apply_lr
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    data = load_shards(args.dataset, max_positions=args.max_positions, seed=args.seed)
    manifest_path = Path(args.dataset) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    n = data['x'].shape[0]
    order = rng.permutation(n)
    holdout = int(round(n * args.holdout))
    val_idx, train_idx = order[:holdout], order[holdout:]
    if len(train_idx) < args.batch:
        raise ValueError(f'{len(train_idx)} training rows is fewer than one batch of {args.batch}')
    bins = np.fromiter((ply_bin(int(p)) for p in data['ply'][train_idx].tolist()), dtype=np.int8,
                       count=len(train_idx))
    sampler = StratifiedSampler(bins)
    steps_per_epoch = max(1, len(train_idx) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    model = create_model(args.arch).to(device)
    use_fp16 = bool(args.fp16) and str(device) == 'cuda'
    model.use_fp16 = use_fp16
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    # Per-OPTIMIZER-STEP cosine over the whole pretraining run, unlike the online
    # trainer's per-ITERATION schedule -- there is no "iteration" concept here,
    # just steps, so the horizon is measured in the unit that actually advances.
    # Linear warm-up before the cosine. Without it, AdamW's first steps at
    # lr 1e-3 saturate the value head's tanh and its gradient reaches exactly
    # zero by step 8 -- measured for BOTH resnet and unet, 5 of 7 unet runs and
    # 1 of 1 resnet run at lr 1e-3, and survival flipped on batch order alone.
    # See docs/history/value-head-check.txt. Clamped so a tiny run still decays.
    # Capped at a quarter of the run as well as by the flag: on a very short run
    # `total_steps - 1` would leave the whole schedule ramping and never decaying.
    warmup_steps = max(0, min(int(getattr(args, 'lr_warmup', 100)), total_steps // 4))
    schedule = LRSchedule(kind='cosine', lr_start=args.lr, lr_min=args.lr_min, horizon=total_steps,
                          warmup=warmup_steps, completed=0, phase=0)
    print(f'pretrain: {args.arch}, {total_steps} steps ({steps_per_epoch}/epoch); '
          f'{schedule.describe()}', flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    def to_device(idx):
        idx = torch.as_tensor(idx)
        return [data[k][idx].to(device) for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask')]

    def evaluate_holdout():
        if holdout == 0:
            return {'val_policy': None, 'val_value': None, 'val_q': None}
        model.eval()
        sums, count, q_sum, q_n = np.zeros(2), 0, 0., 0
        with torch.no_grad():
            for start in range(0, holdout, 1024):
                x, pi, mask, z, q, qm = to_device(val_idx[start:start + 1024])
                _, parts = policy_value_loss(model, x, pi, mask, z.float(), use_fp16=use_fp16, q=q, q_mask=qm)
                sums += np.array([float(parts['policy']), float(parts['value'])]) * len(x)
                count += len(x)
                if parts['q'] is not None:
                    q_sum += float(parts['q']) * len(x); q_n += len(x)
        model.train()
        # float(...): every other metric in this function is cast from its
        # numpy/tensor origin before it reaches the checkpoint. Leaving these
        # two as numpy.float64 pickles a `numpy._core.multiarray.scalar`
        # reference into checkpoint['metrics'], which torch.load(weights_only=
        # True) -- what sttt.learning.load_model() uses on `train --resume` --
        # refuses to unpickle (measured: UnpicklingError on the exact resume
        # path this checkpoint exists to support).
        return {'val_policy': float(sums[0] / count), 'val_value': float(sums[1] / count),
                'val_q': (q_sum / q_n) if q_n else None}

    def value_spread():
        """Std of the value head's output over real positions.

        A saturated value head emits one constant, so std collapses to 0 while
        every other metric still looks plausible -- the only visible tell today
        is val_value being bit-identical across epochs, which nobody watches.
        Measured on the holdout when there is one, else on training rows.
        """
        probe = (val_idx[:2048] if holdout else train_idx[:2048])
        model.eval()
        with torch.no_grad():
            x = data['x'][torch.as_tensor(probe)].to(device)
            _, value, _ = model.forward_all(x)
        model.train()
        return float(value.float().std())

    model.train()
    last = None
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        losses, p_l, v_l, q_l = [], [], [], []
        for _ in range(steps_per_epoch):
            apply_lr(optimizer, schedule.current_lr)
            picks = train_idx[sampler.sample(args.batch, rng)]
            x, pi, mask, z, q, qm = to_device(picks)
            x, pi, mask, q, qm = augment_batch(x, pi, mask, rng, q, qm)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = policy_value_loss(model, x, pi, mask, z.float(), use_fp16=use_fp16, q=q, q_mask=qm)
            if not torch.isfinite(loss):
                raise RuntimeError(f'epoch {epoch}: nonfinite loss; refusing to continue')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update()
            schedule.advance()
            # .item(), not float(): these tensors still require grad here (the
            # graph is freed by backward() but the flag persists), and float()
            # on such a tensor emits a UserWarning -- measured on every
            # optimizer step of a multi-epoch pretraining run.
            losses.append(loss.item()); p_l.append(parts['policy'].item()); v_l.append(parts['value'].item())
            if parts['q'] is not None:
                q_l.append(parts['q'].item())
        spread = value_spread()
        row = {'epoch': epoch, 'lr': float(optimizer.param_groups[0]['lr']),
               'train_loss': float(np.mean(losses)), 'train_policy': float(np.mean(p_l)),
               'train_value': float(np.mean(v_l)), 'train_q': float(np.mean(q_l)) if q_l else None,
               **evaluate_holdout(), 'value_std': spread,
               'seconds': round(time.monotonic() - started, 2)}
        with (output / 'pretrain_metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        # Fail loudly rather than write a checkpoint whose value head is a
        # constant. Such a run reports a falling policy loss and a plausible
        # total loss while MCTS backups carry no positional information at all.
        if spread < VALUE_COLLAPSE_STD:
            raise RuntimeError(
                f'epoch {epoch}: the value head has collapsed -- its output std over real '
                f'positions is {spread:.2e} (threshold {VALUE_COLLAPSE_STD}), i.e. it emits a '
                f'near-constant value and its tanh gradient has vanished. Refusing to write a '
                f'checkpoint that would search blind. This is a learning-rate overshoot in the '
                f'first optimizer steps, not an architecture fault: retry with a lower --lr '
                f'(3e-4 and 1e-4 were both measured safe) or a longer --lr-warmup (currently '
                f'{warmup_steps} steps). See docs/history/value-head-check.txt.')
        last = row

    # Warm replay: a random subset of the dataset in the trainer's row format.
    warm = rng.choice(n, size=min(args.replay_buffer, n), replace=False)
    packed_source = {'format': REPLAY_PACKED_FORMAT, 'count': len(warm),
                     **{k: data[k][torch.as_tensor(warm)] for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask')}}
    rows = unpack_replay(packed_source)
    arch = arch_name(model)
    torch.save({'model': model.state_dict(), 'iteration': 0, 'arch': arch}, output / 'model-0000.pt')
    # lr_schedule is None on purpose: this schedule's horizon was measured in
    # pretraining STEPS, not the online trainer's ITERATIONS, so continuing it
    # would be meaningless. `train --resume` attaches its own fresh cosine
    # phase from --lr-schedule cosine instead (see sttt/training_schedule.py).
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': 0,
                  'arch': arch, 'replay': pack_replay(rows), 'replay_meta': [(0, 'bootstrap')] * len(rows),
                  'lr_schedule': None, 'scaler': scaler.state_dict() if use_fp16 else None,
                  'precision': 'fp16' if use_fp16 else 'fp32', 'training_config': vars(args),
                  'search_config': None, 'backend_info': None, 'population_games': 0,
                  'metrics': last, 'pretrain': {'dataset': str(args.dataset), 'manifest': manifest,
                                                 'epochs': args.epochs, 'positions': n, 'holdout': holdout}}
    torch.save(checkpoint, output / 'latest.tmp')
    (output / 'latest.tmp').replace(output / 'latest.pt')
    print(f'wrote {output / "latest.pt"} ({arch}, {len(rows)} warm replay rows)', flush=True)
    return checkpoint
