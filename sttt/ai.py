"""Run with python -m sttt.ai train|play|evaluate|tournament."""
import argparse
from collections import Counter, deque
from dataclasses import asdict
import json
from pathlib import Path
import os
import time
import numpy as np
import torch
from .env import State
from .training_schedule import LRSchedule, apply_lr, build_schedule
from .learning import arch_name, create_model, encode, load_model, policy_value_loss
from .opponent import Opponent, policies, NAMES
from .search import TreeSearch, SearchConfig
from .selfplay import SelfPlayPool
from .population import (sample_matches, augment_batch, family_report,
                         default_population_config, load_population_config, PopulationConfig)
from .replay_sampling import StratifiedSampler, ply_bin, row_ply, sample_bin_fractions
from .benchmarks.harness import StageTimer
from .bots import TacticalBot, AlphaBetaBot, CheckpointBot, create_bot, Bot
from .reports import new_report, write_report, write_tournament_report
from .tournament import run_tournament, run_simulation_sweep
from .engine_registry import create_configured_engine, load_engine_registry
from .bootstrap import generate_dataset, pretrain
from .reanalysis import reanalyse_cmd

DEFAULT_TRAIN_WORKERS = 8

try:
    from .cpp_env import encode_batch as cpp_encode_batch, is_cpp_available
    _HAS_CPP = is_cpp_available()
except ImportError:
    _HAS_CPP = False
    cpp_encode_batch = None

def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value

def search_config(args):
    return SearchConfig(soft_pruning=not getattr(args, 'no_soft_pruning', False),
                        proofs=not getattr(args, 'no_proofs', False),
                        reuse=not getattr(args, 'no_reuse', False))


class RunOwnership:
    """Exclusive per-output-directory lock held for the whole run.

    Two trainers pointed at one output directory would interleave writes to
    latest.pt and to the same snapshot names, each overwriting the other's
    iterations while both reported progress. The lock is taken before any
    worker is spawned so the loser exits without having started a pool.
    """

    def __init__(self, output):
        self.path = Path(output) / '.run.lock'
        self._handle = None

    def acquire(self):
        import fcntl
        self._handle = open(self.path, 'w')
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                f'another training process already owns {self.path.parent} '
                f'(lock: {self.path}). Use a different --output; two writers '
                f'would overwrite each other\'s checkpoints.')
        self._handle.write(f'{os.getpid()}\n')
        self._handle.flush()
        return self

    def release(self):
        if self._handle is not None:
            import fcntl
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def resolve_device(device):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Use --device cpu or check NVIDIA/PyTorch access.')
    return device

def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    # M2: one explicit resolution, recorded and logged. The old boolean kept no
    # record of what `auto` chose, so worker handshakes and checkpoints could not
    # report the backend actually used.
    from .backends import resolve_backend, describe_backend
    backend = getattr(args, 'backend', 'auto')
    backend_info = resolve_backend(backend)
    args._backend_info = backend_info
    use_cpp = backend_info['actual'] == 'cpp'
    args._use_cpp = use_cpp
    print(describe_backend(backend_info), flush=True)
    if args.resume:
        model, saved = load_model(args.resume)
        arch = arch_name(model)
    else:
        arch = getattr(args, 'arch', 'resnet')
        model = create_model(arch)
        saved = {}
    model.to(device)
    if getattr(args, 'fp16', False):
        model.use_fp16 = True
        if str(device) == 'cuda':
            print('Using FP16 Automatic Mixed Precision on CUDA', flush=True)
    if 'numpy_rng_state' in saved:
        rng.bit_generator.state = saved['numpy_rng_state']
    # Torch's stream drives dropout and any torch-side sampling; without
    # restoring it a resumed run diverged from an uninterrupted one even with
    # the numpy stream restored. Absent in legacy checkpoints, which simply
    # keep the process default.
    if saved.get('torch_rng_state') is not None:
        torch.set_rng_state(torch.as_tensor(saved['torch_rng_state'], dtype=torch.uint8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # M4: one GradScaler per PROCESS. It used to be constructed inside the
    # iteration loop, so every iteration threw away the loss scale the previous
    # one had converged on and restarted the scale-discovery ramp -- silently
    # skipping optimizer updates at the start of each iteration, forever.
    use_fp16 = getattr(args, 'fp16', False) and str(device) == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    if use_fp16 and saved.get('scaler'):
        scaler.load_state_dict(saved['scaler'])
    if 'optimizer' in saved:
        optimizer.load_state_dict(saved['optimizer'])
    # M5: built AFTER optimizer state is restored, so an explicit LR override
    # lands on restored moments rather than discarding them. A legacy resume
    # with no saved schedule inherits its restored optimizer LR, so continuing
    # a decayed run never silently jumps back to 1e-3.
    legacy_lr = optimizer.param_groups[0]['lr'] if 'optimizer' in saved else None
    schedule = build_schedule(args, saved.get('lr_schedule'), legacy_lr=legacy_lr)
    apply_lr(optimizer, schedule.current_lr)
    print(schedule.describe(), flush=True)
    replay = deque(unpack_replay(saved.get('replay')), maxlen=args.buffer)
    # M6: parallel per-sample origin (iteration, family) so sampled-position
    # fractions and age are measurable without touching the target tuple.
    # Legacy checkpoints carry none; their samples are tagged as such.
    legacy_meta = [(int(saved.get('iteration', 0)), 'legacy')] * len(replay)
    replay_meta = deque(saved.get('replay_meta') or legacy_meta, maxlen=args.buffer)
    if len(replay_meta) != len(replay):
        raise ValueError(f'checkpoint replay ({len(replay)}) and replay_meta ({len(replay_meta)}) disagree')
    # Game-phase bins for stratified sampling; derived from each row, so legacy
    # checkpoints need no migration. Kept parallel to `replay` like replay_meta.
    replay_bins = deque((ply_bin(row_ply(row[0])) for row in replay), maxlen=args.buffer)
    population_config, population_games, population_phases = resolve_population_config(args, saved)
    print(f'population config: {population_config.name} sha256={population_config.sha256[:12]} '
          f'cursor={population_games}' + (f' phases={len(population_phases)}' if population_phases else ''),
          flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ownership = RunOwnership(output).acquire()
    engine_registry = load_engine_registry(args.engine_config) if getattr(args, 'engine_config', None) else {}
    if getattr(args, 'population', False):
        from .engine_runtime import activate_runtime
        activate_runtime()
        if not engine_registry.get('utttai') and not engine_registry.get('utttai-low'):
            raise ValueError("Population league requires --engine-config with an 'utttai' or 'utttai-low' entry")
        if not any('{simulations}' in part for part in _utttai_command_parts(engine_registry)):
            raise ValueError("The configured uttt.ai command must include the {simulations} placeholder")
        try:
            import pyspiel  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('Population league requires the pyspiel OpenSpiel package') from exc
        if Path(pyspiel.__file__).suffix == '.py':
            raise RuntimeError('Official compiled OpenSpiel is required; install engines/requirements.txt')
        # Exercise the real wrapper before launching search workers.
        from .population import make_opponent, sample_match
        probe = make_opponent(sample_match(args.seed, engine_registry=engine_registry, kind='utttai',
                                           config=population_config))
        try:
            probe.choose(State(), np.random.default_rng(args.seed))
        finally:
            probe.close()
    print(f'{device}: {sum(p.numel() for p in model.parameters()):,} parameters ({arch}); workers={args.workers}; replay stays in RAM', flush=True)
    try:
        with SelfPlayPool(min(args.workers, args.games), args.inference_batch, args.inference_wait_ms) as pool:
            _train_loop(args, model, saved, arch, optimizer, replay, output, rng, device, pool,
                        engine_registry, scaler=scaler, use_fp16=use_fp16, schedule=schedule,
                        population=(population_config, population_games, population_phases),
                        replay_meta=replay_meta, replay_bins=replay_bins)
    finally:
        ownership.release()


def snapshot_iteration(path):
    """Iteration number encoded in a model-NNNN.pt name, or None."""
    try:
        return int(Path(path).stem.split('-')[1])
    except (ValueError, IndexError):
        return None


def prune_snapshots(output, iteration, window=500, keep=10):
    """Delete model-*.pt snapshots outside the retention policy.

    Two independent caps, both applied: drop anything older than `window`
    iterations, and keep at most the most recent `keep` snapshots. Either is
    0 to disable that cap. The count cap matters because the age cap only
    yields a predictable file count while --save-every divides --keep-
    checkpoint-window; an unaligned --eval-every adds extra snapshots.

    best.pt and latest.pt are never matched, so a pinned best and the resume
    point always survive. Returns the deleted paths.
    """
    output = Path(output)
    snapshots = []
    for path in output.glob('model-*.pt'):
        found = snapshot_iteration(path)
        if found is not None:
            snapshots.append((found, path))
    snapshots.sort()
    doomed = set()
    if window and window > 0:
        cutoff = iteration - window
        doomed.update(path for found, path in snapshots if found <= cutoff)
    if keep and keep > 0 and len(snapshots) > keep:
        doomed.update(path for _, path in snapshots[:-keep])
    for path in sorted(doomed):
        path.unlink(missing_ok=True)
    return sorted(doomed)


REPLAY_PACKED_FORMAT = 'packed-v2'
REPLAY_PACKED_FORMATS = ('packed-v1', 'packed-v2')


def widen_row(row):
    """Legacy (x, pi, mask, z) -> (x, pi, mask, z, q, q_mask) with no Q targets."""
    if len(row) == 6:
        return row
    x, pi, mask, z = row
    return (x, pi, mask, z, torch.zeros(81, dtype=torch.float32), torch.zeros(81, dtype=torch.bool))


def pack_replay(replay):
    """Stack the replay's (x, pi, mask, z, q, q_mask) rows into tensors for saving.

    Pickling 200k positions as 600k tiny tensors cost ~10 s per checkpoint
    write, a quarter of every iteration. Contiguous tensors serialise at copy
    speed. Only the on-disk form changes; the in-RAM deque of tuples that the
    optimizer samples from is untouched.

    The Q/Q-mask block (~65 MB + ~16 MB at a 200,000-row buffer) is omitted
    entirely when no row has a set q_mask bit: for `resnet`, ResNet.forward_all
    always returns q=None and policy_value_loss never reads a Q target back,
    so those bytes would be structurally zero on every write of latest.pt
    (measured: ~320 MB -> ~400 MB). `unet` rows carry real targets, so their
    Q block is written as before.
    """
    rows = [widen_row(r) for r in replay]
    if not rows:
        return {'format': REPLAY_PACKED_FORMAT, 'count': 0}
    packed = {'format': REPLAY_PACKED_FORMAT, 'count': len(rows),
              'x': torch.stack([r[0] for r in rows]),
              'pi': torch.stack([r[1] for r in rows]),
              'mask': torch.stack([r[2] for r in rows]),
              'z': torch.tensor([r[3] for r in rows], dtype=torch.float64)}
    q_mask = torch.stack([r[5] for r in rows])
    if bool(q_mask.any()):
        packed['q'] = torch.stack([r[4] for r in rows])
        packed['q_mask'] = q_mask
    return packed


def unpack_replay(saved):
    """Rows from packed-v1, packed-v2 or a legacy list; always 6-tuples."""
    if saved is None:
        return []
    if isinstance(saved, dict) and saved.get('format') in REPLAY_PACKED_FORMATS:
        n = int(saved['count'])
        if n == 0:
            return []
        # Zero-fill whenever either key is ABSENT, not keyed on the format
        # string: a packed-v2 dict with no real Q targets (pack_replay omits
        # the keys in that case) must load exactly like a packed-v1 one.
        q, q_mask = saved.get('q'), saved.get('q_mask')
        if q is None or q_mask is None:
            q = torch.zeros(n, 81, dtype=torch.float32)
            q_mask = torch.zeros(n, 81, dtype=torch.bool)
        # clone() so a row does not keep the whole packed block alive after
        # the deque has evicted its neighbours.
        return [(x.clone(), pi.clone(), mask.clone(), z, qq.clone(), qm.clone())
                for x, pi, mask, z, qq, qm in
                zip(saved['x'].unbind(0), saved['pi'].unbind(0), saved['mask'].unbind(0),
                    saved['z'].tolist(), q.unbind(0), q_mask.unbind(0))]
    return [widen_row(r) for r in saved]


def replay_length(saved):
    if isinstance(saved, dict) and saved.get('format') in REPLAY_PACKED_FORMATS:
        return int(saved['count'])
    return len(saved or [])


def resolve_population_config(args, saved):
    """M6: which curriculum this run trains under, and where its quota cursor is.

    Rule: an omitted --population-config inherits the saved configuration; an
    explicit one that differs from the saved one is a recorded phase change and
    restarts the 100-game cursor, because the old cursor indexes a cycle built
    from the old quotas. Legacy checkpoints carry no config and inherit the
    default while keeping their cursor.

    Returns (config, population_games_cursor, phases).
    """
    explicit = getattr(args, 'population_config', None)
    saved_dict = saved.get('population_config')
    saved_config = PopulationConfig.from_dict(saved_dict) if saved_dict else None
    cursor = int(saved.get('population_games', 0))
    phases = list(saved.get('population_phases', []))
    if explicit:
        config = load_population_config(explicit)
    elif saved_config is not None:
        config = saved_config
    else:
        config = default_population_config()
    if saved_config is not None and config.sha256 != saved_config.sha256:
        phases.append({'iteration': int(saved.get('iteration', 0)),
                       'from_name': saved_config.name, 'from_sha256': saved_config.sha256,
                       'to_name': config.name, 'to_sha256': config.sha256,
                       'source': str(explicit)})
        cursor = 0
    return config, cursor, phases


def _utttai_command_parts(engine_registry):
    config = (engine_registry or {}).get('utttai') or (engine_registry or {}).get('utttai-low') or {}
    command = config.get('command', ())
    return command.split() if isinstance(command, str) else tuple(command)


def _train_loop(args, model, saved, arch, optimizer, replay, output, rng, device, pool,
                engine_registry=None, scaler=None, use_fp16=False, schedule=None,
                population=None, replay_meta=None, replay_bins=None):
    start_iteration = saved.get('iteration', 0)
    if population is None:
        population = resolve_population_config(args, saved)
    population_config, population_games, population_phases = population
    if replay_meta is None:
        replay_meta = deque([(start_iteration, 'legacy')] * len(replay), maxlen=replay.maxlen)
    if replay_bins is None:
        replay_bins = deque((ply_bin(row_ply(row[0])) for row in replay), maxlen=replay.maxlen)
    max_iter = getattr(args, 'max_iterations', None)
    end_iteration = min(start_iteration + args.iterations, max_iter) if max_iter else (start_iteration + args.iterations)
    for iteration in range(start_iteration + 1, end_iteration + 1):
        started = time.monotonic()
        # M8: non-overlapping wall-clock stages. Worker time is recorded as an
        # aggregate, never summed into elapsed runtime.
        stage_timer = StageTimer()
        seeds = rng.integers(0, 10**9, size=args.games)
        matches = None
        if getattr(args, 'population', False):
            history = getattr(args, 'population_checkpoints', None) or str(output)
            matches = sample_matches(seeds, history, engine_registry, offset=population_games,
                                     config=population_config)
        use_cpp = getattr(args, '_use_cpp', False)
        with stage_timer.stage('selfplay'):
            results, inference_stats = pool.run(
                model, seeds, args.simulations, search_config(args), args.leaf_batch,
                matches=matches, use_cpp=use_cpp
            )
        selfplay_seconds = stage_timer.stages['selfplay']
        # Owner-side inference happens inside the self-play stage; it is a
        # component of it, not an extra stage.
        stage_timer.add_aggregate('owner_inference', inference_stats['inference_seconds'], workers=1)
        stage_timer.add_aggregate('worker_game_time', sum(r[2].get('seconds', 0.) for r in results),
                                  workers=min(args.workers, args.games))
        if matches is not None:
            population_games += len(matches)
        replay_started = time.monotonic()
        for game_idx, (trajectory, outcome, stats) in enumerate(results):
            origin = (iteration, stats['match']['kind'])
            if trajectory and use_cpp and _HAS_CPP and cpp_encode_batch is not None:
                traj_states = [s for s, *_ in trajectory]
                encoded_batch = cpp_encode_batch(traj_states)
                for i, (state, pi, q, q_mask) in enumerate(trajectory):
                    mask = np.zeros(81, dtype=bool)
                    mask[state.legal_actions()] = True
                    replay.append((torch.from_numpy(encoded_batch[i].copy()), torch.from_numpy(pi),
                                   torch.from_numpy(mask), float(outcome * state.turn),
                                   torch.from_numpy(q), torch.from_numpy(q_mask)))
                    replay_meta.append(origin)
                    replay_bins.append(ply_bin(row_ply(replay[-1][0])))
            else:
                for state, pi, q, q_mask in trajectory:
                    mask = np.zeros(81, dtype=bool)
                    mask[state.legal_actions()] = True
                    replay.append((torch.from_numpy(encode(state)), torch.from_numpy(pi),
                                   torch.from_numpy(mask), float(outcome * state.turn),
                                   torch.from_numpy(q), torch.from_numpy(q_mask)))
                    replay_meta.append(origin)
                    replay_bins.append(ply_bin(row_ply(replay[-1][0])))
            print(f'iteration {iteration}: game {game_idx+1}/{args.games}, {len(trajectory)} training positions, '
                  f'opponent={stats["match"]["kind"]}', flush=True)

        stage_timer.add('replay_preparation', time.monotonic() - replay_started)
        reanalysis_report = _review_iteration(args, iteration, seeds, results, model, pool, use_cpp,
                                              replay, replay_meta, replay_bins, stage_timer)

        model.train()
        optimization_started = time.monotonic()
        if schedule is not None:
            apply_lr(optimizer, schedule.current_lr)
        losses = []
        policy_losses, value_losses, q_losses, grad_norms = [], [], [], []
        optimizer_updates = 0
        skipped_updates = 0
        sampled_kinds, sampled_ages = Counter(), []
        sampler = StratifiedSampler(np.fromiter(replay_bins, dtype=np.int8, count=len(replay_bins)))
        sampled_bins = []
        for _ in range(args.steps):
            if getattr(args, 'replay_sampling', 'uniform') == 'stratified':
                indices = sampler.sample(min(args.batch, len(replay)), rng)
            else:
                indices = rng.choice(len(replay), size=min(args.batch, len(replay)), replace=False)
            sampled_bins.append(indices)
            batch = [replay[int(i)] for i in indices]
            # M6: what the optimizer actually saw this iteration, by family and
            # age. Game quota is not replay quota; this is the replay side.
            for i in indices:
                origin_iteration, origin_kind = replay_meta[int(i)]
                sampled_kinds[origin_kind] += 1
                sampled_ages.append(iteration - origin_iteration)
            x, pi, mask, q, q_mask = [torch.stack([row[k] for row in batch]).to(device) for k in (0, 1, 2, 4, 5)]
            if getattr(args, 'augment_symmetry', False):
                x, pi, mask, q, q_mask = augment_batch(x, pi, mask, rng, q, q_mask)
            z = torch.tensor([row[3] for row in batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = policy_value_loss(model, x, pi, mask, z, use_fp16=use_fp16, q=q, q_mask=q_mask)
            policy_loss, value_loss = parts['policy'], parts['value']

            # Never commit an iteration built on a nonfinite loss: the optimizer
            # would poison every parameter and the run would continue reporting
            # progress.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f'iteration {iteration}: nonfinite training loss '
                    f'(policy={policy_loss.item()}, value={value_loss.item()}); '
                    f'refusing to apply the update')

            if use_fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Measured BEFORE clipping, so the number reports the real
                # gradient magnitude rather than the clip threshold.
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # step() is a no-op when the unscaled gradients are nonfinite;
                # AMP signals that by lowering the scale.
                if scaler.get_scale() < scale_before:
                    skipped_updates += 1
                else:
                    optimizer_updates += 1
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer_updates += 1
            losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            if parts['q'] is not None:
                # .item(), not float(): parts['q'] still requires grad here
                # (the graph is freed but the flag persists), and float() on
                # such a tensor emits a UserWarning -- measured on every
                # optimizer step of a multi-hour run.
                q_losses.append(parts['q'].item())
            grad_norms.append(float(grad_norm))
        # CUDA kernels are asynchronous: without this the optimization stage
        # would appear instant and its cost would land in whatever synchronized
        # next.
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        stage_timer.add('optimization', time.monotonic() - optimization_started)

        if args.steps and optimizer_updates == 0:
            raise RuntimeError(
                f'iteration {iteration}: all {args.steps} optimizer updates were '
                f'skipped by the AMP scaler; no training occurred. Refusing to '
                f'save this iteration as progress.')

        lr = optimizer.param_groups[0]['lr']
        # Advance only now: the optimization block finished, so this iteration
        # is genuinely complete. An interrupted iteration must not move the
        # curve, or a resume would skip a step.
        next_lr = schedule.advance() if schedule is not None else lr
        print(f'iteration {iteration}: loss={np.mean(losses):.4f} '
              f'policy={np.mean(policy_losses):.4f} value={np.mean(value_losses):.4f} '
              f'lr={lr:.3e} grad_norm={np.mean(grad_norms):.3f} '
              f'next_lr={next_lr:.3e} '
              f'updates={optimizer_updates} skipped={skipped_updates}'
              + (f' q_loss={np.mean(q_losses):.4f}' if q_losses else '')
              + (f' scale={scaler.get_scale():.0f}' if use_fp16 else ''), flush=True)

        # M4: the checkpoint schema now carries everything a full resume needs.
        # Scaler and torch RNG state were absent, so resuming an fp16 run
        # restarted loss-scale discovery and drew a different augmentation
        # stream than an uninterrupted run would have.
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'iteration': iteration, 'replay': pack_replay(replay), 'arch': arch,
                      'training_config': vars(args), 'search_config': asdict(search_config(args)),
                      'numpy_rng_state': rng.bit_generator.state,
                      'torch_rng_state': torch.get_rng_state(),
                      'scaler': scaler.state_dict() if use_fp16 else None,
                      'precision': 'fp16' if use_fp16 else 'fp32',
                      'lr_schedule': schedule.state_dict() if schedule is not None else None,
                      'backend_info': getattr(args, '_backend_info', None),
                      'population_games': population_games,
                      # M6: the curriculum this checkpoint was trained under, so a
                      # budget/mix change is reproducible from the checkpoint alone.
                      'population_config': population_config.to_dict(),
                      'population_config_sha256': population_config.sha256,
                      'population_phases': population_phases,
                      'replay_meta': list(replay_meta),
                      'metrics': {'loss': float(np.mean(losses)),
                                  'policy_loss': float(np.mean(policy_losses)),
                                  'value_loss': float(np.mean(value_losses)),
                                  # None for models without a Q head (or a batch with
                                  # no visited-action targets) rather than omitted, so
                                  # a later reader can distinguish "no Q head" from
                                  # "key not written yet".
                                  'q_loss': float(np.mean(q_losses)) if q_losses else None,
                                  'lr': float(lr),
                                  'next_lr': float(next_lr),
                                  'grad_norm': float(np.mean(grad_norms)),
                                  'optimizer_updates': optimizer_updates,
                                  'skipped_updates': skipped_updates}}
        checkpoint_started = time.monotonic()
        temporary = output / 'latest.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(output / 'latest.pt')
        # Small historical weights retained separately for evaluation and milestones.
        save_snapshot = (getattr(args, 'save_every', 50) and iteration % args.save_every == 0) or (args.eval_every and iteration % args.eval_every == 0)
        if save_snapshot:
            snapshot = output / f'model-{iteration:04d}.pt'
            torch.save({'model': model.state_dict(), 'iteration': iteration, 'arch': arch}, snapshot.with_suffix('.tmp'))
            snapshot.with_suffix('.tmp').replace(snapshot)
            prune_snapshots(output, iteration,
                            window=getattr(args, 'keep_checkpoint_window', 500),
                            keep=getattr(args, 'keep_checkpoints', 10))
        stage_timer.add('checkpoint_write', time.monotonic() - checkpoint_started)
        report = {'iteration': iteration, 'positions': len(replay), 'loss': float(np.mean(losses)),
                  'q_loss': float(np.mean(q_losses)) if q_losses else None,
                  'seconds': round(time.monotonic() - started, 2), 'selfplay_seconds': selfplay_seconds,
                  'games': args.games, 'simulations': args.simulations, 'leaf_batch': args.leaf_batch,
                  'search_config': asdict(search_config(args)), **inference_stats,
                  'population_matches': [r[2]['match'] for r in results],
                  'population_match_counts': dict(Counter(r[2]['match']['kind'] for r in results)),
                  'population_requested_counts': dict(Counter(
                      r[2]['match'].get('requested_kind') or r[2]['match']['kind'] for r in results)),
                  'population_config_sha256': population_config.sha256,
                  'population_config_name': population_config.name,
                  'population_games_cursor': population_games,
                  'population_by_family': family_report(results),
                  'replay_sample_kind_fractions': {k: v / max(1, sum(sampled_kinds.values()))
                                                   for k, v in sampled_kinds.items()},
                  'replay_sample_age': ({'min': int(min(sampled_ages)), 'mean': float(np.mean(sampled_ages)),
                                         'max': int(max(sampled_ages))} if sampled_ages else None),
                  'replay_sampling': getattr(args, 'replay_sampling', 'uniform'),
                  'replay_depth_histogram': sampler.histogram(),
                  'replay_sample_depth_fractions': sample_bin_fractions(
                      sampler.bins, np.concatenate(sampled_bins) if sampled_bins else np.empty(0, dtype=int)),
                  'completed_simulations': sum(r[2]['completed_simulations'] for r in results),
                  'max_search_depth': max(r[2]['max_depth'] for r in results),
                  'soft_rechecks': sum(r[2]['soft_rechecks'] for r in results),
                  'hard_pruned_choices': sum(r[2]['hard_pruned_choices'] for r in results),
                  'retained_visits': sum(r[2]['retained_visits'] for r in results)}
        if device.startswith('cuda'):
            report['peak_gpu_mb'] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            report['reserved_gpu_mb'] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)
        # M8: throughput denominators are completed work, and the search rate
        # uses simulations actually completed, not the budget requested.
        elapsed = time.monotonic() - started
        learner_positions = sum(len(r[0]) for r in results)
        report['throughput'] = {
            'selfplay_games_per_sec': args.games / selfplay_seconds if selfplay_seconds else None,
            'learner_positions_per_sec': learner_positions / selfplay_seconds if selfplay_seconds else None,
            'learner_positions': learner_positions,
            'completed_simulations_per_sec': (report['completed_simulations'] / selfplay_seconds
                                              if selfplay_seconds else None),
            'requested_simulations': args.games * args.simulations,
            'inference_batch_occupancy': (inference_stats['mean_inference_batch']
                                          / max(1, args.inference_batch)),
            'iteration_seconds': elapsed,
        }
        report['stage_seconds'] = stage_timer.report(elapsed)
        if reanalysis_report is not None:
            report['reanalysis'] = reanalysis_report
        with (output / 'metrics.jsonl').open('a') as file:
            file.write(json.dumps(report) + '\n')
        print(json.dumps(report), flush=True)
        if args.eval_every and iteration % args.eval_every == 0:
            evaluation_started = time.monotonic()
            evaluation = argparse.Namespace(**vars(args))
            evaluation.checkpoint = str(output / f'model-{iteration:04d}.pt')
            evaluation.output = str(output / 'evaluations')
            evaluation.games, evaluation.simulations = args.eval_games, args.eval_simulations
            evaluation.opponent = 'alphabeta'
            evaluation.opponent_checkpoint = None
            evaluation.opponent_depth = getattr(args, 'eval_opponent_depth', 4)
            evaluation.opponent_nodes = getattr(args, 'eval_opponent_nodes', 50000)
            evaluation.opponent_simulations = 256
            evaluate(evaluation)
            # Periodic evaluation is real wall time the ETA must include; it
            # lands after the metrics row, so it is appended to its own file.
            with (output / 'metrics.jsonl').open('a') as file:
                file.write(json.dumps({'iteration': iteration,
                                       'evaluation_overhead_seconds':
                                           time.monotonic() - evaluation_started}) + '\n')

def _review_iteration(args, iteration, seeds, results, model, pool, use_cpp,
                      replay, replay_meta, replay_bins, stage_timer):
    """Opt-in game records and loss-review reanalysis for one iteration.

    Returns the metrics entry, or None when both are off (the default), in which
    case nothing here runs and no RNG is touched.
    """
    records_dir = getattr(args, 'save_game_records', None)
    if not records_dir and not getattr(args, 'reanalyse', False):
        return None
    from .reanalysis import (game_record, game_report, reanalysis_rows, select_tasks,
                             strong_opponents, summarize, update_opponent_scores, write_jsonl)
    records = [game_record(iteration, i, seeds[i], *result) for i, result in enumerate(results)]
    # ponytail: in-memory per-run scores, rebuilt after a resume; persist in the checkpoint if that matters.
    scores = update_opponent_scores(args.__dict__.setdefault('_opponent_scores', {}), records)
    strong = strong_opponents(scores, getattr(args, 'reanalyse_strong_threshold', .5))
    if records_dir:
        write_jsonl(Path(records_dir) / f'games-{iteration:04d}.jsonl', records)
    if not getattr(args, 'reanalyse', False):
        return None
    started = time.monotonic()
    new_positions = sum(len(r[0]) for r in results)
    budget = int(args.reanalyse_fraction * new_positions)
    # Own stream, so enabling review never shifts the self-play/sampling RNG.
    rng = np.random.default_rng(np.random.SeedSequence([int(args.seed), int(iteration), 5501]))
    tasks = select_tasks(records, args.reanalyse_outcomes, args.reanalyse_sides, budget, rng,
                         args.reanalyse_margin, priority=strong)
    simulations = args.reanalyse_simulations or 4 * args.simulations
    reviews, inference = pool.reanalyse(model, tasks, simulations, search_config(args),
                                        args.leaf_batch, use_cpp=use_cpp)
    rows = reanalysis_rows(tasks, reviews, args.value_target, args.value_lambda)
    for row in rows:
        replay.append(row)
        replay_meta.append((iteration, 'reanalysis'))
        replay_bins.append(ply_bin(row_ply(row[0])))
    if records_dir:
        write_jsonl(Path(records_dir) / f'reanalysis-{iteration:04d}.jsonl',
                    [game_report(t, r) for t, r in zip(tasks, reviews)])
    seconds = time.monotonic() - started
    stage_timer.add('reanalysis', seconds)
    return {**summarize(reviews), 'games': len(tasks), 'budget': budget,
            'new_positions': new_positions, 'rows_added': len(rows), 'simulations': simulations,
            'value_target': args.value_target, 'seconds': round(seconds, 3),
            'strong_opponents': sorted(strong),
            'opponent_scores': {k: round(v, 3) for k, v in sorted(scores.items())},
            'inference_positions': inference.get('inference_positions', 0)}


def play(args):
    model,_ = load_model(args.checkpoint)
    model.to(resolve_device(args.device))
    opponent = Opponent(args.profile)
    rng = np.random.default_rng(args.seed)
    state = State()
    human = 1 if args.side == 'X' else -1
    tree = TreeSearch(model, rng, search_config(args),
                      opponent=None if args.no_adapt else opponent, agent_side=-human)
    print('Enter board and cell numbers 1–9, e.g. 5 3. Enter q to quit.')
    while state.result is None:
        print('\n'+state.render())
        if state.turn == human:
            line = input('Your move: ').strip()
            if line.lower() == 'q':
                return
            try:
                board,cell = map(int,line.split())
                if not (1 <= board <= 9 and 1 <= cell <= 9):
                    raise ValueError()
                action = (board-1)*9+cell-1
                opponent.observe(state,action)
                tree.advance(action)
                if not args.no_adapt:
                    tree.reset()  # Human-policy update invalidates expected-response statistics.
                state = state.play(action)
            except ValueError:
                print('Enter a legal board and cell, both between 1 and 9.')
        else:
            pi = tree.run(state, args.simulations, batch_size=args.leaf_batch)
            action = int(pi.argmax())
            print(f'AI plays board {action//9+1}, cell {action%9+1}')
            print(f'Search: {tree.stats}')
            tree.advance(action)
            state = state.play(action)
    print(state.render())
    print('Draw' if state.result == 0 else ('You win' if state.result == human else 'AI wins'))

def evaluation_opening(seed, pair, plies):
    """Repeat an opening with swapped agent sides; keep it fixed across budgets."""
    if not 0 <= plies <= 8:
        raise ValueError('Opening moves must be between 0 and 8')
    rng = np.random.default_rng(np.random.SeedSequence([seed, pair, 1741]))
    state, actions = State(), []
    for _ in range(plies):
        action = int(rng.choice(state.legal_actions()))
        actions.append(action)
        state = state.play(action)
    return state, actions


def evaluate(args):
    model,checkpoint = load_model(args.checkpoint)
    model.to(resolve_device(getattr(args, 'device', 'cpu')))
    config = search_config(args)
    leaf_batch = getattr(args, 'leaf_batch', 1)
    opening_moves = getattr(args, 'opening_moves', 2)
    opponent_model = None
    bot = None
    if args.opponent == 'checkpoint':
        if not args.opponent_checkpoint:
            raise ValueError('--opponent checkpoint requires --opponent-checkpoint')
        opponent_model, _ = load_model(args.opponent_checkpoint)
        opponent_model.to(next(model.parameters()).device)
    elif args.opponent == 'tactical':
        bot = TacticalBot()
    elif args.opponent == 'alphabeta':
        bot = AlphaBetaBot(args.opponent_depth, args.opponent_nodes)
    opponent_name = {'tactical': 'tactical-v2', 'alphabeta': 'alphabeta-v1',
                     'legacy-tactical': 'legacy-tactical-v1'}.get(args.opponent, args.opponent)
    report = new_report(args.checkpoint, checkpoint.get('iteration'), opponent_name,
                        args.simulations, args.seed, args.games)
    report.update(search_config=asdict(config), leaf_batch=leaf_batch, opening_moves=opening_moves,
                  device=str(next(model.parameters()).device), search_version='batched-puct-v2',
                  opponent_depth=bot.depth if bot else None,
                  opponent_nodes=bot.node_budget if bot else None,
                  opponent_simulations=args.opponent_simulations if opponent_model is not None else None)
    if opponent_model is not None:
        identity = new_report(args.opponent_checkpoint, None, '', 0, 0, 0)
        report['opponent_checkpoint'] = identity['checkpoint']
        report['opponent_checkpoint_sha256'] = identity['checkpoint_sha256']
    output = args.output or str(Path(args.checkpoint).parent / 'evaluations')
    path = write_report(report, output)
    try:
        for game in range(args.games):
            streams = np.random.SeedSequence([args.seed, game]).spawn(2)
            rng, opponent_rng = [np.random.default_rng(stream) for stream in streams]
            tree = TreeSearch(model, rng, config)
            opponent_tree = TreeSearch(opponent_model, opponent_rng) if opponent_model is not None else None
            started = time.monotonic()
            state, opening_actions = evaluation_opening(args.seed, game // 2, opening_moves)
            agent = 1 if game%2 == 0 else -1
            moves, agent_moves, search_seconds = len(opening_actions), 0, 0.
            search_totals = dict(completed_simulations=0, neural_positions=0, max_depth=0,
                                 hard_pruned_choices=0, soft_rechecks=0, retained_visits=0)
            while state.result is None:
                if state.turn == agent:
                    search_start = time.monotonic()
                    action = int(tree.run(state, args.simulations, batch_size=leaf_batch).argmax())
                    for key in search_totals:
                        search_totals[key] = (max(search_totals[key], tree.stats[key]) if key == 'max_depth'
                                              else search_totals[key] + tree.stats[key])
                    search_seconds += time.monotonic() - search_start
                    agent_moves += 1
                elif opponent_tree is not None:
                    action = int(opponent_tree.run(state, args.opponent_simulations, batch_size=leaf_batch).argmax())
                elif bot is not None:
                    action = bot.choose(state, opponent_rng)
                else:
                    policy_idx = NAMES.index(args.opponent) if args.opponent in NAMES else 4
                    p = policies(state)[policy_idx]
                    action = int(opponent_rng.choice(81, p=p))
                tree.advance(action)
                if opponent_tree is not None:
                    opponent_tree.advance(action)
                state = state.play(action)
                moves += 1
            key = 'draws' if state.result == 0 else ('wins' if state.result == agent else 'losses')
            report['games'].append({'game': game+1, 'agent_side': 'X' if agent == 1 else 'O',
                                    'opening_pair': game // 2 + 1, 'opening_actions': opening_actions,
                                    'outcome': key, 'score': {'wins': 1., 'draws': .5, 'losses': 0.}[key],
                                    'moves': moves, 'seconds': time.monotonic()-started,
                                    'agent_moves': agent_moves, 'agent_search_seconds': search_seconds,
                                    **search_totals})
            write_report(report, output)
            print(f'game {game+1}/{args.games}: {key}',flush=True)
        report['status'] = 'complete'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        print('\nEvaluation interrupted; completed games have been saved.')
    except Exception:
        report['status'] = 'failed'
        raise
    finally:
        write_report(report, output)
        print(f'Reports: {path} (plus _games.csv and _summary.csv)', flush=True)
    print(json.dumps(report['summary']))
    return report


def tournament_cmd(args):
    device = resolve_device(getattr(args, 'device', 'cpu'))
    if getattr(args, 'output', None):
        output_dir = Path(args.output)
    elif getattr(args, 'checkpoint', None):
        output_dir = Path(args.checkpoint).parent / 'tournaments'
    else:
        output_dir = Path('runs/tournaments')
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, 'simulation_budgets', None):
        checkpoint_path = args.checkpoint if getattr(args, 'checkpoint', None) else ""
        opp_specs = args.opponents if getattr(args, 'opponents', None) else ["tactical", "alphabeta"]
        sweep_summary = run_simulation_sweep(
            checkpoint_path=checkpoint_path,
            opponent_specs=opp_specs,
            budgets=args.simulation_budgets,
            games=args.games,
            opening_plies=args.opening_plies,
            seed=args.seed,
            device=device,
            output_dir=output_dir,
        )
        lines = [
            "=" * 110,
            "                               SIMULATION BUDGET SCALING BENCHMARK",
            "=" * 110,
            f"{'Budget':<10} {'Bayesian Elo (95% CI)':<25} {'Glicko-2 (±RD)':<22} {'Win Rate':<10} {'Score Rate':<12} {'W':<5} {'D':<5} {'L':<5} {'Games':<6}",
            "-" * 110,
        ]
        for row in sweep_summary.get("scaling", []):
            b = row["budget"]
            elo_str = f"{row['elo']:.1f} [±{row['elo_ci95']:.1f}]"
            g_str = f"{row['glicko2']:.1f} (±{row['glicko2_rd']:.1f})"
            win_pct = f"{row['win_rate'] * 100:.1f}%"
            score_pct = f"{row['score_rate'] * 100:.1f}%"
            lines.append(
                f"{b:<10} {elo_str:<25} {g_str:<22} {win_pct:<10} {score_pct:<12} {row['wins']:<5} {row['draws']:<5} {row['losses']:<5} {row['games']:<6}"
            )
        lines.append("=" * 110)
        scoreboard_text = "\n".join(lines)
        (output_dir / "scoreboard.txt").write_text(scoreboard_text + "\n", encoding="utf-8")
        print(scoreboard_text)
        return sweep_summary

    engine_registry = load_engine_registry(args.engine_config) if getattr(args, 'engine_config', None) else {}
    participants = {}
    if getattr(args, 'checkpoint', None):
        ckpt_bot = CheckpointBot(
            path=args.checkpoint,
            simulations=args.simulations,
            device=device,
        )
        participants[ckpt_bot.name] = ckpt_bot

    opp_list = getattr(args, 'opponents', None) or []
    for opp in opp_list:
        bot = opp if isinstance(opp, Bot) else (create_configured_engine(opp, engine_registry) or create_bot(opp))
        name = bot.name
        if name in participants:
            suffix = 2
            while f"{name}_{suffix}" in participants:
                suffix += 1
            name = f"{name}_{suffix}"
        participants[name] = bot

    if len(participants) < 2:
        for default_bot in ("alphabeta", "tactical"):
            if len(participants) >= 2:
                break
            b = create_bot(default_bot)
            if b.name not in participants:
                participants[b.name] = b

    tourn = run_tournament(
        bots=participants,
        games_per_matchup=args.games,
        opening_plies=args.opening_plies,
        seed=args.seed,
    )
    ratings_dict = {}
    for p in tourn["participants"]:
        elo_r = tourn["elo_ratings"].get(p)
        glicko_r = tourn["glicko_ratings"].get(p)
        ratings_dict[p] = {
            "elo": round(float(elo_r.elo), 2) if elo_r is not None else 1500.0,
            "elo_ci95": round(float(elo_r.error_margin), 2) if elo_r is not None else 0.0,
            "glicko2": round(float(glicko_r.rating), 2) if glicko_r is not None else 1500.0,
            "glicko2_rd": round(float(glicko_r.rd), 2) if glicko_r is not None else 350.0,
            "volatility": round(float(glicko_r.volatility), 4) if glicko_r is not None else 0.06,
            "games": elo_r.games if elo_r is not None else 0,
            "wins": elo_r.wins if elo_r is not None else 0,
            "draws": elo_r.draws if elo_r is not None else 0,
            "losses": elo_r.losses if elo_r is not None else 0,
            "win_rate": round(elo_r.wins / elo_r.games, 4) if elo_r and elo_r.games > 0 else 0.0,
            "score_rate": round((elo_r.wins + 0.5 * elo_r.draws) / elo_r.games, 4) if elo_r and elo_r.games > 0 else 0.0,
        }
    tourn["ratings"] = ratings_dict
    tourn["matchups"] = [
        {
            "game_id": r.game_id,
            "pair_id": r.pair_id,
            "opening_plies": r.opening_plies,
            "opening_moves": list(r.opening_moves),
            "player_x": r.player_x,
            "player_o": r.player_o,
            "winner": r.winner,
            "moves": r.moves,
        }
        for r in tourn["results"]
    ]
    write_tournament_report(tourn, output_dir)
    print(tourn["scoreboard"])
    return tourn


tournament = tournament_cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    t = commands.add_parser('train')
    for name,default in [('iterations',20),('games',8),('simulations',64),('steps',100),
                         ('batch',128),('buffer',50000)]:
        t.add_argument('--'+name,type=positive,default=default)
    t.add_argument('--arch', choices=['resnet', 'mlp', 'unet'], default='resnet')
    t.add_argument('--workers', type=positive, default=DEFAULT_TRAIN_WORKERS,
                   help=f'Self-play worker processes (default: {DEFAULT_TRAIN_WORKERS})')
    t.add_argument('--backend', choices=['auto', 'cpp', 'python'], default='auto',
                   help='Search backend: cpp uses C++ bitboard engine, python uses pure Python (default: auto)')
    t.add_argument('--fp16', action='store_true', help='Use automatic mixed precision (FP16) on CUDA')
    t.add_argument('--lr', type=float, default=None,
                   help='Initial learning rate (default: 1e-3 fresh; a resume keeps its schedule)')
    t.add_argument('--lr-schedule', choices=['constant', 'cosine'], default=None,
                   help='Learning-rate policy. The cosine horizon is measured in COMPLETED ITERATIONS')
    t.add_argument('--lr-min', type=float, default=0.0,
                   help='Cosine floor; the schedule clamps here and never climbs back')
    t.add_argument('--lr-iterations', type=int, default=0,
                   help='Cosine horizon in completed iterations (not optimizer updates)')
    t.add_argument('--reset-lr-schedule', action='store_true',
                   help='Start a new, recorded schedule phase instead of continuing the saved one')
    t.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    t.add_argument('--output', default='runs/default')
    t.add_argument('--resume')
    t.add_argument('--population', action='store_true', help='Quota-controlled league of self-play and strong opponents')
    t.add_argument('--augment-symmetry', action='store_true', help='Random rotations/reflections of training positions')
    t.add_argument('--replay-sampling', choices=['uniform', 'stratified'], default='uniform',
                   help='stratified: equal share of each nine-ply game phase per batch (uttt.ai-style '
                        'depth balancing); uniform: the historical FIFO sampler')
    t.add_argument('--population-checkpoints', help='Directory of frozen model-*.pt and best.pt opponents; defaults to output')
    t.add_argument('--population-config', default=None,
                   help='Versioned JSON curriculum (quotas summing to 100 + per-family budgets). '
                        'Omitted on resume inherits the saved one; a different file is a recorded phase change. '
                        'Presets: configs/population/*.json')
    t.add_argument('--engine-config', default=str(Path(__file__).resolve().parent.parent / 'engines/registry.json'),
                   help='JSON registry (defaults to bundled uttt.ai wrapper)')
    t.add_argument('--inference-batch', type=positive, default=128,
                   help='Maximum positions evaluated together by the inference owner')
    t.add_argument('--inference-wait-ms', type=float, default=2.,
                   help='Maximum wait for other workers to fill an inference batch')
    t.add_argument('--save-every', type=int, default=100, help='Save snapshot model checkpoint every N iterations; 0 disables')
    t.add_argument('--keep-checkpoint-window', type=int, default=500,
                   help='Prune model-*.pt snapshots older than N iterations; 0 disables')
    t.add_argument('--keep-checkpoints', type=int, default=10,
                   help='Keep at most the N most recent model-*.pt snapshots; 0 disables. '
                        'Applied together with --keep-checkpoint-window; best.pt and '
                        'latest.pt are never pruned')
    t.add_argument('--eval-every', type=int, default=0, help='Evaluate every N iterations; 0 disables')
    t.add_argument('--eval-games', type=positive, default=20)
    t.add_argument('--eval-simulations', type=positive, default=512)
    t.add_argument('--eval-opponent-depth', type=positive, default=4,
                   help='Alpha-beta depth for the periodic in-training evaluation (default 4). '
                        'Depth 3 was the old default and is too weak to be informative once the '
                        'model is past the opening stages')
    t.add_argument('--eval-opponent-nodes', type=positive, default=50000,
                   help='Node budget for that opponent. Must exceed the depth\'s typical node '
                        'count or the depth is nominal only: d3 ~1,453, d4 ~3,479, d5 ~20,514 '
                        '(docs/engineering/cpp-benchmarks.md). The old 3,000 truncated anything above d3')
    t.add_argument('--max-iterations', type=positive, default=None,
                   help='Stop training once total cumulative iterations reach N')
    from .reanalysis import OUTCOMES, SIDES, VALUE_TARGETS
    t.add_argument('--save-game-records', metavar='DIR', default=None,
                   help='Write every game (actions, movers, learner root search) to '
                        'DIR/games-NNNN.jsonl per iteration; off by default')
    t.add_argument('--reanalyse', action=argparse.BooleanOptionalAction, default=True,
                   help='Loss review (default on): re-search sampled decision points (both sides) '
                        'of lost games with a stronger budget and add refreshed targets to replay; '
                        'losses to opponents that beat us consistently are reviewed first')
    t.add_argument('--reanalyse-strong-threshold', type=float, default=.5,
                   help='An opponent counts as strong while our running score against it is '
                        'below this (default 0.5)')
    t.add_argument('--reanalyse-simulations', type=positive, default=None,
                   help='Reanalysis search budget (default: 4 x --simulations)')
    t.add_argument('--reanalyse-fraction', type=float, default=.25,
                   help='Cap on reanalysed rows per iteration as a fraction of new self-play '
                        'positions (default 0.25)')
    t.add_argument('--reanalyse-outcomes', nargs='+', choices=OUTCOMES, default=['loss'],
                   help='Learner outcomes whose games are reviewed (default: loss; decisive '
                        'self-play games count as losses)')
    t.add_argument('--reanalyse-sides', nargs='+', choices=SIDES, default=list(SIDES),
                   help='Whose decision points are reviewed (default: both)')
    t.add_argument('--reanalyse-margin', type=float, default=.3,
                   help='Q gap to the best alternative that flags a suspected mistake')
    t.add_argument('--value-target', choices=VALUE_TARGETS, default='outcome',
                   help='Value target of reanalysed rows: outcome (game result), search '
                        '(reanalysis root value) or mix; normal rows always use the outcome')
    t.add_argument('--value-lambda', type=float, default=.5,
                   help='mix: lambda * outcome + (1 - lambda) * search (default 0.5)')
    p = commands.add_parser('play')
    p.add_argument('--profile',default='profiles/player.json')
    p.add_argument('--side',choices=['X','O'],default='X')
    p.add_argument('--no-adapt',action='store_true')
    e = commands.add_parser('evaluate')
    e.add_argument('--games',type=positive,default=20)
    e.add_argument('--opponent', choices=['random', 'center', 'corners', 'local-win', 'global-win', 'legacy-tactical', 'tactical', 'alphabeta', 'checkpoint'], default='alphabeta')
    e.add_argument('--opponent-checkpoint')
    e.add_argument('--opponent-depth', type=positive, default=3)
    e.add_argument('--opponent-nodes', type=positive, default=3000)
    e.add_argument('--opponent-simulations', type=positive, default=256)
    e.add_argument('--opening-moves', type=int, default=2,
                   help='0–8 seeded opening plies, paired with sides swapped; 0 starts empty')
    e.add_argument('--seeds', nargs='+', type=int, help='Run separate reports for each seed')
    e.add_argument('--simulation-budgets', nargs='+', type=positive,
                   help='Run separate reports for each search budget')
    e.add_argument('--output',help='Report directory (default: checkpoint directory/evaluations)')
    tourn = commands.add_parser('tournament', help='Execute automated round-robin tournament or simulation sweep')
    tourn.add_argument('--checkpoint', help='Path to checkpoint model (optional)')
    tourn.add_argument('--opponents', nargs='+', help='Bot identifiers (e.g. alphabeta, tactical, corners, etc.)')
    tourn.add_argument('--games', type=positive, default=50, help='Games per matchup in [20, 500] (default: 50)')
    tourn.add_argument('--simulations', type=positive, default=512, help='MCTS simulation budget (default: 512)')
    tourn.add_argument('--simulation-budgets', nargs='+', type=positive, help='Simulation budgets for scaling sweep')
    tourn.add_argument('--opening-plies', type=int, default=2, help='Opening plies in [0, 4] (default: 2)')
    tourn.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    tourn.add_argument('--output', help='Report directory (default: runs/<run>/tournaments/ or runs/tournaments/)')
    tourn.add_argument('--engine-config', help='JSON registry of named external engines')
    tourn.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    gen = commands.add_parser('generate-dataset', help='Stage-1 bootstrap: network-free native search games')
    gen.add_argument('--output', required=True)
    gen.add_argument('--games', type=positive, default=20000)
    gen.add_argument('--workers', type=positive, default=DEFAULT_TRAIN_WORKERS,
                     help=f'Dataset generation worker processes (default: {DEFAULT_TRAIN_WORKERS})')
    gen.add_argument('--simulations', type=positive, default=512)
    gen.add_argument('--leaf-batch', type=positive, default=64)
    gen.add_argument('--shard-games', type=positive, default=500)
    gen.add_argument('--alphabeta-share', type=float, default=0.5)
    gen.add_argument('--depths', nargs='+', type=positive, default=[4, 5, 6])
    gen.add_argument('--seed', type=int, default=1)
    pre = commands.add_parser('pretrain', help='Stage-2 bootstrap: supervised fit to a generated dataset')
    pre.add_argument('--dataset', required=True)
    pre.add_argument('--output', required=True)
    pre.add_argument('--arch', choices=['resnet', 'mlp', 'unet'], default='unet')
    pre.add_argument('--epochs', type=positive, default=10)
    pre.add_argument('--batch', type=positive, default=1024)
    pre.add_argument('--lr', type=float, default=3e-4,
                     help='Peak learning rate after warm-up. Default 3e-4, NOT 1e-3: at 1e-3 '
                          "AdamW's first steps saturate the value head's tanh and its gradient "
                          'reaches zero permanently (measured for both resnet and unet; see '
                          'docs/history/value-head-check.txt)')
    pre.add_argument('--lr-warmup', type=int, default=100,
                     help='Optimizer steps of linear warm-up before the cosine decay; clamped to '
                          'one less than the total step count. 0 disables it')
    pre.add_argument('--lr-min', type=float, default=1e-5)
    pre.add_argument('--weight-decay', type=float, default=1e-4)
    pre.add_argument('--holdout', type=float, default=0.02)
    pre.add_argument('--replay-buffer', type=positive, default=200000)
    pre.add_argument('--max-positions', type=positive, default=1000000,
                     help='Maximum bootstrap positions loaded for pretraining; shards are streamed '
                          'one at a time (default: 1000000)')
    pre.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    pre.add_argument('--fp16', action='store_true')
    pre.add_argument('--seed', type=int, default=0)
    rea = commands.add_parser('reanalyse', help='Offline loss review of saved game records')
    rea.add_argument('--records', required=True,
                     help='games-*.jsonl file or a directory of them (--save-game-records output)')
    rea.add_argument('--checkpoint', required=True)
    rea.add_argument('--simulations', type=positive, default=2048)
    rea.add_argument('--output', required=True, help='JSON report path')
    rea.add_argument('--outcomes', nargs='+', choices=OUTCOMES, default=['loss'])
    rea.add_argument('--sides', nargs='+', choices=SIDES, default=list(SIDES))
    rea.add_argument('--margin', type=float, default=.3)
    rea.add_argument('--backend', choices=['auto', 'cpp', 'python'], default='auto')
    rea.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    for sub in (p,e):
        sub.add_argument('--device', choices=['auto','cpu','cuda'], default='cpu')
        sub.add_argument('--checkpoint',required=True)
        sub.add_argument('--simulations',type=positive,default=128)
    for sub in (t,p,e,rea):
        sub.add_argument('--seed',type=int,default=0)
        sub.add_argument('--leaf-batch', type=positive, default=16,
                         help='Pending leaf positions per tree before an inference request')
        sub.add_argument('--no-soft-pruning', action='store_true')
        sub.add_argument('--no-proofs', action='store_true')
        sub.add_argument('--no-reuse', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.seed < 0 or (args.command == 'evaluate' and args.seeds and min(args.seeds) < 0):
        parser.error('Seeds must be nonnegative')
    if args.command == 'train' and (args.eval_every < 0 or not np.isfinite(args.inference_wait_ms) or args.inference_wait_ms < 0):
        parser.error('Evaluation interval and inference wait must be nonnegative and finite')
    if args.command == 'evaluate' and not 0 <= args.opening_moves <= 8:
        parser.error('Opening moves must be between 0 and 8')
    if args.command == 'tournament':
        if not 20 <= args.games <= 500:
            parser.error('Games per matchup must be between 20 and 500')
        if not 0 <= args.opening_plies <= 4:
            parser.error('Opening plies must be between 0 and 4')
    if args.command == 'generate-dataset' and not 0. <= args.alphabeta_share <= 1.:
        parser.error('--alphabeta-share must be between 0 and 1')
    if args.command == 'pretrain' and not 0. <= args.holdout < 1.:
        parser.error('--holdout must be at least 0 and less than 1')
    if args.command == 'train':
        if not 0. <= args.reanalyse_fraction <= 1. or not 0. <= args.value_lambda <= 1.:
            parser.error('--reanalyse-fraction and --value-lambda must be between 0 and 1')
        if args.value_target != 'outcome' and not args.reanalyse:
            parser.error('--value-target search/mix needs --reanalyse')
    try:
        if args.command == 'evaluate':
            for seed in args.seeds or [args.seed]:
                for budget in args.simulation_budgets or [args.simulations]:
                    args.seed, args.simulations = seed, budget
                    if evaluate(args)['status'] != 'complete':
                        return  # Ctrl+C ends the whole sweep, not just one budget.
        else:
            {'train': train, 'play': play, 'evaluate': evaluate, 'tournament': tournament_cmd,
             'generate-dataset': generate_dataset, 'pretrain': pretrain,
             'reanalyse': reanalyse_cmd}[args.command](args)
    except (KeyboardInterrupt,EOFError):
        print('\nStopped. Completed training iterations and observed human moves are saved.')

if __name__ == '__main__':
    main()
