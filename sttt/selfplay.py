"""CPU search workers with a single batched inference owner (CPU or CUDA).

Workers are spawned, never forked after CUDA initialization. The main process
services inference requests while games run, then trains only after all finish.
"""
import multiprocessing as mp
from multiprocessing.connection import wait
import time
import sys
import traceback
from dataclasses import asdict
import os
import numpy as np
from .env import State
from .search import TreeSearch, root_action_values, root_value
from .population import MatchSpec, make_opponent

try:
    from .cpp_env import FastState as CppFastState, CppTreeSearch, is_cpp_available
    _HAS_CPP = is_cpp_available()
except ImportError:
    _HAS_CPP = False
    CppFastState = None
    CppTreeSearch = None


class RemoteEvaluator:
    def __init__(self, connection, timeout=120):
        self.connection, self.timeout = connection, timeout

    def evaluate_many(self, states):
        self.connection.send(('infer', states))
        if not self.connection.poll(self.timeout):
            raise TimeoutError('Timed out waiting for the inference owner')
        kind, result = self.connection.recv()
        if kind != 'prediction':
            raise RuntimeError(f'Unexpected inference response: {kind}')
        return result

    def evaluate(self, state):
        return self.evaluate_many([state])[0]


def play_game(evaluator, simulations, seed, config, leaf_batch, match=None, use_cpp=False):
    rng = np.random.default_rng(seed)
    # M2: `use_cpp` is a REQUIREMENT here, not a preference. Each spawned worker
    # re-evaluates its own _HAS_CPP, so a worker whose extension failed to load
    # used to fall through to the Python TreeSearch while the owner went on
    # believing the whole iteration ran natively. Fail loudly instead; the pool
    # handshake below is meant to catch this before any game is dispatched.
    if use_cpp:
        if not (_HAS_CPP and CppTreeSearch is not None):
            raise RuntimeError(
                'C++ backend required for self-play but sttt_cpp is unavailable '
                'in this worker process. Run "make -C cpp"; refusing to silently '
                'fall back to the Python search.')
        tree = CppTreeSearch(evaluator, rng, config)
    else:
        tree = TreeSearch(evaluator, rng, config)
    match = match or MatchSpec()
    opponent = make_opponent(match)
    try:
        return _play_game(tree, rng, simulations, seed, leaf_batch, match, opponent, use_cpp=use_cpp and _HAS_CPP)
    finally:
        if opponent is not None:
            opponent.close()


def _play_game(tree, rng, simulations, seed, leaf_batch, match, opponent, use_cpp=False):
    import time
    started = time.monotonic()
    state = CppFastState() if (use_cpp and CppFastState is not None) else State()
    trajectory = []
    # Full game record for loss review. Kept beside the trajectory (whose
    # 4-tuples several consumers unpack) and computed without touching any RNG.
    actions, movers, root_values = [], [], []
    opponent_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 731]))
    for _ in range(match.opening_moves):
        if state.result is not None:
            break
        action = int(opponent_rng.choice(state.legal_actions()))
        actions.append(action); movers.append('opening')
        state = state.play(action)
        if opponent is not None:
            opponent.advance(action)
    ply = match.opening_moves
    totals = dict(completed_simulations=0, neural_positions=0, max_depth=0,
                  hard_pruned_choices=0, soft_rechecks=0, retained_visits=0)
    while state.result is None:
        if opponent is not None and state.turn != match.learner_side:
            if opponent_rng.random() < match.epsilon:
                action = int(opponent_rng.choice(state.legal_actions()))
                opponent.advance(action)
            else:
                action = opponent.choose(state, opponent_rng)
            actions.append(action); movers.append('opponent')
            tree.advance(action)
            state = state.play(action)
            ply += 1
            continue
        pi = tree.run(state, simulations, batch_size=leaf_batch, noise=True)
        for key in totals:
            totals[key] = (max(totals[key], tree.stats[key]) if key == 'max_depth'
                           else totals[key] + tree.stats[key])
        q, q_mask = root_action_values(tree.root)
        trajectory.append((state, pi, q, q_mask))
        root_values.append(root_value(tree.root))
        # Normalize in float64 to avoid categorical sampler tolerance differences.
        p = pi.astype(float); p /= p.sum()
        action = int(rng.choice(81, p=p)) if ply < 8 else int(pi.argmax())
        actions.append(action); movers.append('learner')
        tree.advance(action)
        if opponent is not None:
            opponent.advance(action)
        state = state.play(action)
        ply += 1
    totals['match'] = asdict(match)
    totals['plies'] = ply
    totals['seconds'] = time.monotonic() - started   # M6: wall time by family
    totals['record'] = {'actions': actions, 'movers': movers, 'root_values': root_values}
    return trajectory, state.result, totals


def _worker(connection):
    # Import here; no model or CUDA objects are sent to worker processes.
    import torch
    torch.set_num_threads(1)
    evaluator = RemoteEvaluator(connection)
    try:
        while True:
            message = connection.recv()
            if message[0] == 'stop':
                break
            if message[0] == 'handshake':
                # Report what THIS process can actually do, before it is asked
                # to play anything.
                try:
                    from .cpp_env import get_cpp_provenance
                    provenance = get_cpp_provenance()
                except ImportError:
                    provenance = {'available': False, 'build_id': None, 'version': None}
                connection.send(('capabilities', {
                    'cpp_available': bool(_HAS_CPP and CppTreeSearch is not None),
                    'cpp_build_id': provenance.get('build_id'),
                    'cpp_version': provenance.get('version'),
                    'pid': os.getpid(),
                }))
                continue
            try:
                if message[0] == 'reanalyse':
                    from .reanalysis import review_game
                    _, index, task, simulations, config, leaf_batch, use_cpp = message
                    payload = (review_game(evaluator, task, simulations, config, leaf_batch,
                                           use_cpp=use_cpp),)
                else:
                    _, index, simulations, seed, config, leaf_batch, match, use_cpp = message
                    payload = play_game(evaluator, simulations, seed, config, leaf_batch, match, use_cpp=use_cpp)
            except Exception:
                # Per-game failure (e.g. an external engine timeout): report it
                # and stay alive so the pool can retry this one game.
                connection.send(('game_error', index, traceback.format_exc()))
                continue
            connection.send(('game', index, *payload))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class SelfPlayPool:
    def __init__(self, workers, batch_size=128, wait_ms=2., timeout=120, game_retries=1):
        if workers < 1 or batch_size < 1 or wait_ms < 0 or game_retries < 0:
            raise ValueError('Invalid inference pool configuration')
        self.batch_size, self.wait_ms, self.timeout = batch_size, wait_ms, timeout
        self.game_retries = game_retries
        self.connections, self.processes = [], []
        self.worker_capabilities = None
        ctx = mp.get_context('spawn')
        try:
            for _ in range(workers):
                owner, child = ctx.Pipe()
                process = ctx.Process(target=_worker, args=(child,))
                process.start()
                child.close()
                self.connections.append(owner)
                self.processes.append(process)
        except BaseException:
            self.close()
            raise

    def verify_backend(self, use_cpp, timeout=30):
        """Handshake every worker before games are dispatched.

        Returns the per-worker capability reports. A strict native run aborts
        here rather than discovering a degraded worker mid-iteration -- by which
        point part of the data would already be Python-search output labelled as
        native.
        """
        reports = []
        for connection in self.connections:
            connection.send(('handshake',))
        for connection in self.connections:
            if not connection.poll(timeout):
                raise TimeoutError('Self-play worker did not answer the backend handshake')
            message = connection.recv()
            if message[0] == 'error':
                raise RuntimeError('Self-play worker failed during handshake:\n' + message[1])
            if message[0] != 'capabilities':
                raise RuntimeError(f'Unexpected handshake reply: {message[0]!r}')
            reports.append(message[1])

        if use_cpp:
            degraded = [r for r in reports if not r['cpp_available']]
            if degraded:
                raise RuntimeError(
                    f'{len(degraded)} of {len(reports)} self-play workers lack the '
                    f'native search (pids {[r["pid"] for r in degraded]}). Refusing '
                    f'to start: those workers would run the Python search while the '
                    f'run reported "cpp".')
            builds = {r['cpp_build_id'] for r in reports}
            if len(builds) > 1:
                raise RuntimeError(f'Self-play workers loaded different native builds: {builds}')
        self.worker_capabilities = reports
        return reports

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for connection in self.connections:
            try:
                connection.send(('stop',))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()

    def run(self, model, seeds, simulations, config, leaf_batch, matches=None, use_cpp=False):
        """Collect one iteration with fixed model weights. Queue waiting is bounded."""
        if not len(seeds):
            raise ValueError('At least one game is required')
        if matches is not None and len(matches) != len(seeds):
            raise ValueError('One match specification is required per seed')

        def message(index, attempt):
            match = matches[index] if matches is not None else None
            # A retry gets a derived seed so it is not a replay of the failure.
            seed = int(seeds[index]) + 7919 * attempt
            return ('start', index, simulations, seed, config, leaf_batch, match, use_cpp)
        return self._execute(model, len(seeds), message, use_cpp)

    def reanalyse(self, model, tasks, simulations, config, leaf_batch, use_cpp=False):
        """Loss-review searches (sttt.reanalysis.review_game), one task per game,
        through the same workers and batched inference as self-play."""
        if not tasks:
            return [], {}
        results, metrics = self._execute(
            model, len(tasks),
            lambda index, attempt: ('reanalyse', index, tasks[index], simulations, config,
                                    leaf_batch, use_cpp),
            use_cpp)
        return [r[0] for r in results], metrics

    def _execute(self, model, count, make_message, use_cpp):
        model.eval()
        # Verified once per pool; workers are long-lived across iterations.
        if getattr(self, 'worker_capabilities', None) is None:
            self.verify_backend(use_cpp)
        results, active = {}, set()
        next_game, retry, attempts = 0, [], {}
        metrics = dict(inference_batches=0, inference_positions=0, max_inference_batch=0,
                       inference_seconds=0., game_retries=0)

        def dispatch(connection):
            nonlocal next_game
            if retry:
                index = retry.pop()
            elif next_game < count:
                index, next_game = next_game, next_game + 1
            else:
                return
            connection.send(make_message(index, attempts.get(index, 0)))
            active.add(connection)

        for connection in self.connections:
            dispatch(connection)
        while active:
            ready = wait(list(active), timeout=self.timeout)
            if not ready:
                raise TimeoutError('Self-play workers stalled; check CPU/inference load')
            requests = []
            deadline = time.monotonic() + self.wait_ms / 1000
            while ready:
                for connection in ready:
                    try:
                        message = connection.recv()
                    except EOFError as error:
                        raise RuntimeError('Self-play worker exited unexpectedly') from error
                    if message[0] == 'error':
                        raise RuntimeError('Self-play worker failed:\n' + message[1])
                    if message[0] == 'game_error':
                        _, index, trace = message
                        attempts[index] = attempts.get(index, 0) + 1
                        if attempts[index] > self.game_retries:
                            raise RuntimeError(f'Self-play game {index} failed {attempts[index]} times:\n' + trace)
                        print(f'WARNING: self-play game {index} failed '
                              f'(attempt {attempts[index]}/{self.game_retries + 1}); retrying:\n{trace}',
                              file=sys.stderr, flush=True)
                        metrics['game_retries'] += 1
                        retry.append(index)
                        active.remove(connection)
                        dispatch(connection)
                        continue
                    if message[0] == 'game':
                        results[message[1]] = tuple(message[2:])
                        active.remove(connection)
                        dispatch(connection)
                    elif message[0] == 'infer':
                        requests.append((connection, message[1]))
                    else:
                        raise RuntimeError(f'Unexpected worker message: {message[0]}')
                blocked = {connection for connection, _ in requests}
                available = list(active - blocked)
                if (not available or sum(len(states) for _, states in requests) >= self.batch_size
                        or time.monotonic() >= deadline):
                    break
                ready = wait(available, timeout=max(0., deadline - time.monotonic()))
            if requests:
                states = [s for _, batch in requests for s in batch]
                outputs = []
                for start in range(0, len(states), self.batch_size):
                    chunk = states[start:start+self.batch_size]
                    begin = time.monotonic()
                    outputs.extend(model.evaluate_many(chunk))
                    metrics['inference_seconds'] += time.monotonic() - begin
                    metrics['inference_batches'] += 1
                    metrics['inference_positions'] += len(chunk)
                    metrics['max_inference_batch'] = max(metrics['max_inference_batch'], len(chunk))
                start = 0
                for connection, batch in requests:
                    connection.send(('prediction', outputs[start:start+len(batch)]))
                    start += len(batch)
        metrics['mean_inference_batch'] = metrics['inference_positions'] / max(1, metrics['inference_batches'])
        return [results[i] for i in range(count)], metrics
