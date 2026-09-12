"""CPU search workers with a single batched inference owner (CPU or CUDA).

Workers are spawned, never forked after CUDA initialization. The main process
services inference requests while games run, then trains only after all finish.
"""
import multiprocessing as mp
from multiprocessing.connection import wait
import time
import traceback
from dataclasses import asdict
import numpy as np
from .env import State
from .search import TreeSearch
from .population import MatchSpec, make_opponent


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


def play_game(evaluator, simulations, seed, config, leaf_batch, match=None):
    rng = np.random.default_rng(seed)
    tree = TreeSearch(evaluator, rng, config)
    match = match or MatchSpec()
    opponent = make_opponent(match)
    try:
        return _play_game(tree, rng, simulations, seed, leaf_batch, match, opponent)
    finally:
        if opponent is not None:
            opponent.close()


def _play_game(tree, rng, simulations, seed, leaf_batch, match, opponent):
    state, trajectory = State(), []
    opponent_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 731]))
    for _ in range(match.opening_moves):
        if state.result is not None:
            break
        action = int(opponent_rng.choice(state.legal_actions()))
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
            tree.advance(action)
            state = state.play(action)
            ply += 1
            continue
        pi = tree.run(state, simulations, batch_size=leaf_batch, noise=True)
        for key in totals:
            totals[key] = (max(totals[key], tree.stats[key]) if key == 'max_depth'
                           else totals[key] + tree.stats[key])
        trajectory.append((state, pi))
        # Normalize in float64 to avoid categorical sampler tolerance differences.
        p = pi.astype(float); p /= p.sum()
        action = int(rng.choice(81, p=p)) if ply < 15 else int(pi.argmax())
        tree.advance(action)
        if opponent is not None:
            opponent.advance(action)
        state = state.play(action)
        ply += 1
    totals['match'] = asdict(match)
    totals['plies'] = ply
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
            _, index, simulations, seed, config, leaf_batch, match = message
            trajectory, outcome, stats = play_game(evaluator, simulations, seed, config, leaf_batch, match)
            connection.send(('game', index, trajectory, outcome, stats))
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
    def __init__(self, workers, batch_size=128, wait_ms=2., timeout=120):
        if workers < 1 or batch_size < 1 or wait_ms < 0:
            raise ValueError('Invalid inference pool configuration')
        self.batch_size, self.wait_ms, self.timeout = batch_size, wait_ms, timeout
        self.connections, self.processes = [], []
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

    def run(self, model, seeds, simulations, config, leaf_batch, matches=None):
        """Collect one iteration with fixed model weights. Queue waiting is bounded."""
        if not len(seeds):
            raise ValueError('At least one game is required')
        if matches is not None and len(matches) != len(seeds):
            raise ValueError('One match specification is required per seed')
        model.eval()
        results, active = {}, set()
        next_game = 0
        metrics = dict(inference_batches=0, inference_positions=0, max_inference_batch=0,
                       inference_seconds=0.)

        def dispatch(connection):
            nonlocal next_game
            if next_game < len(seeds):
                match = matches[next_game] if matches is not None else None
                connection.send(('start', next_game, simulations, int(seeds[next_game]), config, leaf_batch, match))
                next_game += 1
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
                    if message[0] == 'game':
                        _, index, trajectory, outcome, stats = message
                        results[index] = (trajectory, outcome, stats)
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
        return [results[i] for i in range(len(seeds))], metrics
