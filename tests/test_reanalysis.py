"""Loss review / reanalysis: records, perspective, blunder detection, targets,
replay mixing, default-path identity, game-level hold-out, checkpoint compat."""
import argparse
import json
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sttt.ai import _train_loop, pack_replay, unpack_replay
from sttt.bootstrap import game_ids, split_by_game
from sttt.env import State
from sttt.learning import Network, encode
from sttt.population import MatchSpec
from sttt.reanalysis import (game_record, load_records, reanalysis_rows, replay_states,
                             review_game, review_position, select_tasks, value_target, write_jsonl)
from sttt.search import SearchConfig
from sttt.selfplay import play_game


class Uniform:
    """Stub network: uniform policy, value 0."""
    def evaluate_many(self, states):
        return [(np.full(81, 1 / 81), 0.) for _ in states]

    def evaluate(self, state):
        return self.evaluate_many([state])[0]


def blunder_position(turn=1):
    """X to move in board 4. O owns boards 0 and 1 and has two in a row in
    board 2 (cell 20 open). Sending O to board 0/1 (free choice) or 2 lets O
    win at once: actions 36, 37, 38 blunder; 39..43 are safe."""
    cells = [0] * 81
    for c in (0, 1, 2, 9, 10, 11, 18, 19, 58):
        cells[c] = -1
    for c in (44, 30, 50, 54, 64, 74):
        cells[c] = 1
    boards = [-1, -1] + [0] * 7
    if turn == -1:   # the same position with colours swapped and O to move
        cells, boards = [-v for v in cells], [-v for v in boards]
    return State(tuple(cells), tuple(boards), turn, 4, None)


def winning_position(turn=1):
    """Player to move owns boards 0 and 1 and wins by playing 20."""
    cells = [0] * 81
    for c in (0, 1, 2, 9, 10, 11, 18, 19):
        cells[c] = turn
    for c in (30, 31, 40, 50, 60, 70):
        cells[c] = -turn
    return State(tuple(cells), tuple([turn, turn] + [0] * 7), turn, 2, None)


def win_or_lose_position():
    """X to move in board 2: 20 wins at once (boards 0, 1, 2). O owns boards 6
    and 7 and threatens 74 in board 8, so 24/25 (free choice) and 26 (board 8)
    lose at once."""
    cells = [0] * 81
    for c in (0, 1, 2, 9, 10, 11, 18, 19):
        cells[c] = 1
    for c in (54, 55, 56, 63, 64, 65, 72, 73):
        cells[c] = -1
    boards = [1, 1, 0, 0, 0, 0, -1, -1, 0]
    return State(tuple(cells), tuple(boards), 1, 2, None)


BLUNDERS, SAFE = (36, 37, 38), (39, 40, 41, 42, 43)


class RecordTests(unittest.TestCase):
    def test_record_round_trips_and_replays_to_the_recorded_result(self):
        for side in (1, -1):
            seed = 7 + side
            trajectory, result, stats = play_game(Uniform(), 8, seed, SearchConfig(), 2,
                                                  MatchSpec(kind='style', learner_side=side, opening_moves=2))
            record = game_record(3, 0, seed, trajectory, result, stats)
            with tempfile.TemporaryDirectory() as tmp:
                write_jsonl(Path(tmp, 'games-0003.jsonl'), [record])
                [loaded] = load_records(tmp)
            self.assertEqual(loaded, json.loads(json.dumps(record)))
            self.assertEqual(replay_states(loaded['actions'])[-1].result, result)
            self.assertEqual(loaded['movers'][:2], ['opening', 'opening'])
            states = replay_states(loaded['actions'])
            for ply, mover in enumerate(loaded['movers'][2:], 2):
                self.assertEqual(mover, 'learner' if states[ply].turn == side else 'opponent')
            self.assertEqual(len(loaded['learner_search']), len(trajectory))
            for entry, (state, pi, _, q_mask) in zip(loaded['learner_search'], trajectory):
                self.assertEqual(states[entry['ply']], state)
                self.assertEqual([a for a, _ in entry['policy']], np.flatnonzero(pi).tolist())
                self.assertEqual([a for a, _ in entry['q']], np.flatnonzero(q_mask).tolist())
                self.assertTrue(-1. <= entry['root_value'] <= 1.)
            self.assertEqual(loaded['learner_outcome'],
                             'draw' if result == 0 else ('win' if result == side else 'loss'))


class ReviewTests(unittest.TestCase):
    def review(self, state, played, sims=300):
        return review_position(Uniform(), state, played, sims, SearchConfig(), leaf_batch=1)

    def test_value_is_from_the_player_to_move_for_both_colours(self):
        for turn in (1, -1):
            r = self.review(winning_position(turn), 21)
            self.assertEqual(r['best'], 20)
            self.assertEqual(r['value'], 1.)        # the mover wins
            self.assertEqual(r['q_best'], 1.)
            self.assertTrue(r['suspected_mistake'])  # 21 skipped the win
            # Outcome target of that parent if the mover went on to win the game:
            self.assertEqual(value_target('outcome', turn * winning_position(turn).turn, r['value']), 1.)
            self.assertEqual(value_target('outcome', -turn * winning_position(turn).turn, r['value']), -1.)

    def test_planted_blunder_is_flagged_and_the_policy_prefers_a_safe_move(self):
        for turn in (1, -1):
            state = blunder_position(turn)
            r = self.review(state, 36)
            self.assertTrue(r['suspected_mistake'])
            self.assertEqual(r['q_played'], -1.)
            self.assertTrue(r['q_played_exact'])
            # The safe alternatives are only estimated, so this is not "proven".
            self.assertFalse(r['proven'])
            self.assertIn(r['best'], SAFE)
            self.assertGreater(r['gap'], .5)
            self.assertIn(int(np.argmax(r['pi'])), SAFE)
            self.assertEqual(float(r['pi'][list(BLUNDERS)].sum()), 0.)
            safe = self.review(state, 40)
            self.assertFalse(safe['suspected_mistake'])
            self.assertFalse(safe['proven'])

    def test_proven_only_when_both_sides_of_the_gap_are_solved(self):
        state = win_or_lose_position()
        r = self.review(state, 26, sims=50)   # root is proven won before 26 is searched
        self.assertEqual((r['best'], r['q_best'], r['q_played']), (20, 1., -1.))
        self.assertTrue(r['suspected_mistake'] and r['proven'])
        self.assertEqual(r['value'], 1.)
        unproven = self.review(blunder_position(1), 36)
        self.assertFalse(unproven['proven'])

    def test_native_search_agrees_on_the_planted_positions(self):
        from sttt.cpp_env import is_cpp_available, to_fast_state
        if not is_cpp_available():
            self.skipTest('native extension not built')
        for turn in (1, -1):
            r = review_position(Uniform(), to_fast_state(blunder_position(turn)), 36, 300,
                                SearchConfig(), 4, use_cpp=True)
            self.assertTrue(r['suspected_mistake'])
            self.assertIn(int(np.argmax(r['pi'])), SAFE)
        r = review_position(Uniform(), to_fast_state(win_or_lose_position()), 26, 50,
                            SearchConfig(), 4, use_cpp=True)
        self.assertTrue(r['proven'])
        [row] = reanalysis_rows([{'result': 1}], [[dict(r, state=to_fast_state(win_or_lose_position()))]])
        np.testing.assert_array_equal(row[0].numpy(), encode(win_or_lose_position()))

    def test_opponent_positions_are_never_one_hot_imitated(self):
        state = blunder_position(1)
        # The opponent (X) played the safe move 40 from this parent and won.
        task = {'id': 'g', 'result': 1}
        reviews = [dict(self.review(state, 40), ply=0, mover='opponent', state=state)]
        [row] = reanalysis_rows([task], [reviews])
        pi = row[1].numpy()
        np.testing.assert_allclose(pi, reviews[0]['pi'])
        self.assertLess(pi.max(), .99)
        self.assertGreater(int((pi > 0).sum()), 1)
        self.assertTrue(row[5].any())
        # And through review_game on a real record, the target is the search policy too.
        game = review_game(Uniform(), {'actions': [40, 36], 'movers': ['opponent', 'learner'],
                                       'plies': [0, 1]}, 64, SearchConfig())
        self.assertEqual([g['mover'] for g in game], ['opponent', 'learner'])
        self.assertGreater(int((game[0]['pi'] > 0).sum()), 1)

    def test_value_modes(self):
        self.assertEqual(value_target('outcome', -1., .4), -1.)
        self.assertEqual(value_target('search', -1., .4), .4)
        self.assertAlmostEqual(value_target('mix', -1., .4, lam=.25), .25 * -1 + .75 * .4)
        self.assertAlmostEqual(value_target('mix', 1., -.2), .4)
        with self.assertRaises(ValueError):
            value_target('bogus', 0., 0.)
        state = winning_position(-1)
        review = dict(self.review(state, 20), ply=0, mover='learner', state=state)
        task = {'result': 1}   # X eventually won: O-to-move outcome target is -1
        for mode, expected in (('outcome', -1.), ('search', 1.), ('mix', 0.)):
            [row] = reanalysis_rows([task], [[review]], mode, .5)
            self.assertAlmostEqual(row[3], expected)
            np.testing.assert_array_equal(row[0].numpy(), encode(state))

    def test_selection_budget_outcomes_and_sides(self):
        records = [{'id': 'a', 'actions': [0] * 10, 'result': -1, 'learner_outcome': 'loss',
                    'movers': ['opening'] * 2 + ['learner', 'opponent'] * 4},
                   {'id': 'b', 'actions': [0] * 10, 'result': 1, 'learner_outcome': 'win',
                    'movers': ['opening'] * 2 + ['learner', 'opponent'] * 4}]
        tasks = select_tasks(records, ('loss',))
        self.assertEqual([t['id'] for t in tasks], ['a'])
        self.assertEqual(tasks[0]['plies'], list(range(2, 10)))
        self.assertEqual(select_tasks(records, ('loss',), ('opponent',))[0]['plies'], [3, 5, 7, 9])
        picked = select_tasks(records, ('loss', 'win'), budget=5, rng=np.random.default_rng(1))
        self.assertEqual(sum(len(t['plies']) for t in picked), 5)
        # External record: only actions (+ opening count); result derived by replay.
        [ext] = select_tasks([{'actions': [40, 36], 'opening_moves': 1}])
        self.assertEqual(ext['plies'], [1])
        self.assertIsNone(ext['result'])


class FakePool:
    """In-process stand-in for SelfPlayPool (python search, real play_game)."""
    def __init__(self):
        self.reanalyse_calls = 0

    def run(self, model, seeds, simulations, config, leaf_batch, matches=None, use_cpp=False):
        results = [play_game(model, simulations, int(s), config, leaf_batch) for s in seeds]
        return results, dict(inference_batches=0, inference_positions=0, max_inference_batch=0,
                             inference_seconds=0., game_retries=0, mean_inference_batch=0.)

    def reanalyse(self, model, tasks, simulations, config, leaf_batch, use_cpp=False):
        self.reanalyse_calls += 1
        return [review_game(model, t, simulations, config, leaf_batch) for t in tasks], {}


def train_args(output, **extra):
    return argparse.Namespace(iterations=1, games=3, simulations=4, leaf_batch=2, steps=2, batch=8,
                              buffer=10000, eval_every=0, save_every=0, workers=1, inference_batch=8,
                              seed=0, output=str(output), **extra)


REANALYSIS_DEFAULTS = dict(save_game_records=None, reanalyse=False, reanalyse_simulations=None,
                           reanalyse_fraction=.25, reanalyse_outcomes=['loss'],
                           reanalyse_sides=['learner', 'opponent'], reanalyse_margin=.3,
                           value_target='outcome', value_lambda=.5)


def run_loop(args, saved=None, replay=()):
    Path(args.output).mkdir(parents=True)
    torch.manual_seed(0)
    model = Network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    buffer = deque(replay, maxlen=args.buffer)
    meta = deque([(0, 'legacy')] * len(buffer), maxlen=args.buffer)
    rng = np.random.default_rng(args.seed)
    pool = FakePool()
    _train_loop(args, model, saved or {}, 'mlp', optimizer, buffer, Path(args.output), rng, 'cpu', pool,
                replay_meta=meta)
    return model, buffer, meta, rng, pool


class TrainingIntegrationTests(unittest.TestCase):
    def test_default_flags_leave_training_output_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            # An old namespace (no new attributes at all) versus the new defaults.
            m1, r1, meta1, rng1, pool1 = run_loop(train_args(Path(tmp, 'a')))
            m2, r2, meta2, rng2, pool2 = run_loop(train_args(Path(tmp, 'b'), **REANALYSIS_DEFAULTS))
            self.assertEqual((pool1.reanalyse_calls, pool2.reanalyse_calls), (0, 0))
            self.assertEqual(len(r1), len(r2))
            for a, b in zip(r1, r2):
                for x, y in zip(a, b):
                    self.assertTrue(torch.equal(x, y) if torch.is_tensor(x) else x == y)
            self.assertEqual(list(meta1), list(meta2))
            self.assertEqual({k for _, k in meta2}, {'self'})
            self.assertEqual(rng1.bit_generator.state, rng2.bit_generator.state)
            for k, v in m1.state_dict().items():
                self.assertTrue(torch.equal(v, m2.state_dict()[k]))
            self.assertTrue(all(row[3] in (-1., 0., 1.) for row in r2))
            row = json.loads(Path(tmp, 'b', 'metrics.jsonl').read_text().splitlines()[-1])
            self.assertNotIn('reanalysis', row)
            self.assertEqual(sorted(p.name for p in Path(tmp, 'b').iterdir()),
                             ['.run.lock', 'latest.pt', 'metrics.jsonl'] if Path(tmp, 'b', '.run.lock').exists()
                             else ['latest.pt', 'metrics.jsonl'])

    def test_reanalysis_mixing_respects_the_cap_and_logs_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = {**REANALYSIS_DEFAULTS, 'reanalyse': True, 'reanalyse_outcomes': ['loss', 'draw', 'win'],
                        'save_game_records': str(Path(tmp, 'records')), 'value_target': 'mix'}
            args = train_args(Path(tmp, 'run'), **settings)
            _, replay, meta, _, pool = run_loop(args)
            self.assertEqual(pool.reanalyse_calls, 1)
            kinds = [k for _, k in meta]
            new = kinds.count('self')
            added = kinds.count('reanalysis')
            self.assertGreater(added, 0)
            self.assertLessEqual(added, int(.25 * new))
            self.assertEqual(len(replay), new + added)
            row = json.loads(Path(tmp, 'run', 'metrics.jsonl').read_text().splitlines()[-1])
            info = row['reanalysis']
            self.assertEqual(info['rows_added'], added)
            self.assertEqual(info['positions'], added)
            self.assertEqual(info['simulations'], 16)
            self.assertGreaterEqual(info['seconds'], 0.)
            self.assertIn('reanalysis', row['stage_seconds']['stages'])
            records = load_records(Path(tmp, 'records'))
            self.assertEqual(len(records), 3)
            report = Path(tmp, 'records', 'reanalysis-0001.jsonl').read_text().splitlines()
            self.assertTrue(report)
            # The iteration's checkpoint (reanalysed rows included) reloads.
            saved = torch.load(Path(tmp, 'run', 'latest.pt'), weights_only=False)
            self.assertEqual(len(unpack_replay(saved['replay'])), len(replay))

    def test_old_checkpoint_resumes_with_reanalysis_on(self):
        legacy_rows = [(torch.zeros(289), torch.full((81,), 1 / 81), torch.ones(81, dtype=torch.bool), 0.)] * 4
        packed = pack_replay(unpack_replay(legacy_rows))
        replay = unpack_replay(packed)
        with tempfile.TemporaryDirectory() as tmp:
            args = train_args(Path(tmp, 'run'), **{**REANALYSIS_DEFAULTS, 'reanalyse': True,
                                                  'reanalyse_outcomes': ['loss', 'draw', 'win']})
            _, buffer, meta, _, _ = run_loop(args, saved={'iteration': 7}, replay=replay)
            saved = torch.load(Path(tmp, 'run', 'latest.pt'), weights_only=False)
            self.assertEqual(saved['iteration'], 8)
            rows = unpack_replay(saved['replay'])
            self.assertEqual(len(rows), len(buffer))
            self.assertTrue(all(len(r) == 6 for r in rows))
            self.assertEqual([k for _, k in meta][:4], ['legacy'] * 4)


class GameSplitTests(unittest.TestCase):
    def test_holdout_never_shares_a_game_with_training(self):
        rng = np.random.default_rng(3)
        plies, truth = [], []
        for game in range(200):
            start, step = int(rng.integers(0, 5)), int(rng.choice([1, 2]))
            length = int(rng.integers(8, 40))
            plies += list(range(start, start + step * length, step))
            truth += [game] * length
        plies, truth = np.array(plies), np.array(truth)
        np.testing.assert_array_equal(game_ids(plies), truth)
        val, train = split_by_game(plies, .1, np.random.default_rng(0))
        self.assertEqual(len(val) + len(train), len(plies))
        self.assertFalse(set(truth[val]) & set(truth[train]))
        self.assertEqual(len(set(truth[val])), 20)
        # Subsampling rows can merge games but never split one.
        keep = np.sort(np.random.default_rng(1).choice(len(plies), len(plies) // 3, replace=False))
        val, train = split_by_game(plies[keep], .1, np.random.default_rng(0))
        self.assertFalse(set(truth[keep][val]) & set(truth[keep][train]))


if __name__ == '__main__':
    unittest.main()
