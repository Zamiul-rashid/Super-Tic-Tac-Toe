import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from sttt.env import State
from sttt.learning import Network, encode
from sttt.search import SearchConfig
from sttt.selfplay import SelfPlayPool, play_game
from sttt.population import (MatchSpec, POPULATION_WEIGHTS, augment_batch,
                              population_quota_counts, sample_match, sample_matches,
                              SYMMETRIES)


class PopulationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_sampler_varies_and_reproduces_without_live_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'latest.pt').touch()
            specs = [sample_match(i, directory) for i in range(1000)]
            self.assertEqual(specs, [sample_match(i, directory) for i in range(1000)])
            self.assertNotIn('history', {s.kind for s in specs})
            Path(directory, 'model-0001.pt').touch()
            specs = [sample_match(i, directory) for i in range(1000)]
            self.assertEqual({s.kind for s in specs},
                             {'self', 'history', 'alphabeta', 'tactical', 'threat',
                              'openspiel', 'utttai', 'style'})
            self.assertEqual({s.learner_side for s in specs}, {-1, 1})
            self.assertEqual({s.depth for s in specs}, {1, 2, 3, 4, 5, 6, 7, 8})
            self.assertTrue(all(s.checkpoint.endswith('model-0001.pt') for s in specs if s.kind == 'history'))

    def test_iteration_schedule_has_meaningful_quotas(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'model-0001.pt').touch()
            Path(directory, 'best.pt').touch()
            config = {'utttai': {'command': ['wrapper', '--simulations', '{simulations}'],
                                 'protocol': 'action_index', 'timeout': 5.}}
            matches = sample_matches(range(16), directory, config)
            counts = {kind: sum(match.kind == kind for match in matches)
                      for kind in POPULATION_WEIGHTS}
            self.assertEqual(counts, population_quota_counts(16))
            totals = dict.fromkeys(POPULATION_WEIGHTS, 0)
            for offset in range(0, 400, 16):
                for kind, count in population_quota_counts(16, offset).items():
                    totals[kind] += count
            self.assertEqual(totals, {kind: round(400 * weight)
                                     for kind, weight in POPULATION_WEIGHTS.items()})
            self.assertTrue(all(match.opening_moves == 0 for match in matches if match.kind == 'utttai'))
            self.assertTrue(all(any('{simulations}' in part for part in match.engine_command)
                                for match in matches if match.kind == 'utttai'))
            self.assertTrue(all(match.simulations in {64, 128, 256}
                                for match in matches if match.kind == 'utttai'))

    def test_external_turns_never_become_policy_targets(self):
        model = Network().eval()
        for side in (-1, 1):
            trajectory, result, stats = play_game(model, 2, 91, SearchConfig(), 2,
                MatchSpec(kind='style', learner_side=side, opening_moves=3))
            self.assertTrue(trajectory)
            self.assertTrue(all(state.turn == side for state, _ in trajectory))
            self.assertIn(result, (-1, 0, 1))
            self.assertGreater(stats['plies'], len(trajectory))
            for state, pi in trajectory:
                self.assertAlmostEqual(float(pi.sum()), 1., places=5)
                self.assertTrue(set(np.flatnonzero(pi)).issubset(state.legal_actions()))

    def test_symmetry_respects_rules_and_forced_board(self):
        state = State()
        for a in [0, 1, 9, 2, 18]:
            state = state.play(a)
        for inputs, cells in SYMMETRIES:
            transformed = State()
            for a in [0, 1, 9, 2, 18]:
                transformed = transformed.play(int(cells[a]))
            expected = np.empty_like(encode(state))
            expected[inputs] = encode(state)
            np.testing.assert_array_equal(expected, encode(transformed))
            self.assertEqual(set(cells[state.legal_actions()]), set(transformed.legal_actions()))
        x = torch.tensor(np.stack([encode(state)] * 16))
        pi = torch.zeros(16, 81)
        pi[:, state.legal_actions()] = 1 / len(state.legal_actions())
        mask = pi > 0
        xx, pp, mm = augment_batch(x, pi, mask, np.random.default_rng(4))
        self.assertTrue(torch.equal(pp > 0, mm))
        self.assertTrue(torch.allclose(pp.sum(1), torch.ones(16)))
        self.assertTrue(torch.allclose(xx.sum(1), x.sum(1)))

    def test_pool_mixes_history_and_live_inference_and_reuses_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'model-0001.pt')
            torch.save({'model': Network().state_dict(), 'arch': 'mlp', 'iteration': 1}, path)
            specs = [MatchSpec(), MatchSpec(kind='history', checkpoint=str(path), simulations=2),
                     MatchSpec(kind='alphabeta', depth=1, nodes=50),
                     MatchSpec(kind='tactical', nodes=50)]
            with SelfPlayPool(2, batch_size=4) as pool:
                results, stats = pool.run(Network().eval(), [1, 2, 3, 4], 2, SearchConfig(), 2, specs)
                self.assertEqual([r[2]['match']['kind'] for r in results], [s.kind for s in specs])
                self.assertGreater(stats['inference_positions'], 0)
                pool.run(Network().eval(), [5], 2, SearchConfig(), 2)
                with self.assertRaises(ValueError):
                    pool.run(Network().eval(), [1], 2, SearchConfig(), 2, [])


if __name__ == '__main__':
    unittest.main()
