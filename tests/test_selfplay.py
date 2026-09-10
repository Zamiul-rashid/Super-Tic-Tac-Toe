import unittest
import numpy as np
import torch
from sttt.learning import Network
from sttt.search import SearchConfig
from sttt.selfplay import SelfPlayPool, play_game
from sttt.env import State


class SelfPlayTests(unittest.TestCase):
    def test_worker_pool_routes_batches_and_keeps_workers_for_next_iteration(self):
        torch.set_num_threads(1)
        model = Network().eval()
        with SelfPlayPool(2, batch_size=8) as pool:
            pids = [p.pid for p in pool.processes]
            results, stats = pool.run(model,[3,4],8,SearchConfig(),4)
            self.assertEqual(len(results),2)
            self.assertGreater(stats['max_inference_batch'],1)
            self.assertLessEqual(stats['max_inference_batch'],8)
            for trajectory, outcome, _ in results:
                for state, pi in trajectory:
                    self.assertAlmostEqual(float(pi.sum()),1.,places=5)
                    self.assertTrue(set(np.flatnonzero(pi)).issubset(state.legal_actions()))
                self.assertIn(outcome,(-1,0,1))
            pool.run(model,[9],4,SearchConfig(),2)
            self.assertEqual([p.pid for p in pool.processes],pids)
        self.assertTrue(all(not p.is_alive() for p in pool.processes))

    def test_worker_exception_propagates_and_cleanup_does_not_hang(self):
        with SelfPlayPool(1, batch_size=4) as pool:
            with self.assertRaisesRegex(RuntimeError,'positive budgets'):
                pool.run(Network().eval(),[0],0,SearchConfig(),4)
        self.assertTrue(all(not p.is_alive() for p in pool.processes))


if __name__ == '__main__':
    unittest.main()
