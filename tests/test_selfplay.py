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
                for state, pi, q, q_mask in trajectory:
                    self.assertAlmostEqual(float(pi.sum()),1.,places=5)
                    self.assertTrue(set(np.flatnonzero(pi)).issubset(state.legal_actions()))
                    self.assertEqual(q.shape, (81,)); self.assertTrue(q_mask.any())
                self.assertIn(outcome,(-1,0,1))
            pool.run(model,[9],4,SearchConfig(),2)
            self.assertEqual([p.pid for p in pool.processes],pids)
        self.assertTrue(all(not p.is_alive() for p in pool.processes))

    def test_worker_exception_propagates_and_cleanup_does_not_hang(self):
        with SelfPlayPool(1, batch_size=4) as pool:
            with self.assertRaisesRegex(RuntimeError,'positive budgets'):
                pool.run(Network().eval(),[0],0,SearchConfig(),4)
        self.assertTrue(all(not p.is_alive() for p in pool.processes))

    def test_failed_game_is_retried_once_then_succeeds(self):
        import os, sys, tempfile
        from sttt.population import MatchSpec
        marker = os.path.join(tempfile.mkdtemp(), 'crashed')
        # Dies on its first move request ever, then plays the first legal move.
        engine = ("import os,sys\n"
                  f"m={marker!r}\n"
                  "while True:\n"
                  " sys.stdin.readline(); n=int(sys.stdin.readline())\n"
                  " legal=[sys.stdin.readline().strip() for _ in range(n)]\n"
                  " if not os.path.exists(m): open(m,'w').close(); sys.exit(1)\n"
                  " print(legal[0], flush=True)\n")
        match = MatchSpec(kind='utttai', learner_side=1, engine_command=(sys.executable, '-c', engine),
                          engine_timeout=10.)
        with SelfPlayPool(1, batch_size=4) as pool:
            results, stats = pool.run(Network().eval(), [5], 4, SearchConfig(), 2, matches=[match])
        self.assertEqual(stats['game_retries'], 1)
        self.assertIn(results[0][1], (-1, 0, 1))


if __name__ == '__main__':
    unittest.main()
