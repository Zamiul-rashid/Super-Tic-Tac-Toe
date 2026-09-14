import tempfile, unittest
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sttt.cpp_env import is_cpp_available
from sttt.bootstrap import (generate_rows, load_shards, pack_arrays, sample_bootstrap_match,
                            write_shard)
from sttt.ai import unpack_replay, replay_length
from sttt.replay_sampling import row_ply


class MatchSamplingTests(unittest.TestCase):
    def test_share_and_ranges_are_respected(self):
        kinds = Counter()
        for i in range(400):
            spec = sample_bootstrap_match(1, i, alphabeta_share=0.5, depths=(4, 5, 6))
            kinds[spec.kind] += 1
            self.assertIn(spec.kind, ('self', 'alphabeta'))
            self.assertIn(spec.opening_moves, range(0, 5))
            if spec.kind == 'alphabeta':
                self.assertIn(spec.depth, (4, 5, 6))
                self.assertIn(spec.learner_side, (1, -1))
                self.assertTrue(0. <= spec.epsilon <= .05)
        self.assertGreater(kinds['alphabeta'], 150)
        self.assertGreater(kinds['self'], 150)

    def test_is_deterministic_in_seed_and_stream(self):
        self.assertEqual(sample_bootstrap_match(3, 7, 0.5, (4,)), sample_bootstrap_match(3, 7, 0.5, (4,)))
        self.assertNotEqual(sample_bootstrap_match(3, 7, 0.5, (4, 5, 6)), sample_bootstrap_match(3, 8, 0.5, (4, 5, 6)))


class GenerationTests(unittest.TestCase):
    def setUp(self):
        if not is_cpp_available():
            self.fail('native extension required: make -C cpp clean && make -C cpp PYTHON=$PY')

    def test_rows_are_consistent_training_targets(self):
        rows = generate_rows(seed=11, games=3, simulations=16, leaf_batch=4, alphabeta_share=0.5, depths=(2,))
        n = rows['x'].shape[0]
        self.assertGreater(n, 20)
        self.assertEqual(rows['games'], 3)
        self.assertEqual(rows['x'].shape, (n, 289)); self.assertEqual(rows['x'].dtype, np.float32)
        self.assertEqual(rows['pi'].shape, (n, 81)); self.assertEqual(rows['mask'].dtype, bool)
        self.assertEqual(rows['q'].shape, (n, 81)); self.assertEqual(rows['q_mask'].dtype, bool)
        self.assertEqual(rows['z'].shape, (n,)); self.assertTrue(np.isin(rows['z'], [-1., 0., 1.]).all())
        np.testing.assert_allclose(rows['pi'].sum(1), 1., atol=1e-5)
        self.assertTrue((rows['pi'] * ~rows['mask'] == 0).all())
        self.assertTrue((rows['q_mask'] <= rows['mask']).all())
        self.assertTrue(rows['q_mask'].any(1).all())
        for i in range(n):
            self.assertEqual(int(rows['ply'][i]), row_ply(rows['x'][i]))

    def test_shard_roundtrip_is_loadable_by_the_replay_code(self):
        rows = generate_rows(seed=2, games=2, simulations=8, leaf_batch=4, alphabeta_share=0., depths=(2,))
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_shard(Path(tmp, 'shard-0000.pt'), rows, {'note': 'test'})
            self.assertEqual(summary['positions'], rows['x'].shape[0])
            shard = torch.load(Path(tmp, 'shard-0000.pt'), map_location='cpu', weights_only=False)
            self.assertEqual(shard['format'], 'bootstrap-v1')
            self.assertEqual(replay_length(shard['replay']), rows['x'].shape[0])
            back = unpack_replay(shard['replay'])
            self.assertEqual(len(back[0]), 6)
            data = load_shards(tmp)
            self.assertEqual(data['x'].shape[0], rows['x'].shape[0])
            self.assertEqual(data['ply'].dtype, torch.int16)
