import os, tempfile, unittest
from collections import Counter
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from sttt.cpp_env import is_cpp_available
from sttt.bootstrap import (generate_rows, load_shards, pack_arrays, sample_bootstrap_match,
                            write_shard, _worker_init)
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


class GenerateDatasetCommandTests(unittest.TestCase):
    def test_writes_shards_and_a_manifest(self):
        import json, subprocess, sys
        if not is_cpp_available():
            self.fail('native extension required')
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, '-m', 'sttt.ai', 'generate-dataset', '--output', tmp, '--games', '6',
                   '--workers', '2', '--simulations', '8', '--leaf-batch', '4', '--shard-games', '3',
                   '--alphabeta-share', '0.5', '--depths', '2', '--seed', '9']
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            shards = sorted(Path(tmp).glob('shard-*.pt'))
            manifest = json.loads(Path(tmp, 'manifest.json').read_text())
        self.assertEqual(len(shards), 2)
        self.assertEqual(manifest['games'], 6)
        self.assertGreater(manifest['positions'], 40)
        self.assertEqual(sum(manifest['ply_histogram'].values()), manifest['positions'])
        self.assertGreater(manifest['positions_per_second'], 0)
        self.assertIn('native_build', manifest)

    def test_refuses_more_than_ten_workers(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, '-m', 'sttt.ai', 'generate-dataset', '--output', tmp,
                                   '--games', '1', '--workers', '11'], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('workers', proc.stderr)

    def test_refuses_a_directory_that_already_holds_shards(self):
        # A stale shard-*.pt from a different seed/simulation/depth config would
        # otherwise be silently merged by load_shards, and manifest.json would be
        # overwritten so it no longer describes the directory -- wrong data plus
        # wrong provenance. Must be refused before anything is written, so a
        # bogus placeholder file (no native extension needed) is enough to prove it.
        from sttt.bootstrap import generate_dataset
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'shard-0000.pt').write_bytes(b'not a real shard')
            args = argparse.Namespace(output=tmp, workers=1, games=1, simulations=8, leaf_batch=4,
                                      shard_games=1, alphabeta_share=0.5, depths=[2], seed=1)
            with self.assertRaises(RuntimeError) as ctx:
                generate_dataset(args)
        self.assertIn(tmp, str(ctx.exception))
        self.assertIn('shard', str(ctx.exception))
        # Refusal must precede the native-extension check reordering concern:
        # the manifest must not have been touched either.
        self.assertFalse(Path(tmp, 'manifest.json').exists())


class WorkerInitTests(unittest.TestCase):
    def test_niceness_and_thread_cap_are_applied(self):
        # A later refactor could silently drop os.nice(10) or the env-var cap and nothing
        # would fail until the machine hangs at 32 x 100 %, so pin both side effects down.
        # os.nice is not reversible downward by an unprivileged process (and could already
        # be capped at 19 if the test runner itself was launched under `nice -n 19`), so we
        # record the call instead of letting it actually renice this test process.
        saved = os.environ.pop('OMP_NUM_THREADS', None)
        try:
            with mock.patch('sttt.bootstrap.os.nice') as nice_mock:
                _worker_init()
            nice_mock.assert_called_once_with(10)
            self.assertEqual(os.environ.get('OMP_NUM_THREADS'), '1')
        finally:
            os.environ.pop('OMP_NUM_THREADS', None)
            if saved is not None:
                os.environ['OMP_NUM_THREADS'] = saved


class PretrainCommandTests(unittest.TestCase):
    def test_pretrain_writes_a_checkpoint_that_train_resumes(self):
        import json, subprocess, sys
        if not is_cpp_available():
            self.fail('native extension required')
        with tempfile.TemporaryDirectory() as tmp:
            data, run, cont = Path(tmp, 'data'), Path(tmp, 'boot'), Path(tmp, 'cont')
            subprocess.run([sys.executable, '-m', 'sttt.ai', 'generate-dataset', '--output', str(data),
                            '--games', '6', '--workers', '2', '--simulations', '8', '--leaf-batch', '4',
                            '--shard-games', '3', '--depths', '2', '--seed', '4'],
                           check=True, capture_output=True, text=True, timeout=600)
            subprocess.run([sys.executable, '-m', 'sttt.ai', 'pretrain', '--dataset', str(data),
                            '--output', str(run), '--arch', 'unet', '--epochs', '2', '--batch', '16',
                            '--holdout', '0.2', '--replay-buffer', '50', '--device', 'cpu', '--seed', '0'],
                           check=True, capture_output=True, text=True, timeout=600)
            rows = [json.loads(l) for l in Path(run, 'pretrain_metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in rows], [1, 2])
            for r in rows:
                for k in ('train_loss', 'train_policy', 'train_value', 'train_q', 'val_policy', 'val_value', 'val_q'):
                    self.assertTrue(np.isfinite(r[k]), k)
            self.assertLess(rows[1]['lr'], rows[0]['lr'])
            ckpt = torch.load(Path(run, 'latest.pt'), map_location='cpu', weights_only=False)
            self.assertEqual(ckpt['arch'], 'unet'); self.assertEqual(ckpt['iteration'], 0)
            self.assertEqual(replay_length(ckpt['replay']), 50)
            self.assertEqual(len(ckpt['replay_meta']), 50)
            self.assertIsNone(ckpt['lr_schedule'])
            self.assertTrue(Path(run, 'model-0000.pt').is_file())
            # The online trainer resumes it and attaches a fresh cosine phase.
            subprocess.run([sys.executable, '-m', 'sttt.ai', 'train', '--resume', str(run / 'latest.pt'),
                            '--output', str(cont), '--backend', 'python', '--device', 'cpu', '--iterations', '1',
                            '--games', '1', '--simulations', '4', '--steps', '2', '--batch', '8', '--workers', '1',
                            '--leaf-batch', '2', '--save-every', '0', '--lr-schedule', 'cosine', '--lr', '0.001',
                            '--lr-min', '0.00001', '--lr-iterations', '10', '--replay-sampling', 'stratified'],
                           check=True, capture_output=True, text=True, timeout=600)
            out = [json.loads(l) for l in Path(cont, 'metrics.jsonl').read_text().splitlines()]
        out = [r for r in out if 'q_loss' in r]
        self.assertEqual(out[-1]['iteration'], 1)
        self.assertGreater(out[-1]['positions'], 50)          # warm buffer plus the new game
        self.assertIsNotNone(out[-1]['q_loss'])
