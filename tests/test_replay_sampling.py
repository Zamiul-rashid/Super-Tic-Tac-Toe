import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from sttt.env import State
from sttt.learning import encode
from sttt.replay_sampling import (BIN_LABELS, PLY_BINS, StratifiedSampler, ply_bin,
                                  row_ply, sample_bin_fractions)


class RowPlyTests(unittest.TestCase):
    def test_ply_counts_moves_played_on_random_games(self):
        rng = np.random.default_rng(3)
        for _ in range(30):
            state, plies = State(), 0
            while state.result is None and plies < 70:
                state = state.play(int(rng.choice(state.legal_actions())))
                plies += 1
                self.assertEqual(row_ply(encode(state)), plies)
                self.assertEqual(row_ply(torch.from_numpy(encode(state))), plies)

    def test_bins_cover_all_plies_and_labels_match(self):
        self.assertEqual(len(BIN_LABELS), PLY_BINS)
        self.assertEqual(ply_bin(0), 0)
        self.assertEqual(ply_bin(8), 0)
        self.assertEqual(ply_bin(9), 1)
        self.assertEqual(ply_bin(62), 6)
        self.assertEqual(ply_bin(63), 7)
        self.assertEqual(ply_bin(80), 7)
        self.assertEqual(BIN_LABELS[0], '0-8')
        self.assertEqual(BIN_LABELS[-1], '63-80')


class StratifiedSamplerTests(unittest.TestCase):
    def test_equal_share_per_bin_when_every_bin_is_deep(self):
        bins = np.repeat(np.arange(PLY_BINS), 1000)
        picks = StratifiedSampler(bins).sample(512, np.random.default_rng(0))
        self.assertEqual(len(picks), 512)
        self.assertEqual(len(set(picks.tolist())), 512)
        counts = np.bincount(bins[picks], minlength=PLY_BINS)
        self.assertTrue((counts == 64).all(), counts)

    def test_thin_bins_are_exhausted_and_their_share_redistributed(self):
        # 5 endgame rows, 2000 rows in each of two midgame bins, nothing else.
        bins = np.concatenate([np.full(5, 7), np.full(2000, 3), np.full(2000, 4)])
        picks = StratifiedSampler(bins).sample(300, np.random.default_rng(1))
        self.assertEqual(len(picks), 300)
        self.assertEqual(len(set(picks.tolist())), 300)
        counts = np.bincount(bins[picks], minlength=PLY_BINS)
        self.assertEqual(counts[7], 5)
        self.assertEqual(counts[3] + counts[4], 295)
        self.assertLessEqual(abs(counts[3] - counts[4]), 1)

    def test_batch_larger_than_buffer_returns_every_row_once(self):
        bins = np.array([0, 0, 3, 7, 7, 7])
        picks = StratifiedSampler(bins).sample(50, np.random.default_rng(2))
        self.assertEqual(sorted(picks.tolist()), [0, 1, 2, 3, 4, 5])

    def test_single_bin_degenerates_to_uniform_without_replacement(self):
        bins = np.zeros(100, dtype=np.int8)
        picks = StratifiedSampler(bins).sample(40, np.random.default_rng(3))
        self.assertEqual(len(set(picks.tolist())), 40)

    def test_empty_buffer_returns_empty(self):
        picks = StratifiedSampler(np.zeros(0, dtype=np.int8)).sample(16, np.random.default_rng(4))
        self.assertEqual(len(picks), 0)

    def test_histogram_and_fractions_are_labelled_by_bin(self):
        bins = np.array([0, 0, 0, 3, 7])
        sampler = StratifiedSampler(bins)
        self.assertEqual(sampler.histogram(), {'0-8': 3, '27-35': 1, '63-80': 1})
        fractions = sample_bin_fractions(bins, np.array([0, 3, 4, 1]))
        self.assertAlmostEqual(fractions['0-8'], 0.5)
        self.assertAlmostEqual(fractions['27-35'], 0.25)
        self.assertAlmostEqual(fractions['63-80'], 0.25)


class TrainerWiringTests(unittest.TestCase):
    def _run(self, sampling, output):
        cmd = [sys.executable, '-m', 'sttt.ai', 'train', '--backend', 'python', '--device', 'cpu',
               '--iterations', '1', '--games', '2', '--simulations', '4', '--steps', '3',
               '--batch', '8', '--workers', '1', '--leaf-batch', '2', '--save-every', '0',
               '--replay-sampling', sampling, '--output', output, '--seed', '5']
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        rows = [json.loads(l) for l in (Path(output) / 'metrics.jsonl').read_text().splitlines()]
        return [r for r in rows if 'replay_sampling' in r][-1]

    def test_stratified_run_reports_buffer_histogram_and_sampled_fractions(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run('stratified', tmp)
        self.assertEqual(report['replay_sampling'], 'stratified')
        self.assertEqual(sum(report['replay_depth_histogram'].values()), report['positions'])
        self.assertAlmostEqual(sum(report['replay_sample_depth_fractions'].values()), 1.0, places=6)
        self.assertGreaterEqual(len(report['replay_sample_depth_fractions']), 2)

    def test_uniform_is_the_default_and_still_reports_composition(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, '-m', 'sttt.ai', 'train', '--backend', 'python', '--device', 'cpu',
                   '--iterations', '1', '--games', '1', '--simulations', '4', '--steps', '2',
                   '--batch', '8', '--workers', '1', '--leaf-batch', '2', '--save-every', '0',
                   '--output', tmp, '--seed', '5']
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            rows = [json.loads(l) for l in (Path(tmp) / 'metrics.jsonl').read_text().splitlines()]
        report = [r for r in rows if 'replay_sampling' in r][-1]
        self.assertEqual(report['replay_sampling'], 'uniform')
        self.assertEqual(sum(report['replay_depth_histogram'].values()), report['positions'])
