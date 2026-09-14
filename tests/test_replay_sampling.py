import unittest
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
