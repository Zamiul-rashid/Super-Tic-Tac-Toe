import unittest
import numpy as np
import torch
from sttt.env import State
from sttt.learning import encode
from sttt.population import SYMMETRIES
from sttt.unet import CELL_TO_GRID, GRID_TO_CELL, planes_from_flat


class PlaneGeometryTests(unittest.TestCase):
    def test_index_tables_are_inverse_permutations(self):
        self.assertEqual(sorted(CELL_TO_GRID.tolist()), list(range(81)))
        self.assertTrue(torch.equal(GRID_TO_CELL[CELL_TO_GRID], torch.arange(81)))

    def test_stone_lands_at_the_geometric_cell(self):
        for action in [0, 8, 40, 72, 80, 13, 67]:
            state = State().play(action)                      # X played; now O to move
            planes = planes_from_flat(torch.from_numpy(encode(state))[None])
            b, c = action // 9, action % 9
            row, col = (b // 3) * 3 + c // 3, (b % 3) * 3 + c % 3
            self.assertEqual(planes.shape, (1, 8, 9, 9))
            self.assertEqual(float(planes[0, 2, row, col]), 1.0)   # opponent-of-mover plane
            self.assertEqual(float(planes[0, 2].sum()), 1.0)
            self.assertEqual(float(planes[0, 0].sum()), 80.0)     # empties

    def test_board_and_forced_planes_are_tiled_over_their_mini_board(self):
        state = State().play(40)                              # forces board 4 (centre)
        planes = planes_from_flat(torch.from_numpy(encode(state))[None])
        forced = planes[0, 7]
        self.assertEqual(float(forced.sum()), 9.0)
        self.assertTrue(torch.equal(forced[3:6, 3:6], torch.ones(3, 3)))
        boards_open = planes[0, 3]
        self.assertEqual(float(boards_open.sum()), 81.0)      # all nine boards still open

    def test_free_move_forced_plane_is_all_ones(self):
        state = State()
        planes = planes_from_flat(torch.from_numpy(encode(state))[None])
        self.assertTrue(torch.equal(planes[0, 7], torch.ones(9, 9)))

    def test_symmetry_table_acts_on_planes_as_a_whole_board_transform(self):
        rng = np.random.default_rng(5)
        state = State()
        for _ in range(17):
            state = state.play(int(rng.choice(state.legal_actions())))
        x = torch.from_numpy(encode(state))
        base = planes_from_flat(x[None])[0]
        grid = np.arange(81).reshape(9, 9)
        expected_ops = []
        for mirror in (False, True):
            b = np.fliplr(grid) if mirror else grid
            for k in range(4):
                expected_ops.append(torch.as_tensor(np.rot90(b, k).copy().ravel()))
        for inputs, _ in SYMMETRIES:
            xt = torch.empty_like(x); xt[torch.as_tensor(inputs)] = x
            transformed = planes_from_flat(xt[None])[0]
            matches = any(torch.equal(transformed.flatten(1)[:, op], base.flatten(1)) for op in expected_ops)
            self.assertTrue(matches)
