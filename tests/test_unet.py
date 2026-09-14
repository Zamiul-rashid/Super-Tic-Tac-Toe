import unittest
import numpy as np
import torch
from sttt.env import State
from sttt.learning import encode, create_model, load_model, arch_name
from sttt.population import SYMMETRIES
from sttt.unet import CELL_TO_GRID, GRID_TO_CELL, planes_from_flat, UNet


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


class UNetModuleTests(unittest.TestCase):
    def test_output_shapes_and_ranges(self):
        torch.manual_seed(0)
        model = UNet()
        x = torch.from_numpy(np.stack([encode(State()), encode(State().play(40))]))
        logits, value, q = model.forward_all(x)
        self.assertEqual(logits.shape, (2, 81))
        self.assertEqual(value.shape, (2,))
        self.assertEqual(q.shape, (2, 81))
        self.assertTrue((value.abs() <= 1).all() and (q.abs() <= 1).all())
        l2, v2 = model(x)
        self.assertTrue(torch.equal(l2, logits) and torch.equal(v2, value))

    def test_parameter_count_is_in_the_resnet_class(self):
        n = sum(p.numel() for p in UNet().parameters())
        self.assertGreater(n, 1_000_000)
        self.assertLess(n, 3_000_000)

    def test_policy_logits_are_in_cell_order(self):
        # Zero every weight, then bias the policy conv so grid position (row 4,
        # col 4) is hot: that is board 4, cell 4, i.e. action index 40.
        model = UNet()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        feats = torch.zeros(1, 64, 9, 9); feats[0, 0, 4, 4] = 1.0
        with torch.no_grad():
            model.policy.weight[0, 0] = 1.0
        logits = model.policy(feats).flatten(1)[:, CELL_TO_GRID]
        self.assertEqual(int(logits.argmax()), 40)

    def test_evaluate_many_works_and_masks_illegal_moves(self):
        torch.manual_seed(0)
        model = UNet().eval()
        state = State().play(40)
        (probs, value), = model.evaluate_many([state])
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        illegal = np.setdiff1d(np.arange(81), state.legal_actions())
        self.assertEqual(float(probs[illegal].sum()), 0.0)

    def test_registration_and_checkpoint_roundtrip(self):
        self.assertIsInstance(create_model('unet'), UNet)
        self.assertEqual(arch_name(create_model('unet')), 'unet')
        self.assertEqual(arch_name(create_model('resnet')), 'resnet')
        self.assertEqual(arch_name(create_model('mlp')), 'mlp')
        import tempfile
        from pathlib import Path
        model = UNet()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'm.pt')
            torch.save({'model': model.state_dict(), 'arch': 'unet', 'iteration': 0}, path)
            loaded, ckpt = load_model(path)
            self.assertIsInstance(loaded, UNet)
            torch.save({'model': model.state_dict(), 'iteration': 0}, path)   # no arch field
            loaded, _ = load_model(path)
            self.assertIsInstance(loaded, UNet)

    def test_fp16_autocast_forward_is_finite(self):
        if not torch.cuda.is_available():
            self.fail('CUDA required for this gate; do not skip on the training box')
        model = UNet().cuda()
        x = torch.from_numpy(np.stack([encode(State())] * 16)).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            logits, value, q = model.forward_all(x)
        self.assertTrue(torch.isfinite(logits).all() and torch.isfinite(value).all() and torch.isfinite(q).all())
