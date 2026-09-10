import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from sttt.env import State, winner
from sttt.learning import Network, encode, INPUTS
from sttt.search import search
from sttt.opponent import Opponent

class Tests(unittest.TestCase):
    def test_random_games(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            s = State()
            for ply in range(81):
                legal = s.legal_actions()
                self.assertTrue(legal)
                a = int(rng.choice(legal))
                child = s.play(a)
                self.assertEqual(sum(a != b for a,b in zip(s.cells,child.cells)),1)
                self.assertEqual(child.forced,a%9 if child.boards[a%9] == 0 else -1)
                s = child
                if s.result is not None:
                    self.assertEqual(s.legal_actions(),[])
                    break
            self.assertIsNotNone(s.result)

    def test_rules(self):
        self.assertEqual(len(State().legal_actions()),81)
        self.assertEqual(State().play(4).forced,4)
        with self.assertRaises(ValueError):
            State().play(4).play(0)
        self.assertEqual(winner((0,0,0,1,1,1,0,0,0)),1)
        self.assertEqual(winner((2,2,2,0,0,0,0,0,0)),0)
        cells = [0]*81
        cells[:9] = [1,-1,1,1,-1,-1,-1,1,0]
        child = State(tuple(cells),forced=0).play(8)
        self.assertEqual(child.boards[0],2)
        boards = (0,0,0,0,2,0,0,0,0)
        self.assertEqual(State(boards=boards).play(4).forced,-1)

    def test_search_win(self):
        cells = [0]*81
        cells[18:20] = [1,1]
        s = State(tuple(cells),(1,1,0,0,0,0,0,0,0),forced=2)
        torch.set_num_threads(1)
        torch.manual_seed(1)
        p = search(s,Network(),100,np.random.default_rng(2))
        self.assertEqual(int(p.argmax()),20)
        self.assertAlmostEqual(float(p.sum()),1,places=6)
        self.assertEqual(len(encode(s)),INPUTS)

    def test_adaptation(self):
        o = Opponent()
        before = o.weights().copy()
        for _ in range(20):
            o.observe(State(),4)
        self.assertGreater(o.weights()[1],before[1])
        base = np.ones(81)/81
        self.assertGreater(o.predict(State(),base)[4],base[4])

    def test_profile_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'player.json'
            o = Opponent(path)
            o.observe(State(),4)
            restored = Opponent(path)
            np.testing.assert_allclose(restored.weights(),o.weights())
            self.assertEqual(restored.observations,1)

    def test_finished_game(self):
        cells = [0]*81
        cells[18:20] = [1,1]
        s = State(tuple(cells),(1,1,0,0,0,0,0,0,0),forced=2).play(20)
        self.assertEqual(s.result,1)
        self.assertEqual(s.legal_actions(),[])
        with self.assertRaises(ValueError):
            s.play(30)

    def test_color_symmetry(self):
        s = State().play(4).play(40).play(36)
        swapped = State(tuple(-x for x in s.cells),
                        tuple(-b if b in (-1,1) else b for b in s.boards),
                        -s.turn,s.forced,s.result)
        np.testing.assert_array_equal(encode(s),encode(swapped))
        self.assertEqual(s.legal_actions(),swapped.legal_actions())

    def test_resnet(self):
        from sttt.learning import ResNet, create_model, load_model
        model = create_model('resnet')
        self.assertIsInstance(model, ResNet)
        param_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(param_count, 1_500_000)
        self.assertLess(param_count, 2_500_000)
        s = State()
        p, v = model.evaluate(s)
        self.assertEqual(p.shape, (81,))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        self.assertTrue(-1.0 <= v <= 1.0)

if __name__ == '__main__':
    unittest.main()
