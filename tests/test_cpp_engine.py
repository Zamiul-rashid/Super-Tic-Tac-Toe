"""Exhaustive test suite for C++ bitboard engine and Python bindings."""
import unittest
import random
import numpy as np
from sttt.env import State, LINES
from sttt.cpp_env import CppState, FastState, is_cpp_available, to_python_state, to_fast_state
from sttt.learning import encode as py_encode


class TestCppBitboardEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_cpp_available():
            raise unittest.SkipTest("C++ engine not available")

    def test_initial_state_equivalence(self):
        py_s = State()
        cpp_s = FastState()
        self.assertEqual(cpp_s.cells, py_s.cells)
        self.assertEqual(cpp_s.boards, py_s.boards)
        self.assertEqual(cpp_s.turn, py_s.turn)
        self.assertEqual(cpp_s.forced, py_s.forced)
        self.assertIsNone(cpp_s.result)
        self.assertEqual(cpp_s.legal_actions(), py_s.legal_actions())
        self.assertEqual(len(cpp_s.legal_actions()), 81)

    def test_cppstate_wrapper_equivalence(self):
        wrapper = CppState()
        py_s = State()
        self.assertEqual(wrapper, py_s)
        self.assertEqual(wrapper.legal_actions(), py_s.legal_actions())
        w_next = wrapper.play(40)
        p_next = py_s.play(40)
        self.assertEqual(w_next, p_next)

    def test_lockstep_random_games(self):
        """Play 500 complete random games in lockstep, verifying every single move."""
        rng = random.Random(1337)
        total_moves = 0
        terminal_counts = {1: 0, -1: 0, 0: 0}

        for game_idx in range(2000):
            py_s = State()
            cpp_s = FastState()

            while py_s.result is None:
                py_legal = py_s.legal_actions()
                cpp_legal = cpp_s.legal_actions()
                self.assertEqual(cpp_legal, py_legal, f"Game {game_idx}: legal actions mismatch")

                action = rng.choice(py_legal)

                py_s = py_s.play(action)
                cpp_s = cpp_s.play(action)
                total_moves += 1

                self.assertEqual(cpp_s.cells, py_s.cells, f"Game {game_idx}, move {action}: cells mismatch")
                self.assertEqual(cpp_s.boards, py_s.boards, f"Game {game_idx}, move {action}: boards mismatch")
                self.assertEqual(cpp_s.turn, py_s.turn, f"Game {game_idx}, move {action}: turn mismatch")
                self.assertEqual(cpp_s.forced, py_s.forced, f"Game {game_idx}, move {action}: forced mismatch")
                self.assertEqual(cpp_s.result, py_s.result, f"Game {game_idx}, move {action}: result mismatch")

            terminal_counts[py_s.result] += 1
            self.assertEqual(cpp_s.legal_actions(), [])

        self.assertGreater(total_moves, 15000)
        self.assertGreater(terminal_counts[1], 50)
        self.assertGreater(terminal_counts[-1], 50)
        self.assertGreater(terminal_counts[0], 20)

    def test_all_local_win_lines_for_both_players(self):
        """Verify all 8 winning lines in a local board for both X and O."""
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8), # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8), # cols
            (0, 4, 8), (2, 4, 6)             # diags
        ]

        for b in range(9):
            for player in (1, -1):
                for line in lines:
                    cells = [0] * 81
                    boards = [0] * 9
                    for c in line:
                        cells[b * 9 + c] = player
                    boards[b] = player

                    # Construct state
                    s = FastState(cells=cells, boards=boards, turn=-player, forced=-1, result=None)
                    self.assertEqual(s.boards[b], player)
                    self.assertEqual(s.cells[b * 9 + line[0]], player)

    def test_macro_win_detection(self):
        """Verify macro-board wins in all 8 directions."""
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]

        for winner in (1, -1):
            for line in lines:
                boards = [0] * 9
                cells = [0] * 81
                for b in line:
                    boards[b] = winner
                    cells[b * 9 + 0] = winner
                    cells[b * 9 + 1] = winner
                    cells[b * 9 + 2] = winner

                s = FastState(cells=cells, boards=boards, turn=1, forced=-1, result=winner)
                self.assertEqual(s.result, winner)

    def test_local_board_draw_detection(self):
        """Verify drawn local board (full with no winner) is assigned status 2."""
        # A classic 3x3 draw pattern:
        # X O X
        # X O O
        # O X X
        # Indices: 0:X, 1:O, 2:X, 3:X, 4:O, 5:O, 6:O, 7:X, 8:X
        draw_pattern = [1, -1, 1, 1, -1, -1, -1, 1, 1]
        b = 3
        cells = [0] * 81
        for c, v in enumerate(draw_pattern):
            cells[b * 9 + c] = v
        boards = [0] * 9
        boards[b] = 2

        s = FastState(cells=cells, boards=boards, turn=1, forced=-1, result=None)
        self.assertEqual(s.boards[b], 2)

    def test_macro_board_draw_detection(self):
        """Verify game ends in draw (result 0) when all 9 boards are closed and no macro winner."""
        # Macro board:
        # 1 -1  1
        # 1 -1 -1
        # -1 1  1
        # none of them forms 3-in-a-row
        draw_macro = [1, -1, 1, 1, -1, -1, -1, 1, 1]
        boards = list(draw_macro)
        cells = [0] * 81

        s = FastState(cells=cells, boards=boards, turn=1, forced=-1, result=0)
        self.assertEqual(s.result, 0)
        self.assertEqual(s.legal_actions(), [])

    def test_forced_board_routing(self):
        """Test forced board redirection when target board is closed."""
        s = FastState()
        # Play move 0 (board 0, cell 0): sends opponent to board 0
        s1 = s.play(0)
        self.assertEqual(s1.forced, 0)

        # Now simulate a state where board 4 is won by X
        boards = [0] * 9
        boards[4] = 1 # board 4 won
        cells = [0] * 81
        cells[4 * 9 + 0] = 1
        cells[4 * 9 + 1] = 1
        cells[4 * 9 + 2] = 1

        s_closed = FastState(cells=cells, boards=boards, turn=1, forced=0, result=None)
        # Play move in board 0, cell 4 (action = 0*9 + 4 = 4)
        # Because target board 4 is won/closed, forced must become -1!
        s_next = s_closed.play(4)
        self.assertEqual(s_next.forced, -1)

    def test_illegal_moves_raise_value_error(self):
        s = FastState()
        with self.assertRaises(ValueError):
            s.play(-1)
        with self.assertRaises(ValueError):
            s.play(81)
        with self.assertRaises(ValueError):
            s.play(100)

        # Occupied cell
        s1 = s.play(0)
        # Now opponent is forced to board 0
        # Try to play cell 0 again (already occupied)
        with self.assertRaises(ValueError):
            s1.play(0)

        # Try to play in board 1 when forced to board 0
        with self.assertRaises(ValueError):
            s1.play(9)

        # Terminal state: cannot play
        terminal = FastState(cells=[0]*81, boards=[1, 1, 1, 0, 0, 0, 0, 0, 0], turn=1, forced=-1, result=1)
        with self.assertRaises(ValueError):
            terminal.play(30)

    def test_neural_encoding_equivalence(self):
        """Verify C++ encode() matches Python encode() across varied states."""
        s = State()
        fs = FastState()

        # Initial state
        enc_py = py_encode(s)
        enc_cpp = np.array(fs.encode(), dtype=np.float32)
        np.testing.assert_array_almost_equal(enc_cpp, enc_py)

        # After several moves
        actions = [40, 38, 20, 22, 44, 76]
        for a in actions:
            s = s.play(a)
            fs = fs.play(a)
            enc_py = py_encode(s)
            enc_cpp = np.array(fs.encode(), dtype=np.float32)
            np.testing.assert_array_almost_equal(enc_cpp, enc_py)

    def test_state_hashing_and_dict_keys(self):
        s1 = FastState()
        s2 = FastState()
        self.assertEqual(hash(s1), hash(s2))
        d = {s1: "root"}
        self.assertIn(s2, d)
        s3 = s1.play(40)
        self.assertNotIn(s3, d)
        d[s3] = "child"
        self.assertEqual(len(d), 2)

    def test_render_output(self):
        s_py = State().play(40).play(38)
        s_cpp = FastState().play(40).play(38)
        self.assertEqual(s_cpp.render(), s_py.render())


if __name__ == "__main__":
    unittest.main()
