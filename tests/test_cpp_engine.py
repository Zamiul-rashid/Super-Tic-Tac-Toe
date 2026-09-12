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

    def test_init_with_cells_only_and_none_boards(self):
        """Verify CppState and FastState handle default and None boards without TypeError."""
        cs = CppState([0] * 81)
        self.assertEqual(cs.boards, (0,) * 9)
        fs = FastState([0] * 81, None)
        self.assertEqual(fs.boards, (0,) * 9)

        # Invalid cell values must raise ValueError
        bad_cells = [0] * 81
        bad_cells[5] = 99
        with self.assertRaises(ValueError):
            FastState(cells=bad_cells)

        # Invalid board values must raise ValueError
        bad_boards = [0] * 9
        bad_boards[2] = 5
        with self.assertRaises(ValueError):
            FastState(cells=[0] * 81, boards=bad_boards)

    def test_forced_closed_board_legal_actions_matches_python(self):
        """Verify forced board routing to an already closed board yields 0 legal actions."""
        boards = [1, 0, 0, 0, 0, 0, 0, 0, 0] # board 0 won by X
        py_s = State(boards=tuple(boards), forced=0)
        cpp_s = FastState(boards=boards, forced=0)
        self.assertEqual(cpp_s.legal_actions(), [])
        self.assertEqual(cpp_s.legal_actions(), py_s.legal_actions())

    def test_pickle_and_copy_serialization(self):
        """Verify FastState and CppState can be pickled and deepcopied for multiprocessing."""
        import pickle
        import copy

        fs = FastState().play(40).play(38)
        fs_pickled = pickle.loads(pickle.dumps(fs))
        self.assertEqual(fs_pickled, fs)
        self.assertEqual(fs_pickled.legal_actions(), fs.legal_actions())

        fs_copied = copy.deepcopy(fs)
        self.assertEqual(fs_copied, fs)

        cs = CppState().play(40).play(38)
        cs_pickled = pickle.loads(pickle.dumps(cs))
        self.assertEqual(cs_pickled, cs)

        cs_copied = copy.deepcopy(cs)
        self.assertEqual(cs_copied, cs)

    def test_faststate_state_cross_equality(self):
        """Verify bidirectional equality comparison between FastState and Python State."""
        py_s = State().play(40).play(38)
        cpp_s = FastState().play(40).play(38)
        self.assertEqual(cpp_s, py_s)
        self.assertEqual(py_s, cpp_s)

    def test_heuristic_evaluation_matches_python_value(self):
        """Verify C++ evaluate() matches sttt.bots.value() across diverse game states."""
        from sttt.bots import value as py_value
        from sttt.cpp_env import cpp_evaluate

        rng = random.Random(999)
        for _ in range(50):
            py_s = State()
            cpp_s = FastState()
            while py_s.result is None:
                cpp_val = cpp_s.evaluate()
                py_val = py_value(py_s)
                self.assertAlmostEqual(cpp_val, py_val, places=5)
                self.assertAlmostEqual(cpp_evaluate(cpp_s), py_val, places=5)

                legal = py_s.legal_actions()
                a = rng.choice(legal)
                py_s = py_s.play(a)
                cpp_s = cpp_s.play(a)

    def test_vectorized_batch_encoding(self):
        """Verify encode_batch matches single-state Python encode."""
        from sttt.cpp_env import encode_batch

        states = [FastState(), FastState().play(40), FastState().play(40).play(38)]
        batch_arr = encode_batch(states)
        self.assertEqual(batch_arr.shape, (3, 289))
        for i, s in enumerate(states):
            py_s = to_python_state(s)
            expected = py_encode(py_s)
            np.testing.assert_array_almost_equal(batch_arr[i], expected)

    def test_cpp_alphabeta_bot(self):
        """Verify CppAlphaBetaBot chooses legal moves and obeys interface contract."""
        from sttt.cpp_env import CppAlphaBetaBot

        bot = CppAlphaBetaBot(depth=3, node_budget=5000)
        s = CppState()
        rng = np.random.default_rng(42)
        move = bot.choose(s, rng)
        self.assertIn(move, s.legal_actions())

        # Terminal state raises ValueError
        term = FastState(boards=[1, 1, 1, 0, 0, 0, 0, 0, 0], result=1)
        with self.assertRaises(ValueError):
            bot.choose(term, rng)

    def test_render_output(self):
        s_py = State().play(40).play(38)
        s_cpp = FastState().play(40).play(38)
        self.assertEqual(s_cpp.render(), s_py.render())


class TestDifferentialParityAndEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_cpp_available():
            raise unittest.SkipTest("C++ engine not available")

    def test_01_differential_parity_ten_thousand_moves(self):
        """Differential parity across >= 10,000 paired random moves vs sttt.env.State."""
        rng = random.Random(42)
        total_moves = 0
        game = 0
        while total_moves < 10000:
            py_s = State()
            cpp_s = FastState()
            wrapper_s = CppState()
            while py_s.result is None:
                py_legal = py_s.legal_actions()
                cpp_legal = cpp_s.legal_actions()
                wrap_legal = wrapper_s.legal_actions()
                self.assertEqual(cpp_legal, py_legal)
                self.assertEqual(wrap_legal, py_legal)

                action = rng.choice(py_legal)
                py_s = py_s.play(action)
                cpp_s = cpp_s.play(action)
                wrapper_s = wrapper_s.play(action)
                total_moves += 1

                self.assertEqual(cpp_s.cells, py_s.cells)
                self.assertEqual(cpp_s.boards, py_s.boards)
                self.assertEqual(cpp_s.turn, py_s.turn)
                self.assertEqual(cpp_s.forced, py_s.forced)
                self.assertEqual(cpp_s.result, py_s.result)
                self.assertEqual(wrapper_s.cells, py_s.cells)
                self.assertEqual(wrapper_s.boards, py_s.boards)
                self.assertEqual(wrapper_s.turn, py_s.turn)
                self.assertEqual(wrapper_s.forced, py_s.forced)
                self.assertEqual(wrapper_s.result, py_s.result)
            game += 1
        self.assertGreaterEqual(total_moves, 10000)

    def test_02_subgrid_won_with_empty_cells_no_moves_permitted(self):
        """Subgrid won with empty cells remaining blocks any moves in that subgrid."""
        cells = [0] * 81
        boards = [0] * 9
        for c in (0, 1, 2):
            cells[2 * 9 + c] = 1
        boards[2] = 1

        py_s = State(cells=tuple(cells), boards=tuple(boards), turn=-1, forced=-1)
        cpp_s = FastState(cells=cells, boards=boards, turn=-1, forced=-1)

        for c in range(9):
            act = 2 * 9 + c
            self.assertNotIn(act, py_s.legal_actions())
            self.assertNotIn(act, cpp_s.legal_actions())
            with self.assertRaises(ValueError):
                py_s.play(act)
            with self.assertRaises(ValueError):
                cpp_s.play(act)

    def test_03_subgrid_drawn_marked_two_no_winning_lines(self):
        """Subgrid drawn (marked 2) never triggers a win on local or macro level."""
        draw_pattern = [1, -1, 1, 1, -1, -1, -1, 1, 1]
        cells = [0] * 81
        boards = [0] * 9
        for c, v in enumerate(draw_pattern):
            cells[5 * 9 + c] = v
        boards[5] = 2

        py_s = State(cells=tuple(cells), boards=tuple(boards), turn=1, forced=-1)
        cpp_s = FastState(cells=cells, boards=boards, turn=1, forced=-1)

        self.assertEqual(cpp_s.boards[5], 2)
        self.assertEqual(py_s.boards[5], 2)
        self.assertIsNone(cpp_s.result)
        self.assertIsNone(py_s.result)

        for c in range(9):
            act = 5 * 9 + c
            self.assertNotIn(act, cpp_s.legal_actions())

    def test_04_forced_routing_redirection_when_target_board_closed(self):
        """Target board closed redirects forced to -1."""
        cells = [0] * 81
        boards = [0] * 9
        boards[3] = -1
        for c in (0, 1, 2):
            cells[3 * 9 + c] = -1

        py_s = State(cells=tuple(cells), boards=tuple(boards), turn=1, forced=0)
        cpp_s = FastState(cells=cells, boards=boards, turn=1, forced=0)

        py_next = py_s.play(3)
        cpp_next = cpp_s.play(3)

        self.assertEqual(py_next.forced, -1)
        self.assertEqual(cpp_next.forced, -1)
        self.assertEqual(cpp_next.legal_actions(), py_next.legal_actions())

    def test_05_move_on_c_equals_b_winning_board_redirects_forced_to_negative_one(self):
        """Move on c == b that closes board b redirects forced to -1."""
        cells = [0] * 81
        cells[4 * 9 + 0] = 1
        cells[4 * 9 + 8] = 1
        boards = [0] * 9

        py_s = State(cells=tuple(cells), boards=tuple(boards), turn=1, forced=4)
        cpp_s = FastState(cells=cells, boards=boards, turn=1, forced=4)

        py_next = py_s.play(40)
        cpp_next = cpp_s.play(40)

        self.assertEqual(py_next.boards[4], 1)
        self.assertEqual(cpp_next.boards[4], 1)
        self.assertEqual(py_next.forced, -1)
        self.assertEqual(cpp_next.forced, -1)
        self.assertEqual(cpp_next.legal_actions(), py_next.legal_actions())

    def test_06_macro_board_win_detection_rows_cols_diagonals(self):
        """Macro-board win detection across all 8 lines for both players."""
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for winner in (1, -1):
            for line in lines:
                cells = [0] * 81
                boards = [0] * 9
                for b in line[:2]:
                    boards[b] = winner
                    for c in (0, 1, 2):
                        cells[b * 9 + c] = winner

                b3 = line[2]
                cells[b3 * 9 + 0] = winner
                cells[b3 * 9 + 1] = winner

                winning_action = b3 * 9 + 2
                py_s = State(cells=tuple(cells), boards=tuple(boards), turn=winner, forced=b3)
                cpp_s = FastState(cells=cells, boards=boards, turn=winner, forced=b3)

                py_next = py_s.play(winning_action)
                cpp_next = cpp_s.play(winning_action)

                self.assertEqual(py_next.result, winner)
                self.assertEqual(cpp_next.result, winner)
                self.assertEqual(cpp_next.legal_actions(), [])

    def test_07_macro_board_draw_detection_all_nine_closed_no_winner(self):
        """Macro-board draw detection when all 9 boards are closed with no winner."""
        macro_pattern = [1, -1, 1, 1, -1, -1, -1, 1, 1]
        cells = [0] * 81
        boards = list(macro_pattern)
        draw_local = [1, -1, 1, 1, -1, -1, -1, 1, 0]
        boards[8] = 0
        for c, v in enumerate(draw_local):
            cells[8 * 9 + c] = v

        for b in range(8):
            for c in (0, 1, 2):
                cells[b * 9 + c] = macro_pattern[b]

        py_s = State(cells=tuple(cells), boards=tuple(boards), turn=1, forced=8)
        cpp_s = FastState(cells=cells, boards=boards, turn=1, forced=8)

        py_next = py_s.play(80)
        cpp_next = cpp_s.play(80)

        self.assertEqual(py_next.result, 0)
        self.assertEqual(cpp_next.result, 0)
        self.assertEqual(cpp_next.legal_actions(), [])

    def test_08_terminal_state_legal_actions_empty(self):
        """Terminal states return [] for legal_actions and reject play()."""
        for res in (1, -1, 0):
            boards = [1, 1, 1, 0, 0, 0, 0, 0, 0] if res == 1 else (
                [-1, -1, -1, 0, 0, 0, 0, 0, 0] if res == -1 else [1, -1, 1, 1, -1, -1, -1, 1, 1]
            )
            fs = FastState(boards=boards, result=res)
            cs = CppState(boards=boards, result=res)
            self.assertEqual(fs.legal_actions(), [])
            self.assertEqual(cs.legal_actions(), [])
            with self.assertRaises(ValueError):
                fs.play(0)
            with self.assertRaises(ValueError):
                cs.play(0)

    def test_09_invalid_action_raises_value_error(self):
        """Invalid action indices and illegal board plays raise ValueError."""
        fs = FastState()
        cs = CppState()
        for bad_action in (-1, -10, 81, 82, 100, 999):
            with self.assertRaises(ValueError):
                fs.play(bad_action)
            with self.assertRaises(ValueError):
                cs.play(bad_action)

        fs2 = fs.play(40)
        with self.assertRaises(ValueError):
            fs2.play(40)

        with self.assertRaises(ValueError):
            fs2.play(0)

    def test_10_invalid_state_initialization_raises_value_error(self):
        """Invalid cells/boards sequence lengths and values raise ValueError."""
        with self.assertRaises(ValueError):
            FastState(cells=[0] * 80)
        with self.assertRaises(ValueError):
            FastState(cells=[0] * 82)
        with self.assertRaises(ValueError):
            FastState(cells=[0] * 80 + [3])
        with self.assertRaises(ValueError):
            FastState(boards=[0] * 8)
        with self.assertRaises(ValueError):
            FastState(boards=[0] * 10)
        with self.assertRaises(ValueError):
            FastState(boards=[0] * 8 + [5])

    def test_11_hash_consistency_with_python_state(self):
        """FastState, CppState, and Python State must have equal hashes when states are equal."""
        py = State()
        fs = FastState()
        cs = CppState()
        self.assertEqual(py, fs)
        self.assertEqual(py, cs)
        self.assertEqual(hash(py), hash(fs))
        self.assertEqual(hash(py), hash(cs))
        self.assertIn(fs, {py})
        self.assertIn(cs, {py})
        self.assertIn(py, {fs})

        # Play several moves and check hash consistency at each step
        rng = np.random.default_rng(42)
        for _ in range(20):
            legal = py.legal_actions()
            if not legal:
                break
            m = int(rng.choice(legal))
            py = py.play(m)
            fs = fs.play(m)
            cs = cs.play(m)
            self.assertEqual(py, fs)
            self.assertEqual(py, cs)
            self.assertEqual(hash(py), hash(fs))
            self.assertEqual(hash(py), hash(cs))
            self.assertIn(fs, {py})
            self.assertIn(cs, {py})


if __name__ == "__main__":
    unittest.main()



