"""Unit tests for OpenSpiel engine adapter, state/action translations, and bot integration."""
import unittest
import numpy as np
import pyspiel

from sttt.env import State, LINES, winner
from sttt.bots import (
    OpenSpielBot,
    create_bot,
    sttt_to_openspiel_action,
    openspiel_to_sttt_action,
    sttt_to_openspiel_state,
    openspiel_to_sttt_state,
    action_to_coord,
    coord_to_action,
    TacticalBot,
    _find_action_sequence,
    _find_action_sequence_to_state,
    _verify_action_sequence,
)
from sttt.tournament import _play_single_game, run_matchup


class TestOpenSpielCore(unittest.TestCase):
    """Verify pyspiel loads and complies with OpenSpiel specifications."""

    def test_pyspiel_load_game(self):
        game = pyspiel.load_game("ultimate_tic_tac_toe")
        self.assertIsNotNone(game)
        self.assertEqual(game.num_distinct_actions(), 81)
        self.assertEqual(game.num_players(), 2)
        self.assertEqual(game.min_utility(), -1.0)
        self.assertEqual(game.max_utility(), 1.0)

        state = game.new_initial_state()
        self.assertEqual(state.current_player(), 0)
        self.assertFalse(state.is_terminal())
        self.assertEqual(state.returns(), [0.0, 0.0])
        self.assertEqual(len(state.legal_actions()), 81)

    def test_registered_games_and_player_ids(self):
        self.assertIn("ultimate_tic_tac_toe", pyspiel.registered_names())
        self.assertEqual(pyspiel.PlayerId.TERMINAL, -4)
        self.assertEqual(pyspiel.PlayerId.INVALID, -1)


class TestActionTranslation(unittest.TestCase):
    """Verify bidirectional 1-to-1 bijection between sttt (0..80) and OpenSpiel (0..80)."""

    def test_bijection_all_actions(self):
        for a in range(81):
            os_a = sttt_to_openspiel_action(a)
            sttt_a = openspiel_to_sttt_action(os_a)
            self.assertEqual(sttt_a, a, f"Failed round-trip for sttt action {a}")

            # Reverse round-trip
            self.assertEqual(sttt_to_openspiel_action(openspiel_to_sttt_action(a)), a)

    def test_specific_coordinates(self):
        # Action 0: Board 0, cell 0 -> Row 0, Col 0 -> OpenSpiel 0
        self.assertEqual(sttt_to_openspiel_action(0), 0)
        self.assertEqual(openspiel_to_sttt_action(0), 0)

        # Action 3: Board 0, cell 3 -> Row 1, Col 0 -> OpenSpiel 9
        self.assertEqual(sttt_to_openspiel_action(3), 9)
        self.assertEqual(openspiel_to_sttt_action(9), 3)

        # Action 9: Board 1, cell 0 -> Row 0, Col 3 -> OpenSpiel 3
        self.assertEqual(sttt_to_openspiel_action(9), 3)
        self.assertEqual(openspiel_to_sttt_action(3), 9)

        # Action 80: Board 8, cell 8 -> Row 8, Col 8 -> OpenSpiel 80
        self.assertEqual(sttt_to_openspiel_action(80), 80)
        self.assertEqual(openspiel_to_sttt_action(80), 80)


class TestStateMapping(unittest.TestCase):
    """Verify bidirectional state mapping and legal actions consistency."""

    def setUp(self):
        self.game = pyspiel.load_game("ultimate_tic_tac_toe")

    def test_initial_state_mapping(self):
        sttt_state = State()
        os_state = sttt_to_openspiel_state(sttt_state, self.game)

        self.assertFalse(os_state.is_terminal())
        self.assertEqual(os_state.current_player(), 0)
        self.assertEqual(len(os_state.legal_actions()), 81)

        mapped_sttt = openspiel_to_sttt_state(os_state)
        self.assertEqual(mapped_sttt, sttt_state)
        self.assertEqual(
            sorted(map(sttt_to_openspiel_action, sttt_state.legal_actions())),
            sorted(os_state.legal_actions()),
        )

    def test_midgame_legal_actions_consistency(self):
        # Play a sequence of legal moves
        moves = [0, 1, 13, 44, 72, 5]
        sttt_state = State()
        os_state = self.game.new_initial_state()

        for m in moves:
            sttt_state = sttt_state.play(m)
            os_state.apply_action(sttt_to_openspiel_action(m))

            # Check legal actions match
            sttt_legal_in_os = sorted(map(sttt_to_openspiel_action, sttt_state.legal_actions()))
            self.assertEqual(sttt_legal_in_os, sorted(os_state.legal_actions()))

            # Check round-trip conversion
            self.assertEqual(openspiel_to_sttt_state(os_state), sttt_state)

    def test_terminal_win_state_mapping(self):
        # Construct a synthetic winning state for X
        cells = [0] * 81
        boards = [0] * 9

        # Boards 0, 1, 2 won by X (1)
        for b in (0, 1, 2):
            boards[b] = 1
            for c in (0, 1, 2):
                cells[b * 9 + c] = 1

        sttt_state = State(cells=tuple(cells), boards=tuple(boards), turn=-1, forced=-1, result=1)
        os_state = sttt_to_openspiel_state(sttt_state, self.game)

        self.assertTrue(os_state.is_terminal())
        self.assertEqual(os_state.returns(), [1.0, -1.0])
        self.assertEqual(os_state.legal_actions(), [])

        mapped = openspiel_to_sttt_state(os_state)
        self.assertEqual(mapped.result, 1)

    def test_full_game_parallel_rollouts(self):
        rng = np.random.default_rng(12345)
        for _ in range(25):
            sttt_state = State()
            os_state = self.game.new_initial_state()

            while sttt_state.result is None:
                # Legal actions must match at every step
                sttt_legal = set(map(sttt_to_openspiel_action, sttt_state.legal_actions()))
                os_legal = set(os_state.legal_actions())
                self.assertEqual(sttt_legal, os_legal)

                # State translation round-trip must match at every step
                self.assertEqual(openspiel_to_sttt_state(os_state), sttt_state)
                converted_os = sttt_to_openspiel_state(sttt_state, self.game)
                self.assertEqual(set(converted_os.legal_actions()), os_legal)

                # Pick random legal action
                chosen_sttt = int(rng.choice(sttt_state.legal_actions()))
                chosen_os = sttt_to_openspiel_action(chosen_sttt)

                sttt_state = sttt_state.play(chosen_sttt)
                os_state.apply_action(chosen_os)

            # Terminal verification
            self.assertTrue(os_state.is_terminal())
            self.assertEqual(openspiel_to_sttt_state(os_state), sttt_state)
            if sttt_state.result == 1:
                self.assertEqual(os_state.returns(), [1.0, -1.0])
            elif sttt_state.result == -1:
                self.assertEqual(os_state.returns(), [-1.0, 1.0])
            else:
                self.assertEqual(os_state.returns(), [0.0, 0.0])


class TestOpenSpielBot(unittest.TestCase):
    """Verify OpenSpielBot implementation conforming to the Bot interface."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_choose_legal_moves_mcts(self):
        bot = OpenSpielBot(algorithm="mcts", simulations=20, seed=42)
        state = State()
        action = bot.choose(state, self.rng)
        self.assertIn(action, state.legal_actions())

    def test_choose_legal_moves_random(self):
        bot = OpenSpielBot(algorithm="random", seed=42)
        state = State()
        for _ in range(10):
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
            state = state.play(action)
            if state.result is not None:
                break

    def test_choose_terminal_state_raises(self):
        bot = OpenSpielBot(algorithm="random")
        terminal_state = State(result=1)
        with self.assertRaises(ValueError):
            bot.choose(terminal_state, self.rng)

    def test_advance_and_reset_semantics(self):
        bot = OpenSpielBot(algorithm="mcts", simulations=10, seed=42)
        state = State()

        # Play move 0
        bot.advance(0)
        state = state.play(0)
        self.assertIn(sttt_to_openspiel_action(0), bot.os_state.history())

        # Reset
        bot.reset()
        self.assertEqual(bot.os_state.history(), [])
        self.assertFalse(bot.os_state.is_terminal())

    def test_sync_on_arbitrary_state(self):
        bot = OpenSpielBot(algorithm="random")
        # Generate arbitrary state by playing 5 legal moves without informing bot
        state = State()
        for m in (0, 1, 13, 44, 72):
            state = state.play(m)

        # choose() should detect mismatch, resync, and return legal move
        action = bot.choose(state, self.rng)
        self.assertIn(action, state.legal_actions())

    def test_factory_registration(self):
        # openspiel-mcts
        bot_mcts = create_bot("openspiel-mcts")
        self.assertIsInstance(bot_mcts, OpenSpielBot)
        self.assertEqual(bot_mcts.name, "openspiel-mcts")
        self.assertEqual(bot_mcts.algorithm, "mcts")

        # openspiel-random
        bot_rnd = create_bot("openspiel-random")
        self.assertIsInstance(bot_rnd, OpenSpielBot)
        self.assertEqual(bot_rnd.name, "openspiel-random")
        self.assertEqual(bot_rnd.algorithm, "random")

        # openspiel-mcts with simulation spec
        bot_sims = create_bot("openspiel-mcts:64")
        self.assertEqual(bot_sims.simulations, 64)
        self.assertEqual(bot_sims.name, "openspiel-mcts-64")

        # custom name override
        bot_named = create_bot("openspiel-mcts", name="deepmind_bot")
        self.assertEqual(bot_named.name, "deepmind_bot")

    def test_single_game_play_against_tactical(self):
        bot_os = OpenSpielBot(algorithm="mcts", simulations=25, seed=123)
        bot_tac = TacticalBot()

        result, moves = _play_single_game(
            bot_x=bot_os,
            bot_o=bot_tac,
            initial_state=State(),
            opening_moves=[],
            game_rng=self.rng,
        )
        self.assertIn(result, (-1, 0, 1))
        self.assertGreater(moves, 5)

    def test_paired_matchup_execution(self):
        res = run_matchup(
            bot_a="openspiel-random",
            bot_b="tactical",
            games=2,
            opening_plies=2,
            seed=42,
        )
        self.assertEqual(len(res), 2)
        for r in res:
            self.assertIn(r.winner, (-1, 0, 1))
            self.assertGreater(r.moves, 0)

    def test_unsupported_algorithm_raises(self):
        with self.assertRaises(ValueError):
            OpenSpielBot(algorithm="unknown_algo")

    def test_advance_illegal_action_does_not_crash(self):
        bot = OpenSpielBot(algorithm="random")
        # Action 50 is not legal on initial state if 0..80 was restricted,
        # but even if an invalid move is passed to advance, bot handles gracefully
        bot.advance(999)  # completely out of bounds
        # choose() should still succeed cleanly
        s = State()
        action = bot.choose(s, self.rng)
        self.assertIn(action, s.legal_actions())

    def test_mcts_budget_scaling(self):
        s = State()
        for budget in (5, 20, 50):
            bot = OpenSpielBot(algorithm="mcts", simulations=budget, seed=42)
            act = bot.choose(s, self.rng)
            self.assertIn(act, s.legal_actions())

    def test_observation_string_and_tensor(self):
        game = pyspiel.load_game("ultimate_tic_tac_toe")
        state = game.new_initial_state()
        obs_str = state.observation_string()
        self.assertIn("|", obs_str)
        self.assertIn("+", obs_str)
        tensor = state.observation_tensor()
        self.assertEqual(len(tensor), 3 * 81)
        # Initially, channel 2 (empty) should have 1s
        self.assertEqual(sum(tensor[:81]), 0.0)
        self.assertEqual(sum(tensor[81:162]), 0.0)
        self.assertEqual(sum(tensor[162:]), 81.0)

    def test_legal_actions_mask(self):
        game = pyspiel.load_game("ultimate_tic_tac_toe")
        state = game.new_initial_state()
        mask = state.legal_actions_mask()
        self.assertEqual(len(mask), 81)
        self.assertEqual(sum(mask), 81)

        state.apply_action(0)
        mask2 = state.legal_actions_mask()
        self.assertEqual(sum(mask2), len(state.legal_actions()))

    def test_paired_mcts_matchup(self):
        res = run_matchup(
            bot_a="openspiel-mcts:15",
            bot_b="tactical",
            games=2,
            opening_plies=2,
            seed=100,
        )
        self.assertEqual(len(res), 2)
        for r in res:
            self.assertIn(r.winner, (-1, 0, 1))


class TestEdgeCasesAndFallbacks(unittest.TestCase):
    """Deep verification of edge cases, fallbacks, and robustness."""

    def test_action_out_of_bounds_raises(self):
        for bad_act in (-10, -1, 81, 99, 1000):
            with self.assertRaises(ValueError):
                sttt_to_openspiel_action(bad_act)
            with self.assertRaises(ValueError):
                openspiel_to_sttt_action(bad_act)

    def test_forced_subgame_on_completed_board_normalized(self):
        game = pyspiel.load_game("ultimate_tic_tac_toe")
        os_state = game.new_initial_state()
        os_state._subgames[0] = "x"
        os_state._forced_subgame = 0

        sttt_state = openspiel_to_sttt_state(os_state)
        # Completed subgame must be normalized to -1 to allow play on all open subgames
        self.assertEqual(sttt_state.forced, -1)
        self.assertEqual(sttt_state.boards[0], 1)
        self.assertEqual(len(sttt_state.legal_actions()), len(os_state.legal_actions()))
        self.assertEqual(len(sttt_state.legal_actions()), 72)

    def test_subgame_draw_symbol_mapping(self):
        game = pyspiel.load_game("ultimate_tic_tac_toe")
        os_state = game.new_initial_state()
        os_state._subgames[4] = "="
        os_state._forced_subgame = 4

        sttt_state = openspiel_to_sttt_state(os_state)
        self.assertEqual(sttt_state.boards[4], 2)
        self.assertEqual(sttt_state.forced, -1)

        # Convert back
        os_converted = sttt_to_openspiel_state(sttt_state, game)
        self.assertEqual(os_converted._subgames[4], "=")
        self.assertEqual(os_converted._forced_subgame, -1)

    def test_fallback_without_board_attribute(self):
        class MockCppState:
            def __init__(self):
                self._history = []
                self._is_term = False
            def is_terminal(self):
                return self._is_term
            def apply_action(self, a):
                self._history.append(a)
            def history(self):
                return list(self._history)
            def legal_actions(self):
                return list(range(81))

        class MockGame:
            def new_initial_state(self):
                return MockCppState()

        mock_game = MockGame()

        # 1. Initial state returns clean state
        os_init = sttt_to_openspiel_state(State(), game=mock_game)
        self.assertEqual(os_init.history(), [])

        # 2. Replay with explicit action history
        s = State().play(0).play(1).play(13)
        os_with_acts = sttt_to_openspiel_state(s, game=mock_game, actions=[0, 1, 13])
        self.assertEqual(os_with_acts.history(), [sttt_to_openspiel_action(a) for a in [0, 1, 13]])

        # 3. Fallback search without explicit actions
        os_searched = sttt_to_openspiel_state(s, game=mock_game)
        self.assertEqual(len(os_searched.history()), 3)

        # 4. Irreconstructible state raises ValueError
        invalid_state = State(cells=(1,) * 81)  # 81 X pieces, 0 O pieces: impossible
        with self.assertRaises(ValueError):
            sttt_to_openspiel_state(invalid_state, game=mock_game)

    def test_openspiel_bot_rng_determinism(self):
        bot1 = OpenSpielBot(algorithm="mcts", simulations=20, seed=None)
        bot2 = OpenSpielBot(algorithm="mcts", simulations=20, seed=None)
        state = State()

        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)

        act1 = bot1.choose(state, rng1)
        act2 = bot2.choose(state, rng2)
        self.assertEqual(act1, act2)

    def test_bounded_action_sequence_search_bounds(self):
        # 1. State with more moves than max_depth terminates immediately without hanging
        s = State()
        rng = np.random.default_rng(999)
        for _ in range(15):
            acts = s.legal_actions()
            if not acts or s.result is not None:
                break
            s = s.play(rng.choice(acts))

        # Default max_depth=10; 15 moves should immediately return None
        res = _find_action_sequence_to_state(s, max_depth=10)
        self.assertIsNone(res)

        # 2. Node budget bound limits exploration on wide/unreachable states
        unreachable = State(cells=(1,) * 5 + (0,) * 76, turn=-1)
        res_unreach = _find_action_sequence_to_state(unreachable, max_depth=5, node_budget=50)
        self.assertIsNone(res_unreach)

    def test_verify_action_sequence(self):
        s = State().play(0).play(1).play(13)
        self.assertTrue(_verify_action_sequence(s, [0, 1, 13]))
        self.assertFalse(_verify_action_sequence(s, [0, 1]))  # incomplete / stale prefix
        self.assertFalse(_verify_action_sequence(s, [0, 1, 14]))  # wrong move
        self.assertFalse(_verify_action_sequence(s, [999]))  # out of bounds
        self.assertTrue(_verify_action_sequence(State(), []))

    def test_sync_state_multi_ply_jump_maintains_history(self):
        bot = OpenSpielBot(algorithm="random")
        s0 = State()
        s1 = s0.play(0)
        bot.advance(0)
        self.assertEqual(bot.history, [0])

        # State jumps 2 plies forward without advance() being called
        s2 = s1.play(1)
        s3 = s2.play(13)

        # choose() must recognize the 2-ply continuation and sync both history and engine
        rng = np.random.default_rng(42)
        chosen = bot.choose(s3, rng)
        self.assertEqual(bot.history[:3], [0, 1, 13])
        self.assertEqual(bot.history[3], chosen)
        self.assertEqual(len(bot.history), 4)
        self.assertEqual(
            openspiel_to_sttt_state(bot.os_state),
            s3.play(chosen),
        )

    def test_choose_with_none_rng_succeeds(self):
        bot_rnd = OpenSpielBot(algorithm="random")
        s = State()
        # rng=None defaults to np.random.default_rng()
        act_rnd = bot_rnd.choose(s, None)
        self.assertIn(act_rnd, s.legal_actions())

        bot_mcts = OpenSpielBot(algorithm="mcts", simulations=10)
        act_mcts = bot_mcts.choose(s, None)
        self.assertIn(act_mcts, s.legal_actions())

    def test_invalid_forced_subgame_index_normalized(self):
        game = pyspiel.load_game("ultimate_tic_tac_toe")
        os_state = game.new_initial_state()
        os_state._forced_subgame = 15  # Out of range forced subgame
        sttt_state = openspiel_to_sttt_state(os_state)
        self.assertEqual(sttt_state.forced, -1)

    def test_find_action_sequence_with_completed_subgame(self):
        s0 = State()
        # Sequence of 5 moves where board 0 is completed/won by X
        moves = [1, 9, 4, 36, 7]
        target = s0
        for m in moves:
            target = target.play(m)
        self.assertEqual(target.boards[0], 1)

        # _find_action_sequence must successfully reconstruct the 5-move path
        seq = _find_action_sequence(s0, target)
        self.assertEqual(seq, moves)

        # _find_action_sequence_to_state must also reconstruct it
        seq_to = _find_action_sequence_to_state(target)
        self.assertEqual(seq_to, moves)

    def test_mock_cpp_state_reconstruction_with_won_subgame(self):
        class MockCppState:
            def __init__(self):
                self._history = []
                self._is_term = False
            def is_terminal(self):
                return self._is_term
            def apply_action(self, a):
                self._history.append(a)
            def history(self):
                return list(self._history)
            def legal_actions(self):
                return list(range(81))

        class MockGame:
            def new_initial_state(self):
                return MockCppState()

        mock_game = MockGame()
        moves = [1, 9, 4, 36, 7]
        target = State()
        for m in moves:
            target = target.play(m)

        # Fallback search without explicit actions must succeed for states with won subgames
        os_reconstructed = sttt_to_openspiel_state(target, game=mock_game)
        expected_os_moves = [sttt_to_openspiel_action(m) for m in moves]
        self.assertEqual(os_reconstructed.history(), expected_os_moves)

    def test_turn_parity_mismatch_rejected(self):
        s0 = State()
        s1 = s0.play(0)
        # s1 has 1 move played, but we artificially tamper turn to 1
        tampered = State(cells=s1.cells, boards=s1.boards, turn=1, forced=s1.forced)
        self.assertIsNone(_find_action_sequence(s0, tampered))

    def test_history_populated_when_actions_omitted(self):
        s = State().play(0).play(1).play(13)
        game = pyspiel.load_game("ultimate_tic_tac_toe")
        os_state = sttt_to_openspiel_state(s, game=game)
        self.assertEqual(os_state.history(), [sttt_to_openspiel_action(m) for m in [0, 1, 13]])

    def test_close_releases_resources(self):
        bot = OpenSpielBot(algorithm="mcts", simulations=10)
        self.assertIsNotNone(bot._bot)
        self.assertIsNotNone(bot.os_state)
        bot.close()
        self.assertIsNone(bot._bot)
        self.assertIsNone(bot.os_state)

    def test_engine_without_board_and_without_history_raises(self):
        class BrokenState:
            pass

        with self.assertRaises(ValueError):
            openspiel_to_sttt_state(BrokenState())


if __name__ == "__main__":
    unittest.main()
