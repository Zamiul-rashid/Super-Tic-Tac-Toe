"""Tests against official OpenSpiel's two-stage Ultimate Tic-Tac-Toe."""
import unittest
import numpy as np
from sttt.engine_runtime import activate_runtime
activate_runtime()
import pyspiel
from sttt.env import State
from sttt.bots import (OpenSpielBot, create_bot, sttt_to_openspiel_state,
                       openspiel_to_sttt_state, action_to_coord)


class OpenSpielTests(unittest.TestCase):
    def test_official_extension_and_initial_move(self):
        self.assertFalse(pyspiel.__file__.endswith('.py'))
        s = pyspiel.load_game('ultimate_tic_tac_toe').new_initial_state()
        self.assertEqual(s.legal_actions(), list(range(9)))
        s.apply_action(4)
        self.assertEqual(s.current_player(), 0)
        with self.assertRaises(ValueError):
            openspiel_to_sttt_state(s)
        s.apply_action(2)
        self.assertEqual(openspiel_to_sttt_state(s), State().play(38))

    def test_full_games_match_board_legality_and_outcomes(self):
        rng = np.random.default_rng(91)
        for _ in range(20):
            local, actions = State(), []
            official = pyspiel.load_game('ultimate_tic_tac_toe').new_initial_state()
            while local.result is None:
                if local.forced < 0:
                    legal = []
                    for board in official.legal_actions():
                        branch = official.clone()
                        branch.apply_action(board)
                        legal.extend(9 * board + cell for cell in branch.legal_actions())
                else:
                    legal = [9 * local.forced + cell for cell in official.legal_actions()]
                self.assertEqual(set(legal), set(local.legal_actions()))
                action = int(rng.choice(legal))
                if local.forced < 0:
                    official.apply_action(action // 9)
                official.apply_action(action % 9)
                local = local.play(action)
                actions.append(action)
                self.assertEqual(openspiel_to_sttt_state(official), local)
                self.assertEqual(official.is_terminal(), local.result is not None)
                board_text = ''.join(line.replace(' ', '') for line in str(official).splitlines()[:11])
                for i, cell in enumerate(local.cells):
                    row, col = action_to_coord(i)
                    self.assertEqual(board_text[row * 9 + col], {0: '.', 1: 'x', -1: 'o'}[cell])
            self.assertEqual(official.returns()[0], local.result)
            self.assertEqual(str(sttt_to_openspiel_state(local, actions=actions)), str(official))

    def test_mcts_and_random_complete_games_both_seats(self):
        for algorithm in ('mcts', 'random'):
            for side in (1, -1):
                bot = OpenSpielBot(algorithm=algorithm, simulations=4, seed=7)
                rng = np.random.default_rng(7)
                state = State().play(0).play(1).play(9)
                while state.result is None:
                    if state.turn == side:
                        action = bot.choose(state, rng)
                    else:
                        bot._sync_state(state)
                        action = int(rng.choice(state.legal_actions()))
                        bot.advance(action)
                    self.assertIn(action, state.legal_actions())
                    state = state.play(action)
                bot.reset()
                self.assertEqual(bot.history, [])
                bot.close()

    def test_factory_seed_and_invalid_states(self):
        a = create_bot('openspiel-mcts:8', seed=12)
        b = create_bot('openspiel-mcts:8', seed=12)
        self.assertEqual(a.choose(State()), b.choose(State()))
        with self.assertRaises(ValueError):
            a.choose(State(result=1))
        with self.assertRaises(ValueError):
            OpenSpielBot(algorithm='bogus')
        a.close()
        b.close()
