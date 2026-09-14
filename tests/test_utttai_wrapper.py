"""Real model/protocol integration; no tactical fallback or mock network."""
from dataclasses import replace
import unittest
import numpy as np
import torch
from sttt.engine_runtime import ROOT
from sttt.engine_registry import load_engine_registry
from sttt.population import sample_match, make_opponent, MatchSpec
from sttt.selfplay import play_game, SelfPlayPool
from sttt.learning import Network
from sttt.search import SearchConfig
from sttt.env import State
from sttt.utttai_wrapper import to_official


class WrapperTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.registry = load_engine_registry(ROOT / 'engines/registry.json')

    def test_official_state_mapping_all_game_phases(self):
        from sttt.engine_runtime import activate_utttai
        activate_utttai()
        from utttpy.game.action import Action
        rng = np.random.default_rng(71)
        for _ in range(20):
            state = State()
            official = to_official(state)
            while state.result is None:
                a = int(rng.choice(state.legal_actions()))
                official.execute(Action(official.next_symbol, a))
                state = state.play(a)
                self.assertEqual(official.state, to_official(state).state)

    def test_json_midgame_reset_and_strict_failure(self):
        spec = replace(sample_match(4, engine_registry=self.registry, kind='utttai'), simulations=4)
        bot = make_opponent(spec)
        rng = np.random.default_rng(4)
        state = State()
        for _ in range(20):
            state = state.play(int(rng.choice(state.legal_actions())))
        try:
            self.assertIn(bot.choose(state, rng), state.legal_actions())
            bot.reset()
            self.assertIn(bot.choose(state, rng), state.legal_actions())
            self.assertEqual(bot.fallback, 'raise')
        finally:
            bot.close()
        self.assertIsNone(bot.process)

    def test_both_protocols_both_seats_and_noisy_training(self):
        for protocol in ('state_json', 'action_index'):
            for side in (1, -1):
                spec = sample_match(5, engine_registry=self.registry, kind='utttai')
                cmd = tuple(protocol if part == 'state_json' else part for part in spec.engine_command)
                spec = replace(spec, simulations=4, learner_side=side,
                               engine_command=cmd, engine_protocol=protocol,
                               opening_moves=3 if protocol == 'state_json' else 0,
                               epsilon=.35 if protocol == 'state_json' else 0.)
                trajectory, outcome, stats = play_game(Network().eval(), 2, 17,
                                                        SearchConfig(), 2, spec)
                self.assertTrue(trajectory)
                self.assertTrue(all(s.turn == side for s, *_ in trajectory))
                self.assertIn(outcome, (-1, 0, 1))

    def test_spawned_workers_mix_real_engines_and_reuse(self):
        spec = replace(sample_match(5, engine_registry=self.registry, kind='utttai'), simulations=4)
        matches = [spec, MatchSpec(kind='openspiel', simulations=4, opening_moves=4, epsilon=.3)]
        with SelfPlayPool(2, batch_size=4) as pool:
            results, metrics = pool.run(Network().eval(), [3, 4], 2, SearchConfig(), 2, matches)
            self.assertEqual([r[2]['match']['kind'] for r in results], ['utttai', 'openspiel'])
            self.assertGreater(metrics['inference_positions'], 0)
            results, _ = pool.run(Network().eval(), [5], 2, SearchConfig(), 2, [spec])
            self.assertEqual(len(results), 1)
