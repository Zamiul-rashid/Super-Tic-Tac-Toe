"""Unit tests for uniform Bot subsystem, ExternalProcessBot, StyleBot, CheckpointBot, and factory."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
import torch

from sttt.env import State
from sttt.learning import create_model
from sttt.bots import (
    Bot,
    AlphaBetaBot,
    TacticalBot,
    ThreatBlockBot,
    StyleBot,
    ExternalProcessBot,
    CheckpointBot,
    create_bot,
    get_bot,
    action_to_coord,
    coord_to_action,
)


class BotTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1337)

    def test_action_coord_conversions(self):
        # Bijective mapping across all 81 positions
        for a in range(81):
            r, c = action_to_coord(a)
            self.assertTrue(0 <= r <= 8 and 0 <= c <= 8)
            recovered = coord_to_action(r, c)
            self.assertEqual(a, recovered)

        # Explicit known coordinates
        self.assertEqual(action_to_coord(0), (0, 0))
        self.assertEqual(action_to_coord(8), (2, 2))
        self.assertEqual(action_to_coord(9), (0, 3))
        self.assertEqual(action_to_coord(40), (4, 4))
        self.assertEqual(action_to_coord(80), (8, 8))

        self.assertEqual(coord_to_action(0, 0), 0)
        self.assertEqual(coord_to_action(2, 2), 8)
        self.assertEqual(coord_to_action(0, 3), 9)
        self.assertEqual(coord_to_action(4, 4), 40)
        self.assertEqual(coord_to_action(8, 8), 80)

    def test_bot_base_class_contract(self):
        # Cannot instantiate abstract base class
        with self.assertRaises(TypeError):
            Bot()

        # Custom dummy bot inheriting from Bot
        class DummyBot(Bot):
            def choose(self, state, rng):
                return state.legal_actions()[0]

        dummy = DummyBot()
        self.assertEqual(dummy.name, "DummyBot")
        self.assertIsNone(dummy.advance(0))
        self.assertIsNone(dummy.reset())
        self.assertIsNone(dummy.close())
        with dummy as b:
            self.assertIs(b, dummy)

    def test_alphabeta_and_tactical_bots(self):
        tactical = TacticalBot()
        self.assertIsInstance(tactical, Bot)
        self.assertEqual(tactical.name, "tactical")

        ab = AlphaBetaBot(depth=2, node_budget=500, name="custom-ab")
        self.assertIsInstance(ab, Bot)
        self.assertEqual(ab.name, "custom-ab")

        ab_default = AlphaBetaBot(depth=3)
        self.assertEqual(ab_default.name, "alphabeta-d3")

        threat = ThreatBlockBot(node_budget=500)
        self.assertIsInstance(threat, Bot)
        self.assertIn(threat.choose(State(), self.rng), State().legal_actions())

        state = State()
        action_t = tactical.choose(state, self.rng)
        self.assertIn(action_t, state.legal_actions())

        action_ab = ab.choose(state, self.rng)
        self.assertIn(action_ab, state.legal_actions())

        # Terminal state raises ValueError
        # Play a game until terminal
        s = state
        while s.result is None:
            a = s.legal_actions()[0]
            s = s.play(a)
        with self.assertRaises(ValueError):
            tactical.choose(s, self.rng)
        with self.assertRaises(ValueError):
            ab.choose(s, self.rng)
        with self.assertRaises(ValueError):
            threat.choose(s, self.rng)

    def test_style_bot_all_policies(self):
        styles = ('random', 'center', 'corners', 'local-win', 'global-win')
        for style in styles:
            bot_stoch = StyleBot(style, deterministic=False)
            bot_det = StyleBot(style, deterministic=True)
            self.assertIsInstance(bot_stoch, Bot)
            self.assertEqual(bot_stoch.name, f"style-{style}")

            state = State()
            action_s = bot_stoch.choose(state, self.rng)
            self.assertIn(action_s, state.legal_actions())

            action_d = bot_det.choose(state, self.rng)
            self.assertIn(action_d, state.legal_actions())

        with self.assertRaises(ValueError):
            StyleBot("nonexistent-style")

        # Terminal state raises ValueError
        terminal_state = State(result=1)
        with self.assertRaises(ValueError):
            StyleBot("center").choose(terminal_state, self.rng)

    def test_style_bot_behavioral_bias(self):
        # Empty board: center style prefers center cell within board (c == 4)
        center_bot = StyleBot('center', deterministic=True)
        empty_state = State()
        action = center_bot.choose(empty_state, self.rng)
        b, c = divmod(action, 9)
        self.assertEqual(c, 4)

        # Empty board: corners style prefers corner cells (c in 0, 2, 6, 8)
        corners_bot = StyleBot('corners', deterministic=True)
        action = corners_bot.choose(empty_state, self.rng)
        b, c = divmod(action, 9)
        self.assertIn(c, (0, 2, 6, 8))

        # Local-win style takes immediate local board win
        # Construct state where Player 1 can win local board 0: cells 0 and 1 have mark 1, cell 2 is empty
        cells = list((0,) * 81)
        cells[0] = 1
        cells[1] = 1
        state = State(cells=tuple(cells), turn=1, forced=0)
        local_bot = StyleBot('local-win', deterministic=True)
        chosen = local_bot.choose(state, self.rng)
        self.assertEqual(chosen, 2)

        # Global-win style takes immediate global win
        # Boards 0 and 1 are won by Player 1, board 2 can be won by cell 20
        boards = (1, 1, 0, 0, 0, 0, 0, 0, 0)
        cells = list((0,) * 81)
        cells[18] = 1
        cells[19] = 1
        state = State(cells=tuple(cells), boards=boards, turn=1, forced=2)
        global_bot = StyleBot('global-win', deterministic=True)
        chosen = global_bot.choose(state, self.rng)
        self.assertEqual(chosen, 20)

    def test_external_process_bot_codingame_echo(self):
        # A mock python script implementing CodinGame protocol: echoing the first legal move
        code = (
            "import sys\n"
            "while True:\n"
            "    line1 = sys.stdin.readline()\n"
            "    if not line1:\n"
            "        break\n"
            "    line2 = sys.stdin.readline()\n"
            "    if not line2:\n"
            "        break\n"
            "    n = int(line2.strip())\n"
            "    moves = [sys.stdin.readline().strip() for _ in range(n)]\n"
            "    if moves:\n"
            "        sys.stdout.write(moves[0] + '\\n')\n"
            "        sys.stdout.flush()\n"
        )
        cmd = [sys.executable, "-u", "-c", code]
        with ExternalProcessBot(cmd, timeout=3.0, protocol="codingame", name="echo-codingame") as bot:
            self.assertIsInstance(bot, Bot)
            self.assertEqual(bot.name, "echo-codingame")

            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())

            # Simulate next turn with opponent move
            opp_move = state.play(action).legal_actions()[0]
            next_state = state.play(action).play(opp_move)
            bot.advance(opp_move)

            second_action = bot.choose(next_state, self.rng)
            self.assertIn(second_action, next_state.legal_actions())

            # Test reset
            bot.reset()
            reset_action = bot.choose(state, self.rng)
            self.assertIn(reset_action, state.legal_actions())

    def test_external_process_bot_action_protocol(self):
        # A mock python script implementing action protocol
        code = (
            "import sys\n"
            "while True:\n"
            "    opp = sys.stdin.readline()\n"
            "    if not opp:\n"
            "        break\n"
            "    n_line = sys.stdin.readline()\n"
            "    if not n_line:\n"
            "        break\n"
            "    n = int(n_line.strip())\n"
            "    actions = [int(sys.stdin.readline().strip()) for _ in range(n)]\n"
            "    if actions:\n"
            "        sys.stdout.write(str(actions[0]) + '\\n')\n"
            "        sys.stdout.flush()\n"
        )
        cmd = [sys.executable, "-u", "-c", code]
        with ExternalProcessBot(cmd, timeout=3.0, protocol="action", name="echo-action") as bot:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())

    def test_external_process_bot_timeout_non_freezing(self):
        # Mock script that sleeps indefinitely on stdin
        code = (
            "import sys, time\n"
            "sys.stdin.readline()\n"
            "time.sleep(10.0)\n"
        )
        cmd = [sys.executable, "-u", "-c", code]

        # Fast timeout of 0.1s; must complete well under 1s and return tactical fallback
        t0 = time.time()
        with ExternalProcessBot(cmd, timeout=0.1, fallback="tactical") as bot:
            state = State()
            action = bot.choose(state, self.rng)
            elapsed = time.time() - t0
            self.assertLess(elapsed, 1.5, "ExternalProcessBot hung instead of timing out")
            self.assertIn(action, state.legal_actions())

        # Test fallback="raise" raises RuntimeError on timeout
        with ExternalProcessBot(cmd, timeout=0.1, fallback="raise") as bot:
            with self.assertRaises(RuntimeError):
                bot.choose(State(), self.rng)

    def test_external_process_bot_crash_and_recovery(self):
        # Mock script that exits immediately with error code 1
        code = "import sys; sys.exit(1)"
        cmd = [sys.executable, "-u", "-c", code]

        with ExternalProcessBot(cmd, fallback="tactical", auto_restart=True) as bot:
            state = State()
            # First turn: process exits, falls back to tactical
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())

            # Same game: next move continues with fallback
            next_state = state.play(action)
            opp_act = next_state.legal_actions()[0]
            turn2_state = next_state.play(opp_act)
            action2 = bot.choose(turn2_state, self.rng)
            self.assertIn(action2, turn2_state.legal_actions())

            # Test recovery across games via reset()
            bot.reset()
            # Replace cmd with a working echo script
            working_code = (
                "import sys\n"
                "while True:\n"
                "    l1 = sys.stdin.readline()\n"
                "    if not l1: break\n"
                "    l2 = sys.stdin.readline()\n"
                "    if not l2: break\n"
                "    n = int(l2.strip())\n"
                "    moves = [sys.stdin.readline().strip() for _ in range(n)]\n"
                "    sys.stdout.write(moves[0] + '\\n')\n"
                "    sys.stdout.flush()\n"
            )
            bot.cmd = [sys.executable, "-u", "-c", working_code]
            recovered_action = bot.choose(State(), self.rng)
            self.assertIn(recovered_action, State().legal_actions())

        # Test crash with fallback="raise"
        with ExternalProcessBot(cmd, fallback="raise") as bot:
            with self.assertRaises(RuntimeError):
                bot.choose(State(), self.rng)

    def test_external_process_bot_invalid_output(self):
        # Mock script that outputs garbage text
        code = (
            "import sys\n"
            "sys.stdin.readline()\n"
            "sys.stdout.write('INVALID_GARBAGE_TOKEN\\n')\n"
            "sys.stdout.flush()\n"
        )
        cmd = [sys.executable, "-u", "-c", code]
        with ExternalProcessBot(cmd, fallback="tactical") as bot:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())

        # Mock script that outputs illegal move (e.g. 99 99 or out-of-bounds)
        code_illegal = (
            "import sys\n"
            "sys.stdin.readline()\n"
            "sys.stdout.write('99 99\\n')\n"
            "sys.stdout.flush()\n"
        )
        cmd_illegal = [sys.executable, "-u", "-c", code_illegal]
        with ExternalProcessBot(cmd_illegal, fallback="tactical") as bot:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())

    def test_external_process_bot_empty_command_cleanup(self):
        # Empty command must raise ValueError and clean up without AttributeError in __del__
        with self.assertRaises(ValueError):
            ExternalProcessBot("")
        with self.assertRaises(ValueError):
            ExternalProcessBot([])

    def test_external_process_bot_partial_stdout_timeout(self):
        # Subprocess writes partial token without newline, then sleeps
        code = (
            "import sys, time\n"
            "sys.stdin.readline()\n"
            "sys.stdout.write('4')\n"
            "sys.stdout.flush()\n"
            "time.sleep(2.0)\n"
        )
        cmd = [sys.executable, "-u", "-c", code]
        t0 = time.time()
        with ExternalProcessBot(cmd, timeout=0.2, fallback="tactical") as bot:
            state = State()
            action = bot.choose(state, self.rng)
            elapsed = time.time() - t0
            self.assertLess(elapsed, 1.0, f"choose() blocked for {elapsed:.2f}s on partial stdout (timeout=0.2s)")
            self.assertIn(action, state.legal_actions())

    def test_checkpoint_bot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pt"
            model = create_model("mlp")
            torch.save({"model": model.state_dict(), "arch": "mlp", "iteration": 42}, model_path)

            bot = CheckpointBot(model_path, simulations=16, device="cpu")
            self.assertIsInstance(bot, Bot)
            self.assertEqual(bot.name, "ckpt-iter0042")

            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
            self.assertIsNotNone(bot.tree)
            self.assertIsNotNone(bot.tree.root)

            # Subtree reuse verification
            next_state = state.play(action)
            opp_act = next_state.legal_actions()[0]
            turn2_state = next_state.play(opp_act)
            bot.advance(opp_act)

            action2 = bot.choose(turn2_state, self.rng)
            self.assertIn(action2, turn2_state.legal_actions())

            # Reset clears search tree
            bot.reset()
            self.assertIsNone(bot.tree)

            # Non-existent file raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                CheckpointBot(Path(tmpdir) / "nonexistent.pt")

    def test_create_bot_factory(self):
        # Tactical
        bot = create_bot("tactical")
        self.assertIsInstance(bot, TacticalBot)

        bot_threat = create_bot("threat")
        self.assertIsInstance(bot_threat, ThreatBlockBot)

        # AlphaBeta
        bot_ab = create_bot("alphabeta", depth=4)
        self.assertIsInstance(bot_ab, AlphaBetaBot)
        self.assertEqual(bot_ab.depth, 4)

        bot_ab2 = create_bot("alphabeta:5")
        self.assertIsInstance(bot_ab2, AlphaBetaBot)
        self.assertEqual(bot_ab2.depth, 5)

        # Style bots
        for style in ('random', 'center', 'corners', 'local-win', 'global-win'):
            bot_s = create_bot(style)
            self.assertIsInstance(bot_s, StyleBot)
            self.assertEqual(bot_s.style, style)

        bot_sc = create_bot("style:corners", deterministic=True)
        self.assertIsInstance(bot_sc, StyleBot)
        self.assertEqual(bot_sc.style, "corners")
        self.assertTrue(bot_sc.deterministic)

        # External commands
        bot_cmd = create_bot("cmd:echo 0 0")
        self.assertIsInstance(bot_cmd, ExternalProcessBot)

        bot_ext = create_bot("external:echo 0 0")
        self.assertIsInstance(bot_ext, ExternalProcessBot)

        # Existing bot instance returns itself
        self.assertIs(create_bot(bot), bot)

        # Checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "factory_test.pt"
            model = create_model("mlp")
            torch.save({"model": model.state_dict(), "arch": "mlp", "iteration": 7}, model_path)

            bot_ckpt = create_bot(f"checkpoint:{model_path}", simulations=8)
            self.assertIsInstance(bot_ckpt, CheckpointBot)

            bot_ckpt2 = create_bot(str(model_path), simulations=8)
            self.assertIsInstance(bot_ckpt2, CheckpointBot)

        # Alias get_bot
        self.assertIs(get_bot, create_bot)

        # Errors
        with self.assertRaises(ValueError):
            create_bot("completely_invalid_bot_spec")
        with self.assertRaises(TypeError):
            create_bot(12345)


if __name__ == "__main__":
    unittest.main()
