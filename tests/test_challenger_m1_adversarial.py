"""Empirical Challenger 2 adversarial stress tests for Milestone 1.

Verifies:
1. 50+ randomized games across all 5 StyleBot policies verify moves are strictly legal.
2. choose() on terminal states raises ValueError across all Bot implementations.
3. CheckpointBot CPU inference does not allocate GPU VRAM (torch.cuda.memory_allocated() == 0).
4. advance(action) preserves search subtree consistency and statistics.
"""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

from sttt.env import State
from sttt.learning import create_model
from sttt.bots import (
    Bot,
    StyleBot,
    AlphaBetaBot,
    TacticalBot,
    CheckpointBot,
    ExternalProcessBot,
    create_bot,
)
from sttt.opponent import NAMES


class ChallengerAdversarialM1Tests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260911)

    def test_style_bot_50_plus_randomized_games_all_styles(self):
        """Stress-test StyleBot across all 5 styles over 100+ games verifying move legality."""
        total_games = 0
        total_moves = 0

        # Sub-test 1: 10 games per style in self-play (50 games)
        for style in NAMES:
            bot_x = StyleBot(style, deterministic=False)
            bot_o = StyleBot(style, deterministic=False)
            for g in range(10):
                total_games += 1
                state = State()
                while state.result is None:
                    current_bot = bot_x if state.turn == 1 else bot_o
                    legal = state.legal_actions()
                    self.assertGreater(len(legal), 0, "Nonterminal state must have legal actions")

                    action = current_bot.choose(state, self.rng)
                    self.assertIn(action, legal, f"StyleBot({style}) selected illegal action {action}")
                    total_moves += 1
                    state = state.play(action)

                self.assertIn(state.result, (-1, 0, 1))

        # Sub-test 2: 10 games per style vs 'random' (50 games)
        random_bot = StyleBot('random', deterministic=False)
        for style in NAMES:
            bot = StyleBot(style, deterministic=False)
            for g in range(10):
                total_games += 1
                state = State()
                # Alternate seats
                bot_x = bot if g % 2 == 0 else random_bot
                bot_o = random_bot if g % 2 == 0 else bot
                while state.result is None:
                    current_bot = bot_x if state.turn == 1 else bot_o
                    legal = state.legal_actions()
                    action = current_bot.choose(state, self.rng)
                    self.assertIn(action, legal)
                    total_moves += 1
                    state = state.play(action)

                self.assertIn(state.result, (-1, 0, 1))

        # Sub-test 3: Deterministic vs stochastic cross-style matchups (20 matchups across all distinct pairs)
        for style_a in NAMES:
            for style_b in NAMES:
                if style_a == style_b:
                    continue
                bot_a = StyleBot(style_a, deterministic=True)
                bot_b = StyleBot(style_b, deterministic=False)
                total_games += 1
                state = State()
                while state.result is None:
                    bot = bot_a if state.turn == 1 else bot_b
                    legal = state.legal_actions()
                    action = bot.choose(state, self.rng)
                    self.assertIn(action, legal)
                    total_moves += 1
                    state = state.play(action)
                self.assertIn(state.result, (-1, 0, 1))

        self.assertGreaterEqual(total_games, 50, f"Expected >= 50 games, played {total_games}")
        self.assertEqual(total_games, 120, f"Expected exactly 120 games, played {total_games}")
        self.assertGreater(total_moves, 3000, f"Expected > 3000 moves, verified {total_moves}")

    def test_choose_on_terminal_state_raises_value_error(self):
        """Verify calling choose() on any terminal state raises ValueError across all bot classes."""
        # Create lightweight checkpoint for CheckpointBot testing
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "term_test.pt"
            torch.save({"model": create_model("mlp").state_dict(), "arch": "mlp"}, model_path)

            mock_echo_code = (
                "import sys\n"
                "while True:\n"
                "    l = sys.stdin.readline()\n"
                "    if not l: break\n"
                "    n = int(sys.stdin.readline().strip())\n"
                "    moves = [sys.stdin.readline().strip() for _ in range(n)]\n"
                "    if moves: sys.stdout.write(moves[0] + '\\n'); sys.stdout.flush()\n"
            )
            echo_cmd = [sys.executable, "-u", "-c", mock_echo_code]

            bots_to_test: list[Bot] = []
            for style in NAMES:
                bots_to_test.append(StyleBot(style, deterministic=False))
                bots_to_test.append(StyleBot(style, deterministic=True))
            bots_to_test.append(AlphaBetaBot(depth=1))
            bots_to_test.append(TacticalBot())
            bots_to_test.append(CheckpointBot(model_path, simulations=16, device="cpu"))
            if Path("runs/big_run/latest.pt").exists():
                bots_to_test.append(CheckpointBot("runs/big_run/latest.pt", simulations=16, device="cpu"))
            bots_to_test.append(ExternalProcessBot(echo_cmd, timeout=1.0, protocol="codingame"))

            # Distinct terminal state variants
            terminal_states = [
                State(result=1),    # X win
                State(result=-1),   # O win
                State(result=0),    # Draw
            ]

            # Generate a naturally occurring terminal state by playing a quick game
            nat_state = State()
            st_bot = StyleBot('random', deterministic=False)
            while nat_state.result is None:
                nat_state = nat_state.play(st_bot.choose(nat_state, self.rng))
            self.assertIsNotNone(nat_state.result)
            terminal_states.append(nat_state)

            for bot in bots_to_test:
                for term_state in terminal_states:
                    with self.subTest(bot=bot.name, result=term_state.result):
                        with self.assertRaises(ValueError, msg=f"{bot.name} failed to raise ValueError on terminal state"):
                            bot.choose(term_state, self.rng)

            # Clean up external processes
            for b in bots_to_test:
                b.close()

    def test_checkpoint_bot_cpu_inference_zero_gpu_vram(self):
        """Verify CheckpointBot CPU inference does not allocate GPU VRAM (torch.cuda.memory_allocated() == 0)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this platform")

        # Verify initial CUDA allocation is clean
        self.assertEqual(torch.cuda.memory_allocated(), 0, "Initial CUDA memory should be 0")
        self.assertEqual(torch.cuda.max_memory_allocated(), 0, "Initial peak CUDA memory should be 0")

        # Test both lightweight model and real latest.pt checkpoint if present
        checkpoints = []
        with tempfile.TemporaryDirectory() as tmpdir:
            lightweight_path = Path(tmpdir) / "mlp_cpu.pt"
            torch.save({"model": create_model("mlp").state_dict(), "arch": "mlp", "iteration": 10}, lightweight_path)
            checkpoints.append(str(lightweight_path))

            if Path("runs/big_run/latest.pt").exists():
                checkpoints.append("runs/big_run/latest.pt")

            for ckpt_path in checkpoints:
                bot = CheckpointBot(ckpt_path, simulations=32, device="cpu")

                # Verify model parameters are all strictly on CPU
                for param in bot.model.parameters():
                    self.assertEqual(param.device.type, "cpu", f"Parameter on {param.device}, expected cpu")

                # Verify VRAM is 0 after model loading
                self.assertEqual(torch.cuda.memory_allocated(), 0, "VRAM allocated after CheckpointBot init")
                self.assertEqual(torch.cuda.max_memory_allocated(), 0, "Peak VRAM > 0 after CheckpointBot init")

                # Execute choose() across 3 plies
                state = State()
                for ply in range(3):
                    action = bot.choose(state, self.rng)
                    self.assertIn(action, state.legal_actions())
                    self.assertEqual(torch.cuda.memory_allocated(), 0, f"VRAM allocated during ply {ply}")
                    self.assertEqual(torch.cuda.max_memory_allocated(), 0, f"Peak VRAM > 0 during ply {ply}")
                    state = state.play(action)
                    # Opponent reply
                    if state.result is None:
                        opp_action = state.legal_actions()[0]
                        bot.advance(opp_action)
                        state = state.play(opp_action)

                bot.reset()
                self.assertEqual(torch.cuda.memory_allocated(), 0, "VRAM allocated after bot reset")

    def test_advance_action_subtree_integrity(self):
        """Verify advance(action) correctly updates root state and preserves search subtree statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "tree_test.pt"
            torch.save({"model": create_model("mlp").state_dict(), "arch": "mlp"}, model_path)

            bot = CheckpointBot(model_path, simulations=200, device="cpu", backend="python")

            # 1. Advance before any choose() does not crash and leaves tree None
            bot.advance(4)
            self.assertIsNone(bot.tree)

            # 2. First move choose: root must advance to state.play(chosen_action)
            s0 = State()
            a0 = bot.choose(s0, self.rng)
            s1 = s0.play(a0)
            self.assertIsNotNone(bot.tree)
            self.assertIsNotNone(bot.tree.root)
            self.assertEqual(bot.tree.root.state, s1)

            # 3. Subtree preservation when opponent plays an explored child
            # Check if there are explored children from s1
            visited = [(act, node) for act, node in bot.tree.root.children.items() if node.n > 0]
            if visited:
                opp_action, opp_child = max(visited, key=lambda pair: pair[1].n)
                saved_visits = opp_child.n
            else:
                opp_action = s1.legal_actions()[0]
                opp_child = bot.tree.root.children.get(opp_action)
                saved_visits = opp_child.n if opp_child else 0

            # Advance opponent action
            bot.advance(opp_action)
            s2 = s1.play(opp_action)
            self.assertEqual(bot.tree.root.state, s2)
            if opp_child is not None:
                self.assertIs(bot.tree.root, opp_child)

            # Next choose: retained_visits in stats must match saved_visits
            a2 = bot.choose(s2, self.rng)
            self.assertEqual(bot.tree.stats["retained_visits"], saved_visits)
            s3 = s2.play(a2)
            self.assertEqual(bot.tree.root.state, s3)

            # 4. Caller recovery when advance(opp_action) is omitted
            opp_action2 = s3.legal_actions()[0]
            s4 = s3.play(opp_action2)
            # Caller does NOT call bot.advance(opp_action2)
            a4 = bot.choose(s4, self.rng)
            self.assertIn(a4, s4.legal_actions())
            self.assertEqual(bot.tree.root.state, s4.play(a4))

            # 5. Advance with unvisited action gracefully creates new root
            bot.reset()
            self.assertIsNone(bot.tree)
            _ = bot.choose(State(), self.rng)
            # Pick an action that was not explored by wiping children
            bot.tree.root.children = {}
            test_act = bot.tree.root.state.legal_actions()[0]
            expected_next = bot.tree.root.state.play(test_act)
            bot.advance(test_act)
            self.assertEqual(bot.tree.root.state, expected_next)
            self.assertEqual(bot.tree.root.n, 0)

            # 6. Reset clears tree
            bot.reset()
            self.assertIsNone(bot.tree)

            # 7. StyleBot and AlphaBetaBot advance() no-op behavior
            sb = StyleBot("center")
            self.assertIsNone(sb.advance(10))
            ab = AlphaBetaBot(depth=2)
            self.assertIsNone(ab.advance(10))


if __name__ == "__main__":
    unittest.main()
