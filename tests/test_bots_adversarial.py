"""Empirical adversarial stress-test harness for ExternalProcessBot.

Tests:
1. Subprocess hanging, timeout <= 5.0s, busy CPU loops, process reaping, partial output.
2. Subprocess crash, abrupt SIGKILL, non-existent binaries, broken pipes, restart semantics.
3. Malformed/illegal stdout (NaN, Inf, negatives, floats, out-of-bounds, illegal moves, huge output).
4. Stderr buffer saturation and non-blocking I/O edge cases.
"""
import os
import signal
import sys
import tempfile
import time
import unittest
import numpy as np

from sttt.env import State
from sttt.bots import (
    ExternalProcessBot,
    coord_to_action,
    action_to_coord,
)


class AdversarialExternalProcessBotTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2026)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_script(self, name: str, code: str) -> str:
        path = os.path.join(self.temp_dir.name, name)
        with open(path, "w") as f:
            f.write(code)
        return path

    # =========================================================================
    # 1. Subprocess Hanging & Timeout Tests
    # =========================================================================

    def test_mock_bot_sleep_10s_timeout_enforced(self):
        """Mock bot sleeps 10s. Timeout 0.2s must return in < 1.0s with legal fallback."""
        script = self._write_script(
            "sleep10.py",
            "import sys, time\n"
            "sys.stdin.readline()\n"
            "time.sleep(10.0)\n"
        )
        t0 = time.time()
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=0.2, fallback="tactical")
        try:
            state = State()
            action = bot.choose(state, self.rng)
            elapsed = time.time() - t0
            self.assertLess(elapsed, 1.2, f"Bot hung for {elapsed:.2f}s instead of <= 0.2s timeout")
            self.assertIn(action, state.legal_actions())
        finally:
            bot.close()

    def test_configured_timeout_is_honored(self):
        """A 30s budget must stay 30s; a hidden cap once killed a training run."""
        script = self._write_script("sleep10.py", "import time\ntime.sleep(10.0)\n")
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=30.0)
        try:
            self.assertEqual(bot.timeout, 30.0)
        finally:
            bot.close()

    def test_infinite_busy_cpu_loop_killed(self):
        """Subprocess spinning 100% CPU in while True must be terminated and reaped."""
        script = self._write_script(
            "busy_loop.py",
            "import sys\n"
            "sys.stdin.readline()\n"
            "while True:\n"
            "    pass\n"
        )
        t0 = time.time()
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=0.2)
        try:
            pid = bot.process.pid
            state = State()
            action = bot.choose(state, self.rng)
            elapsed = time.time() - t0
            self.assertLess(elapsed, 1.2)
            self.assertIn(action, state.legal_actions())

            # Verify the process is dead and reaped
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
                # If still exists, check if zombie
                if os.path.exists(f"/proc/{pid}/status"):
                    with open(f"/proc/{pid}/status") as sf:
                        status = sf.read()
                        self.assertIn("State:\tZ", status, "Busy loop process was not killed")
            except ProcessLookupError:
                pass  # Cleanly reaped
        finally:
            bot.close()

    def test_subsequent_turns_in_same_game_skip_hanging(self):
        """Once crashed/timed out in a game, subsequent turns in the same game must not re-hang."""
        script = self._write_script(
            "sleep_each.py",
            "import sys, time\n"
            "while True:\n"
            "    sys.stdin.readline()\n"
            "    time.sleep(5.0)\n"
        )
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=0.2)
        try:
            s1 = State()
            a1 = bot.choose(s1, self.rng)
            self.assertIn(a1, s1.legal_actions())

            # Turn 2 in same game
            s2 = s1.play(a1)
            opp_a = s2.legal_actions()[0]
            s3 = s2.play(opp_a)

            t0 = time.time()
            a2 = bot.choose(s3, self.rng)
            elapsed = time.time() - t0
            # Should immediately return fallback without waiting for another timeout
            self.assertLess(elapsed, 0.1, f"Second turn waited {elapsed:.2f}s instead of instant fallback")
            self.assertIn(a2, s3.legal_actions())
        finally:
            bot.close()

    def test_partial_stdout_then_hang_vulnerability(self):
        """Adversarial stress test: subprocess writes partial token without newline, then hangs.
        
        Tests whether select.select readability combined with blocking readline()
        causes a hang beyond the configured timeout.
        """
        script = self._write_script(
            "partial_hang.py",
            "import sys, time\n"
            "sys.stdin.readline()\n"
            "# Write partial token without newline and flush\n"
            "sys.stdout.write('4')\n"
            "sys.stdout.flush()\n"
            "time.sleep(2.0)\n"
        )
        t0 = time.time()
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=0.2)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            elapsed = time.time() - t0
            # If readline() blocks on the missing newline, elapsed will exceed 2.0s
            # rather than honoring timeout=0.2s!
            self.assertLess(
                elapsed,
                1.0,
                f"POTENTIAL VULNERABILITY: choose() blocked for {elapsed:.2f}s on partial stdout without newline (timeout was 0.2s)"
            )
            self.assertIn(action, state.legal_actions())
        finally:
            bot.close()

    # =========================================================================
    # 2. Subprocess Crash, Abrupt Exit & Recovery Tests
    # =========================================================================

    def test_immediate_exit_code_1(self):
        """Process exits immediately with code 1."""
        script = self._write_script("exit1.py", "import sys\nsys.exit(1)\n")
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            bot.close()

    def test_unhandled_exception_crash(self):
        """Process raises an unhandled ZeroDivisionError."""
        script = self._write_script(
            "divzero.py",
            "import sys\n"
            "sys.stdin.readline()\n"
            "x = 1 / 0\n"
        )
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            bot.close()

    def test_sigkill_mid_execution(self):
        """Process killed abruptly via SIGKILL mid-game."""
        script = self._write_script(
            "echo_then_wait.py",
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
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0)
        try:
            s1 = State()
            a1 = bot.choose(s1, self.rng)
            self.assertIn(a1, s1.legal_actions())

            # Kill child with SIGKILL
            if bot.process is not None:
                os.kill(bot.process.pid, signal.SIGKILL)
                time.sleep(0.05)

            # Next turn must recover with fallback
            s2 = s1.play(a1)
            opp_a = s2.legal_actions()[0]
            s3 = s2.play(opp_a)
            a2 = bot.choose(s3, self.rng)
            self.assertIn(a2, s3.legal_actions())
        finally:
            bot.close()

    def test_nonexistent_executable_binary(self):
        """Nonexistent executable path falls back gracefully without unhandled crash."""
        bot = ExternalProcessBot(["/bin/nonexistent_bot_binary_xyz123"], timeout=0.5, fallback="tactical")
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            bot.close()

    def test_clean_restart_after_reset(self):
        """Process crashes in Game 1. Calling bot.reset() restores a working process for Game 2."""
        toggle_file = os.path.join(self.temp_dir.name, "toggle.txt")
        with open(toggle_file, "w") as f:
            f.write("crash")

        script = self._write_script(
            "conditional_crash.py",
            f"import sys\n"
            f"with open(r'{toggle_file}') as f:\n"
            f"    mode = f.read().strip()\n"
            f"if mode == 'crash':\n"
            f"    sys.exit(1)\n"
            f"while True:\n"
            f"    l1 = sys.stdin.readline()\n"
            f"    if not l1: break\n"
            f"    l2 = sys.stdin.readline()\n"
            f"    if not l2: break\n"
            f"    n = int(l2.strip())\n"
            f"    moves = [sys.stdin.readline().strip() for _ in range(n)]\n"
            f"    sys.stdout.write(moves[0] + '\\n')\n"
            f"    sys.stdout.flush()\n"
        )
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0)
        try:
            s1 = State()
            a1 = bot.choose(s1, self.rng)
            self.assertIn(a1, s1.legal_actions())
            self.assertTrue(bot._crashed_in_game)

            # Fix script for Game 2
            with open(toggle_file, "w") as f:
                f.write("work")

            bot.reset()
            self.assertFalse(bot._crashed_in_game)

            s2 = State()
            a2 = bot.choose(s2, self.rng)
            self.assertIn(a2, s2.legal_actions())
            self.assertFalse(bot._crashed_in_game)
            self.assertIsNotNone(bot.process)
            self.assertIsNone(bot.process.poll())
        finally:
            bot.close()

    def test_fallback_raise_mode(self):
        """fallback='raise' raises RuntimeError on crash or timeout."""
        script = self._write_script("exit_err.py", "import sys\nsys.exit(2)\n")
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=0.5, fallback="raise")
        try:
            with self.assertRaises(RuntimeError):
                bot.choose(State(), self.rng)
        finally:
            bot.close()

    # =========================================================================
    # 3. Malformed & Illegal Output Tests
    # =========================================================================

    def _test_corrupted_output(self, output_line: str, protocol: str = "codingame"):
        script = self._write_script(
            f"corrupt_{abs(hash(output_line))}.py",
            f"import sys\n"
            f"sys.stdin.readline()\n"
            f"sys.stdout.write({repr(output_line)} + '\\n')\n"
            f"sys.stdout.flush()\n"
        )
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0, protocol=protocol)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions(), f"Failed on output {repr(output_line)}")
        finally:
            bot.close()

    def test_empty_string_output(self):
        self._test_corrupted_output("")

    def test_whitespace_only_output(self):
        self._test_corrupted_output("   \t  ")

    def test_negative_numbers(self):
        self._test_corrupted_output("-1 -1")
        self._test_corrupted_output("-5")
        self._test_corrupted_output("-1 0")

    def test_nan_and_infinity(self):
        self._test_corrupted_output("NaN NaN")
        self._test_corrupted_output("inf inf")
        self._test_corrupted_output("-inf")

    def test_floating_point_numbers(self):
        self._test_corrupted_output("4.5 4.5")
        self._test_corrupted_output("1e3 0")

    def test_out_of_bounds_action_and_coords(self):
        self._test_corrupted_output("99 99")
        self._test_corrupted_output("9 0")
        self._test_corrupted_output("0 9")
        self._test_corrupted_output("81", protocol="action")
        self._test_corrupted_output("999", protocol="action")

    def test_illegal_occupied_cell_move(self):
        """Bot attempts to play an already occupied cell."""
        # State with cell 40 played
        s = State().play(40)
        r, c = action_to_coord(40)
        script = self._write_script(
            "occupied.py",
            f"import sys\n"
            f"sys.stdin.readline()\n"
            f"sys.stdout.write('{r} {c}\\n')\n"
            f"sys.stdout.flush()\n"
        )
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0)
        try:
            action = bot.choose(s, self.rng)
            self.assertIn(action, s.legal_actions())
            self.assertNotEqual(action, 40)
        finally:
            bot.close()

    def test_illegal_wrong_board_when_forced(self):
        """Bot attempts to play in board 0 when forced to board 4."""
        # State with forced board 4
        s = State(forced=4)
        # Choose cell in board 0 (action 0, coord 0 0)
        script = self._write_script(
            "wrong_board.py",
            "import sys\n"
            "sys.stdin.readline()\n"
            "sys.stdout.write('0 0\\n')\n"
            "sys.stdout.flush()\n"
        )
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0)
        try:
            action = bot.choose(s, self.rng)
            self.assertIn(action, s.legal_actions())
            # Action must be in forced board 4 (action in [36..44])
            b, _ = divmod(action, 9)
            self.assertEqual(b, 4)
        finally:
            bot.close()

    def test_huge_stdout_output(self):
        """Process writes 100,000 characters of garbage on stdout."""
        script = self._write_script(
            "huge_output.py",
            "import sys\n"
            "sys.stdin.readline()\n"
            "sys.stdout.write('A' * 100000 + '\\n')\n"
            "sys.stdout.flush()\n"
        )
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            bot.close()

    def test_codingame_output_with_trailing_comment(self):
        """CodinGame protocol allows trailing comments (e.g. '0 0 my message')."""
        script = self._write_script(
            "trailing_msg.py",
            "import sys\n"
            "sys.stdin.readline()\n"
            "sys.stdout.write('0 0 thinking complete!\\n')\n"
            "sys.stdout.flush()\n"
        )
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertEqual(action, 0)
        finally:
            bot.close()

    def test_rapid_consecutive_games_reset_lifecycle(self):
        """Verify 10 consecutive games with reset() do not leak file descriptors or processes."""
        script = self._write_script(
            "rapid_echo.py",
            "import sys\n"
            "while True:\n"
            "    l1 = sys.stdin.readline()\n"
            "    if not l1: break\n"
            "    l2 = sys.stdin.readline()\n"
            "    if not l2: break\n"
            "    n = int(l2.strip())\n"
            "    moves = [sys.stdin.readline().strip() for _ in range(n)]\n"
            "    if moves:\n"
            "        sys.stdout.write(moves[0] + '\\n')\n"
            "        sys.stdout.flush()\n"
        )
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=1.0)
        try:
            for game_idx in range(10):
                bot.reset()
                state = State()
                action = bot.choose(state, self.rng)
                self.assertIn(action, state.legal_actions())
        finally:
            bot.close()

    # =========================================================================
    # 4. Stderr Isolation & Pipe Buffer Saturation
    # =========================================================================

    def test_stderr_saturation_does_not_deadlock(self):
        """Subprocess writes 128KB to stderr (exceeding standard 64KB pipe buffer).
        
        If stderr pipe buffer is filled and never read by parent, child will block
        on stderr write. ExternalProcessBot must handle this via timeout rather
        than deadlocking forever.
        """
        script = self._write_script(
            "stderr_flood.py",
            "import sys\n"
            "sys.stdin.readline()\n"
            "sys.stderr.write('E' * 131072)\n"
            "sys.stderr.flush()\n"
            "sys.stdout.write('0 0\\n')\n"
            "sys.stdout.flush()\n"
        )
        t0 = time.time()
        bot = ExternalProcessBot([sys.executable, "-u", script], timeout=0.5)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            elapsed = time.time() - t0
            self.assertLess(elapsed, 1.5, f"Bot deadlocked on stderr flood ({elapsed:.2f}s)")
            self.assertIn(action, state.legal_actions())
        finally:
            bot.close()


if __name__ == "__main__":
    unittest.main()
