"""Empirical Challenger 2 Adversarial Stress Suite for Milestone 5.

Stress-tests:
1. Live multi-agent CLI tournament combining all bot types simultaneously
   (TacticalBot, AlphaBetaBot, StyleBot center, StyleBot corners, ExternalProcessBot).
2. Validation of scoreboards, Bayesian Elo MAP, Glicko-2, side statistics,
   and CSV/JSON report data integrity.
3. CheckpointBot + ExternalProcessBot live round-robin tournament interaction.
4. Memory stability (RSS) across multi-round sweeps without leaks.
5. CPU thread confinement (torch.set_num_threads(1), /proc/self/status thread counts).
6. ExternalProcessBot process lifecycle and zombie process prevention.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sttt.bots import (
    AlphaBetaBot,
    Bot,
    CheckpointBot,
    ExternalProcessBot,
    StyleBot,
    TacticalBot,
    create_bot,
)
from sttt.env import State
from sttt.tournament import (
    run_matchup,
    run_simulation_sweep,
    run_tournament,
)


def _get_resident_mb() -> float:
    """Read resident memory in MB via /proc/self/statm."""
    try:
        pagesize = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return (resident_pages * pagesize) / (1024.0 * 1024.0)
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _get_os_threads() -> int:
    """Read thread count of the current process via /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except Exception:
        pass
    import threading
    return threading.active_count()


def _get_child_pids() -> list[int]:
    """Read active child process PIDs via /proc/<pid>/task/<pid>/children."""
    pid = os.getpid()
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            return [int(p) for p in f.read().split()]
    except Exception:
        res = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
        if res.returncode == 0:
            return [int(p) for p in res.stdout.split()]
        return []


class ChallengerM5MultiAgentTournamentTests(unittest.TestCase):
    """Adversarial validation of comprehensive multi-agent tournament interactions."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.python_bin = sys.executable
        cls.mock_bot_path = cls.repo_root / "tests" / "mock_codingame_bot.py"
        assert cls.mock_bot_path.is_file(), f"Mock bot not found at {cls.mock_bot_path}"
        cls.starter_checkpoint = cls.repo_root / "runs" / "starter" / "latest.pt"

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="sttt_challenger_m5_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_live_cli_tournament_all_bot_types_simultaneous(self):
        """Run a live CLI tournament combining all bot types simultaneously.

        Bot types combined:
        - tactical
        - alphabeta
        - center
        - corners
        - external bot (mock codingame protocol)
        """
        output_dir = Path(self.test_dir) / "all_bots_tourn"
        cmd = [
            self.python_bin,
            "-m",
            "sttt.ai",
            "tournament",
            "--opponents",
            "tactical",
            "alphabeta",
            "center",
            "corners",
            f"cmd:{self.python_bin} {self.mock_bot_path}",
            "--games",
            "20",
            "--opening-plies",
            "2",
            "--seed",
            "42",
            "--output",
            str(output_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"CLI tournament failed with code {result.returncode}:\n{result.stderr}",
        )

        # Verify stdout prints scoreboard and summary statistics
        stdout = result.stdout
        self.assertIn("TOURNAMENT LEADERBOARD & RATINGS", stdout)
        self.assertIn("alphabeta-d3", stdout)
        self.assertIn("tactical", stdout)
        self.assertIn("style-center", stdout)
        self.assertIn("style-corners", stdout)
        self.assertIn("python", stdout)
        self.assertIn("Total games played: 200", stdout)
        self.assertIn("Pairwise color bias delta: 0.00%", stdout)

        # 1. Validate files generated
        for fname in ("summary.csv", "matchups.csv", "games.csv", "tournament.json", "scoreboard.txt"):
            fpath = output_dir / fname
            self.assertTrue(fpath.is_file(), f"Expected artifact {fname} not found in {output_dir}")
            self.assertGreater(fpath.stat().st_size, 0, f"Artifact {fname} is empty")

        # 2. Validate summary.csv
        with open(output_dir / "summary.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 5, "Expected 5 participants in summary.csv")
        participants = {r["participant"] for r in rows}
        expected_participants = {"alphabeta-d3", "tactical", "style-center", "style-corners", "python"}
        self.assertEqual(participants, expected_participants)

        # Validate statistics per participant
        for r in rows:
            games = int(r["games"])
            wins = int(r["wins"])
            draws = int(r["draws"])
            losses = int(r["losses"])
            self.assertEqual(games, 80, f"Expected 80 games per participant, got {games}")
            self.assertEqual(wins + draws + losses, games, f"Wins+Draws+Losses != Games for {r['participant']}")
            win_rate = float(r["win_rate"])
            score_rate = float(r["score_rate"])
            self.assertAlmostEqual(win_rate, wins / games, places=4)
            self.assertAlmostEqual(score_rate, (wins + 0.5 * draws) / games, places=4)

            # Elo & Glicko-2 ratings
            elo = float(r["elo"])
            elo_ci95 = float(r["elo_ci95"])
            glicko2 = float(r["glicko2"])
            glicko2_rd = float(r["glicko2_rd"])
            volatility = float(r["volatility"])

            self.assertTrue(np.isfinite(elo), f"Elo is non-finite for {r['participant']}")
            self.assertGreater(elo_ci95, 0.0, f"Elo 95% CI is non-positive for {r['participant']}")
            self.assertTrue(np.isfinite(glicko2), f"Glicko-2 is non-finite for {r['participant']}")
            self.assertGreater(glicko2_rd, 0.0, f"Glicko-2 RD is non-positive for {r['participant']}")
            self.assertGreater(volatility, 0.0, f"Volatility is non-positive for {r['participant']}")

        # Validate relative strength: AlphaBeta and Tactical should dominate weak styles
        elo_by_name = {r["participant"]: float(r["elo"]) for r in rows}
        self.assertGreater(elo_by_name["alphabeta-d3"], elo_by_name["tactical"])
        self.assertGreater(elo_by_name["tactical"], elo_by_name["style-center"])
        self.assertGreater(elo_by_name["tactical"], elo_by_name["style-corners"])

        # 3. Validate matchups.csv
        with open(output_dir / "matchups.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            matchup_rows = list(reader)
        self.assertEqual(len(matchup_rows), 200, "Expected exactly 200 games in matchups.csv")

        # Check paired symmetry in matchups
        for i in range(0, 200, 2):
            g1, g2 = matchup_rows[i], matchup_rows[i + 1]
            self.assertEqual(g1["pair_id"], g2["pair_id"])
            self.assertEqual(g1["opening_moves"], g2["opening_moves"])
            self.assertEqual(g1["player_x"], g2["player_o"])
            self.assertEqual(g1["player_o"], g2["player_x"])

        # 4. Validate tournament.json
        with open(output_dir / "tournament.json", mode="r", encoding="utf-8") as f:
            tjson = json.load(f)

        self.assertIn("participants", tjson)
        self.assertIn("ratings", tjson)
        self.assertIn("side_stats", tjson)
        self.assertIn("matchups", tjson)

        side_stats = tjson["side_stats"]
        self.assertEqual(side_stats["total_games"], 200)
        self.assertEqual(
            side_stats["x_wins"] + side_stats["o_wins"] + side_stats["draws"],
            200,
        )
        self.assertAlmostEqual(
            side_stats["x_win_rate"] + side_stats["o_win_rate"] + side_stats["draw_rate"],
            1.0,
            places=5,
        )

    def test_checkpoint_and_external_in_tournament(self):
        """Pit a real CheckpointBot, ExternalProcessBot, and heuristic bots in a tournament."""
        if not self.starter_checkpoint.is_file():
            self.skipTest(f"Starter checkpoint not found at {self.starter_checkpoint}")

        ckpt_bot = CheckpointBot(
            path=str(self.starter_checkpoint),
            simulations=64,
            leaf_batch=8,
            device="cpu",
            name="ckpt-starter",
        )
        ext_bot = ExternalProcessBot(
            command=[self.python_bin, str(self.mock_bot_path)],
            timeout=2.0,
            name="external-mock",
        )
        tactical = TacticalBot(name="tactical-test")
        alphabeta = AlphaBetaBot(depth=2, node_budget=500, name="ab-d2")

        bots = {
            ckpt_bot.name: ckpt_bot,
            ext_bot.name: ext_bot,
            tactical.name: tactical,
            alphabeta.name: alphabeta,
        }

        try:
            tourn = run_tournament(
                bots=bots,
                games_per_matchup=20,
                opening_plies=2,
                seed=123,
            )

            self.assertEqual(len(tourn["results"]), 120)  # 6 pairs * 20 games = 120
            self.assertEqual(len(tourn["participants"]), 4)

            # Verify CheckpointBot played all 60 games
            ckpt_elo = tourn["elo_ratings"][ckpt_bot.name]
            self.assertEqual(ckpt_elo.games, 60)
            self.assertTrue(np.isfinite(ckpt_elo.elo))

            # Verify external bot played all 60 games
            ext_elo = tourn["elo_ratings"][ext_bot.name]
            self.assertEqual(ext_elo.games, 60)
            self.assertTrue(np.isfinite(ext_elo.elo))

        finally:
            ext_bot.close()
            ckpt_bot.reset()


class ChallengerM5StressResourceTests(unittest.TestCase):
    """Stress-test memory stability and CPU thread confinement during multi-round sweeps."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.python_bin = sys.executable
        cls.mock_bot_path = cls.repo_root / "tests" / "mock_codingame_bot.py"
        cls.starter_checkpoint = cls.repo_root / "runs" / "starter" / "latest.pt"

    def test_memory_stability_across_multi_round_sweeps(self):
        """Verify RSS memory does not exhibit monotonic unbounded growth across 6 rounds."""
        rss_samples = []

        bots_specs = ["tactical", "style:center", "style:corners"]

        for round_idx in range(6):
            bots = {f"b_{i}": create_bot(spec) for i, spec in enumerate(bots_specs)}
            tourn = run_tournament(
                bots=bots,
                games_per_matchup=20,
                opening_plies=2,
                seed=1000 + round_idx * 100,
            )
            self.assertEqual(len(tourn["results"]), 60)  # 3 pairs * 20 games

            # Clean up bots
            for b in bots.values():
                b.reset()
                b.close()
            del bots
            del tourn

            import gc
            gc.collect()

            rss_mb = _get_resident_mb()
            rss_samples.append(rss_mb)

        # Memory should stabilize after warm-up (round 1 to round 5)
        # Check that memory from round 2 to round 5 does not grow by more than 35 MB
        delta_rss = rss_samples[-1] - rss_samples[1]
        self.assertLess(
            delta_rss,
            35.0,
            f"Memory growth across rounds exceeded 35 MB: samples={rss_samples}, delta={delta_rss:.2f} MB",
        )

    def test_cpu_thread_confinement_during_sweeps(self):
        """Verify torch and OS threads remain confined to 1 during simulation sweeps."""
        # Save baseline
        baseline_torch_threads = torch.get_num_threads()
        baseline_os_threads = _get_os_threads()

        torch.set_num_threads(1)
        self.assertEqual(torch.get_num_threads(), 1)

        # Run simulation sweep across multiple budgets
        with tempfile.TemporaryDirectory(prefix="sweep_thread_test_") as temp_dir:
            sweep = run_simulation_sweep(
                checkpoint_path=str(self.starter_checkpoint) if self.starter_checkpoint.is_file() else "",
                opponent_specs=["tactical", "alphabeta"],
                budgets=[16, 32, 64],
                games=20,
                opening_plies=2,
                seed=42,
                device="cpu",
                output_dir=temp_dir,
            )

            self.assertEqual(len(sweep["scaling"]), 3)

        # Verify PyTorch thread count remained strictly 1
        current_torch_threads = torch.get_num_threads()
        self.assertEqual(
            current_torch_threads,
            1,
            f"PyTorch threads altered during sweep: expected 1, got {current_torch_threads}",
        )

        # Verify OS thread count has not exploded
        final_os_threads = _get_os_threads()
        self.assertLessEqual(
            final_os_threads,
            baseline_os_threads + 2,
            f"OS threads exploded: baseline={baseline_os_threads}, final={final_os_threads}",
        )

    def test_external_bot_zombie_process_prevention(self):
        """Verify ExternalProcessBot terminates cleanly without leaving zombie children."""
        initial_children = _get_child_pids()

        bots = []
        for i in range(3):
            bot = ExternalProcessBot(
                command=[self.python_bin, str(self.mock_bot_path)],
                timeout=1.0,
                name=f"ext_{i}",
            )
            bots.append(bot)

        # Ensure processes are alive
        for bot in bots:
            self.assertIsNotNone(bot.process)
            self.assertIsNone(bot.process.poll())

        # Play a quick game with each
        rng = np.random.default_rng(42)
        state = State()
        for bot in bots:
            action = bot.choose(state, rng)
            self.assertIn(action, state.legal_actions())

        # Close all bots
        for bot in bots:
            bot.close()

        # Check child processes
        import time
        time.sleep(0.2)
        final_children = _get_child_pids()
        leaked = [pid for pid in final_children if pid not in initial_children]
        self.assertEqual(
            len(leaked),
            0,
            f"Detected orphaned child processes: {leaked}",
        )


if __name__ == "__main__":
    unittest.main()
