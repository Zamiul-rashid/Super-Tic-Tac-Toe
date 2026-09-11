"""Unit and integration tests for Super Tic-Tac-Toe tournament CLI command and reporting."""
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sttt.ai import tournament_cmd, tournament
from sttt.learning import create_model
from sttt.reports import write_tournament_report
from sttt.tournament import run_tournament, MatchResult, EloRating, GlickoRating


class TestTournamentCLIArguments(unittest.TestCase):
    """Test argument parsing and validation for the tournament CLI subcommand."""

    def test_tournament_help_displays_all_arguments(self):
        """Verify python -m sttt.ai tournament --help shows all expected options."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("--checkpoint", out)
        self.assertIn("--opponents", out)
        self.assertIn("--games", out)
        self.assertIn("--simulations", out)
        self.assertIn("--simulation-budgets", out)
        self.assertIn("--opening-plies", out)
        self.assertIn("--seed", out)
        self.assertIn("--output", out)
        self.assertIn("--device", out)

    def test_rejection_games_under_20(self):
        """Verify game counts < 20 are rejected with specific error."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--games", "19"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Games per matchup must be between 20 and 500", res.stderr)

    def test_rejection_games_over_500(self):
        """Verify game counts > 500 are rejected with specific error."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--games", "501"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Games per matchup must be between 20 and 500", res.stderr)

    def test_rejection_opening_plies_under_0(self):
        """Verify opening plies < 0 are rejected with specific error."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--opening-plies", "-1"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Opening plies must be between 0 and 4", res.stderr)

    def test_rejection_opening_plies_over_4(self):
        """Verify opening plies > 4 are rejected with specific error."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--opening-plies", "5"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Opening plies must be between 0 and 4", res.stderr)

    def test_rejection_negative_seed(self):
        """Verify negative seeds are rejected."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--seed", "-1"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Seeds must be nonnegative", res.stderr)


class TestTournamentCLIExecution(unittest.TestCase):
    """Test full execution of tournament subcommand via CLI and direct dispatch."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_direct_tournament_cmd_execution(self):
        """Verify tournament_cmd direct invocation creates all artifacts and prints scoreboard."""
        import argparse
        args = argparse.Namespace(
            command="tournament",
            checkpoint=None,
            opponents=["alphabeta", "tactical"],
            games=20,
            simulations=128,
            simulation_budgets=None,
            opening_plies=2,
            seed=42,
            output=self.temp_dir,
            device="cpu",
        )
        f = io.StringIO()
        with patch("sys.stdout", f):
            res = tournament_cmd(args)
        
        self.assertIn("participants", res)
        self.assertIn("ratings", res)
        self.assertIn("matchups", res)
        self.assertIn("scoreboard", res)
        self.assertIn("TOURNAMENT LEADERBOARD", f.getvalue())

        # Verify files on disk
        self.assertTrue((Path(self.temp_dir) / "tournament.json").is_file())
        self.assertTrue((Path(self.temp_dir) / "summary.csv").is_file())
        self.assertTrue((Path(self.temp_dir) / "matchups.csv").is_file())
        self.assertTrue((Path(self.temp_dir) / "scoreboard.txt").is_file())

    def test_subprocess_tournament_execution_files_and_schemas(self):
        """Verify subprocess tournament execution generates valid schemas."""
        cmd = [
            sys.executable, "-m", "sttt.ai", "tournament",
            "--opponents", "tactical", "corners",
            "--games", "20",
            "--opening-plies", "1",
            "--output", self.temp_dir,
            "--device", "cpu",
            "--seed", "123",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Subprocess failed:\n{res.stderr}")

        # Validate tournament.json
        json_path = Path(self.temp_dir) / "tournament.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data["participants"]), 2)
        self.assertIn("tactical", data["participants"])
        corner_bot_name = [p for p in data["participants"] if "corners" in p][0]
        self.assertIn("tactical", data["ratings"])
        self.assertIn(corner_bot_name, data["ratings"])
        self.assertIn("elo", data["ratings"]["tactical"])
        self.assertIn("glicko2", data["ratings"]["tactical"])
        self.assertEqual(len(data["matchups"]), 20)

        # Validate summary.csv
        csv_path = Path(self.temp_dir) / "summary.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertIn("participant", [h.lower() for h in reader.fieldnames])
            self.assertIn("elo", reader.fieldnames)
            self.assertIn("glicko2", reader.fieldnames)

        # Validate matchups.csv
        matchups_path = Path(self.temp_dir) / "matchups.csv"
        with open(matchups_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            m_rows = list(reader)
            self.assertEqual(len(m_rows), 20)
            self.assertIn("game_id", reader.fieldnames)
            self.assertIn("player_x", reader.fieldnames)
            self.assertIn("player_o", reader.fieldnames)
            self.assertIn("winner", reader.fieldnames)

        # Validate scoreboard.txt
        board_path = Path(self.temp_dir) / "scoreboard.txt"
        with open(board_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("TOURNAMENT LEADERBOARD", content)
            self.assertIn("tactical", content)
            self.assertIn("corners", content)

    def test_simulation_budgets_sweep_outputs(self):
        """Verify simulation budgets sweep generates scaling summary files."""
        cmd = [
            sys.executable, "-m", "sttt.ai", "tournament",
            "--opponents", "tactical",
            "--games", "20",
            "--simulation-budgets", "64", "128",
            "--output", self.temp_dir,
            "--device", "cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Sweep failed:\n{res.stderr}")

        json_file = Path(self.temp_dir) / "scaling_summary.json"
        csv_file = Path(self.temp_dir) / "scaling_summary.csv"
        board_file = Path(self.temp_dir) / "scoreboard.txt"

        self.assertTrue(json_file.is_file())
        self.assertTrue(csv_file.is_file())
        self.assertTrue(board_file.is_file())

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["sweep_budgets"], [64, 128])
            self.assertEqual(len(data["scaling"]), 2)
            self.assertEqual(data["scaling"][0]["budget"], 64)
            self.assertEqual(data["scaling"][1]["budget"], 128)

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[0]["budget"]), 64)
            self.assertEqual(int(rows[1]["budget"]), 128)

    def test_checkpoint_integration_tournament(self):
        """Verify running tournament with a real neural model checkpoint."""
        ckpt_path = Path(self.temp_dir) / "test_model.pt"
        model = create_model("resnet")
        torch.save({"model": model.state_dict(), "iteration": 5, "arch": "resnet"}, ckpt_path)

        cmd = [
            sys.executable, "-m", "sttt.ai", "tournament",
            "--checkpoint", str(ckpt_path),
            "--opponents", "tactical",
            "--games", "20",
            "--simulations", "16",
            "--output", self.temp_dir,
            "--device", "cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Checkpoint tournament failed:\n{res.stderr}")

        json_file = Path(self.temp_dir) / "tournament.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(any("ckpt" in p or "test_model" in p for p in data["participants"]))
        self.assertIn("tactical", data["participants"])


class TestWriteTournamentReportUnit(unittest.TestCase):
    """Unit tests for write_tournament_report helper in sttt.reports."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_tournament_report_atomic_no_tmp(self):
        """Verify write_tournament_report writes all files atomically leaving no .tmp files."""
        sample_tourn = {
            "participants": ["BotA", "BotB"],
            "games_per_matchup": 20,
            "results": [
                MatchResult(0, 0, 2, [0, 1], "BotA", "BotB", 1, 30),
                MatchResult(1, 0, 2, [0, 1], "BotB", "BotA", -1, 30),
            ],
            "elo_ratings": {
                "BotA": EloRating("BotA", 1600.0, 50.0, 2, 2, 0, 0),
                "BotB": EloRating("BotB", 1400.0, 50.0, 2, 0, 0, 2),
            },
            "glicko_ratings": {
                "BotA": GlickoRating("BotA", 1620.0, 100.0, 0.06, 2),
                "BotB": GlickoRating("BotB", 1380.0, 100.0, 0.06, 2),
            },
            "matrix": {"BotA": {"BotB": {"wins": 2, "draws": 0, "losses": 0}}},
            "side_stats": {"x_win_rate": 0.5, "o_win_rate": 0.5, "draw_rate": 0.0},
            "scoreboard": "SAMPLE SCOREBOARD",
        }

        ret = write_tournament_report(sample_tourn, self.temp_dir)
        self.assertTrue(ret.is_file())
        
        # Check no .tmp files exist
        for f in os.listdir(self.temp_dir):
            self.assertFalse(f.endswith(".tmp"), f"Found left-over temp file: {f}")

        # Check files exist
        for fname in ("tournament.json", "summary.csv", "matchups.csv", "games.csv", "scoreboard.txt"):
            self.assertTrue((Path(self.temp_dir) / fname).is_file(), f"Missing file: {fname}")


if __name__ == "__main__":
    unittest.main()
