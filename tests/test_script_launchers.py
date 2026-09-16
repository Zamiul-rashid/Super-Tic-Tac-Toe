import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class TestTrainLauncher(unittest.TestCase):
    def test_help_is_available(self):
        result = subprocess.run(
            ["bash", "scripts/train.sh", "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--dry-run", result.stdout)

    def test_dry_run_accepts_paths_without_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "latest.pt"
            checkpoint.touch()
            output = root / "next"
            result = subprocess.run(
                [
                    "bash", "scripts/train.sh",
                    "--checkpoint", str(checkpoint),
                    "--output", str(output),
                    "--iterations", "2",
                    "--device", "cpu",
                    "--no-fp16",
                    "--no-population",
                    "--dry-run",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(checkpoint), result.stdout)
            self.assertIn(str(output), result.stdout)
            self.assertIn("--iterations 2", result.stdout)
            self.assertFalse(output.exists())


class TestEvaluationLauncher(unittest.TestCase):
    def test_help_lists_all_modes(self):
        result = subprocess.run(
            ["bash", "scripts/evaluate.sh", "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for mode in ("full", "championship", "sweep", "compare"):
            self.assertIn(mode, result.stdout)

    def test_compare_requires_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "latest.pt"
            checkpoint.touch()
            result = subprocess.run(
                [
                    "bash", "scripts/evaluate.sh",
                    "--mode", "compare",
                    "--checkpoint", str(checkpoint),
                    "--output", str(root / "out"),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--reference is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
