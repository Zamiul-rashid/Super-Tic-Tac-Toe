"""Adversarial stress tests for training readiness runner and directory isolation.

Authored by challenger_m1_2.
Verifies:
1. Output directory validation rejects runs/big_run, runs/run_v2, ., ./runs,
   relative traversals, subdirectories, symlinks, and edge cases with ValueError.
2. Valid isolated directories under runs/readiness/ or temporary locations are accepted.
3. Invalid CLI flags (--backend, --device, --stage, unrecognized flags) fail cleanly.
4. Unsupported backend/device combinations (--device cuda without CUDA, --backend python --stage native,
   --device cpu --stage gpu) are handled deterministically (fail or skip with manifest records).
5. Zero artifacts leak into runs/big_run or runs/run_v2 under valid or adversarial runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_training_ready import (
    ReadinessRunner,
    freeze_checkpoint,
    validate_output_dir,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_dir_file_hashes(directory: Path) -> dict[str, str]:
    """Compute sha256 hashes for all files under directory relative to directory."""
    hashes = {}
    if not directory.exists():
        return hashes
    for root, _, files in os.walk(directory):
        for f in files:
            p = Path(root) / f
            rel = p.relative_to(directory).as_posix()
            try:
                hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
            except Exception:
                pass
    return hashes


class TestOutputDirectoryAdversarialIsolation(unittest.TestCase):
    """Stress test validate_output_dir against malicious, relative, and symlinked paths."""

    def setUp(self):
        self.repo_root = _REPO_ROOT
        self.big_run = (self.repo_root / "runs" / "big_run").resolve()
        self.run_v2 = (self.repo_root / "runs" / "run_v2").resolve()
        self.runs_dir = (self.repo_root / "runs").resolve()

    def test_direct_protected_roots_raise_value_error(self):
        """Verify protected roots raise ValueError."""
        protected = [
            self.repo_root,
            Path("."),
            self.runs_dir,
            Path("runs"),
            Path("./runs"),
            self.big_run,
            Path("runs/big_run"),
            Path("./runs/big_run"),
            self.run_v2,
            Path("runs/run_v2"),
            Path("./runs/run_v2"),
        ]
        for p in protected:
            with self.subTest(path=str(p)):
                with self.assertRaises(ValueError) as cm:
                    validate_output_dir(p, self.repo_root)
                self.assertIn("Run isolation violation", str(cm.exception))

    def test_trailing_slashes_and_dots_raise_value_error(self):
        """Verify trailing slashes, redundant dots, and internal dot segments are resolved and rejected."""
        paths = [
            Path("runs/big_run/"),
            Path("runs/run_v2/"),
            Path("./runs/"),
            Path("./"),
            Path("runs/./big_run"),
            Path("runs/run_v2/."),
            Path("runs/big_run/././"),
            Path("./runs/././run_v2/."),
        ]
        for p in paths:
            with self.subTest(path=str(p)):
                with self.assertRaises(ValueError) as cm:
                    validate_output_dir(p, self.repo_root)
                self.assertIn("Run isolation violation", str(cm.exception))

    def test_directory_traversal_backtracking_raises_value_error(self):
        """Verify path traversal (..) resolving into protected roots is rejected."""
        paths = [
            Path("runs/readiness/../big_run"),
            Path("runs/readiness/../../runs/big_run"),
            Path("runs/readiness/../../runs/run_v2"),
            Path("runs/run_v2/../run_v2"),
            Path("runs/big_run/../../runs"),
            Path("runs/readiness/../.."),  # resolves to repo root
            Path("tests/../runs/big_run"),
            Path("tests/../runs/run_v2"),
        ]
        for p in paths:
            with self.subTest(path=str(p)):
                with self.assertRaises(ValueError) as cm:
                    validate_output_dir(p, self.repo_root)
                self.assertIn("Run isolation violation", str(cm.exception))

    def test_protected_subdirectories_raise_value_error(self):
        """Verify subdirectories inside big_run and run_v2 are rejected."""
        paths = [
            Path("runs/big_run/evaluations"),
            Path("runs/big_run/checkpoints/deep/nested"),
            Path("runs/big_run/latest.pt"),
            Path("runs/run_v2/models"),
            Path("runs/run_v2/sub1/sub2/sub3"),
            Path("runs/run_v2/best.pt"),
            self.big_run / "nested_dir",
            self.run_v2 / "nested_dir",
        ]
        for p in paths:
            with self.subTest(path=str(p)):
                with self.assertRaises(ValueError) as cm:
                    validate_output_dir(p, self.repo_root)
                self.assertIn("Run isolation violation", str(cm.exception))
                self.assertIn("cannot be inside protected", str(cm.exception))

    def test_symlink_attacks_raise_value_error(self):
        """Verify symlinks pointing to protected directories or their subdirectories are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            symlink_big_run = tmp_path / "link_to_big_run"
            symlink_run_v2 = tmp_path / "link_to_run_v2"
            symlink_runs = tmp_path / "link_to_runs"
            symlink_root = tmp_path / "link_to_root"

            try:
                symlink_big_run.symlink_to(self.big_run)
                symlink_run_v2.symlink_to(self.run_v2)
                symlink_runs.symlink_to(self.runs_dir)
                symlink_root.symlink_to(self.repo_root)
            except OSError:
                self.skipTest("Symlinks not supported on filesystem")

            symlinks = [symlink_big_run, symlink_run_v2, symlink_runs, symlink_root]
            for sl in symlinks:
                with self.subTest(symlink=str(sl)):
                    with self.assertRaises(ValueError) as cm:
                        validate_output_dir(sl, self.repo_root)
                    self.assertIn("Run isolation violation", str(cm.exception))

            # Nested inside a symlinked target
            nested_symlink = symlink_big_run / "evaluations"
            with self.assertRaises(ValueError) as cm:
                validate_output_dir(nested_symlink, self.repo_root)
            self.assertIn("Run isolation violation", str(cm.exception))

    def test_empty_string_path_resolves_to_root_and_raises(self):
        """Empty string resolves to current directory (repo root) and must be rejected."""
        with self.assertRaises(ValueError) as cm:
            validate_output_dir(Path(""), self.repo_root)
        self.assertIn("Run isolation violation", str(cm.exception))

    def test_legitimate_readiness_directories_accepted(self):
        """Ensure valid directories under runs/readiness/ are accepted without error."""
        valid_paths = [
            self.repo_root / "runs" / "readiness" / "run_abc123",
            self.repo_root / "runs" / "readiness" / "20260913_test",
            Path("runs/readiness/my_readiness_test"),
        ]
        for p in valid_paths:
            with self.subTest(path=str(p)):
                res = validate_output_dir(p, self.repo_root)
                self.assertEqual(res, p.resolve())


class TestCliFlagsAndCombinations(unittest.TestCase):
    """Test CLI argument parsing, invalid flags, and unsupported combinations."""

    def test_invalid_backend_flag_rejected(self):
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
            "--backend",
            "invalid_backend",
        ]
        res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 2)
        self.assertIn("invalid choice", res.stderr)

    def test_invalid_device_flag_rejected(self):
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
            "--device",
            "tpu",
        ]
        res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 2)
        self.assertIn("invalid choice", res.stderr)

    def test_invalid_stage_flag_rejected(self):
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
            "--stage",
            "nonexistent_stage",
        ]
        res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 2)
        self.assertIn("invalid choice", res.stderr)

    def test_unrecognized_argument_rejected(self):
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
            "--unknown-custom-flag",
        ]
        res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 2)
        self.assertIn("unrecognized arguments", res.stderr)

    def test_cli_output_dir_targeting_protected_run_exits_nonzero(self):
        """CLI invocation targeting runs/big_run exits with code 1 and prints ValueError."""
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
            "--output-dir",
            "runs/big_run",
        ]
        res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("Run isolation violation", res.stderr)

    def test_device_cuda_without_cuda_fails_build_stage(self):
        """--device cuda when torch.cuda.is_available() is False must fail build stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "runs" / "readiness" / "test_cuda_fail"
            cmd = [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
                "--backend",
                "cpp",
                "--device",
                "cuda",
                "--stage",
                "build",
                "--output-dir",
                str(out_dir),
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True, env=env)
            self.assertEqual(res.returncode, 1)
            manifest_file = out_dir / "summary_manifest.json"
            self.assertTrue(manifest_file.is_file())
            with open(manifest_file) as f:
                data = json.load(f)
            self.assertEqual(data["overall_status"], "failed")
            self.assertIn("cuda", data["stages"]["build"]["error"].lower())

    def test_backend_python_skips_native_stage_cleanly(self):
        """--backend python --stage native must skip cleanly with a clear reason."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "runs" / "readiness" / "test_py_native"
            cmd = [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
                "--backend",
                "python",
                "--device",
                "cpu",
                "--stage",
                "native",
                "--output-dir",
                str(out_dir),
            ]
            res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0)
            manifest_file = out_dir / "native" / "manifest.json"
            self.assertTrue(manifest_file.is_file())
            with open(manifest_file) as f:
                data = json.load(f)
            self.assertEqual(data["status"], "skipped")
            self.assertIn("Python backend specified", data["reason"])

    def test_device_cpu_skips_gpu_stage_cleanly(self):
        """--device cpu --stage gpu must skip cleanly with a clear reason."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "runs" / "readiness" / "test_cpu_gpu"
            cmd = [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
                "--backend",
                "cpp",
                "--device",
                "cpu",
                "--stage",
                "gpu",
                "--output-dir",
                str(out_dir),
            ]
            res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0)
            manifest_file = out_dir / "gpu" / "manifest.json"
            self.assertTrue(manifest_file.is_file())
            with open(manifest_file) as f:
                data = json.load(f)
            self.assertEqual(data["status"], "skipped")
            self.assertIn("CPU device specified", data["reason"])

    def test_backend_python_build_stage_passes(self):
        """--backend python --device cpu --stage build executes and passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "runs" / "readiness" / "test_py_build"
            cmd = [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
                "--backend",
                "python",
                "--device",
                "cpu",
                "--stage",
                "build",
                "--output-dir",
                str(out_dir),
            ]
            res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Python build stage failed:\n{res.stderr}\n{res.stdout}")
            manifest_file = out_dir / "build" / "manifest.json"
            self.assertTrue(manifest_file.is_file())
            with open(manifest_file) as f:
                data = json.load(f)
            self.assertEqual(data["status"], "passed")


class TestProtectedRunsImmutability(unittest.TestCase):
    """Verify that running check_training_ready.py leaves zero unwanted artifacts in runs/big_run or runs/run_v2."""

    def test_zero_artifacts_in_protected_runs_after_runner_execution(self):
        big_run_dir = _REPO_ROOT / "runs" / "big_run"
        run_v2_dir = _REPO_ROOT / "runs" / "run_v2"

        hashes_big_run_before = _get_dir_file_hashes(big_run_dir)
        hashes_run_v2_before = _get_dir_file_hashes(run_v2_dir)

        # 1. Run manifest-only
        subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "check_training_ready.py"), "--manifest-only"],
            cwd=_REPO_ROOT,
            capture_output=True,
            check=True,
        )

        # 2. Run adversarial attacks attempting to write to protected runs
        for target in ["runs/big_run", "runs/run_v2", ".", "./runs"]:
            subprocess.run(
                [
                    sys.executable,
                    str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
                    "--output-dir",
                    target,
                ],
                cwd=_REPO_ROOT,
                capture_output=True,
            )

        # 3. Run build stage in isolated readiness dir
        subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
                "--backend",
                "cpp",
                "--device",
                "cpu",
                "--stage",
                "build",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            check=True,
        )

        hashes_big_run_after = _get_dir_file_hashes(big_run_dir)
        hashes_run_v2_after = _get_dir_file_hashes(run_v2_dir)

        # Assert zero additions, zero deletions, zero modifications
        self.assertEqual(
            set(hashes_big_run_before.keys()),
            set(hashes_big_run_after.keys()),
            "Files in runs/big_run were added or deleted!",
        )
        for k, v in hashes_big_run_before.items():
            self.assertEqual(
                v,
                hashes_big_run_after[k],
                f"File content in runs/big_run/{k} was mutated!",
            )

        self.assertEqual(
            set(hashes_run_v2_before.keys()),
            set(hashes_run_v2_after.keys()),
            "Files in runs/run_v2 were added or deleted!",
        )
        for k, v in hashes_run_v2_before.items():
            self.assertEqual(
                v,
                hashes_run_v2_after[k],
                f"File content in runs/run_v2/{k} was mutated!",
            )


if __name__ == "__main__":
    unittest.main()
