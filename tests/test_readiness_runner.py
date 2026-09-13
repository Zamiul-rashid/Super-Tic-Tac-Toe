"""Unit tests for training readiness runner, C++ build provenance, and run isolation."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sttt.cpp_env as ce
from scripts.check_training_ready import (
    ReadinessRunner,
    freeze_checkpoint,
    validate_output_dir,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCppProvenance(unittest.TestCase):
    """Verify C++ build provenance macros, module constants, and accessors."""

    def test_provenance_dict_structure(self):
        prov = ce.get_cpp_provenance()
        self.assertIsInstance(prov, dict)
        expected_keys = {
            "available",
            "version",
            "build_id",
            "source_revision",
            "compiler_flags",
            "binary_path",
        }
        self.assertTrue(expected_keys.issubset(prov.keys()))

    def test_provenance_types_when_available(self):
        if not ce.is_cpp_available():
            self.skipTest("C++ extension not available")
        prov = ce.get_cpp_provenance()
        self.assertTrue(prov["available"])
        self.assertIsInstance(prov["version"], str)
        self.assertTrue(len(prov["version"]) > 0)
        self.assertIsInstance(prov["build_id"], str)
        self.assertTrue(len(prov["build_id"]) > 0)
        self.assertIsInstance(prov["source_revision"], str)
        self.assertTrue(len(prov["source_revision"]) > 0)
        self.assertIsInstance(prov["compiler_flags"], str)
        self.assertTrue(len(prov["compiler_flags"]) > 0)
        self.assertIsInstance(prov["binary_path"], str)
        self.assertTrue(Path(prov["binary_path"]).is_file())

    def test_module_constants_on_cpp_env(self):
        if not ce.is_cpp_available():
            self.skipTest("C++ extension not available")
        self.assertIsNotNone(ce.CPP_VERSION)
        self.assertIsNotNone(ce.CPP_BUILD_ID)
        self.assertIsNotNone(ce.CPP_SOURCE_REVISION)
        self.assertIsNotNone(ce.CPP_COMPILER_FLAGS)
        self.assertIsNotNone(ce.CPP_BINARY_PATH)

    def test_module_constants_on_sttt_cpp(self):
        if not ce.is_cpp_available():
            self.skipTest("C++ extension not available")
        import sttt_cpp

        self.assertTrue(hasattr(sttt_cpp, "__version__"))
        self.assertTrue(hasattr(sttt_cpp, "BUILD_ID"))
        self.assertTrue(hasattr(sttt_cpp, "SOURCE_REVISION"))
        self.assertTrue(hasattr(sttt_cpp, "COMPILER_FLAGS"))
        self.assertIsInstance(sttt_cpp.__version__, str)
        self.assertIsInstance(sttt_cpp.BUILD_ID, str)
        self.assertIsInstance(sttt_cpp.SOURCE_REVISION, str)
        self.assertIsInstance(sttt_cpp.COMPILER_FLAGS, str)


class TestRunIsolation(unittest.TestCase):
    """Verify run isolation and protected directory protection."""

    def test_valid_readiness_output_dir(self):
        target = _REPO_ROOT / "runs" / "readiness" / "test_run_123"
        validated = validate_output_dir(target, _REPO_ROOT)
        self.assertEqual(validated, target.resolve())

    def test_reject_big_run_root(self):
        target = _REPO_ROOT / "runs" / "big_run"
        with self.assertRaises(ValueError) as cm:
            validate_output_dir(target, _REPO_ROOT)
        self.assertIn("Run isolation violation", str(cm.exception))

    def test_reject_big_run_subdirectory(self):
        target = _REPO_ROOT / "runs" / "big_run" / "nested" / "output"
        with self.assertRaises(ValueError) as cm:
            validate_output_dir(target, _REPO_ROOT)
        self.assertIn("Run isolation violation", str(cm.exception))
        self.assertIn("runs/big_run", str(cm.exception))

    def test_reject_run_v2_root(self):
        target = _REPO_ROOT / "runs" / "run_v2"
        with self.assertRaises(ValueError) as cm:
            validate_output_dir(target, _REPO_ROOT)
        self.assertIn("Run isolation violation", str(cm.exception))

    def test_reject_run_v2_subdirectory(self):
        target = _REPO_ROOT / "runs" / "run_v2" / "models"
        with self.assertRaises(ValueError) as cm:
            validate_output_dir(target, _REPO_ROOT)
        self.assertIn("Run isolation violation", str(cm.exception))
        self.assertIn("runs/run_v2", str(cm.exception))

    def test_reject_runs_root(self):
        target = _REPO_ROOT / "runs"
        with self.assertRaises(ValueError) as cm:
            validate_output_dir(target, _REPO_ROOT)
        self.assertIn("Run isolation violation", str(cm.exception))


class TestCheckpointFreezer(unittest.TestCase):
    """Verify safe checkpoint freezing and SHA-256 verification."""

    def test_freeze_checkpoint_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            dest_dir = Path(tmpdir) / "dest"
            src_dir.mkdir()
            src_file = src_dir / "model.pt"
            src_file.write_bytes(b"dummy checkpoint binary content 12345")

            dest_path, sha = freeze_checkpoint(src_file, dest_dir)
            self.assertTrue(dest_path.is_file())
            self.assertEqual(dest_path.name, "model.pt")
            self.assertEqual(dest_path.parent, dest_dir.resolve())
            self.assertEqual(dest_path.read_bytes(), b"dummy checkpoint binary content 12345")
            self.assertTrue(len(sha) == 64)

    def test_freeze_checkpoint_missing_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does_not_exist.pt"
            dest_dir = Path(tmpdir) / "dest"
            with self.assertRaises(FileNotFoundError):
                freeze_checkpoint(nonexistent, dest_dir)


class TestReadinessRunnerBuildStage(unittest.TestCase):
    """Verify ReadinessRunner execution and build stage checks."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "runs" / "readiness" / "test_build_run"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_capability_manifest_structure(self):
        runner = ReadinessRunner(
            backend="cpp",
            device="cpu",
            stage="build",
            output_dir=self.output_dir,
            manifest_only=True,
        )
        manifest = runner.get_capability_manifest()
        self.assertIn("timestamp", manifest)
        self.assertIn("git_revision", manifest)
        self.assertIn("python", manifest)
        self.assertIn("torch", manifest)
        self.assertIn("cpp_provenance", manifest)
        self.assertIn("suite_discovery", manifest)
        self.assertGreaterEqual(manifest["suite_discovery"]["total_tests"], 264)

    def test_build_stage_passes(self):
        runner = ReadinessRunner(
            backend="cpp",
            device="cpu",
            stage="build",
            output_dir=self.output_dir,
        )
        result = runner.run_stage_build()
        self.assertEqual(result["status"], "passed")
        self.assertIn("checks", result)
        self.assertEqual(result["checks"]["native_tests"]["skips"], 0)
        self.assertGreater(result["checks"]["native_tests"]["total_run"], 0)
        self.assertEqual(result["checks"]["native_tests"]["failures"], 0)
        self.assertEqual(result["checks"]["native_tests"]["errors"], 0)

        # Check manifest written to disk
        manifest_path = self.output_dir / "build" / "manifest.json"
        self.assertTrue(manifest_path.is_file())
        with open(manifest_path) as f:
            data = json.load(f)
        self.assertEqual(data["status"], "passed")

    def test_cli_execution_build_stage(self):
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "check_training_ready.py"),
            "--backend",
            "cpp",
            "--device",
            "cpu",
            "--stage",
            "build",
            "--output-dir",
            str(self.output_dir),
        ]
        res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI build stage failed:\n{res.stderr}\n{res.stdout}")
        summary_path = self.output_dir / "summary_manifest.json"
        self.assertTrue(summary_path.is_file())
        with open(summary_path) as f:
            summary = json.load(f)
        self.assertEqual(summary["overall_status"], "passed")
        self.assertIn("build", summary["stages"])
        self.assertEqual(summary["stages"]["build"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
