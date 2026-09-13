"""Empirical Challenger adversarial tests for Milestone 1 (M0 & M9 / Requirement R1).

Adversarially tests:
1. Build system behavior under varying PYTHON and include paths; -Werror verification; Makefile flags.
2. sttt_cpp module constants and sttt.cpp_env.get_cpp_provenance() return valid non-empty strings and correct boolean status.
3. Import behavior: extension loaded from site-packages vs cpp/, sys.path precedence, and missing binary fallback handling.
4. check_training_ready.py runner robustness: strict run isolation, failure on invalid paths, skip rejection, checkpoint freezer.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sttt.cpp_env as ce
from scripts.check_training_ready import (
    ReadinessRunner,
    freeze_checkpoint,
    validate_output_dir,
)


class TestBuildSystemAdversarial(unittest.TestCase):
    """Stress-test C++ build system under varied configurations, include paths, and compiler flags."""

    def test_makefile_werror_flag_present(self):
        """Verify -Werror is explicitly present in Makefile CXXFLAGS and cpp_env provenance."""
        makefile = (_REPO_ROOT / "cpp" / "Makefile").read_text()
        self.assertIn("-Werror", makefile)
        self.assertIn("-Wall", makefile)
        self.assertIn("-Wextra", makefile)

        prov = ce.get_cpp_provenance()
        if prov["available"]:
            self.assertIn("-Werror", prov["compiler_flags"])

    def test_compiler_werror_enforcement_empirically(self):
        """Empirically prove that -Werror converts warnings to errors during compilation."""
        # A simple snippet with an unused variable warning
        c_code = "int foo() { int unused_var = 42; return 0; }\n"
        cmd = [
            "g++",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-std=c++17",
            "-c",
            "-x",
            "c++",
            "-",
            "-o",
            "/dev/null",
        ]
        res = subprocess.run(cmd, input=c_code, text=True, capture_output=True)
        self.assertNotEqual(res.returncode, 0, "Expected compilation to fail under -Werror on unused variable")
        self.assertIn("unused-variable", res.stderr.lower())

    def test_build_with_explicit_python_variable(self):
        """Verify make -C cpp succeeds when PYTHON is explicitly passed."""
        cmd = ["make", "-C", "cpp", f"PYTHON={sys.executable}", "sttt_cpp.so"]
        res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"make failed with explicit PYTHON:\n{res.stderr}\n{res.stdout}")

    def test_build_fails_with_invalid_include_path(self):
        """Verify make -C cpp fails with non-zero exit code when given an invalid PY_INCLUDE."""
        cmd = [
            "make",
            "-C",
            "cpp",
            "-B",
            "sttt_cpp.so",
            "PY_INCLUDE=/invalid/include/path/does_not_exist",
        ]
        res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0, "Build should fail when PY_INCLUDE is invalid")
        self.assertIn("Python.h", res.stderr)

        # Restore clean build
        restore_res = subprocess.run(["make", "-C", "cpp"], cwd=_REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(restore_res.returncode, 0)

    def test_setup_py_contains_werror_and_provenance_defines(self):
        """Verify setup.py includes -Werror and required provenance macros in extra_compile_args."""
        setup_py = (_REPO_ROOT / "setup.py").read_text()
        self.assertIn("-Werror", setup_py)
        self.assertIn("STTT_CPP_VERSION", setup_py)
        self.assertIn("STTT_BUILD_ID", setup_py)
        self.assertIn("STTT_SOURCE_REVISION", setup_py)
        self.assertIn("STTT_COMPILER_FLAGS", setup_py)


class TestCppProvenanceMetadataAdversarial(unittest.TestCase):
    """Verify sttt_cpp module constants and sttt.cpp_env.get_cpp_provenance() integrity."""

    def test_provenance_non_empty_strings_and_boolean(self):
        """Verify provenance fields return valid non-empty strings and boolean status."""
        prov = ce.get_cpp_provenance()
        self.assertIsInstance(prov["available"], bool)
        self.assertTrue(prov["available"])

        for key in ["version", "build_id", "source_revision", "compiler_flags", "binary_path"]:
            val = prov.get(key)
            self.assertIsInstance(val, str, f"Key '{key}' must be a string, got {type(val)}")
            self.assertGreater(len(val), 0, f"Key '{key}' must be non-empty")

        self.assertTrue(Path(prov["binary_path"]).is_file(), f"binary_path '{prov['binary_path']}' does not exist")
        self.assertTrue(prov["binary_path"].endswith(".so"), "binary_path must be a .so shared library")

    def test_sttt_cpp_module_constants_match_provenance(self):
        """Verify module constants in sttt_cpp exactly match get_cpp_provenance()."""
        import sttt_cpp

        prov = ce.get_cpp_provenance()
        self.assertEqual(sttt_cpp.__version__, prov["version"])
        self.assertEqual(sttt_cpp.VERSION, prov["version"])
        self.assertEqual(sttt_cpp.BUILD_ID, prov["build_id"])
        self.assertEqual(sttt_cpp.SOURCE_REVISION, prov["source_revision"])
        self.assertEqual(sttt_cpp.COMPILER_FLAGS, prov["compiler_flags"])
        self.assertEqual(sttt_cpp.__file__, prov["binary_path"])

    def test_build_id_format(self):
        """Verify BUILD_ID contains timestamp and git revision separator."""
        prov = ce.get_cpp_provenance()
        build_id = prov["build_id"]
        # Format: YYYYMMDDHHMMSS_<git_rev> or non-empty string
        self.assertIn("_", build_id, f"BUILD_ID '{build_id}' expected to contain '_' separator")
        parts = build_id.split("_", 1)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].isdigit(), f"Timestamp part '{parts[0]}' should be numeric digits")


class TestExtensionImportBehavior(unittest.TestCase):
    """Adversarially test extension import behavior: site-packages vs cpp/ precedence and fallback."""

    def test_site_packages_precedence_over_cpp_dir(self):
        """Verify that an extension found in site-packages takes precedence over cpp/ directory."""
        cpp_so = _REPO_ROOT / "cpp" / "sttt_cpp.so"
        if not cpp_so.is_file():
            # Check with extension suffix
            candidates = list((_REPO_ROOT / "cpp").glob("sttt_cpp*.so"))
            self.assertTrue(len(candidates) > 0, "No compiled sttt_cpp.so found in cpp/")
            cpp_so = candidates[0]

        with tempfile.TemporaryDirectory() as tmp_site:
            mock_so = Path(tmp_site) / "sttt_cpp.so"
            shutil.copy2(cpp_so, mock_so)

            code = (
                "import sys\n"
                f"sys.path.insert(0, {repr(tmp_site)})\n"
                f"sys.path.append({repr(str(_REPO_ROOT))})\n"
                "import sttt.cpp_env as ce\n"
                "prov = ce.get_cpp_provenance()\n"
                "print('BINARY_PATH:', prov['binary_path'])\n"
                f"assert {repr(tmp_site)} in prov['binary_path'], f'Expected {tmp_site} in {{prov[\"binary_path\"]}}'\n"
                "print('SITE_PACKAGES_PRECEDENCE_SUCCESS')\n"
            )
            res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Subprocess failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
            self.assertIn("SITE_PACKAGES_PRECEDENCE_SUCCESS", res.stdout)

    def test_fallback_to_cpp_when_not_in_site_packages(self):
        """Verify that when extension is not in standard sys.path, it falls back to cpp/."""
        code = (
            "import sys\n"
            f"sys.path.insert(0, {repr(str(_REPO_ROOT))})\n"
            "import sttt.cpp_env as ce\n"
            "prov = ce.get_cpp_provenance()\n"
            "print('BINARY_PATH:', prov['binary_path'])\n"
            "assert 'cpp/sttt_cpp' in prov['binary_path'], f'Expected cpp/sttt_cpp in {prov[\"binary_path\"]}'\n"
            "print('CPP_FALLBACK_SUCCESS')\n"
        )
        res = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Subprocess failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        self.assertIn("CPP_FALLBACK_SUCCESS", res.stdout)

    def test_graceful_handling_when_extension_completely_missing(self):
        """Verify graceful degradation when sttt_cpp is absent from both sys.path and cpp/."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_sttt = Path(tmp_dir) / "sttt"
            shutil.copytree(_REPO_ROOT / "sttt", tmp_sttt)
            # Empty cpp directory without any .so
            (Path(tmp_dir) / "cpp").mkdir()

            code = (
                "import sys\n"
                "import sttt.cpp_env as ce\n"
                "prov = ce.get_cpp_provenance()\n"
                "assert prov['available'] is False, 'Expected available=False'\n"
                "assert prov['version'] is None\n"
                "assert prov['build_id'] is None\n"
                "assert prov['source_revision'] is None\n"
                "assert prov['compiler_flags'] is None\n"
                "assert prov['binary_path'] is None\n"
                "assert ce.is_cpp_available() is False\n"
                "assert ce.FastState is None\n"
                "assert ce.CppTreeSearch is None\n"
                "try:\n"
                "    ce.encode_batch([])\n"
                "    assert False, 'encode_batch should have raised RuntimeError'\n"
                "except RuntimeError as e:\n"
                "    assert 'not compiled or available' in str(e)\n"
                "try:\n"
                "    ce.CppState()\n"
                "    assert False, 'CppState should have raised RuntimeError'\n"
                "except RuntimeError as e:\n"
                "    assert 'Run \\'make -C cpp\\'' in str(e)\n"
                "print('MISSING_EXTENSION_GRACEFUL_SUCCESS')\n"
            )
            res = subprocess.run([sys.executable, "-c", code], cwd=tmp_dir, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Subprocess failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
            self.assertIn("MISSING_EXTENSION_GRACEFUL_SUCCESS", res.stdout)


class TestReadinessRunnerAdversarial(unittest.TestCase):
    """Stress-test check_training_ready.py runner and isolation constraints."""

    def test_run_isolation_edge_cases(self):
        """Verify run isolation strictly blocks all attempts to target protected runs."""
        protected_targets = [
            _REPO_ROOT,
            _REPO_ROOT / "runs",
            _REPO_ROOT / "runs" / "big_run",
            _REPO_ROOT / "runs" / "big_run" / "latest.pt",
            _REPO_ROOT / "runs" / "big_run" / "nested" / "deep",
            _REPO_ROOT / "runs" / "run_v2",
            _REPO_ROOT / "runs" / "run_v2" / "models",
            _REPO_ROOT / "runs" / "run_v2" / "sub" / "dir",
        ]
        for target in protected_targets:
            with self.subTest(target=str(target)):
                with self.assertRaises(ValueError, msg=f"Should have rejected {target}"):
                    validate_output_dir(target, _REPO_ROOT)

    def test_run_isolation_allows_valid_readiness_runs(self):
        """Verify valid readiness subdirectories under runs/readiness/ are allowed."""
        valid_targets = [
            _REPO_ROOT / "runs" / "readiness" / "test_run_01",
            _REPO_ROOT / "runs" / "readiness" / "audit_suite",
        ]
        for target in valid_targets:
            validated = validate_output_dir(target, _REPO_ROOT)
            self.assertEqual(validated, target.resolve())

    def test_runner_rejects_skips_in_native_tests(self):
        """Verify that runner treats test skips in native test suites as failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "runs" / "readiness" / "skip_test"
            runner = ReadinessRunner(
                backend="cpp",
                device="cpu",
                stage="build",
                output_dir=out_dir,
            )

            # Mock test result with 1 skip
            mock_test_res = MagicMock()
            mock_test_res.testsRun = 10
            mock_test_res.failures = []
            mock_test_res.errors = []
            mock_test_res.skipped = [("dummy_test", "skipped reason")]
            mock_test_res.wasSuccessful.return_value = True

            with patch("unittest.TextTestRunner.run", return_value=mock_test_res):
                res = runner.run_stage_build()
                self.assertEqual(res["status"], "failed")
                self.assertIn("Native unit tests were skipped", res["error"])

    def test_checkpoint_freezer_detects_concurrent_mutation(self):
        """Verify freeze_checkpoint detects when source file is mutated during copy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "test.pt"
            dest_dir = Path(tmpdir) / "frozen"
            src.write_bytes(b"initial content")

            mutation_count = 0

            # Mock shutil.copy2 to simulate continuous concurrent file modification
            def mutating_copy(s, d):
                nonlocal mutation_count
                mutation_count += 1
                src.write_bytes(f"mutated_during_copy_{mutation_count}".encode())
                shutil.copyfile(s, d)

            with patch("shutil.copy2", side_effect=mutating_copy):
                with patch("time.sleep", return_value=None):
                    with self.assertRaises(RuntimeError) as cm:
                        freeze_checkpoint(src, dest_dir, max_retries=2)
                    self.assertIn("concurrently modified", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
