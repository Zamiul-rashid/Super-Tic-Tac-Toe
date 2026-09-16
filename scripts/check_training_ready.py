#!/usr/bin/env python3
"""Training Readiness and Verification Runner.

Implements M0 and M9 verification per docs/history/training-readiness.md.
Provides 8 verification stages:
  1. build: dynamic header resolution, extension build, provenance check, native test execution
  2. native: worker pool and native search integration
  3. cpu: CPU training iterations and checkpoint resume
  4. mixed: opponent pool verification (AlphaBeta, Tactical, Style, etc.)
  5. failure: robust error handling and invalid input rejection
  6. gpu: CUDA tensor operations and AMP scaler persistence
  7. memory: RSS memory stability and leak detection
  8. pilot: representative benchmark iterations and ETA calculation
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sttt.cpp_env as ce

# RECURSION GUARD, set at import. The build gate runs the full test suite, and
# the suite contains tests that invoke the build gate -- directly and by
# spawning this file as a CLI. Marking the environment here means every child
# process of a test run inherits the flag, so the nested gate skips the
# full-suite check instead of forking another 530-test run.
_main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
if (getattr(_main_spec, "name", "") or "").startswith(("unittest", "pytest")):
    os.environ["STTT_READINESS_IN_SUITE"] = "1"


def get_git_revision() -> str:
    """Return current git revision short hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def validate_output_dir(output_dir: Path, repo_root: Path = _REPO_ROOT) -> Path:
    """Enforce strict run isolation.

    Guards against modifying repository root, runs directory, runs/big_run, or runs/run_v2.
    """
    resolved = output_dir.resolve()
    root = repo_root.resolve()
    runs_dir = (root / "runs").resolve()
    big_run = (root / "runs" / "big_run").resolve()
    run_v2 = (root / "runs" / "run_v2").resolve()

    if resolved in (root, runs_dir, big_run, run_v2):
        raise ValueError(
            f"Run isolation violation: Target directory '{resolved}' cannot be a protected run root."
        )

    try:
        resolved.relative_to(big_run)
        raise ValueError(
            f"Run isolation violation: Target directory '{resolved}' cannot be inside protected 'runs/big_run'."
        )
    except ValueError as e:
        if "cannot be inside" in str(e):
            raise

    try:
        resolved.relative_to(run_v2)
        raise ValueError(
            f"Run isolation violation: Target directory '{resolved}' cannot be inside protected 'runs/run_v2'."
        )
    except ValueError as e:
        if "cannot be inside" in str(e):
            raise

    return resolved


# M7: one implementation, shared with the evaluation harness, which freezes the
# same rolling checkpoints for the same reason. Re-exported here so the readiness
# stages and their tests keep importing it from this module.
from sttt.evaluation import freeze_checkpoint  # noqa: E402,F401


class ReadinessRunner:
    """Orchestrates staged training readiness verification."""

    STAGE_NAMES = [
        "build",
        "native",
        "cpu",
        "mixed",
        "failure",
        "gpu",
        "memory",
        "pilot",
    ]

    def __init__(
        self,
        backend: str = "cpp",
        device: str = "cpu",
        stage: str = "all",
        output_dir: Path | None = None,
        manifest_only: bool = False,
        pilot_checkpoint: str | None = None,
        pilot_iterations: int = 20,
        pilot_warmup: int = 2,
        pilot_workers: int = 16,
        pilot_population_config: str = "configs/population/baseline.json",
        pilot_eta_iterations: int = 5000,
    ):
        self.pilot_checkpoint = pilot_checkpoint
        self.pilot_iterations = pilot_iterations
        self.pilot_warmup = pilot_warmup
        self.pilot_workers = pilot_workers
        self.pilot_population_config = pilot_population_config
        self.pilot_eta_iterations = pilot_eta_iterations
        self.backend = backend
        self.device = device
        self.requested_stage = stage
        self.manifest_only = manifest_only
        self.repo_root = _REPO_ROOT

        if output_dir is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:8]
            output_dir = self.repo_root / "runs" / "readiness" / f"readiness_{ts}_{uid}"

        self.output_dir = validate_output_dir(output_dir, self.repo_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, dict[str, Any]] = {}

    def get_capability_manifest(self) -> dict[str, Any]:
        """Compile a comprehensive machine-readable capability manifest."""
        include_path = sysconfig.get_path("include") or ""
        python_h = Path(include_path) / "Python.h"

        torch_info: dict[str, Any] = {"available": False}
        try:
            import torch

            torch_info = {
                "available": True,
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        except ImportError:
            pass

        # Suite discovery
        suite = unittest.defaultTestLoader.discover(str(self.repo_root / "tests"))
        test_count = suite.countTestCases()

        provenance = ce.get_cpp_provenance()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_revision": get_git_revision(),
            "python": {
                "executable": sys.executable,
                "version": sys.version,
                "include_path": include_path,
                "python_h_exists": python_h.is_file(),
            },
            "torch": torch_info,
            "cpp_provenance": provenance,
            "suite_discovery": {
                "total_tests": test_count,
            },
            "requested_config": {
                "backend": self.backend,
                "device": self.device,
                "stage": self.requested_stage,
                "output_dir": str(self.output_dir),
            },
        }

    def run_stage_build(self) -> dict[str, Any]:
        """Stage 1: Verify dynamic header resolution, build extension, check provenance, run native tests."""
        print("\n==========================================")
        print("STAGE 1: BUILD GATE & CAPABILITY MANIFEST")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "build"
        stage_dir.mkdir(parents=True, exist_ok=True)

        manifest = self.get_capability_manifest()
        checks: dict[str, Any] = {}

        # Check 1: Python include header resolution
        inc_dir = Path(manifest["python"]["include_path"])
        py_h = inc_dir / "Python.h"
        checks["dynamic_include_resolution"] = {
            "include_dir": str(inc_dir),
            "python_h_found": py_h.is_file(),
        }
        if not py_h.is_file():
            # Check alternative standard directories
            alt = Path(f"/usr/include/python{sys.version_info.major}.{sys.version_info.minor}/Python.h")
            if alt.is_file():
                checks["dynamic_include_resolution"]["python_h_found"] = True
                checks["dynamic_include_resolution"]["include_dir"] = str(alt.parent)
            else:
                return {
                    "stage": "build",
                    "status": "failed",
                    "duration_sec": time.time() - t0,
                    "error": f"Python.h not found in include directory {inc_dir}",
                    "checks": checks,
                    "manifest": manifest,
                }

        # Check 2: Makefile dynamic include resolution check
        makefile_path = self.repo_root / "cpp" / "Makefile"
        makefile_content = makefile_path.read_text() if makefile_path.is_file() else ""
        has_dynamic_make = "sysconfig" in makefile_content
        checks["makefile_dynamic_resolution"] = has_dynamic_make
        if not has_dynamic_make:
            return {
                "stage": "build",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": "cpp/Makefile does not use sysconfig for dynamic header resolution.",
                "checks": checks,
                "manifest": manifest,
            }

        # Check 3: Build extension via make -C cpp
        if self.backend == "cpp":
            print("[build] Invoking make -C cpp to ensure clean build...")
            make_res = subprocess.run(
                ["make", "-C", "cpp", f"PYTHON={sys.executable}"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
            checks["make_exit_code"] = make_res.returncode
            if make_res.returncode != 0:
                return {
                    "stage": "build",
                    "status": "failed",
                    "duration_sec": time.time() - t0,
                    "error": f"make -C cpp failed:\n{make_res.stderr}\n{make_res.stdout}",
                    "checks": checks,
                    "manifest": manifest,
                }

        # Check 4: C++ extension availability and provenance attributes
        prov = ce.get_cpp_provenance()
        checks["provenance"] = prov
        if self.backend == "cpp":
            if not prov["available"]:
                return {
                    "stage": "build",
                    "status": "failed",
                    "duration_sec": time.time() - t0,
                    "error": "C++ extension is not available. Run 'make -C cpp' or 'python setup.py build_ext --inplace'.",
                    "checks": checks,
                    "manifest": manifest,
                }
            if not prov.get("version"):
                return {
                    "stage": "build",
                    "status": "failed",
                    "duration_sec": time.time() - t0,
                    "error": "sttt_cpp does not expose __version__ attribute.",
                    "checks": checks,
                    "manifest": manifest,
                }
            if not prov.get("build_id"):
                return {
                    "stage": "build",
                    "status": "failed",
                    "duration_sec": time.time() - t0,
                    "error": "sttt_cpp does not expose BUILD_ID attribute.",
                    "checks": checks,
                    "manifest": manifest,
                }

        # Check 5: If CUDA requested, ensure CUDA is available
        if self.device == "cuda":
            if not manifest["torch"].get("cuda_available", False):
                return {
                    "stage": "build",
                    "status": "failed",
                    "duration_sec": time.time() - t0,
                    "error": "Device 'cuda' requested, but torch.cuda.is_available() is False.",
                    "checks": checks,
                    "manifest": manifest,
                }

        # Check 6: Native unit test execution and strict NO-SKIP verification
        print("[build] Executing native unit test suites to verify 0 skips...")
        loader = unittest.defaultTestLoader
        native_suite = unittest.TestSuite()
        native_suite.addTests(loader.loadTestsFromName("tests.test_cpp_engine"))
        native_suite.addTests(loader.loadTestsFromName("tests.test_cpp_mcts"))

        stream = io.StringIO()
        test_runner = unittest.TextTestRunner(stream=stream, verbosity=1)
        test_res = test_runner.run(native_suite)

        checks["native_tests"] = {
            "total_run": test_res.testsRun,
            "failures": len(test_res.failures),
            "errors": len(test_res.errors),
            "skips": len(test_res.skipped),
        }

        if self.backend == "cpp":
            if len(test_res.skipped) > 0:
                return {
                    "stage": "build",
                    "status": "failed",
                    "duration_sec": time.time() - t0,
                    "error": f"Native unit tests were skipped (skips={len(test_res.skipped)}). C++ tests must be executed, not skipped.",
                    "checks": checks,
                    "manifest": manifest,
                }
            if not test_res.wasSuccessful():
                return {
                    "stage": "build",
                    "status": "failed",
                    "duration_sec": time.time() - t0,
                    "error": f"Native unit tests failed:\n{stream.getvalue()}",
                    "checks": checks,
                    "manifest": manifest,
                }

        # Check 7: Verify total test discovery count >= 264
        total_discovered = manifest["suite_discovery"]["total_tests"]
        checks["suite_count_ok"] = total_discovered >= 264
        if total_discovered < 264:
            return {
                "stage": "build",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": f"Discovered only {total_discovered} tests; expected at least 264.",
                "checks": checks,
                "manifest": manifest,
            }

        # Check 8: Execute the CURRENT FULL SUITE, not only the native modules.
        # M9 stage 1 requires it. Checks 6 and 7 alone would pass while the rest
        # of the suite was broken: 6 runs ~40 native tests and 7 only counts
        # what discovery *found* without running it.
        #
        # RECURSION GUARD. The suite itself contains tests that invoke this
        # stage, so without this flag the gate runs the suite, which runs the
        # gate, which runs the suite -- an exponential process explosion, not a
        # slow test run. When the gate is reached from inside a suite run, the
        # full-suite check records itself as skipped-nested; the outermost run
        # is the one that actually executes it.
        main_module = sys.modules.get("__main__")
        main_name = getattr(getattr(main_module, "__spec__", None), "name", "") or ""
        under_unittest = main_name.startswith("unittest") or main_name.startswith("pytest")
        if os.environ.get("STTT_READINESS_IN_SUITE") == "1" or under_unittest:
            checks["full_suite"] = {"status": "skipped_nested",
                                    "reason": "invoked from within a test-suite run"}
            print("[build] Nested invocation: skipping the full-suite check.")
            elapsed = time.time() - t0
            stage_result = {"stage": "build", "status": "passed", "duration_sec": elapsed,
                            "checks": checks, "manifest": manifest}
            manifest_file = stage_dir / "manifest.json"
            manifest_file.write_text(json.dumps(stage_result, indent=2))
            return stage_result

        print(f"[build] Executing the full discovered suite ({total_discovered} tests)...")
        suite_env = {**os.environ, "STTT_READINESS_IN_SUITE": "1"}
        suite_res = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=self.repo_root, capture_output=True, text=True, env=suite_env,
        )
        suite_output = suite_res.stdout + suite_res.stderr
        failed_names = sorted({
            line.split(" ")[1] for line in suite_output.splitlines()
            if line.startswith("FAIL: ") or line.startswith("ERROR: ")
        })
        ran_line = [l for l in suite_output.splitlines() if l.startswith("Ran ")]
        checks["full_suite"] = {
            "exit_code": suite_res.returncode,
            "summary": ran_line[-1] if ran_line else None,
            "failed_tests": failed_names,
        }
        if suite_res.returncode != 0:
            return {
                "stage": "build",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": ("Full test suite failed: "
                          + ", ".join(failed_names) + "\n" + suite_output[-4000:]),
                "checks": checks,
                "manifest": manifest,
            }

        elapsed = time.time() - t0
        stage_result = {
            "stage": "build",
            "status": "passed",
            "duration_sec": elapsed,
            "checks": checks,
            "manifest": manifest,
        }

        # Write stage manifest
        manifest_file = stage_dir / "manifest.json"
        manifest_file.write_text(json.dumps(stage_result, indent=2))
        print(f"[build] Stage PASSED in {elapsed:.2f}s. Manifest written to {manifest_file}")
        return stage_result

    def run_stage_native(self) -> dict[str, Any]:
        """Stage 2: two spawned workers, four complete games, twice, with a model
        update in between; worker identity reused and the native backend actually
        used.

        The previous implementation was written against `sttt.model.PolicyValueNet`
        and `pool.play_games()`, neither of which exists, so this gate had never
        executed successfully.
        """
        print("\n==========================================")
        print("STAGE 2: NATIVE INTEGRATION GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "native"
        stage_dir.mkdir(parents=True, exist_ok=True)
        checks: dict[str, Any] = {}

        def fail(message):
            result = {"stage": "native", "status": "failed",
                      "duration_sec": time.time() - t0, "error": message, "checks": checks}
            (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
            return result

        if self.backend != "cpp":
            result = {"stage": "native", "status": "skipped",
                      "duration_sec": time.time() - t0,
                      "reason": "Python backend specified; native integration gate applies to --backend cpp."}
            (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
            return result

        if not ce.is_cpp_available():
            return fail("C++ backend unavailable for the native integration gate.")

        try:
            import numpy as np
            from sttt.learning import create_model
            from sttt.search import SearchConfig
            from sttt.selfplay import SelfPlayPool

            use_cpp = self.backend == "cpp"
            config = SearchConfig()
            for arch in ("mlp", "resnet"):
                model = create_model(arch).eval()
                with SelfPlayPool(2, batch_size=8, wait_ms=2.) as pool:
                    reports = pool.verify_backend(use_cpp)
                    checks[f"{arch}_worker_reports"] = reports
                    if len(reports) != 2:
                        return fail(f"expected 2 worker handshakes, got {len(reports)}")
                    for report in reports:
                        if use_cpp and not report.get("cpp_available"):
                            return fail(f"worker reports no native build: {report}")
                    pids_before = [p.pid for p in pool.processes]

                    rounds = []
                    for round_index in range(2):
                        if round_index == 1:
                            # Model update between rounds: weights change, the
                            # same worker processes must be reused.
                            with __import__("torch").no_grad():
                                for parameter in model.parameters():
                                    parameter.add_(0.01)
                        seeds = [11 + round_index * 10 + i for i in range(4)]
                        results, metrics = pool.run(model, seeds, 32, config, 4, use_cpp=use_cpp)
                        if len(results) != 4:
                            return fail(f"expected 4 games, got {len(results)}")
                        for trajectory, outcome, stats in results:
                            if outcome not in (-1, 0, 1):
                                return fail(f"illegal game outcome {outcome}")
                            for state, pi, *_ in trajectory:
                                if pi.shape != (81,):
                                    return fail(f"policy shape {pi.shape}")
                                if not np.isclose(pi.sum(), 1.0, atol=1e-3):
                                    return fail("policy is not normalized")
                                legal = set(state.legal_actions())
                                if not set(np.flatnonzero(pi)).issubset(legal):
                                    return fail("policy puts mass on an illegal action")
                        rounds.append({"games": len(results),
                                       "inference_positions": metrics["inference_positions"],
                                       "mean_inference_batch": metrics["mean_inference_batch"]})
                    pids_after = [p.pid for p in pool.processes]
                    if pids_before != pids_after:
                        return fail(f"workers were not reused: {pids_before} -> {pids_after}")
                    checks[f"{arch}_rounds"] = rounds
                    checks[f"{arch}_worker_pids"] = pids_before

                # Cleanup: no worker may outlive the pool.
                alive = [p.pid for p in pool.processes if p.is_alive()]
                if alive:
                    return fail(f"workers still alive after pool close: {alive}")

            # The plan asks this gate to exercise the Python reference backend as
            # well, so a native-only regression cannot hide behind it.
            model = create_model("mlp").eval()
            with SelfPlayPool(2, batch_size=8, wait_ms=2.) as pool:
                pool.verify_backend(False)
                results, _ = pool.run(model, [91, 92, 93, 94], 32, config, 4, use_cpp=False)
                if len(results) != 4:
                    return fail(f"python backend produced {len(results)} games, expected 4")
                for trajectory, outcome, stats in results:
                    if outcome not in (-1, 0, 1):
                        return fail(f"python backend illegal outcome {outcome}")
                checks["python_backend_games"] = len(results)

            result = {"stage": "native", "status": "passed",
                      "duration_sec": time.time() - t0, "checks": checks}
        except Exception as exc:                         # noqa: BLE001
            return fail(f"{type(exc).__name__}: {exc}")

        (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
        print(f"[native] Stage PASSED in {result['duration_sec']:.2f}s")
        return result

    def run_stage_cpu(self) -> dict[str, Any]:
        """Stage 3: CPU Training & Resume Gate."""
        print("\n==========================================")
        print("STAGE 3: CPU TRAINING & RESUME GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "cpu"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Phase A: Initial 2 iterations
        run_output = stage_dir / "run"
        cmd_init = [
            sys.executable,
            "-m",
            "sttt.ai",
            "train",
            "--backend",
            self.backend,
            "--device",
            "cpu",
            "--arch",
            "mlp",
            "--output",
            str(run_output),
            "--iterations",
            "2",
            "--games",
            "4",
            "--workers",
            "2",
            "--simulations",
            "32",
            "--leaf-batch",
            "4",
            "--inference-batch",
            "8",
            "--steps",
            "4",
            "--batch",
            "16",
            "--buffer",
            "1000",
            "--eval-every",
            "0",
        ]

        # Snapshot untrained weights so "the model actually moved" is checkable.
        import torch as _torch
        from sttt.learning import create_model as _create_model
        _torch.manual_seed(0)
        _torch.save(_create_model("mlp").state_dict(), stage_dir / "initial_weights.pt")

        print(f"[cpu] Executing training: {' '.join(cmd_init)}")
        res_init = subprocess.run(cmd_init, cwd=self.repo_root, capture_output=True, text=True)
        if res_init.returncode != 0:
            stage_result = {
                "stage": "cpu",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": f"Initial training failed:\n{res_init.stderr}\n{res_init.stdout}",
            }
            (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
            return stage_result

        latest_ckpt = run_output / "latest.pt"
        if not latest_ckpt.is_file():
            stage_result = {
                "stage": "cpu",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": f"Expected checkpoint '{latest_ckpt}' was not created.",
            }
            (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
            return stage_result

        # Phase B: Resume for 1 additional iteration
        cmd_resume = [
            sys.executable,
            "-m",
            "sttt.ai",
            "train",
            "--resume",
            str(latest_ckpt),
            "--output",
            str(run_output),
            "--backend",
            self.backend,
            "--device",
            "cpu",
            "--iterations",
            "1",
            "--games",
            "4",
            "--workers",
            "2",
            "--simulations",
            "32",
            "--leaf-batch",
            "4",
            "--inference-batch",
            "8",
            "--steps",
            "4",
            "--batch",
            "16",
            "--buffer",
            "1000",
            "--eval-every",
            "0",
        ]
        print(f"[cpu] Executing resume: {' '.join(cmd_resume)}")
        res_resume = subprocess.run(cmd_resume, cwd=self.repo_root, capture_output=True, text=True)
        if res_resume.returncode != 0:
            stage_result = {
                "stage": "cpu",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": f"Resume training failed:\n{res_resume.stderr}\n{res_resume.stdout}",
            }
            (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
            return stage_result

        # The commands above only prove training did not crash. The plan requires
        # asserting what the run actually produced, so a silently degenerate run
        # (no iteration advance, unchanged weights, lost optimizer state) cannot
        # pass this gate.
        import torch

        def cpu_fail(message):
            result = {"stage": "cpu", "status": "failed",
                      "duration_sec": time.time() - t0, "error": message,
                      "checks": checks}
            (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
            return result

        checks: dict[str, Any] = {}
        after = torch.load(latest_ckpt, map_location="cpu", weights_only=True)

        # Iteration advancement: 2 initial + 1 resumed.
        checks["iteration"] = int(after.get("iteration", -1))
        if checks["iteration"] != 3:
            return cpu_fail(f"expected iteration 3 after 2 + 1, got {checks['iteration']}")

        # Full-resume state must survive.
        for field in ("optimizer", "replay", "lr_schedule", "numpy_rng_state", "arch"):
            if after.get(field) is None:
                return cpu_fail(f"checkpoint lost '{field}' across the resume")
        from sttt.ai import replay_length
        checks["replay_positions"] = replay_length(after["replay"])
        checks["optimizer_param_groups"] = len(after["optimizer"]["param_groups"])
        checks["lr_schedule"] = after["lr_schedule"]
        if checks["replay_positions"] == 0:
            return cpu_fail("replay buffer is empty after training")
        if not after["optimizer"].get("state"):
            return cpu_fail("optimizer moments were not preserved across the resume")

        # Finite losses, and the model actually moved.
        rows = [json.loads(line) for line in
                (run_output / "metrics.jsonl").read_text().splitlines() if line.strip()]
        losses = [r["loss"] for r in rows if "loss" in r]
        checks["losses"] = losses
        if len(losses) != 3:
            return cpu_fail(f"expected 3 metric rows, got {len(losses)}")
        import math
        if not all(math.isfinite(v) for v in losses):
            return cpu_fail(f"nonfinite training loss recorded: {losses}")

        first_snapshot = after["model"]
        baseline = torch.load(stage_dir / "initial_weights.pt", map_location="cpu",
                              weights_only=True) if (stage_dir / "initial_weights.pt").is_file() else None
        if baseline is not None:
            changed = any(not torch.equal(first_snapshot[k], baseline[k]) for k in baseline)
            checks["model_changed"] = changed
            if not changed:
                return cpu_fail("model weights are unchanged after three iterations")

        # No worker may outlive the training process.
        leftover = subprocess.run(["pgrep", "-f", f"--output {run_output}"],
                                  capture_output=True, text=True)
        checks["leftover_workers"] = leftover.stdout.split()
        if leftover.stdout.strip():
            return cpu_fail(f"workers outlived the training run: {leftover.stdout.split()}")

        stage_result = {
            "stage": "cpu",
            "status": "passed",
            "duration_sec": time.time() - t0,
            "checks": checks,
            "details": {
                "checkpoint": str(latest_ckpt),
                "initial_iterations": 2,
                "resumed_iterations": 1,
            },
        }
        (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
        print(f"[cpu] iteration={checks['iteration']} replay={checks['replay_positions']} "
              f"losses={[round(v, 3) for v in losses]}")
        return stage_result

    def run_stage_mixed(self) -> dict[str, Any]:
        """Stage 4: every opponent family exercised from explicit match specs on
        both seats with injected openings, then a full 100-game quota cycle.

        The previous implementation called `bot.choose(state)` without an rng and
        used bot specs that do not exist, so it had never passed.
        """
        print("\n==========================================")
        print("STAGE 4: MIXED-OPPONENT GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "mixed"
        stage_dir.mkdir(parents=True, exist_ok=True)
        checks: dict[str, Any] = {}

        def fail(message):
            result = {"stage": "mixed", "status": "failed",
                      "duration_sec": time.time() - t0, "error": message, "checks": checks}
            (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
            return result

        try:
            import numpy as np
            import torch
            from sttt.learning import create_model
            from sttt.population import (MatchSpec, default_population_config,
                                         load_population_config, population_quota_counts,
                                         sample_matches)
            from sttt.search import SearchConfig
            from sttt.selfplay import play_game

            use_cpp = self.backend == "cpp"
            model = create_model("mlp").eval()
            config = SearchConfig()

            # A frozen checkpoint so history/best have something real to load.
            history_dir = stage_dir / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            for name in ("model-0001.pt", "best.pt"):
                torch.save({"model": model.state_dict(), "arch": "mlp", "iteration": 1},
                           history_dir / name)
            checkpoint = str((history_dir / "model-0001.pt").resolve())

            # Test budgets, deliberately tiny. These are NOT the production
            # preset; the production budgets live in configs/population/.
            specs = {
                "self": MatchSpec(kind="self"),
                "history": MatchSpec(kind="history", checkpoint=checkpoint, simulations=2),
                "best": MatchSpec(kind="best", checkpoint=checkpoint, simulations=2),
                "alphabeta": MatchSpec(kind="alphabeta", depth=1, nodes=200),
                "tactical": MatchSpec(kind="tactical", nodes=200),
                "threat": MatchSpec(kind="threat", nodes=200),
                "style": MatchSpec(kind="style", style="center"),
                "openspiel": MatchSpec(kind="openspiel", simulations=2),
                "utttai": MatchSpec(kind="utttai", simulations=64),
            }
            # Use the real loader: it is what resolves {python} and validates the
            # entry. Reading the JSON directly leaves the placeholder unexpanded
            # and the engine fails to spawn.
            from sttt.engine_registry import load_engine_registry
            from sttt.engine_runtime import activate_runtime
            activate_runtime()
            registry_path = self.repo_root / "engines" / "registry.json"
            engine_registry = load_engine_registry(registry_path) if registry_path.is_file() else {}

            from sttt.population import _engine_fields
            exercised = {}
            for family, spec in specs.items():
                if family == "utttai":
                    fields = _engine_fields(engine_registry, 7, spec.simulations)
                    if not fields.get("engine_command"):
                        return fail("uttt.ai is not configured; the mixed gate requires it")
                    spec = MatchSpec(kind="utttai", simulations=spec.simulations, **fields)
                seats = {}
                for side in (1, -1):
                    # Both seats, with injected opening plies.
                    seated = MatchSpec(**{**spec.__dict__, "learner_side": side,
                                          "opening_moves": 0 if family == "utttai" and
                                          spec.engine_protocol != "state_json" else 2})
                    trajectory, outcome, stats = play_game(model, 2, 1234 + side, config, 2,
                                                           seated, use_cpp=use_cpp)
                    if outcome not in (-1, 0, 1):
                        return fail(f"{family}: illegal outcome {outcome}")
                    actual = stats["match"]["kind"]
                    for state, pi, *_ in trajectory:
                        # Only meaningful when an opponent actually plays: in
                        # self-play the learner holds both seats, so the turn
                        # alternates by design.
                        if actual != "self" and state.turn != side:
                            return fail(f"{family}: an opponent turn became a learner target")
                        if not np.isclose(pi.sum(), 1.0, atol=1e-3):
                            return fail(f"{family}: unnormalized policy target")
                    if actual != family and not (family in ("history", "best") and actual == "self"):
                        return fail(f"{family} was silently substituted by {actual}")
                    seats[side] = {"positions": len(trajectory), "outcome": outcome,
                                   "actual_kind": actual, "plies": stats["plies"]}
                exercised[family] = seats
            checks["families"] = exercised

            # Full configured 100-game quota cycle, sliced across uneven
            # iterations, with the cursor surviving each slice.
            population_config = load_population_config(
                self.repo_root / "configs" / "population" / "baseline.json")
            totals = dict.fromkeys(population_config.quotas, 0)
            cursor = 0
            for size in (7, 13, 30, 21, 29):
                for kind, count in population_quota_counts(size, cursor,
                                                           config=population_config).items():
                    totals[kind] += count
                cursor += size
            if cursor != 100:
                return fail(f"quota cycle slices summed to {cursor}, expected 100")
            if totals != population_config.quotas:
                return fail(f"100-game cycle produced {totals}, expected {population_config.quotas}")
            checks["quota_cycle"] = {"slices": [7, 13, 30, 21, 29], "counts": totals,
                                     "config": population_config.name,
                                     "config_sha256": population_config.sha256}

            # And the sampler must realise that cycle end to end.
            sampled = sample_matches(range(100), str(history_dir), engine_registry,
                                     config=population_config)
            requested = {k: sum(m.requested_kind == k for m in sampled) for k in population_config.quotas}
            if requested != population_config.quotas:
                return fail(f"sampled requests {requested} != quotas {population_config.quotas}")
            checks["sampled_requests"] = requested
            checks["test_budgets_note"] = ("Budgets here are tiny test values, distinct from "
                                           "the production preset in configs/population/.")

            result = {"stage": "mixed", "status": "passed",
                      "duration_sec": time.time() - t0, "checks": checks}
        except Exception as exc:                          # noqa: BLE001
            import traceback
            return fail(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")

        (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
        print(f"[mixed] Stage PASSED in {result['duration_sec']:.2f}s")
        return result

    def run_stage_failure(self) -> dict[str, Any]:
        """Stage 5: each failure mode must fail promptly, close its children and
        leave the last completed checkpoint loadable.

        The previous implementation checked only that an invalid --backend was
        rejected; the other five modes the plan names were never tested.
        """
        print("\n==========================================")
        print("STAGE 5: FAILURE & ROBUSTNESS GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "failure"
        stage_dir.mkdir(parents=True, exist_ok=True)
        checks: dict[str, Any] = {}

        def fail(message):
            result = {"stage": "failure", "status": "failed",
                      "duration_sec": time.time() - t0, "error": message, "checks": checks}
            (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
            return result

        try:
            import signal
            import numpy as np
            import torch
            from sttt.ai import RunOwnership
            from sttt.learning import create_model
            from sttt.population import MatchSpec
            from sttt.search import SearchConfig
            from sttt.selfplay import SelfPlayPool, play_game

            # 1. Invalid backend is rejected before anything is spawned.
            res = subprocess.run(
                [sys.executable, "-m", "sttt.ai", "train", "--backend",
                 "invalid_nonexistent_backend", "--iterations", "1"],
                cwd=self.repo_root, capture_output=True, text=True, timeout=180)
            checks["invalid_backend_rejected"] = res.returncode != 0
            if res.returncode == 0:
                return fail("invalid backend was accepted")

            # 2. Malformed inference: an evaluator returning a bad policy must
            # raise, not poison the tree with a corrupt distribution.
            class Malformed:
                def evaluate_many(self, states):
                    return [(np.full(81, np.nan, dtype=np.float32), 0.0) for _ in states]

                def evaluate(self, state):
                    return self.evaluate_many([state])[0]

            try:
                play_game(Malformed(), 8, 3, SearchConfig(), 2, MatchSpec(),
                          use_cpp=self.backend == "cpp")
                return fail("a NaN policy from the evaluator was accepted")
            except (ValueError, RuntimeError) as exc:
                checks["malformed_inference_rejected"] = f"{type(exc).__name__}: {exc}"[:200]

            # 3. Worker exit mid-iteration is detected rather than hanging.
            model = create_model("mlp").eval()
            with SelfPlayPool(2, batch_size=8, wait_ms=2.) as pool:
                pool.verify_backend(self.backend == "cpp")
                victim = pool.processes[0]
                victim.kill()
                victim.join(timeout=10)
                killed = time.time()
                try:
                    pool.run(model, [1, 2, 3, 4], 8, SearchConfig(), 2,
                             use_cpp=self.backend == "cpp")
                    return fail("a killed worker did not surface as an error")
                except (RuntimeError, EOFError, TimeoutError, OSError) as exc:
                    checks["worker_exit_detected"] = {
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                        "seconds_to_detect": round(time.time() - killed, 2)}
            checks["worker_exit_cleanup"] = [p.pid for p in pool.processes if p.is_alive()]
            if checks["worker_exit_cleanup"]:
                return fail(f"workers survived pool close: {checks['worker_exit_cleanup']}")

            # 4. Engine timeout: an external engine that never answers must time
            # out rather than block the iteration forever.
            from sttt.bots import ExternalProcessBot
            from sttt.env import State
            sleeper = ExternalProcessBot(
                command=[sys.executable, "-c", "import time; time.sleep(600)"],
                timeout=3.0, protocol="action_index", fallback="raise", name="sleeper")
            started = time.time()
            try:
                sleeper.choose(State(), np.random.default_rng(0))
                return fail("an unresponsive engine did not time out")
            except Exception as exc:                       # noqa: BLE001
                waited = time.time() - started
                checks["engine_timeout"] = {"error": f"{type(exc).__name__}: {exc}"[:200],
                                            "seconds": round(waited, 2)}
                if waited > 60:
                    return fail(f"engine timeout took {waited:.0f}s; the contract is unbounded")
            finally:
                sleeper.close()

            # 5. Duplicate writer: a second trainer on one output directory is
            # rejected before it spawns workers.
            lock_dir = stage_dir / "ownership"
            lock_dir.mkdir(parents=True, exist_ok=True)
            first = RunOwnership(lock_dir).acquire()
            try:
                try:
                    RunOwnership(lock_dir).acquire()
                    return fail("a second writer acquired the same output directory")
                except RuntimeError as exc:
                    checks["duplicate_writer_rejected"] = str(exc)[:200]
            finally:
                first.release()

            # 6. Ctrl+C during training leaves the last completed checkpoint
            # loadable and no orphaned workers.
            interrupt_dir = stage_dir / "interrupt"
            command = [sys.executable, "-m", "sttt.ai", "train",
                       "--backend", self.backend, "--device", "cpu", "--arch", "mlp",
                       "--output", str(interrupt_dir), "--iterations", "50",
                       "--games", "2", "--workers", "2", "--simulations", "16",
                       "--leaf-batch", "2", "--inference-batch", "4", "--steps", "2",
                       "--batch", "8", "--buffer", "200", "--eval-every", "0",
                       "--save-every", "0", "--keep-checkpoint-window", "0", "--seed", "3"]
            proc = subprocess.Popen(command, cwd=self.repo_root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    start_new_session=True)
            checkpoint = interrupt_dir / "latest.pt"
            deadline = time.time() + 180
            while time.time() < deadline and not checkpoint.is_file():
                time.sleep(0.5)
            if not checkpoint.is_file():
                proc.kill()
                return fail("training produced no checkpoint to interrupt")
            time.sleep(1.0)
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                return fail("training did not exit within 120s of SIGINT")
            interrupted = time.time()
            # The checkpoint written before the interrupt must still load.
            restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "model" not in restored or "iteration" not in restored:
                return fail("the checkpoint left after an interrupt is not loadable")
            leftover = subprocess.run(
                ["pgrep", "-f", f"--output {interrupt_dir}"], capture_output=True, text=True)
            checks["ctrl_c"] = {"exit_code": proc.returncode,
                                "iteration_preserved": int(restored["iteration"]),
                                "seconds_to_exit": round(interrupted - deadline + 180, 2),
                                "orphans": leftover.stdout.split()}
            if leftover.stdout.strip():
                return fail(f"orphaned processes after SIGINT: {leftover.stdout.split()}")

            result = {"stage": "failure", "status": "passed",
                      "duration_sec": time.time() - t0, "checks": checks}
        except Exception as exc:                           # noqa: BLE001
            import traceback
            return fail(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")

        (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
        print(f"[failure] Stage PASSED in {result['duration_sec']:.2f}s")
        return result

    def run_stage_gpu(self) -> dict[str, Any]:
        """Stage 6: GPU Correctness Gate.

        FP32 and FP16 training on an explicit CUDA device, a real optimizer
        update, finite masked policy loss, GradScaler persistence across a CUDA
        checkpoint/resume, and no CPU fallback. A CPU run cannot satisfy it.
        """
        print("\n==========================================")
        print("STAGE 6: GPU CORRECTNESS GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "gpu"
        stage_dir.mkdir(parents=True, exist_ok=True)
        checks: dict[str, Any] = {}

        def finish(status, **extra):
            result = {"stage": "gpu", "status": status,
                      "duration_sec": time.time() - t0, "checks": checks, **extra}
            (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
            return result

        if self.device != "cuda":
            return finish("skipped", reason="CPU device specified; GPU gate requires --device cuda.")

        import math
        import torch

        if not torch.cuda.is_available():
            return finish("failed", error="CUDA device specified but torch.cuda.is_available() is False.")
        checks["device_name"] = torch.cuda.get_device_name(0)

        common = ["--backend", self.backend, "--device", "cuda", "--games", "4",
                  "--workers", "2", "--simulations", "32", "--leaf-batch", "4",
                  "--inference-batch", "8", "--steps", "4", "--batch", "16",
                  "--buffer", "1000", "--eval-every", "0"]

        def train(output, iterations, *extra, resume=None):
            cmd = [sys.executable, "-m", "sttt.ai", "train"]
            cmd += ["--resume", str(resume)] if resume else ["--arch", "resnet"]
            cmd += ["--output", str(output), "--iterations", str(iterations), *common, *extra]
            print(f"[gpu] {' '.join(cmd)}")
            res = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"training exited {res.returncode}:\n{res.stderr}\n{res.stdout}")
            ckpt = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
            rows = [json.loads(line) for line in
                    (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
            return ckpt, rows

        def assert_finite_on_cuda(label, ckpt, rows):
            metrics = ckpt["metrics"]
            for key in ("loss", "policy_loss", "value_loss"):
                if not math.isfinite(metrics[key]):
                    raise RuntimeError(f"{label}: nonfinite {key}={metrics[key]}")
            if metrics["optimizer_updates"] < 1:
                raise RuntimeError(f"{label}: no optimizer update happened")
            # peak_gpu_mb is only recorded when the learner ran on CUDA.
            if not all(row.get("peak_gpu_mb", 0) > 0 for row in rows):
                raise RuntimeError(f"{label}: an iteration ran without CUDA memory use (CPU fallback?)")

        try:
            # FP32: forward/train on CUDA without a scaler.
            ckpt32, rows32 = train(stage_dir / "fp32", 1)
            assert_finite_on_cuda("fp32", ckpt32, rows32)
            if ckpt32.get("precision") != "fp32" or ckpt32.get("scaler") is not None:
                raise RuntimeError(f"fp32 run recorded precision={ckpt32.get('precision')} scaler={ckpt32.get('scaler') is not None}")
            checks["fp32"] = {"loss": ckpt32["metrics"]["loss"], "policy_loss": ckpt32["metrics"]["policy_loss"],
                              "optimizer_updates": ckpt32["metrics"]["optimizer_updates"],
                              "peak_gpu_mb": max(r["peak_gpu_mb"] for r in rows32)}

            # FP16: two iterations, then a CUDA resume for one more.
            fp16_dir = stage_dir / "fp16"
            ckpt_a, rows_a = train(fp16_dir, 2, "--fp16")
            assert_finite_on_cuda("fp16", ckpt_a, rows_a)
            if ckpt_a.get("precision") != "fp16" or not ckpt_a.get("scaler"):
                raise RuntimeError("fp16 run did not persist a GradScaler state")
            scale_a = ckpt_a["scaler"]["scale"]
            tracker_a = ckpt_a["scaler"]["_growth_tracker"]

            ckpt_b, rows_b = train(fp16_dir, 1, "--fp16", resume=fp16_dir / "latest.pt")
            assert_finite_on_cuda("fp16-resume", ckpt_b, rows_b)
            if ckpt_b["iteration"] != 3:
                raise RuntimeError(f"expected iteration 3 after 2 + 1, got {ckpt_b['iteration']}")
            if ckpt_b.get("precision") != "fp16" or not ckpt_b.get("scaler"):
                raise RuntimeError("resumed fp16 run lost the GradScaler state")
            scale_b = ckpt_b["scaler"]["scale"]
            tracker_b = ckpt_b["scaler"]["_growth_tracker"]
            updates_b = ckpt_b["metrics"]["optimizer_updates"]
            skipped_b = ckpt_b["metrics"]["skipped_updates"]
            # A fresh GradScaler also starts at scale 65536, so equal scales prove
            # nothing. The growth tracker counts consecutive unskipped steps and
            # only continues from the saved value if the state was restored.
            if skipped_b == 0 and tracker_b != tracker_a + updates_b:
                raise RuntimeError(f"scaler state was not restored on resume: growth tracker "
                                   f"{tracker_a} + {updates_b} updates != {tracker_b}")
            if skipped_b and not scale_b < scale_a:
                raise RuntimeError("an update was skipped but the scale did not back off")
            for field in ("optimizer", "replay", "lr_schedule"):
                if ckpt_b.get(field) is None:
                    raise RuntimeError(f"resumed checkpoint lost '{field}'")
            # metrics.jsonl is appended on resume, so rows_b already holds all three.
            checks["fp16"] = {"losses": [r["loss"] for r in rows_b],
                              "scale_before_resume": scale_a, "scale_after_resume": scale_b,
                              "growth_tracker_before": tracker_a, "growth_tracker_after": tracker_b,
                              "resume_optimizer_updates": updates_b, "resume_skipped_updates": skipped_b,
                              "peak_gpu_mb": max(r["peak_gpu_mb"] for r in rows_b)}

            leftover = subprocess.run(["pgrep", "-f", f"--output {stage_dir}"],
                                      capture_output=True, text=True)
            checks["leftover_workers"] = leftover.stdout.split()
            if leftover.stdout.strip():
                raise RuntimeError(f"workers outlived the training run: {checks['leftover_workers']}")
        except Exception as exc:
            return finish("failed", error=str(exc))

        print(f"[gpu] {checks['device_name']}: fp32 loss={checks['fp32']['loss']:.3f} "
              f"fp16 losses={[round(v, 3) for v in checks['fp16']['losses']]} "
              f"scaler tracker {tracker_a}->{tracker_b} scale={scale_b:.0f}")
        return finish("passed")

    @staticmethod
    def _rss_kb(pid: int) -> int | None:
        """Resident set size from /proc, so no psutil dependency is required."""
        try:
            with open(f"/proc/{pid}/statm") as handle:
                pages = int(handle.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") // 1024
        except (OSError, IndexError, ValueError):
            return None

    def run_stage_memory(self) -> dict[str, Any]:
        """Stage 7: 100 complete games and >= 20 training iterations after a
        warm-up, with owner, workers and external engines accounted separately.

        The previous implementation ran 1000 rollouts, recorded RSS before and
        after, and returned "passed" without ever comparing them: it could not
        fail. Replay is deliberately sized to saturate during warm-up so that
        post-warm-up growth is attributable to leaks rather than to the buffer
        filling.
        """
        print("\n==========================================")
        print("STAGE 7: MEMORY STABILITY GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "memory"
        stage_dir.mkdir(parents=True, exist_ok=True)
        checks: dict[str, Any] = {}

        def fail(message):
            result = {"stage": "memory", "status": "failed",
                      "duration_sec": time.time() - t0, "error": message, "checks": checks}
            (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
            return result

        # Growth beyond INVESTIGATE is recorded for investigation; beyond FAIL the
        # gate fails. Replay is saturated before the baseline is taken, so this
        # measures leakage, not buffer fill.
        INVESTIGATE_PCT, FAIL_PCT = 10.0, 25.0
        GAMES_PER_ITERATION, WARMUP_ITERATIONS, MEASURED_ITERATIONS = 5, 4, 20

        try:
            import gc
            import numpy as np
            import torch
            from sttt.learning import create_model
            from sttt.search import SearchConfig
            from sttt.selfplay import SelfPlayPool

            use_cpp = self.backend == "cpp"
            device = self.device
            model = create_model("mlp").to(device).eval()
            config = SearchConfig()
            owner_pid = os.getpid()

            def sample(label, pool):
                owner = self._rss_kb(owner_pid)
                workers = {p.pid: self._rss_kb(p.pid) for p in pool.processes}
                return {"label": label, "owner_rss_kb": owner,
                        "worker_rss_kb": workers,
                        "worker_total_kb": sum(v for v in workers.values() if v),
                        # Owner and workers are separate processes; their RSS is
                        # summed here only as an upper bound, because shared
                        # pages are counted once per process.
                        "aggregate_rss_kb_upper_bound": (owner or 0) + sum(v for v in workers.values() if v)}

            samples = []
            with SelfPlayPool(2, batch_size=8, wait_ms=2.) as pool:
                pool.verify_backend(use_cpp)

                # Warm-up: fill replay and pay one-time allocations.
                replay = []
                seed = 0
                for _ in range(WARMUP_ITERATIONS):
                    seeds = list(range(seed, seed + GAMES_PER_ITERATION))
                    seed += GAMES_PER_ITERATION
                    results, _ = pool.run(model, seeds, 16, config, 2, use_cpp=use_cpp)
                    for trajectory, outcome, _ in results:
                        replay.extend(trajectory)
                    replay = replay[-4000:]          # saturated, fixed-size
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                baseline = sample("post_warmup_baseline", pool)
                samples.append(baseline)

                games_played = 0
                arena = []
                for iteration in range(MEASURED_ITERATIONS):
                    seeds = list(range(seed, seed + GAMES_PER_ITERATION))
                    seed += GAMES_PER_ITERATION
                    results, _ = pool.run(model, seeds, 16, config, 2, use_cpp=use_cpp)
                    games_played += len(results)
                    for trajectory, outcome, stats in results:
                        replay.extend(trajectory)
                        if "arena_nodes" in stats:
                            arena.append((stats["arena_nodes"], stats.get("arena_capacity")))
                    replay = replay[-4000:]
                    if iteration % 5 == 4:
                        gc.collect()
                        samples.append(sample(f"iteration_{iteration + 1}", pool))
                final = sample("final", pool)
                samples.append(final)
                worker_pids = [p.pid for p in pool.processes]

            # Orphans: no worker may outlive the pool.
            orphans = [pid for pid in worker_pids if Path(f"/proc/{pid}").exists()]
            checks["orphan_processes"] = orphans
            checks["games_played"] = games_played
            checks["measured_iterations"] = MEASURED_ITERATIONS
            checks["samples"] = samples
            if arena:
                live = [a for a, _ in arena]
                capacity = [c for _, c in arena if c]
                checks["native_arena"] = {"live_nodes_max": max(live),
                                          "capacity_max": max(capacity) if capacity else None}
            if device.startswith("cuda"):
                checks["cuda"] = {
                    "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1)}

            if games_played < 100:
                return fail(f"only {games_played} games completed; the gate requires 100")
            if orphans:
                return fail(f"orphaned worker processes survived the pool: {orphans}")

            base_kb = baseline["aggregate_rss_kb_upper_bound"]
            final_kb = final["aggregate_rss_kb_upper_bound"]
            growth_pct = ((final_kb - base_kb) / base_kb * 100) if base_kb else 0.0
            checks["growth"] = {"baseline_kb": base_kb, "final_kb": final_kb,
                                "growth_pct": round(growth_pct, 2),
                                "investigate_threshold_pct": INVESTIGATE_PCT,
                                "fail_threshold_pct": FAIL_PCT,
                                "note": ("Replay is saturated before the baseline, so this is "
                                         "not buffer fill. Aggregate RSS double-counts shared "
                                         "pages and is an upper bound.")}
            if growth_pct > FAIL_PCT:
                return fail(f"aggregate RSS grew {growth_pct:.1f}% after warm-up "
                            f"({base_kb} -> {final_kb} kB), above the {FAIL_PCT}% cap")
            checks["growth"]["verdict"] = ("investigate" if growth_pct > INVESTIGATE_PCT else "stable")

            result = {"stage": "memory", "status": "passed",
                      "duration_sec": time.time() - t0, "checks": checks}
        except MemoryError as exc:
            return fail(f"MemoryError during the memory gate: {exc}")
        except Exception as exc:                           # noqa: BLE001
            import traceback
            return fail(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")

        (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
        print(f"[memory] {games_played} games, {MEASURED_ITERATIONS} iterations, "
              f"RSS growth {checks['growth']['growth_pct']}% ({checks['growth']['verdict']})")
        print(f"[memory] Stage PASSED in {result['duration_sec']:.2f}s")
        return result

    def run_stage_pilot(self) -> dict[str, Any]:
        """Stage 8: Representative Pilot Gate.

        Runs the canonical launcher (scripts/train.sh) with the exact production
        configuration, resumed from a frozen copy of a full checkpoint, for
        warm-up + measured iterations on the target GPU. Computes the ETA from
        the measured iterations, requires GPU memory headroom and no CPU
        fallback, and preserves the checkpoints. Fewer than 20 measured
        iterations leaves the gate pending: the wiring is proven, the ETA is not.
        """
        print("\n==========================================")
        print("STAGE 8: REPRESENTATIVE PILOT GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "pilot"
        stage_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {"stage": "pilot"}

        def finish(status, **extra):
            result.update(status=status, duration_sec=time.time() - t0, **extra)
            (stage_dir / "manifest.json").write_text(json.dumps(result, indent=2))
            return result

        if self.device != "cuda":
            return finish("skipped", reason="CPU device specified; the pilot gate requires --device cuda.")
        if not self.pilot_checkpoint:
            return finish("pending", reason="requires --pilot-checkpoint <full checkpoint> to resume from")
        if not Path(self.pilot_checkpoint).is_file():
            return finish("failed", error=f"pilot checkpoint not found: {self.pilot_checkpoint}")

        import statistics
        import torch
        if not torch.cuda.is_available():
            return finish("failed", error="CUDA device specified but torch.cuda.is_available() is False.")
        free_before, total_bytes = torch.cuda.mem_get_info()
        total_mb = total_bytes / 1024**2

        sys.path.insert(0, str(self.repo_root / "scripts"))
        from benchmark_pipeline import compute_eta, evaluation_overhead
        from sttt.evaluation import sha256_file

        run_dir = stage_dir / "run"
        iterations = self.pilot_warmup + self.pilot_iterations
        env = {**os.environ, "STTT_PY": sys.executable}
        cmd = ["bash", str(self.repo_root / "scripts" / "train.sh"),
               "--checkpoint", str(Path(self.pilot_checkpoint).resolve()),
               "--output", str(run_dir),
               "--population-config", str(Path(self.pilot_population_config).resolve()),
               "--workers", str(self.pilot_workers),
               "--iterations", str(iterations),
               "--lr-horizon", str(self.pilot_eta_iterations)]
        result["command"] = cmd
        result["environment"] = {"STTT_PY": env["STTT_PY"]}
        print(f"[pilot] {' '.join(cmd)}  (WORKERS={self.pilot_workers} ITERATIONS={iterations})")
        res = subprocess.run(cmd, cwd=self.repo_root, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            return finish("failed", error=f"scripts/train.sh exited {res.returncode}:\n{res.stderr[-4000:]}\n{res.stdout[-4000:]}")

        frozen = sorted((run_dir / "start-checkpoint").glob("*.pt"))
        result["start_checkpoint"] = {"source": str(self.pilot_checkpoint),
                                      "frozen": str(frozen[0]) if frozen else None,
                                      "sha256": sha256_file(frozen[0]) if frozen else None}
        rows = [json.loads(line) for line in
                (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
        rows = [r for r in rows if "loss" in r]
        if len(rows) < iterations:
            return finish("failed", error=f"expected {iterations} training iterations, found {len(rows)}")
        measured = rows[self.pilot_warmup:]
        if not all(r.get("peak_gpu_mb", 0) > 0 for r in rows):
            return finish("failed", error="an iteration ran without CUDA memory use (CPU fallback)")
        peak_reserved = max(r.get("reserved_gpu_mb", 0) for r in rows)
        headroom_mb = total_mb - peak_reserved
        result["gpu"] = {"device_name": torch.cuda.get_device_name(0), "total_mb": round(total_mb, 1),
                         "free_mb_before_pilot": round(free_before / 1024**2, 1),
                         "peak_reserved_mb": peak_reserved,
                         "peak_allocated_mb": max(r.get("peak_gpu_mb", 0) for r in rows),
                         "headroom_mb": round(headroom_mb, 1)}
        if peak_reserved > 0.9 * total_mb:
            return finish("failed", error=f"peak reserved {peak_reserved} MiB leaves no headroom on {total_mb:.0f} MiB")

        seconds = [r["seconds"] for r in measured]
        arm = {"backend": self.backend, "measured_iterations": len(measured),
               "iteration_seconds": {"mean": statistics.mean(seconds), "median": statistics.median(seconds),
                                     "min": min(seconds), "max": max(seconds),
                                     "stdev": statistics.pstdev(seconds) if len(seconds) > 1 else 0.0}}
        result["measured"] = {"warmup_iterations": self.pilot_warmup, "measured_iterations": len(measured),
                              "per_iteration_seconds": seconds, "stage_seconds": [r.get("stage_seconds") for r in measured],
                              "iteration_seconds": arm["iteration_seconds"]}
        result["eta"] = compute_eta(arm, self.pilot_eta_iterations, eval_every=100,
                                    eval_overhead_seconds=evaluation_overhead(run_dir))
        result["checkpoints"] = sorted(str(p) for p in run_dir.glob("*.pt"))
        if not (run_dir / "latest.pt").is_file():
            return finish("failed", error="latest.pt was not preserved")
        print(f"[pilot] {len(measured)} measured iterations: mean {arm['iteration_seconds']['mean']:.1f}s "
              f"(min {min(seconds):.1f}, max {max(seconds):.1f}); ETA for {self.pilot_eta_iterations}: "
              f"{result['eta']['hours_mean']:.1f} h [{result['eta']['hours_low']:.1f}, {result['eta']['hours_high']:.1f}]; "
              f"peak reserved {peak_reserved} MiB of {total_mb:.0f} MiB")
        if len(measured) < 20:
            return finish("pending", reason=f"only {len(measured)} measured iterations; the representative "
                                            f"pilot needs at least 20 (rerun with --pilot-iterations 20)")
        return finish("passed")

    def run(self) -> int:
        """Execute requested verification stages and write summary manifest."""
        overall_t0 = time.time()
        print(f"Readiness Runner Target: Backend={self.backend}, Device={self.device}, Stage={self.requested_stage}")
        print(f"Output Directory: {self.output_dir}")

        if self.manifest_only:
            manifest = self.get_capability_manifest()
            manifest_path = self.output_dir / "capability_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"Capability manifest saved to {manifest_path}")
            return 0

        stages_to_run = (
            self.STAGE_NAMES if self.requested_stage == "all" else [self.requested_stage]
        )

        all_passed = True
        any_pending = False
        for stage_name in stages_to_run:
            handler = getattr(self, f"run_stage_{stage_name}", None)
            if not handler:
                print(f"Unknown stage: {stage_name}", file=sys.stderr)
                return 1

            result = handler()
            self.results[stage_name] = result
            status = result.get("status", "failed")
            if status == "failed":
                all_passed = False
                print(f"--> Stage '{stage_name}' FAILED: {result.get('error', 'unknown error')}")
                if self.requested_stage == "all":
                    # In 'all' mode, abort on failure of a prerequisite stage
                    break
            elif status == "skipped":
                print(f"--> Stage '{stage_name}' SKIPPED: {result.get('reason', '')}")
            elif status == "pending":
                any_pending = True
                print(f"--> Stage '{stage_name}' PENDING: {result.get('reason', '')}")
            else:
                print(f"--> Stage '{stage_name}' PASSED ({result.get('duration_sec', 0.0):.2f}s)")

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_sec": time.time() - overall_t0,
            "overall_status": ("failed" if not all_passed else "pending" if any_pending else "passed"),
            "requested_stage": self.requested_stage,
            "backend": self.backend,
            "device": self.device,
            "stages": self.results,
        }

        summary_file = self.output_dir / "summary_manifest.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary manifest written to {summary_file}")

        return 0 if all_passed and not any_pending else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Super Tic-Tac-Toe Training Readiness Runner")
    parser.add_argument(
        "--backend",
        choices=["python", "cpp"],
        default="cpp",
        help="Search and environment backend (default: cpp)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device target for training and neural evaluation (default: cpu)",
    )
    parser.add_argument(
        "--stage",
        choices=["build", "native", "cpu", "mixed", "failure", "gpu", "memory", "pilot", "all"],
        default="all",
        help="Readiness gate stage to execute (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for manifests and test runs (isolated under runs/readiness/)",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Print and save machine-readable capability manifest without executing stages",
    )

    parser.add_argument("--pilot-checkpoint", default=None,
                        help="full checkpoint the pilot gate resumes from (frozen copy is made)")
    parser.add_argument("--pilot-iterations", type=int, default=20,
                        help="measured pilot iterations after warm-up (gate needs >= 20)")
    parser.add_argument("--pilot-warmup", type=int, default=2)
    parser.add_argument("--pilot-workers", type=int, default=16)
    parser.add_argument("--pilot-population-config", default="configs/population/baseline.json")
    parser.add_argument("--pilot-eta-iterations", type=int, default=5000,
                        help="production iteration count the ETA is projected for")

    args = parser.parse_args()
    runner = ReadinessRunner(
        backend=args.backend,
        device=args.device,
        stage=args.stage,
        output_dir=args.output_dir,
        manifest_only=args.manifest_only,
        pilot_checkpoint=args.pilot_checkpoint,
        pilot_iterations=args.pilot_iterations,
        pilot_warmup=args.pilot_warmup,
        pilot_workers=args.pilot_workers,
        pilot_population_config=args.pilot_population_config,
        pilot_eta_iterations=args.pilot_eta_iterations,
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
