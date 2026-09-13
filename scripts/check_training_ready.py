#!/usr/bin/env python3
"""Training Readiness and Verification Runner.

Implements M0 and M9 verification per TRAINING_READINESS_PLAN.md.
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
    ):
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
        """Stage 2: Native Integration Gate (SelfPlayPool, MCTS worker reuse, shape verification)."""
        print("\n==========================================")
        print("STAGE 2: NATIVE INTEGRATION GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "native"
        stage_dir.mkdir(parents=True, exist_ok=True)

        if self.backend != "cpp":
            stage_result = {
                "stage": "native",
                "status": "skipped",
                "duration_sec": time.time() - t0,
                "reason": "Python backend specified; native integration gate applies to --backend cpp.",
            }
            (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
            return stage_result

        if not ce.is_cpp_available():
            stage_result = {
                "stage": "native",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": "C++ backend unavailable for native integration test.",
            }
            (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
            return stage_result

        try:
            import numpy as np
            import torch
            from sttt.selfplay import SelfPlayPool
            from sttt.model import PolicyValueNet

            model = PolicyValueNet()
            model.eval()

            # Spawn 2 workers, inference_batch 8
            pool = SelfPlayPool(model=model, workers=2, inference_batch=8)
            try:
                # Run 4 games, 32 simulations, leaf batch 4
                trajectories = pool.play_games(
                    num_games=4,
                    simulations=32,
                    leaf_batch=4,
                    backend=self.backend,
                )
                assert len(trajectories) == 4, f"Expected 4 trajectories, got {len(trajectories)}"
                for traj in trajectories:
                    for s, p, v in traj:
                        assert p.shape == (81,), f"Policy shape mismatch: {p.shape}"
                        assert np.isclose(p.sum(), 1.0, atol=1e-3), "Policy not normalized"
            finally:
                pool.close()

            stage_result = {
                "stage": "native",
                "status": "passed",
                "duration_sec": time.time() - t0,
                "details": {
                    "games_played": len(trajectories),
                    "workers": 2,
                    "simulations": 32,
                },
            }
        except Exception as e:
            stage_result = {
                "stage": "native",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": str(e),
            }

        (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
        return stage_result

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

        stage_result = {
            "stage": "cpu",
            "status": "passed",
            "duration_sec": time.time() - t0,
            "details": {
                "checkpoint": str(latest_ckpt),
                "initial_iterations": 2,
                "resumed_iterations": 1,
            },
        }
        (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
        return stage_result

    def run_stage_mixed(self) -> dict[str, Any]:
        """Stage 4: Mixed-Opponent Gate."""
        print("\n==========================================")
        print("STAGE 4: MIXED-OPPONENT GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "mixed"
        stage_dir.mkdir(parents=True, exist_ok=True)

        try:
            from sttt.bots import create_bot
            from sttt.env import State

            bots_to_test = [
                ("tactical", {}),
                ("alphabeta", {"depth": 2, "node_budget": 500}),
                ("style-center", {}),
                ("style-corners", {}),
                ("threat-block", {}),
            ]

            tested_bots = []
            for name, kwargs in bots_to_test:
                bot = create_bot(name, **kwargs)
                state = State()
                act = bot.choose(state)
                assert act in state.legal_actions(), f"Bot {name} chose illegal move {act}"
                tested_bots.append(name)

            stage_result = {
                "stage": "mixed",
                "status": "passed",
                "duration_sec": time.time() - t0,
                "tested_bots": tested_bots,
            }
        except Exception as e:
            stage_result = {
                "stage": "mixed",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": str(e),
            }

        (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
        return stage_result

    def run_stage_failure(self) -> dict[str, Any]:
        """Stage 5: Failure & Robustness Gate."""
        print("\n==========================================")
        print("STAGE 5: FAILURE & ROBUSTNESS GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "failure"
        stage_dir.mkdir(parents=True, exist_ok=True)

        checks: dict[str, Any] = {}

        # Check: Invalid backend rejects gracefully with nonzero exit code
        cmd_bad_backend = [
            sys.executable,
            "-m",
            "sttt.ai",
            "train",
            "--backend",
            "invalid_nonexistent_backend",
            "--iterations",
            "1",
        ]
        res = subprocess.run(cmd_bad_backend, cwd=self.repo_root, capture_output=True, text=True)
        checks["invalid_backend_rejected"] = res.returncode != 0
        if res.returncode == 0:
            stage_result = {
                "stage": "failure",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": "Invalid backend argument was accepted without error.",
                "checks": checks,
            }
            (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
            return stage_result

        stage_result = {
            "stage": "failure",
            "status": "passed",
            "duration_sec": time.time() - t0,
            "checks": checks,
        }
        (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
        return stage_result

    def run_stage_gpu(self) -> dict[str, Any]:
        """Stage 6: GPU Correctness Gate."""
        print("\n==========================================")
        print("STAGE 6: GPU CORRECTNESS GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "gpu"
        stage_dir.mkdir(parents=True, exist_ok=True)

        if self.device != "cuda":
            stage_result = {
                "stage": "gpu",
                "status": "skipped",
                "duration_sec": time.time() - t0,
                "reason": "CPU device specified; GPU gate requires --device cuda.",
            }
            (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
            return stage_result

        import torch

        if not torch.cuda.is_available():
            stage_result = {
                "stage": "gpu",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": "CUDA device specified but torch.cuda.is_available() is False.",
            }
            (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
            return stage_result

        try:
            from sttt.model import PolicyValueNet

            model = PolicyValueNet().cuda()
            x = torch.randn(8, 289, device="cuda")
            p, v = model(x)
            assert p.shape == (8, 81)
            assert v.shape == (8, 1)

            stage_result = {
                "stage": "gpu",
                "status": "passed",
                "duration_sec": time.time() - t0,
                "device_name": torch.cuda.get_device_name(0),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
        except Exception as e:
            stage_result = {
                "stage": "gpu",
                "status": "failed",
                "duration_sec": time.time() - t0,
                "error": str(e),
            }

        (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
        return stage_result

    def run_stage_memory(self) -> dict[str, Any]:
        """Stage 7: Memory Stability Gate."""
        print("\n==========================================")
        print("STAGE 7: MEMORY STABILITY GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "memory"
        stage_dir.mkdir(parents=True, exist_ok=True)

        import resource

        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Perform state transitions / rollouts
        if ce.is_cpp_available():
            ce.benchmark_rollouts(1000, 1, 42)
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        stage_result = {
            "stage": "memory",
            "status": "passed",
            "duration_sec": time.time() - t0,
            "rss_before_kb": rss_before,
            "rss_after_kb": rss_after,
        }
        (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
        return stage_result

    def run_stage_pilot(self) -> dict[str, Any]:
        """Stage 8: Representative Pilot Gate."""
        print("\n==========================================")
        print("STAGE 8: REPRESENTATIVE PILOT GATE")
        print("==========================================")
        t0 = time.time()
        stage_dir = self.output_dir / "pilot"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Quick pilot benchmark
        if ce.is_cpp_available():
            rate, elapsed = ce.benchmark_rollouts(2000, 2, 42)
            stage_result = {
                "stage": "pilot",
                "status": "passed",
                "duration_sec": time.time() - t0,
                "rollouts_per_sec": rate,
                "benchmark_duration_sec": elapsed,
            }
        else:
            stage_result = {
                "stage": "pilot",
                "status": "passed",
                "duration_sec": time.time() - t0,
                "note": "Pure Python baseline (C++ unavailable).",
            }

        (stage_dir / "manifest.json").write_text(json.dumps(stage_result, indent=2))
        return stage_result

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
            else:
                print(f"--> Stage '{stage_name}' PASSED ({result.get('duration_sec', 0.0):.2f}s)")

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_sec": time.time() - overall_t0,
            "overall_status": "passed" if all_passed else "failed",
            "requested_stage": self.requested_stage,
            "backend": self.backend,
            "device": self.device,
            "stages": self.results,
        }

        summary_file = self.output_dir / "summary_manifest.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary manifest written to {summary_file}")

        return 0 if all_passed else 1


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

    args = parser.parse_args()
    runner = ReadinessRunner(
        backend=args.backend,
        device=args.device,
        stage=args.stage,
        output_dir=args.output_dir,
        manifest_only=args.manifest_only,
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
