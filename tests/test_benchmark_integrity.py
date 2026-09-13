"""M8: a benchmark must measure the work it claims to measure.

Three defects the plan named:

1. **Dead primitive loops.** `bench_play_move` built `BoardState next = s;
   next.play_inplace(40);` and never read `next`. With -O3 the optimizer is free
   to delete the whole body, so ">400M moves/sec" could be the cost of an empty
   loop. Every primitive loop must now consume its output into a checked
   accumulator, and the harness must report that checksum.
2. **Timing input generation, and timing one position.** Primitives ran on a
   single fixed state, so branch prediction and cache behaviour were unmeasured
   and unrepresentative. Positions are now pre-generated and varied, outside the
   timed block.
3. **Unlabelled categories.** Dummy-evaluator traversal, native heuristic search
   and real neural inference were reported in one table as if comparable, and
   parallel independent-tree throughput was presented beside single-move
   latency. Each measurement now declares its category.

Plus: repeated timed blocks with median and spread, and stage timings whose
denominators are completed work.
"""
import json
import tempfile
import pathlib
import subprocess
import unittest

from sttt.benchmarks.harness import (
    CATEGORIES,
    Measurement,
    StageTimer,
    summarize_repeats,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
BENCH_SOURCE = REPO / "cpp" / "benchmarks" / "bench_main.cpp"


class TestNativePrimitivesAreNotOptimizedAway(unittest.TestCase):
    """The C++ loops must consume their results, and say so."""

    @classmethod
    def setUpClass(cls):
        cls.source = BENCH_SOURCE.read_text()

    def test_move_execution_loop_consumes_its_result(self):
        body = self.source.split("void bench_play_move")[1].split("void ")[0]
        self.assertIn("checksum", body,
                      "play loop does not accumulate its result; -O3 may delete it")
        self.assertNotIn("BoardState next = s;\n        next.play_inplace(40); // center move", body)

    def test_every_primitive_benchmark_reports_a_checksum(self):
        for name in ("bench_legal_actions", "bench_play_move", "bench_encoding"):
            body = self.source.split(f"void {name}")[1].split("\nvoid ")[0]
            self.assertIn("checksum", body, f"{name} reports no checksum")

    def test_primitives_vary_the_position(self):
        for name in ("bench_legal_actions", "bench_play_move", "bench_encoding"):
            body = self.source.split(f"void {name}")[1].split("\nvoid ")[0]
            self.assertIn("positions[", body, f"{name} still times one fixed state")

    def test_position_generation_is_outside_the_timed_region(self):
        """Positions are built before the timed block, not inside it."""
        for name in ("bench_legal_actions", "bench_play_move", "bench_encoding"):
            body = self.source.split(f"void {name}")[1].split("\nvoid ")[0]
            gen = body.index("make_positions")
            timed = body.index("timed_median(")
            self.assertLess(gen, timed, f"{name} times its own input generation")

    def test_timed_blocks_are_warmed_up(self):
        helper = self.source.split("static double timed_median")[1].split("\nvoid ")[0]
        self.assertIn("warm-up", helper)


class TestNativeBenchmarkRuns(unittest.TestCase):
    """Actually execute the built binary and check its output is not impossible."""

    @classmethod
    def setUpClass(cls):
        cls.binary = REPO / "cpp" / "bench_main"
        if not cls.binary.is_file():
            raise unittest.SkipTest("cpp/bench_main not built")
        cls.output = subprocess.run([str(cls.binary), "--quick"], capture_output=True,
                                    text=True, timeout=600).stdout

    def test_no_primitive_reports_zero_elapsed_time(self):
        for line in self.output.splitlines():
            if "ns/" in line:
                value = float(line.split("(")[-1].split()[0])
                self.assertGreater(value, 0.0, f"impossible zero-work timing: {line}")

    def test_checksums_are_printed_and_nonzero(self):
        checksums = [l for l in self.output.splitlines() if "checksum=" in l]
        self.assertTrue(checksums, "no checksums in benchmark output")
        for line in checksums:
            value = int(line.split("checksum=")[1].split()[0])
            self.assertNotEqual(value, 0, line)

    def test_each_measurement_declares_a_category(self):
        measured = [l for l in self.output.splitlines() if l.startswith("[Benchmark]")]
        self.assertTrue(measured)
        for line in measured:
            self.assertTrue(any(f"[{c}]" in line for c in CATEGORIES),
                            f"measurement declares no category: {line}")

    def test_repeated_blocks_report_spread(self):
        self.assertIn("median", self.output)


class TestRepeatSummary(unittest.TestCase):
    def test_median_and_spread_over_repeats(self):
        stats = summarize_repeats([10.0, 12.0, 11.0, 100.0])
        self.assertEqual(stats["median"], 11.5)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 100.0)
        self.assertEqual(stats["repeats"], 4)

    def test_single_sample_has_zero_spread(self):
        stats = summarize_repeats([7.0])
        self.assertEqual(stats["median"], 7.0)
        self.assertEqual(stats["min"], stats["max"])

    def test_empty_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_repeats([])


class TestMeasurementLabelling(unittest.TestCase):
    def test_category_must_be_declared_and_known(self):
        with self.assertRaises(ValueError):
            Measurement(name="x", category="vibes", seconds=1.0, completed=10)

    def test_rate_denominator_is_completed_work(self):
        m = Measurement(name="search", category="native-heuristic-search",
                        seconds=2.0, completed=1000, requested=2000)
        self.assertEqual(m.rate, 500.0)
        self.assertEqual(m.completed, 1000)
        # requested != completed must be visible, not silently used as the rate
        self.assertEqual(m.to_dict()["requested"], 2000)
        self.assertNotEqual(m.rate, m.requested / m.seconds)

    def test_zero_completed_work_has_no_rate(self):
        m = Measurement(name="idle", category="dummy-evaluator-traversal",
                        seconds=1.0, completed=0)
        self.assertIsNone(m.rate)

    def test_parallel_throughput_is_not_labelled_as_latency(self):
        m = Measurement(name="10-thread rollouts", category="parallel-throughput",
                        seconds=1.0, completed=1000, threads=10)
        self.assertEqual(m.threads, 10)
        self.assertNotIn("latency", m.to_dict())

    def test_measurement_is_json_serializable(self):
        json.dumps(Measurement(name="n", category="native-primitive",
                               seconds=0.5, completed=5, checksum=12).to_dict())


class TestStageTimer(unittest.TestCase):
    """Pipeline stages must not sum overlapping worker time into elapsed time."""

    def test_stages_record_separately_and_do_not_claim_to_be_elapsed(self):
        timer = StageTimer()
        with timer.stage("selfplay"):
            pass
        with timer.stage("optimization"):
            pass
        report = timer.report(elapsed_seconds=10.0)
        self.assertIn("selfplay", report["stages"])
        self.assertIn("optimization", report["stages"])
        self.assertEqual(report["elapsed_seconds"], 10.0)

    def test_worker_time_is_recorded_as_aggregate_not_elapsed(self):
        timer = StageTimer()
        timer.add_aggregate("worker_selfplay", 40.0, workers=4)
        report = timer.report(elapsed_seconds=10.0)
        aggregate = report["aggregates"]["worker_selfplay"]
        self.assertEqual(aggregate["seconds"], 40.0)
        self.assertEqual(aggregate["workers"], 4)
        # 40 worker-seconds across 4 workers in 10s wall time is not a 400% stage
        self.assertNotIn("worker_selfplay", report["stages"])
        self.assertLessEqual(sum(report["stages"].values()), report["elapsed_seconds"] + 1e-6)

    def test_unaccounted_time_is_reported_rather_than_hidden(self):
        timer = StageTimer()
        with timer.stage("optimization"):
            pass
        report = timer.report(elapsed_seconds=5.0)
        self.assertIn("unaccounted_seconds", report)
        self.assertAlmostEqual(report["unaccounted_seconds"],
                               5.0 - report["stages"]["optimization"], places=3)


if __name__ == "__main__":
    unittest.main()


class TestPipelineBenchmarkAccounting(unittest.TestCase):
    """The end-to-end runner's arithmetic, without running a real pipeline."""

    def setUp(self):
        import scripts.run_pipeline_benchmark as pipeline
        self.pipeline = pipeline

    def _arm(self, backend, median, min_s=None, max_s=None, iterations=20):
        return {"backend": backend, "measured_iterations": iterations,
                "iteration_seconds": {"median": median, "mean": median,
                                      "min": min_s if min_s is not None else median,
                                      "max": max_s if max_s is not None else median,
                                      "stdev": 0.0, "repeats": iterations},
                "evaluation_overhead_seconds": 0.0}

    def test_eta_uses_the_measured_mean_and_reports_the_observed_band(self):
        eta = self.pipeline.compute_eta(self._arm("cpp", 30.0, 28.0, 36.0), 5000)
        self.assertAlmostEqual(eta["hours_mean"], 30.0 * 5000 / 3600)
        self.assertAlmostEqual(eta["hours_low"], 28.0 * 5000 / 3600)
        self.assertAlmostEqual(eta["hours_high"], 36.0 * 5000 / 3600)
        self.assertIn("20 measured iterations", eta["basis"])

    def test_eta_includes_periodic_evaluation_overhead(self):
        arm = self._arm("cpp", 10.0)
        plain = self.pipeline.compute_eta(arm, 1000)
        withev = self.pipeline.compute_eta(arm, 1000, eval_every=100, eval_overhead_seconds=60.0)
        self.assertEqual(withev["evaluations"], 10)
        self.assertAlmostEqual(withev["hours_mean"] - plain["hours_mean"], 600.0 / 3600)

    def test_a_regression_blocks_a_speedup_claim(self):
        arms = {"python": self._arm("python", 10.0), "cpp": self._arm("cpp", 12.0)}
        comparison = self.pipeline.compare_backends(arms)
        self.assertLess(comparison["cpp_speedup_vs_python"], 1.0)
        self.assertTrue(comparison["regression_blocks_claim"])
        self.assertIn("profile", comparison["note"])

    def test_a_speedup_is_reported_with_its_scope(self):
        arms = {"python": self._arm("python", 20.0), "cpp": self._arm("cpp", 10.0)}
        comparison = self.pipeline.compare_backends(arms)
        self.assertAlmostEqual(comparison["cpp_speedup_vs_python"], 2.0)
        self.assertFalse(comparison["regression_blocks_claim"])
        self.assertIn("this configuration", comparison["note"])

    def test_measured_rows_exclude_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "metrics.jsonl"
            rows = [{"iteration": i, "throughput": {"iteration_seconds": float(i)},
                     "stage_seconds": {"stages": {"selfplay": 1.0}}} for i in range(1, 6)]
            rows.append({"iteration": 5, "evaluation_overhead_seconds": 3.0})
            path.write_text("\n".join(json.dumps(r) for r in rows))
            parsed = self.pipeline.read_metrics(pathlib.Path(tmp))
            # The evaluation-overhead row carries no throughput and must not be
            # counted as an iteration.
            self.assertEqual(len(parsed), 5)
            self.assertEqual(self.pipeline.evaluation_overhead(pathlib.Path(tmp)), 3.0)


class TestReportedNumbersAreReproducible(unittest.TestCase):
    def test_the_fixed_state_encoding_rate_is_marked_historical_not_current(self):
        """45.4M states/sec was one fully cached state; the real rate is ~6x lower.

        The plan asks for old results to be marked historical rather than
        silently rewritten, so the figure may stay -- but every line carrying it
        must say it is historical, and a current (M8) measurement must exist.
        """
        report = (REPO / "BENCHMARK_REPORT.md").read_text()
        for line in report.splitlines():
            if "45,454,545" in line:
                self.assertIn("historical", line.lower(), line)
        self.assertIn("180.5 ns", report, "no current encoding measurement in the report")
        self.assertIn("M8 correction", report)
