"""M6: the curriculum is configuration, and its coverage is measurable.

Before this, the opponent mix lived in a hardcoded dict and the per-family
budget ranges in literals inside sample_match, so a curriculum experiment meant
editing source, and nothing recorded which mix a checkpoint was trained under.
`history -> self` fallback also happened silently inside the sampler, so the
"10% history" quota could be 0% history with no trace in the actual counts.
"""
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from sttt.population import (
    FAMILIES,
    POPULATION_WEIGHTS,
    PopulationConfig,
    default_population_config,
    family_report,
    load_population_config,
    population_quota_counts,
    sample_match,
    sample_matches,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
PRESETS = REPO / "configs" / "population"
BASELINE_QUOTAS = {"self": 30, "utttai": 25, "alphabeta": 25, "history": 10, "best": 5,
                   "tactical": 2, "openspiel": 1, "threat": 1, "style": 1}


def write_config(directory, **overrides):
    data = default_population_config().to_dict()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    path = pathlib.Path(directory) / "population.json"
    path.write_text(json.dumps(data))
    return path


class TestDefaultAndPresets(unittest.TestCase):
    def test_default_config_is_the_legacy_weights_as_integer_quotas(self):
        config = default_population_config()
        self.assertEqual(config.quotas, {k: round(100 * v) for k, v in POPULATION_WEIGHTS.items()})
        self.assertEqual(sum(config.quotas.values()), 100)

    def test_baseline_preset_matches_the_plan(self):
        config = load_population_config(PRESETS / "baseline.json")
        self.assertEqual(config.quotas, BASELINE_QUOTAS)
        self.assertEqual(config.families["utttai"]["simulations"], [64, 128, 256])

    def test_candidate_preset_changes_only_the_utttai_alphabeta_split(self):
        baseline = load_population_config(PRESETS / "baseline.json")
        candidate = load_population_config(PRESETS / "candidate-utttai35.json")
        self.assertEqual(candidate.quotas["utttai"], 35)
        self.assertEqual(candidate.quotas["alphabeta"], 15)
        for family in FAMILIES:
            if family not in ("utttai", "alphabeta"):
                self.assertEqual(candidate.quotas[family], baseline.quotas[family], family)
        # Mix-only comparison: opponent budgets must be unchanged.
        self.assertEqual(candidate.families, baseline.families)
        self.assertNotEqual(candidate.sha256, baseline.sha256)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def assert_rejected(self, message, **overrides):
        path = write_config(self.tmp.name, **overrides)
        with self.assertRaises(ValueError, msg=message) as ctx:
            load_population_config(path)
        return str(ctx.exception)

    def test_rejects_unknown_family(self):
        quotas = {**BASELINE_QUOTAS, "self": 29, "wizard": 1}
        self.assertIn("wizard", self.assert_rejected("unknown family", quotas=quotas))

    def test_rejects_negative_quota(self):
        quotas = {**BASELINE_QUOTAS, "self": 35, "style": -4}
        self.assert_rejected("negative", quotas=quotas)

    def test_rejects_sum_other_than_100(self):
        quotas = {**BASELINE_QUOTAS, "self": 31}
        self.assertIn("101", self.assert_rejected("sum", quotas=quotas))

    def test_rejects_invalid_budgets(self):
        self.assert_rejected("zero simulations", families={"utttai": {"simulations": [0, 128]}})
        self.assert_rejected("negative nodes", families={"alphabeta": {"nodes": [-1]}})
        self.assert_rejected("empty depth list", families={"alphabeta": {"depth": []}})
        self.assert_rejected("epsilon above 1", families={"tactical": {"epsilon": [0.0, 1.5]}})
        self.assert_rejected("epsilon reversed", families={"tactical": {"epsilon": [0.5, 0.1]}})

    def test_rejects_unsupported_version(self):
        self.assert_rejected("version", version=2)


class TestDeterministicResolution(unittest.TestCase):
    def test_same_file_resolves_to_the_same_hash_and_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp)
            a, b = load_population_config(path), load_population_config(path)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(a.sha256, b.sha256)
        self.assertEqual(PopulationConfig.from_dict(a.to_dict()).sha256, a.sha256)

    def test_hash_changes_when_a_quota_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = load_population_config(write_config(tmp))
            changed = load_population_config(write_config(
                tmp, quotas={**BASELINE_QUOTAS, "utttai": 35, "alphabeta": 15}))
        self.assertNotEqual(base.sha256, changed.sha256)


class TestQuotaCycle(unittest.TestCase):
    def test_exact_counts_over_one_cycle_for_the_candidate_preset(self):
        config = load_population_config(PRESETS / "candidate-utttai35.json")
        self.assertEqual(population_quota_counts(100, config=config), config.quotas)

    def test_exact_counts_when_sliced_across_arbitrary_iteration_sizes_and_resumes(self):
        """The cursor is games completed so far; any slicing must re-sum exactly."""
        config = load_population_config(PRESETS / "candidate-utttai35.json")
        rng = np.random.default_rng(0)
        for _ in range(20):
            sizes, offset, totals = [], 0, dict.fromkeys(FAMILIES, 0)
            while offset < 300:
                size = int(rng.integers(1, 40))
                for kind, count in population_quota_counts(size, offset, config=config).items():
                    totals[kind] += count
                offset += size
                sizes.append(size)
            # Everything past the third full cycle is a partial slice; check the
            # full cycles exactly by trimming to 300.
            expected = {k: 3 * q for k, q in config.quotas.items()}
            overshoot = population_quota_counts(offset - 300, 300, config=config) if offset > 300 else {}
            for kind in totals:
                totals[kind] -= overshoot.get(kind, 0)
            self.assertEqual(totals, expected, sizes)

    def test_default_argument_preserves_legacy_behaviour(self):
        self.assertEqual(population_quota_counts(16), population_quota_counts(16, config=default_population_config()))


class TestSamplerUsesConfig(unittest.TestCase):
    def test_family_budgets_come_from_the_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_population_config(write_config(
                tmp, families={"utttai": {"simulations": [999]}, "alphabeta": {"depth": [4], "nodes": [12345]}}))
        registry = {"utttai": {"command": ["w", "--simulations", "{simulations}"], "protocol": "state_json"}}
        for seed in range(40):
            spec = sample_match(seed, engine_registry=registry, kind="utttai", config=config)
            self.assertEqual(spec.simulations, 999)
            spec = sample_match(seed, kind="alphabeta", config=config)
            self.assertEqual((spec.depth, spec.nodes), (4, 12345))

    def test_schedule_follows_configured_quotas(self):
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "model-0001.pt").touch()
            pathlib.Path(tmp, "best.pt").touch()
            config = load_population_config(PRESETS / "candidate-utttai35.json")
            registry = {"utttai": {"command": ["w", "{simulations}"], "protocol": "state_json"}}
            matches = sample_matches(range(100), tmp, registry, config=config)
        counts = {k: sum(m.requested_kind == k for m in matches) for k in FAMILIES}
        self.assertEqual(counts, config.quotas)

    def test_sampling_is_deterministic_for_a_config(self):
        config = load_population_config(PRESETS / "candidate-utttai35.json")
        a = [sample_match(s, kind="alphabeta", config=config) for s in range(50)]
        b = [sample_match(s, kind="alphabeta", config=config) for s in range(50)]
        self.assertEqual(a, b)


class TestExplicitFallback(unittest.TestCase):
    def test_missing_history_falls_back_to_self_but_records_the_request(self):
        with tempfile.TemporaryDirectory() as tmp:          # no checkpoints at all
            spec = sample_match(3, tmp, kind="history")
        self.assertEqual(spec.kind, "self")
        self.assertEqual(spec.requested_kind, "history")

    def test_every_non_fallback_spec_has_requested_equal_to_actual(self):
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "model-0001.pt").touch()
            pathlib.Path(tmp, "best.pt").touch()
            registry = {"utttai": {"command": ["w", "{simulations}"], "protocol": "state_json"}}
            matches = sample_matches(range(200), tmp, registry)
        for m in matches:
            self.assertEqual(m.requested_kind, m.kind)

    def test_no_family_is_ever_silently_substituted_by_tactical(self):
        with tempfile.TemporaryDirectory() as tmp:          # missing history/best
            registry = {"utttai": {"command": ["w", "{simulations}"], "protocol": "state_json"}}
            matches = sample_matches(range(200), tmp, registry)
        for m in matches:
            if m.kind == "tactical":
                self.assertEqual(m.requested_kind, "tactical")
            if m.kind != m.requested_kind:
                self.assertEqual((m.requested_kind in ("history", "best"), m.kind), (True, "self"))


class TestFamilyReport(unittest.TestCase):
    def _result(self, kind, requested, side, outcome, positions, seconds, **budget):
        match = {"kind": kind, "requested_kind": requested, "learner_side": side,
                 "simulations": budget.get("simulations", 128), "depth": budget.get("depth", 3),
                 "nodes": budget.get("nodes", 0)}
        stats = {"match": match, "seconds": seconds}
        trajectory = [None] * positions
        return trajectory, outcome, stats

    def test_counts_positions_time_and_score_by_family(self):
        results = [
            self._result("alphabeta", "alphabeta", 1, 1, 30, 2.0, depth=4, nodes=100000),   # learner win
            self._result("alphabeta", "alphabeta", -1, 1, 28, 3.0, depth=6, nodes=250000),  # learner loss
            self._result("self", "history", 1, 0, 40, 1.0),                                  # fallback, draw
        ]
        report = family_report(results)
        ab = report["alphabeta"]
        self.assertEqual(ab["games"], 2)
        self.assertEqual(ab["positions"], 58)
        self.assertAlmostEqual(ab["seconds"], 5.0)
        self.assertAlmostEqual(ab["score_rate"], 0.5)
        self.assertEqual(ab["budgets"]["depth"], {"4": 1, "6": 1})
        self.assertEqual(report["self"]["games"], 1)
        self.assertEqual(report["self"]["requested_as"], {"history": 1})
        self.assertNotIn("history", report)                  # actual families only

    def test_report_is_json_serializable(self):
        report = family_report([self._result("style", "style", 1, -1, 20, 0.5)])
        json.dumps(report)


class TestResumeResolution(unittest.TestCase):
    """Resolution rule from the plan: inherit the saved config unless one is
    explicitly given; an explicit change records a phase and restarts the
    100-game cursor."""

    def setUp(self):
        from sttt.ai import resolve_population_config
        self.resolve = resolve_population_config
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.candidate_path = PRESETS / "candidate-utttai35.json"
        self.baseline = default_population_config()
        self.candidate = load_population_config(self.candidate_path)

    def _saved(self, config, games=37, phases=()):
        return {"iteration": 12, "population_config": config.to_dict(),
                "population_config_sha256": config.sha256, "population_games": games,
                "population_phases": list(phases)}

    def test_fresh_run_without_flag_uses_the_default(self):
        config, cursor, phases = self.resolve(SimpleNamespace(population_config=None), {})
        self.assertEqual(config.sha256, self.baseline.sha256)
        self.assertEqual((cursor, phases), (0, []))

    def test_resume_without_flag_inherits_saved_config_and_cursor(self):
        config, cursor, phases = self.resolve(SimpleNamespace(population_config=None),
                                              self._saved(self.candidate))
        self.assertEqual(config.sha256, self.candidate.sha256)
        self.assertEqual(cursor, 37)
        self.assertEqual(phases, [])

    def test_resume_with_the_same_config_is_not_a_phase_change(self):
        config, cursor, phases = self.resolve(SimpleNamespace(population_config=str(self.candidate_path)),
                                              self._saved(self.candidate))
        self.assertEqual(cursor, 37)
        self.assertEqual(phases, [])

    def test_resume_with_a_different_config_records_a_phase_and_restarts_the_cursor(self):
        config, cursor, phases = self.resolve(SimpleNamespace(population_config=str(self.candidate_path)),
                                              self._saved(self.baseline))
        self.assertEqual(config.sha256, self.candidate.sha256)
        self.assertEqual(cursor, 0)
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0]["iteration"], 12)
        self.assertEqual(phases[0]["from_sha256"], self.baseline.sha256)
        self.assertEqual(phases[0]["to_sha256"], self.candidate.sha256)

    def test_legacy_checkpoint_without_config_inherits_the_default_and_keeps_its_cursor(self):
        config, cursor, phases = self.resolve(SimpleNamespace(population_config=None),
                                              {"iteration": 5, "population_games": 61})
        self.assertEqual(config.sha256, self.baseline.sha256)
        self.assertEqual(cursor, 61)


if __name__ == "__main__":
    unittest.main()
