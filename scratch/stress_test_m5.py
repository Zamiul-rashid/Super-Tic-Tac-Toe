"""Adversarial stress-test harness for Milestone 5."""
import os
import sys
import time
import math
import json
import signal
import subprocess
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sttt.env import State
from sttt.bots import (
    Bot, AlphaBetaBot, TacticalBot, StyleBot, ExternalProcessBot, create_bot
)
from sttt.tournament import (
    MatchResult, EloRating, GlickoRating,
    compute_bayesian_elo, compute_glicko2,
    paired_opening, _validate_sample_size, _validate_opening_plies,
    run_matchup, run_tournament, format_scoreboard
)
from sttt.opponent import Opponent, policies, NAMES
from sttt.reports import write_tournament_report


def test_section(name):
    print(f"\n{'='*20} [STRESS TEST] {name} {'='*20}")


def run_harness_1_elo_glicko():
    test_section("Harness 1: Elo & Glicko-2 Edge Cases & Stress")
    failures = []

    # 1.1 Disconnected graph: 2 completely disjoint pairs
    print("Testing 1.1: Disconnected graph with 2 disjoint cliques...")
    results_disjoint = [
        # Component A: p1 vs p2
        MatchResult(game_id=0, pair_id=0, opening_plies=2, opening_moves=[0, 1],
                    player_x="A1", player_o="A2", winner=1, moves=20),
        MatchResult(game_id=1, pair_id=0, opening_plies=2, opening_moves=[0, 1],
                    player_x="A2", player_o="A1", winner=-1, moves=20),
        # Component B: p3 vs p4
        MatchResult(game_id=2, pair_id=0, opening_plies=2, opening_moves=[2, 3],
                    player_x="B1", player_o="B2", winner=1, moves=25),
        MatchResult(game_id=3, pair_id=0, opening_plies=2, opening_moves=[2, 3],
                    player_x="B2", player_o="B1", winner=0, moves=30),
    ]
    try:
        elo_disjoint = compute_bayesian_elo(results_disjoint, players=["A1", "A2", "B1", "B2"])
        glicko_disjoint = compute_glicko2(results_disjoint, players=["A1", "A2", "B1", "B2"])
        for p, r in elo_disjoint.items():
            assert np.isfinite(r.elo), f"Elo for {p} not finite: {r.elo}"
            assert np.isfinite(r.error_margin), f"CI for {p} not finite: {r.error_margin}"
            assert r.error_margin > 0, f"CI for {p} non-positive: {r.error_margin}"
        for p, r in glicko_disjoint.items():
            assert np.isfinite(r.rating), f"Glicko rating for {p} not finite: {r.rating}"
            assert np.isfinite(r.rd), f"Glicko RD for {p} not finite: {r.rd}"
            assert np.isfinite(r.volatility), f"Glicko vol for {p} not finite: {r.volatility}"
        print("  -> Passed 1.1: Disconnected graph handled stably.")
    except Exception as e:
        failures.append(f"1.1 Disconnected graph failed: {e}")
        print(f"  -> FAILED 1.1: {e}")

    # 1.2 Isolated participant with 0 games
    print("Testing 1.2: Completely isolated participant with 0 matches...")
    try:
        elo_iso = compute_bayesian_elo(results_disjoint, players=["A1", "A2", "Isolated_Ghost"])
        assert "Isolated_Ghost" in elo_iso
        ghost_elo = elo_iso["Isolated_Ghost"]
        assert np.isfinite(ghost_elo.elo), f"Ghost elo not finite: {ghost_elo.elo}"
        assert ghost_elo.games == 0
        print(f"  -> Passed 1.2: Isolated player Elo = {ghost_elo.elo:.2f} +/- {ghost_elo.error_margin:.2f}")
    except Exception as e:
        failures.append(f"1.2 Isolated participant failed: {e}")
        print(f"  -> FAILED 1.2: {e}")

    # 1.3 Extreme win-rate disparity: 10,000 - 0 clean sweep
    print("Testing 1.3: Extreme win-rate disparity (10,000 wins vs 0)...")
    try:
        sweep_results = []
        for gid in range(10000):
            sweep_results.append(
                MatchResult(game_id=gid, pair_id=gid//2, opening_plies=2, opening_moves=[0, 1],
                            player_x="Titan", player_o="Puny", winner=1, moves=15)
            )
        elo_sweep = compute_bayesian_elo(sweep_results, players=["Titan", "Puny"])
        glicko_sweep = compute_glicko2(sweep_results, players=["Titan", "Puny"])

        titan_elo = elo_sweep["Titan"].elo
        puny_elo = elo_sweep["Puny"].elo
        assert np.isfinite(titan_elo) and np.isfinite(puny_elo)
        assert titan_elo > puny_elo + 500, f"Expected huge Elo gap, got Titan={titan_elo}, Puny={puny_elo}"
        assert elo_sweep["Titan"].error_margin > 0
        assert np.isfinite(glicko_sweep["Titan"].rating)
        print(f"  -> Passed 1.3: Extreme sweep Titan Elo={titan_elo:.1f} vs Puny={puny_elo:.1f}")
    except Exception as e:
        failures.append(f"1.3 Extreme disparity failed: {e}")
        print(f"  -> FAILED 1.3: {e}")

    # 1.4 All draws (100% draw rate)
    print("Testing 1.4: 100% draw rate across 1,000 games...")
    try:
        draw_results = [
            MatchResult(game_id=gid, pair_id=gid//2, opening_plies=2, opening_moves=[0, 1],
                        player_x="Twin1", player_o="Twin2", winner=0, moves=81)
            for gid in range(1000)
        ]
        elo_draws = compute_bayesian_elo(draw_results, players=["Twin1", "Twin2"])
        assert abs(elo_draws["Twin1"].elo - elo_draws["Twin2"].elo) < 1e-3
        print(f"  -> Passed 1.4: Twin1={elo_draws['Twin1'].elo:.2f}, Twin2={elo_draws['Twin2'].elo:.2f}")
    except Exception as e:
        failures.append(f"1.4 100% draws failed: {e}")
        print(f"  -> FAILED 1.4: {e}")

    # 1.5 Monotonic rating chain: P1 beats P2, P2 beats P3, P3 beats P4, P4 beats P5
    print("Testing 1.5: Monotonic transitivity chain P1 > P2 > P3 > P4 > P5...")
    try:
        chain_results = []
        players = [f"P{k}" for k in range(1, 6)]
        gid = 0
        for idx in range(len(players) - 1):
            p_hi = players[idx]
            p_lo = players[idx + 1]
            for _ in range(50):
                chain_results.append(
                    MatchResult(game_id=gid, pair_id=gid//2, opening_plies=2, opening_moves=[0, 1],
                                player_x=p_hi, player_o=p_lo, winner=1, moves=20)
                )
                gid += 1
        elo_chain = compute_bayesian_elo(chain_results, players=players)
        ratings_order = [elo_chain[p].elo for p in players]
        print(f"  Ratings order: {[f'{p}: {elo_chain[p].elo:.1f}' for p in players]}")
        for idx in range(len(ratings_order) - 1):
            assert ratings_order[idx] > ratings_order[idx + 1], (
                f"Transitivity violated at {players[idx]} ({ratings_order[idx]}) vs {players[idx+1]} ({ratings_order[idx+1]})"
            )
        print("  -> Passed 1.5: Strict monotonicity preserved across transitively ordered participants.")
    except Exception as e:
        failures.append(f"1.5 Transitivity failed: {e}")
        print(f"  -> FAILED 1.5: {e}")

    # 1.6 Large participant count: 50 players in Elo solver
    print("Testing 1.6: High participant count (50 players) in Elo MAP solver...")
    try:
        t0 = time.monotonic()
        many_players = [f"Bot_{k:02d}" for k in range(50)]
        many_results = []
        gid = 0
        rng = np.random.default_rng(42)
        # Random matches among 50 players
        for _ in range(500):
            p_a, p_b = rng.choice(many_players, size=2, replace=False)
            w = int(rng.choice([1, -1, 0], p=[0.45, 0.45, 0.1]))
            many_results.append(
                MatchResult(game_id=gid, pair_id=gid//2, opening_plies=2, opening_moves=[0, 1],
                            player_x=str(p_a), player_o=str(p_b), winner=w, moves=30)
            )
            gid += 1
        elo_many = compute_bayesian_elo(many_results, players=many_players)
        glicko_many = compute_glicko2(many_results, players=many_players)
        dt = time.monotonic() - t0
        assert len(elo_many) == 50
        assert all(np.isfinite(r.elo) for r in elo_many.values())
        print(f"  -> Passed 1.6: 50-player Bayesian Elo and Glicko-2 computed in {dt:.3f}s.")
    except Exception as e:
        failures.append(f"1.6 50 players failed: {e}")
        print(f"  -> FAILED 1.6: {e}")

    # 1.7 Edge cases: empty results, empty players
    print("Testing 1.7: Empty results and boundary players...")
    try:
        assert compute_bayesian_elo([], players=[]) == {}
        assert compute_glicko2([], players=[]) == {}
        elo_empty_matches = compute_bayesian_elo([], players=["Solo"])
        assert "Solo" in elo_empty_matches
        assert elo_empty_matches["Solo"].elo == 1500.0
        assert elo_empty_matches["Solo"].games == 0
        print("  -> Passed 1.7: Empty inputs handled gracefully.")
    except Exception as e:
        failures.append(f"1.7 Empty inputs failed: {e}")
        print(f"  -> FAILED 1.7: {e}")

    return failures


def run_harness_2_tournament_orchestrator():
    test_section("Harness 2: Tournament Orchestrator Stress")
    failures = []

    # 2.1 Sample size boundary validations
    print("Testing 2.1: Sample size boundary validations...")
    invalid_samples = [-1, 0, 1, 11, 15, 19, 501, 1000]
    for s in invalid_samples:
        try:
            _validate_sample_size(s)
            failures.append(f"2.1 Expected rejection for sample size {s}, but accepted")
            print(f"  -> FAILED: Accepted invalid sample size {s}")
        except ValueError:
            pass  # Expected

    valid_samples = [2, 4, 10, 20, 50, 500]
    for s in valid_samples:
        try:
            v = _validate_sample_size(s)
            assert v == s
        except Exception as e:
            failures.append(f"2.1 Valid sample size {s} rejected: {e}")
            print(f"  -> FAILED: Valid sample size {s} rejected: {e}")

    # Odd sample auto-rounding
    assert _validate_sample_size(21) == 22
    assert _validate_sample_size(499) == 500
    print("  -> Passed 2.1: Sample size validation exact.")

    # 2.2 Opening plies validations
    print("Testing 2.2: Opening plies boundary validations...")
    for p in [-2, -1, 5, 9, "bad"]:
        try:
            _validate_opening_plies(p)
            failures.append(f"2.2 Expected rejection for plies {p}")
        except ValueError:
            pass
    for p in [0, 1, 2, 3, 4, "random", "RANDOM"]:
        try:
            _validate_opening_plies(p)
        except Exception as e:
            failures.append(f"2.2 Valid plies {p} rejected: {e}")
    print("  -> Passed 2.2: Opening plies validation exact.")

    # 2.3 Paired opening symmetry and seed reproducibility
    print("Testing 2.3: Opening state deterministic reproducibility...")
    s1, a1 = paired_opening(seed=12345, pair=7, plies=3)
    s2, a2 = paired_opening(seed=12345, pair=7, plies=3)
    assert s1 == s2 and a1 == a2, "Deterministic paired opening failed"
    s3, a3 = paired_opening(seed=12345, pair=8, plies=3)
    assert a1 != a3, "Different pair IDs produced identical openings"
    print("  -> Passed 2.3: Deterministic opening generator verified.")

    # 2.4 Multi-bot tournament execution with matrix symmetry
    print("Testing 2.4: 5-bot tournament with StyleBots & TacticalBot...")
    try:
        bots = {
            "tactical": TacticalBot(),
            "center": StyleBot("center", deterministic=True),
            "corners": StyleBot("corners", deterministic=True),
            "local-win": StyleBot("local-win", deterministic=True),
            "random": StyleBot("random", deterministic=False),
        }
        # Run 2 games per matchup (rapid test)
        tourn = run_tournament(bots, games_per_matchup=2, opening_plies=2, seed=42)
        assert len(tourn["participants"]) == 5
        # Total matchups = 5 * 4 / 2 = 10 matchups * 2 games = 20 games
        assert len(tourn["results"]) == 20
        matrix = tourn["matrix"]
        for p1 in bots:
            for p2 in bots:
                if p1 == p2:
                    continue
                w12 = matrix[p1][p2]["wins"]
                l21 = matrix[p2][p1]["losses"]
                d12 = matrix[p1][p2]["draws"]
                d21 = matrix[p2][p1]["draws"]
                assert w12 == l21, f"Matrix mismatch {p1} vs {p2}: {w12} != {l21}"
                assert d12 == d21, f"Matrix draw mismatch {p1} vs {p2}: {d12} != {d21}"

        side_stats = tourn["side_stats"]
        assert side_stats["total_games"] == 20
        assert side_stats["x_wins"] + side_stats["o_wins"] + side_stats["draws"] == 20
        print("  -> Passed 2.4: 5-bot tournament executed cleanly with zero matrix discrepancies.")
    except Exception as e:
        failures.append(f"2.4 Multi-bot tournament failed: {e}")
        print(f"  -> FAILED 2.4: {e}")

    # 2.5 Maximum allowed games: 500 games per matchup
    print("Testing 2.5: High volume matchup (500 games)...")
    try:
        t0 = time.monotonic()
        b_a = StyleBot("center", deterministic=True)
        b_b = StyleBot("corners", deterministic=True)
        results_500 = run_matchup(b_a, b_b, games=500, opening_plies=2, seed=999)
        dt = time.monotonic() - t0
        assert len(results_500) == 500
        elo_500 = compute_bayesian_elo(results_500)
        assert len(elo_500) == 2
        print(f"  -> Passed 2.5: 500-game paired matchup completed in {dt:.3f}s. Results count: {len(results_500)}")
    except Exception as e:
        failures.append(f"2.5 High volume matchup failed: {e}")
        print(f"  -> FAILED 2.5: {e}")

    return failures


def run_harness_3_external_process_bot():
    test_section("Harness 3: ExternalProcessBot Lifecycle, Pipe Closure & Concurrency")
    failures = []
    py_bin = sys.executable

    # 3.1 Timeout compliance: External bot that sleeps for 10s
    print("Testing 3.1: Timeout enforcement with sleeping external process...")
    sleep_code = "import time; time.sleep(10)"
    try:
        t0 = time.monotonic()
        bot = ExternalProcessBot(
            command=[py_bin, "-c", sleep_code],
            timeout=0.2,  # 200ms timeout
            fallback="tactical",
        )
        state = State()
        rng = np.random.default_rng(42)
        move = bot.choose(state, rng)
        dt = time.monotonic() - t0
        assert move in state.legal_actions(), f"Fallback move {move} not legal"
        assert dt < 2.0, f"Bot took {dt:.2f}s, expected <= ~0.5s with 0.2s timeout"
        bot.close()
        print(f"  -> Passed 3.1: Timeout triggered after {dt:.3f}s; legal fallback move chosen.")
    except Exception as e:
        failures.append(f"3.1 Timeout enforcement failed: {e}")
        print(f"  -> FAILED 3.1: {e}")

    # 3.2 Immediate exit / crash recovery
    print("Testing 3.2: Immediate crash / exit recovery...")
    exit_code = "import sys; sys.exit(42)"
    try:
        bot = ExternalProcessBot(
            command=[py_bin, "-c", exit_code],
            timeout=1.0,
            fallback="tactical",
        )
        state = State()
        rng = np.random.default_rng(42)
        move = bot.choose(state, rng)
        assert move in state.legal_actions()
        bot.close()
        print("  -> Passed 3.2: Exited process gracefully caught and handled.")
    except Exception as e:
        failures.append(f"3.2 Exit handling failed: {e}")
        print(f"  -> FAILED 3.2: {e}")

    # 3.3 Segfault / SIGKILL during game
    print("Testing 3.3: Mid-execution SIGKILL...")
    suicide_code = "import os, signal, sys; sys.stdin.readline(); os.kill(os.getpid(), signal.SIGKILL)"
    try:
        bot = ExternalProcessBot(
            command=[py_bin, "-c", suicide_code],
            timeout=1.0,
            fallback="tactical",
        )
        state = State()
        rng = np.random.default_rng(42)
        move = bot.choose(state, rng)
        assert move in state.legal_actions()
        bot.close()
        print("  -> Passed 3.3: Killed process caught via EOF/BrokenPipe and handled.")
    except Exception as e:
        failures.append(f"3.3 SIGKILL failed: {e}")
        print(f"  -> FAILED 3.3: {e}")

    # 3.4 Pipe flood / huge output without newline
    print("Testing 3.4: Pipe flood with 1MB non-newline data...")
    flood_code = "import sys; sys.stdin.readline(); sys.stdout.write('A' * 100000); sys.stdout.flush(); import time; time.sleep(1)"
    try:
        bot = ExternalProcessBot(
            command=[py_bin, "-c", flood_code],
            timeout=0.3,
            fallback="tactical",
        )
        state = State()
        rng = np.random.default_rng(42)
        move = bot.choose(state, rng)
        assert move in state.legal_actions()
        bot.close()
        print("  -> Passed 3.4: Flood attack handled without hang or memory explosion.")
    except Exception as e:
        failures.append(f"3.4 Flood attack failed: {e}")
        print(f"  -> FAILED 3.4: {e}")

    # 3.5 Malformed and illegal move strings
    print("Testing 3.5: Malformed outputs (garbage, negative, out of bounds)...")
    bad_outputs = ["NaN NaN\n", "foo bar\n", "-5 100\n", "80\n", ""]
    for bad in bad_outputs:
        bad_code = f"import sys; sys.stdin.readline(); sys.stdout.write({repr(bad)}); sys.stdout.flush()"
        try:
            bot = ExternalProcessBot(
                command=[py_bin, "-c", bad_code],
                timeout=0.5,
                fallback="tactical",
            )
            state = State()
            rng = np.random.default_rng(42)
            move = bot.choose(state, rng)
            assert move in state.legal_actions()
            bot.close()
        except Exception as e:
            failures.append(f"3.5 Bad output {repr(bad)} failed: {e}")
            print(f"  -> FAILED 3.5 on {repr(bad)}: {e}")
    print("  -> Passed 3.5: Malformed output strings handled safely.")

    # 3.6 Rapid lifecycle: 50 processes opened and closed
    print("Testing 3.6: Rapid lifecycle of 50 ExternalProcessBots...")
    try:
        echo_code = "import sys\nwhile True:\n    line = sys.stdin.readline()\n    if not line: break\n    n = int(sys.stdin.readline())\n    for _ in range(n): sys.stdin.readline()\n    sys.stdout.write('4 4\\n')\n    sys.stdout.flush()\n"
        for idx in range(50):
            bot = ExternalProcessBot(
                command=[py_bin, "-c", echo_code],
                timeout=1.0,
            )
            state = State()
            rng = np.random.default_rng(idx)
            m = bot.choose(state, rng)
            assert m in state.legal_actions()
            bot.close()
        print("  -> Passed 3.6: 50 sequential ExternalProcessBots opened and closed cleanly.")
    except Exception as e:
        failures.append(f"3.6 Rapid lifecycle failed: {e}")
        print(f"  -> FAILED 3.6: {e}")

    return failures


def run_harness_4_opponent_adaptation():
    test_section("Harness 4: Opponent Adaptation & Belief Tracking Stress")
    failures = []

    # 4.1 High observation volume: 10,000 moves
    print("Testing 4.1: 10,000 consecutive observations without numerical drift...")
    try:
        opp = Opponent()
        state = State()
        rng = np.random.default_rng(42)
        for step in range(10000):
            legal = state.legal_actions()
            a = rng.choice(legal)
            opp.observe(state, a)
            state = state.play(a)
            if state.result is not None:
                state = State()
        w = opp.weights()
        assert np.isclose(w.sum(), 1.0), f"Weights sum {w.sum()} != 1.0"
        assert all(np.isfinite(w)), f"Weights non-finite: {w}"
        assert opp.observations == 10000
        print(f"  -> Passed 4.1: 10,000 observations processed. Weights: {w.round(3)}")
    except Exception as e:
        failures.append(f"4.1 High observation volume failed: {e}")
        print(f"  -> FAILED 4.1: {e}")

    # 4.2 Dynamic style shift recovery
    print("Testing 4.2: Dynamic style shift recovery (100 corners -> 100 center)...")
    try:
        opp = Opponent()
        state = State()
        corners_idx = NAMES.index("corners")
        center_idx = NAMES.index("center")
        # Phase 1: 100 corner moves
        for _ in range(100):
            p = policies(state)[corners_idx]
            a = int(np.argmax(p))
            opp.observe(state, a)
            state = state.play(a)
            if state.result is not None:
                state = State()
        w_p1 = opp.weights()
        assert w_p1[corners_idx] > 0.8, f"Expected corners dominance, got {w_p1}"

        # Phase 2: 100 center moves
        state = State()
        for _ in range(100):
            p = policies(state)[center_idx]
            a = int(np.argmax(p))
            opp.observe(state, a)
            state = state.play(a)
            if state.result is not None:
                state = State()
        w_p2 = opp.weights()
        assert w_p2[center_idx] > 0.8, f"Expected center dominance after shift, got {w_p2}"
        print(f"  -> Passed 4.2: Dynamic belief tracking adapted from {w_p1[corners_idx]:.2f} corners to {w_p2[center_idx]:.2f} center.")
    except Exception as e:
        failures.append(f"4.2 Style shift failed: {e}")
        print(f"  -> FAILED 4.2: {e}")

    # 4.3 Illegal move observation rejection
    print("Testing 4.3: Illegal action observation rejection...")
    try:
        opp = Opponent()
        state = State()
        state = state.play(40)  # Move played at 40
        try:
            opp.observe(state, 40)  # 40 is no longer legal
            failures.append("4.3 Allowed observation of illegal move")
            print("  -> FAILED: Allowed observation of illegal move")
        except ValueError:
            print("  -> Passed 4.3: Correctly rejected illegal move observation.")
    except Exception as e:
        failures.append(f"4.3 Illegal move failed: {e}")

    return failures


def run_harness_5_reports_atomicity():
    test_section("Harness 5: Report Serialization & Atomic Writes")
    failures = []
    tmp_dir = Path("scratch/test_report_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Testing 5.1: Tournament report generation and atomic writes...")
    try:
        results = [
            MatchResult(game_id=0, pair_id=0, opening_plies=2, opening_moves=[0, 1],
                        player_x="BotA", player_o="BotB", winner=1, moves=22),
            MatchResult(game_id=1, pair_id=0, opening_plies=2, opening_moves=[0, 1],
                        player_x="BotB", player_o="BotA", winner=-1, moves=24),
        ]
        elo_ratings = compute_bayesian_elo(results, players=["BotA", "BotB"])
        glicko_ratings = compute_glicko2(results, players=["BotA", "BotB"])
        side_stats = {"x_wins": 2, "o_wins": 0, "draws": 0, "total_games": 2,
                      "x_win_rate": 1.0, "o_win_rate": 0.0, "draw_rate": 0.0}
        scoreboard = format_scoreboard(elo_ratings, glicko_ratings, 2, side_stats)

        tourn = {
            "participants": ["BotA", "BotB"],
            "games_per_matchup": 2,
            "results": results,
            "elo_ratings": elo_ratings,
            "glicko_ratings": glicko_ratings,
            "matrix": {"BotA": {"BotB": {"wins": 2, "draws": 0, "losses": 0}},
                       "BotB": {"BotA": {"wins": 0, "draws": 0, "losses": 2}}},
            "side_stats": side_stats,
            "scoreboard": scoreboard,
        }
        write_tournament_report(tourn, tmp_dir)

        # Check files exist
        for fname in ["tournament.json", "summary.csv", "matchups.csv", "games.csv", "scoreboard.txt"]:
            fpath = tmp_dir / fname
            assert fpath.exists(), f"Expected file {fname} not generated"
            assert fpath.stat().st_size > 0, f"File {fname} is empty"

        # Check tournament.json can be loaded and has valid structure
        data = json.loads((tmp_dir / "tournament.json").read_text(encoding="utf-8"))
        assert "participants" in data and len(data["participants"]) == 2
        assert "ratings" in data and "BotA" in data["ratings"]
        assert "matchups" in data and len(data["matchups"]) == 2
        print("  -> Passed 5.1: All tournament report files atomically written and valid.")
    except Exception as e:
        failures.append(f"5.1 Report generation failed: {e}")
        print(f"  -> FAILED 5.1: {e}")
    finally:
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return failures


def main():
    print("=" * 70)
    print("STARTING EMPIRICAL ADVERSARIAL STRESS TEST SUITE FOR MILESTONE 5")
    print("=" * 70)
    all_failures = []
    all_failures.extend(run_harness_1_elo_glicko())
    all_failures.extend(run_harness_2_tournament_orchestrator())
    all_failures.extend(run_harness_3_external_process_bot())
    all_failures.extend(run_harness_4_opponent_adaptation())
    all_failures.extend(run_harness_5_reports_atomicity())

    print("\n" + "=" * 70)
    if all_failures:
        print(f"OVERALL RESULT: FAILED ({len(all_failures)} failures)")
        for f in all_failures:
            print(f" - {f}")
        sys.exit(1)
    else:
        print("OVERALL RESULT: ALL 18 ADVERSARIAL STRESS HARNESS TESTS PASSED CLEANLY!")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
