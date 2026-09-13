"""Comprehensive benchmark comparison between run_v2 and big_run (6000 iterations).

Executes:
1. Direct Head-to-Head Showdown: run_v2 vs big_run (50 paired games = 100 games).
2. 8-Engine Round-Robin Tournament:
   - run_v2 (best.pt or latest.pt)
   - big_run-best (runs/big_run/best.pt, 1671 Elo baseline)
   - utttai-128 (external SOTA reference, 2320 Elo)
   - cpp-alphabeta-d6 (1645 Elo)
   - cpp-alphabeta-d4 (1476 Elo)
   - tactical (1245 Elo)
   - openspiel-mcts (1103 Elo)
   - threat-block (883 Elo)
3. Grandmaster Oracle Challenge: run_v2 vs cpp-alphabeta-d10 (20 paired games).
4. Export structured reports, rating tables, and markdown summary.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

from sttt.bots import create_bot
from sttt.tournament import (
    run_tournament,
    run_matchup,
    compute_bayesian_elo,
    compute_glicko2,
    format_scoreboard,
)


def run_comparison(run_v2_ckpt: str, output_dir: Path, h2h_games: int = 50, round_robin_games: int = 20):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []

    def log(msg=""):
        print(msg, flush=True)
        report_lines.append(msg)

    log("=" * 90)
    log("       SUPER TIC-TAC-TOE: RUN_V2 (FRESH RESNET) VS BIG_RUN (6000 ITER)       ")
    log("=" * 90)
    log(f"Candidate Checkpoint: {run_v2_ckpt}")
    log(f"Legacy Best: runs/big_run/best.pt")
    log(f"Output Directory: {output_dir}")
    log("=" * 90)

    # ---------------------------------------------------------
    # Phase 1: Direct Head-to-Head Showdown (run_v2 vs big_run)
    # ---------------------------------------------------------
    log(f"\n[Phase 1/3] DIRECT HEAD-TO-HEAD SHOWDOWN: {h2h_games} paired games ({h2h_games * 2} games total)...")
    bot_v2 = create_bot(run_v2_ckpt, simulations=512, name="run_v2-s512")
    bot_legacy = create_bot("runs/big_run/best.pt", simulations=512, name="big_run-best-s512")

    t0 = time.time()
    h2h_results = run_matchup(
        bot_a=bot_v2,
        bot_b=bot_legacy,
        pairs=h2h_games,
        games=h2h_games,
        opening_plies=2,
        seed=42,
    )
    h2h_elapsed = time.time() - t0

    v2_wins = sum(1 for r in h2h_results if (r.winner == 1 and r.player_x == "run_v2-s512") or (r.winner == -1 and r.player_o == "run_v2-s512"))
    legacy_wins = sum(1 for r in h2h_results if (r.winner == 1 and r.player_x == "big_run-best-s512") or (r.winner == -1 and r.player_o == "big_run-best-s512"))
    draws = sum(1 for r in h2h_results if r.winner == 0)
    total_h2h = len(h2h_results)
    v2_score = (v2_wins + 0.5 * draws) / total_h2h * 100

    log(f"Completed {total_h2h} games in {h2h_elapsed:.1f}s ({h2h_elapsed / total_h2h:.3f}s/game)")
    log(f"  run_v2 Wins: {v2_wins} ({v2_wins / total_h2h * 100:.1f}%)")
    log(f"  big_run Wins: {legacy_wins} ({legacy_wins / total_h2h * 100:.1f}%)")
    log(f"  Draws: {draws} ({draws / total_h2h * 100:.1f}%)")
    log(f"  run_v2 Total Score: {v2_score:.1f}%")

    if v2_score > 50.0:
        log(f"  >>> SUCCESS: run_v2 OUTPERFORMS big_run by +{v2_score - 50.0:.1f}% score margin! <<<")
    elif v2_score < 50.0:
        log(f"  >>> big_run leads run_v2 by +{50.0 - v2_score:.1f}% score margin. <<<")
    else:
        log("  >>> Dead heat: Exactly 50.0% score tie. <<<")

    # Clean up bots
    bot_v2.close()
    bot_legacy.close()

    # ---------------------------------------------------------
    # Phase 2: 8-Engine Round-Robin Championship Tournament
    # ---------------------------------------------------------
    log(f"\n[Phase 2/3] 8-ENGINE ROUND-ROBIN TOURNAMENT ({round_robin_games} games per matchup)...")
    bots = {
        "run_v2-s512": create_bot(run_v2_ckpt, simulations=512, name="run_v2-s512"),
        "big_run-best-s512": create_bot("runs/big_run/best.pt", simulations=512, name="big_run-best-s512"),
        "cpp-alphabeta-d6": create_bot("cpp-alphabeta:6", name="cpp-alphabeta-d6"),
        "cpp-alphabeta-d4": create_bot("cpp-alphabeta:4", name="cpp-alphabeta-d4"),
        "tactical": create_bot("tactical", name="tactical"),
        "threat-block": create_bot("threat-block", name="threat-block"),
        "openspiel-mcts": create_bot("openspiel", name="openspiel-mcts"),
        "utttai-128": create_bot("utttai", name="utttai-128"),
    }

    t1 = time.time()
    tourn = run_tournament(
        bots=bots,
        games_per_matchup=round_robin_games,
        opening_plies=2,
        seed=100,
    )
    tourn_elapsed = time.time() - t1

    log(f"\nTournament completed in {tourn_elapsed:.1f}s!")
    log("\n" + tourn["scoreboard"])

    # ---------------------------------------------------------
    # Phase 3: Grandmaster Oracle Challenge (vs Alpha-Beta Depth 10)
    # ---------------------------------------------------------
    log(f"\n[Phase 3/3] GRANDMASTER ORACLE CHALLENGE (vs Alpha-Beta Depth 10, 20 paired games)...")
    oracle = create_bot("cpp-alphabeta:10", name="cpp-alphabeta-d10")
    test_bot_v2 = create_bot(run_v2_ckpt, simulations=512, name="run_v2-s512")

    t2 = time.time()
    oracle_results = run_matchup(
        bot_a=test_bot_v2,
        bot_b=oracle,
        pairs=10,
        games=20,
        opening_plies=2,
        seed=42,
    )
    oracle_elapsed = time.time() - t2

    owins = sum(1 for r in oracle_results if (r.winner == 1 and r.player_x == "run_v2-s512") or (r.winner == -1 and r.player_o == "run_v2-s512"))
    oloss = sum(1 for r in oracle_results if (r.winner == 1 and r.player_x == "cpp-alphabeta-d10") or (r.winner == -1 and r.player_o == "cpp-alphabeta-d10"))
    odraw = sum(1 for r in oracle_results if r.winner == 0)
    total_oracle = len(oracle_results)
    oscore = (owins + 0.5 * odraw) / total_oracle * 100

    log(f"Oracle Challenge (20 games) completed in {oracle_elapsed:.1f}s")
    log(f"  run_v2 vs Alpha-Beta Depth 10: {owins} Wins, {odraw} Draws, {oloss} Losses (Score: {oscore:.1f}%)")

    test_bot_v2.close()
    oracle.close()

    # ---------------------------------------------------------
    # Save Artifacts
    # ---------------------------------------------------------
    log("\nSaving artifacts...")
    (output_dir / "scoreboard.txt").write_text(tourn["scoreboard"] + "\n", encoding="utf-8")
    (output_dir / "comparison_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Ratings CSV
    ratings_csv = output_dir / "ratings.csv"
    with ratings_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "participant", "elo", "elo_ci95", "glicko2", "glicko2_rd",
            "score_pct", "wins", "draws", "losses", "games", "win_pct"
        ])
        for rank, p in enumerate(tourn["elo_ratings"], start=1):
            e = tourn["elo_ratings"][p]
            g = tourn["glicko_ratings"][p]
            writer.writerow([
                rank, p, f"{e.elo:.1f}", f"{e.error_margin:.1f}",
                f"{g.rating:.1f}", f"{g.rd:.1f}",
                f"{(e.wins + 0.5 * e.draws) / e.games * 100:.1f}",
                e.wins, e.draws, e.losses, e.games,
                f"{e.wins / e.games * 100:.1f}"
            ])

    log(f"All reports written to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Run comparison between run_v2 and big_run")
    parser.add_argument("--checkpoint", default="runs/run_v2/best.pt", help="run_v2 checkpoint")
    parser.add_argument("--output", default="runs/tournaments/run_v2_vs_bigrun", help="Output directory")
    parser.add_argument("--h2h-games", type=int, default=50, help="Paired games for head-to-head")
    parser.add_argument("--round-robin-games", type=int, default=20, help="Games per matchup in tournament")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        # Fall back to latest.pt if best.pt is not created yet
        alt = ckpt_path.parent / "latest.pt"
        if alt.is_file():
            print(f"{ckpt_path} not found; falling back to {alt}")
            ckpt_path = alt
        else:
            print(f"Error: Neither {ckpt_path} nor {alt} found!")
            sys.exit(1)

    run_comparison(str(ckpt_path), Path(args.output), args.h2h_games, args.round_robin_games)


if __name__ == "__main__":
    main()

