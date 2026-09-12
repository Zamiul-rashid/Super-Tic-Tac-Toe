"""Thorough benchmark and grand championship tournament runner.

Executes:
1. Multi-bot round-robin tournament across neural checkpoints, C++ Alpha-Beta depths,
   tactical heuristics, OpenSpiel, and utttai.
2. Oracle challenge against C++ Alpha-Beta Depth 10.
3. Full ratings computation (Bayesian Elo with 95% CI, Glicko-2).
4. Structured report generation.
"""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

from sttt.bots import create_bot
from sttt.tournament import run_tournament, run_matchup, compute_bayesian_elo, compute_glicko2, format_scoreboard


def run_grand_championship(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("           SUPER TIC-TAC-TOE: GRAND CHAMPIONSHIP TOURNAMENT")
    print("=" * 90)

    # 1. Assemble participants
    print("[1/3] Initializing tournament participants...")
    bots = {
        "ckpt-iter6155-s512": create_bot("runs/big_run/latest.pt", simulations=512, name="ckpt-iter6155-s512"),
        "ckpt-iter6000-s512": create_bot("runs/big_run/model-6000.pt", simulations=512, name="ckpt-iter6000-s512"),
        "cpp-alphabeta-d6": create_bot("cpp-alphabeta:6", name="cpp-alphabeta-d6"),
        "cpp-alphabeta-d4": create_bot("cpp-alphabeta:4", name="cpp-alphabeta-d4"),
        "tactical": create_bot("tactical", name="tactical"),
        "threat-block": create_bot("threat-block", name="threat-block"),
        "openspiel-mcts": create_bot("openspiel", name="openspiel-mcts"),
        "utttai-128": create_bot("utttai", name="utttai-128"),
    }

    for name, bot in bots.items():
        print(f"  - {name}: {type(bot).__name__}")

    # 2. Run Round-Robin Tournament
    games_per_matchup = 20
    opening_plies = 2
    seed = 42
    total_matchups = len(bots) * (len(bots) - 1) // 2
    total_games = total_matchups * games_per_matchup

    print(f"\n[2/3] Executing Round-Robin Tournament: {len(bots)} bots, {total_matchups} matchups, {total_games} total games...")
    t0 = time.time()
    tourn = run_tournament(
        bots=bots,
        games_per_matchup=games_per_matchup,
        opening_plies=opening_plies,
        seed=seed,
    )
    elapsed = time.time() - t0
    print(f"\nCompleted {total_games} games in {elapsed:.1f}s ({elapsed / total_games:.3f}s per game)!")

    # Display scoreboard
    print("\n" + tourn["scoreboard"])

    # 3. Save artifacts
    print("\n[3/3] Saving tournament artifacts...")
    # Scoreboard text
    (output_dir / "scoreboard.txt").write_text(tourn["scoreboard"] + "\n", encoding="utf-8")

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

    # Head-to-Head Matrix CSV
    matrix_csv = output_dir / "matrix.csv"
    participants = tourn["participants"]
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["participant"] + participants)
        for p1 in participants:
            row = [p1]
            for p2 in participants:
                if p1 == p2:
                    row.append("-")
                else:
                    cell = tourn["matrix"][p1][p2]
                    row.append(f"{cell['wins']}-{cell['draws']}-{cell['losses']}")
            writer.writerow(row)

    # Tournament JSON summary
    summary_data = {
        "participants": participants,
        "games_per_matchup": games_per_matchup,
        "total_games": total_games,
        "elapsed_seconds": round(elapsed, 2),
        "side_stats": tourn["side_stats"],
        "ratings": {
            p: {
                "elo": round(tourn["elo_ratings"][p].elo, 2),
                "elo_ci95": round(tourn["elo_ratings"][p].error_margin, 2),
                "glicko2": round(tourn["glicko_ratings"][p].rating, 2),
                "glicko2_rd": round(tourn["glicko_ratings"][p].rd, 2),
                "wins": tourn["elo_ratings"][p].wins,
                "draws": tourn["elo_ratings"][p].draws,
                "losses": tourn["elo_ratings"][p].losses,
                "games": tourn["elo_ratings"][p].games,
            }
            for p in participants
        },
        "matrix": tourn["matrix"],
    }
    (output_dir / "tournament.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # Clean up any external bot processes
    for b in bots.values():
        b.close()

    print(f"Artifacts saved cleanly to {output_dir}/")
    return summary_data


def run_oracle_challenge(output_dir: Path):
    print("\n" + "=" * 90)
    print("         GRANDMASTER ORACLE CHALLENGE: LATEST NEURAL vs ALPHA-BETA D10")
    print("=" * 90)

    ab_d10 = create_bot("cpp-alphabeta:10", name="cpp-alphabeta-d10")

    budgets = [512, 2000]
    oracle_results = {}

    for sims in budgets:
        bot_name = f"ckpt-iter6155-s{sims}"
        print(f"\n--- Testing {bot_name} vs cpp-alphabeta-d10 (20 paired games) ---")
        neural_bot = create_bot("runs/big_run/latest.pt", simulations=sims, name=bot_name)

        t0 = time.time()
        results = run_matchup(neural_bot, ab_d10, games=20, opening_plies=2, seed=100 + sims)
        elapsed = time.time() - t0

        neural_wins = sum(1 for r in results if (r.winner == 1 and r.player_x == bot_name) or (r.winner == -1 and r.player_o == bot_name))
        oracle_wins = sum(1 for r in results if (r.winner == 1 and r.player_x == ab_d10.name) or (r.winner == -1 and r.player_o == ab_d10.name))
        draws = sum(1 for r in results if r.winner == 0)

        score_pct = (neural_wins + 0.5 * draws) / len(results) * 100
        print(f"Outcome: {neural_wins} Wins, {draws} Draws, {oracle_wins} Losses (Score: {score_pct:.1f}%) in {elapsed:.1f}s")

        oracle_results[bot_name] = {
            "simulations": sims,
            "neural_wins": neural_wins,
            "draws": draws,
            "oracle_wins": oracle_wins,
            "score_pct": score_pct,
            "elapsed_seconds": round(elapsed, 2),
        }

        neural_bot.close()

    ab_d10.close()

    (output_dir / "oracle_challenge.json").write_text(json.dumps(oracle_results, indent=2), encoding="utf-8")
    print(f"Oracle challenge results saved to {output_dir / 'oracle_challenge.json'}")
    return oracle_results


if __name__ == "__main__":
    out = Path("runs/tournaments/grand_championship")
    run_grand_championship(out)
    run_oracle_challenge(out)

