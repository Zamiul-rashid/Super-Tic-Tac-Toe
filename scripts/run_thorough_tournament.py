"""Round-robin championship and search-budget sweep.

Two independent entry points:

* `run_grand_championship` - round-robin across neural checkpoints, depth-limited
  Alpha-Beta, heuristics, OpenSpiel and uttt.ai, anchored on a fixed opponent.
* `run_budget_sweep` - the same checkpoint at several search budgets against one
  reference opponent, on ONE shared opening corpus.

The sweep is the reason this file was rewritten. It previously seeded each budget
with ``100 + simulations``, so 512 played openings from seed 612 and 2000 from
seed 2100: the arms never met the same positions, and any score difference
between them is confounded with the opening corpus. Budgets now share a seed, so
the difference is attributable to search budget alone.

What this does NOT do: claim an equal-time comparison. Latency is measured and
reported per budget, but varying the simulation count is not a time-controlled
test, and no such claim is derived from these numbers.

Elo figures are local to this pool, these budgets and this opening corpus. The
Alpha-Beta entrant is depth- and node-limited: a reference opponent, not a
solved-game oracle, and its depth cap alone does not prove that depth was
reached.
"""
import argparse
import csv
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from sttt.bots import create_bot
from sttt.evaluation import (
    entrant_name,
    evaluation_manifest,
    freeze_checkpoint,
    pair_bootstrap_difference,
    pair_bootstrap_score,
    write_ratings_csv,
)
from sttt.tournament import run_matchup, run_tournament

DEFAULT_BUDGETS = [512, 1024, 2000]


def _safe_close(bot):
    try:
        bot.close()
    except Exception as exc:                       # noqa: BLE001
        print(f"warning: failed to close {getattr(bot, 'name', bot)}: {exc}", file=sys.stderr)


def open_bot(stack: ExitStack, spec, **kwargs):
    """Create a bot whose close is registered on every exit path."""
    bot = create_bot(spec, **kwargs)
    stack.callback(_safe_close, bot)
    return bot


def game_rows(results) -> list[dict]:
    return [{"game_id": r.game_id, "pair_id": r.pair_id, "opening_plies": r.opening_plies,
             "opening_moves": " ".join(str(m) for m in r.opening_moves),
             "player_x": r.player_x, "player_o": r.player_o,
             "winner": r.winner, "moves": r.moves}
            for r in results]


def write_games_csv(results, path: Path) -> Path:
    rows = game_rows(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                ["game_id", "pair_id", "opening_plies", "opening_moves",
                                 "player_x", "player_o", "winner", "moves"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def model_dtype(bot) -> str:
    """Report the precision actually in use rather than the precision requested."""
    model = getattr(bot, "model", None)
    if model is None:
        return "n/a"
    try:
        return str(next(model.parameters()).dtype)
    except (StopIteration, AttributeError):
        return "unknown"


def run_budget_sweep(checkpoint: str, output_dir: Path, budgets=None, games: int = 20,
                     seed: int = 4242, opponent_depth: int = 10, opponent_nodes: int = 50_000_000,
                     backend: str = "auto", device: str = "cpu", leaf_batch: int = 16) -> dict:
    """Play one checkpoint at several search budgets against one fixed opponent.

    All budgets share `seed`, hence one opening corpus, which is what makes the
    between-budget difference interpretable.
    """
    budgets = list(budgets or DEFAULT_BUDGETS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen, _ = freeze_checkpoint(checkpoint, output_dir / "evaluation-inputs" / "candidate")
    opponent_spec = f"cpp-alphabeta:{opponent_depth}:{opponent_nodes}"
    opponent_name = f"cpp-alphabeta-d{opponent_depth}-n{opponent_nodes}"

    manifest = evaluation_manifest(
        kind="search-budget-sweep",
        checkpoints={"candidate": frozen},
        config={"budgets": budgets, "games_per_budget": games, "opening_plies": 2,
                "backend": backend, "device": device, "leaf_batch": leaf_batch,
                "opponent": opponent_spec, "opponent_depth": opponent_depth,
                "opponent_nodes": opponent_nodes, "opening_seed": seed,
                "checkpoint_source": str(checkpoint),
                "equal_time_comparison": False},
        seeds={"openings": seed, "bootstrap": seed + 1},
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    arms: dict[int, dict] = {}
    raw_results: dict[int, list] = {}

    print("=" * 90)
    print("           SEARCH BUDGET SWEEP (one shared opening corpus)")
    print("=" * 90)
    print(f"Checkpoint: {frozen}")
    print(f"Opponent:   {opponent_spec}")
    print(f"Budgets:    {budgets}   Games per budget: {games}   Opening seed: {seed}")

    with ExitStack() as stack:
        opponent = open_bot(stack, opponent_spec, name=opponent_name)

        for budget in budgets:
            name = entrant_name(frozen, budget)
            candidate = open_bot(stack, str(frozen), simulations=budget, leaf_batch=leaf_batch,
                                 device=device, backend=backend, name=name)

            print(f"\n--- {name} vs {opponent_name}: {games} games "
                  f"({games // 2} mirrored opening pairs) ---")
            started = time.time()
            # Same seed for every budget: this is the corpus, not a per-arm knob.
            results = run_matchup(bot_a=candidate, bot_b=opponent, games=games,
                                  opening_plies=2, seed=seed)
            elapsed = time.time() - started

            stats = pair_bootstrap_score(results, name, seed=seed + 1)
            write_games_csv(results, output_dir / f"budget_{budget}_games.csv")
            raw_results[budget] = results
            arms[budget] = {
                "entrant": name,
                "simulations": budget,
                "precision": model_dtype(candidate),
                "elapsed_seconds": round(elapsed, 3),
                "seconds_per_game": round(elapsed / len(results), 4) if results else None,
                # `games` below is the count, from stats; the rows keep their own key.
                "game_rows": game_rows(results),
                **{k: v for k, v in stats.items() if k != "subject"},
            }
            print(f"score {stats['score_rate'] * 100:.1f}% "
                  f"(95% pair-bootstrap CI {stats['ci_low'] * 100:.1f}%-{stats['ci_high'] * 100:.1f}%) "
                  f"in {elapsed:.1f}s")

    # Paired differences against the smallest budget, on the shared corpus.
    comparisons: dict[str, dict] = {}
    baseline = budgets[0]
    for budget in budgets[1:]:
        # The arms are one checkpoint under budget-derived names, so each arm
        # names its own subject.
        comparisons[f"{baseline}->{budget}"] = pair_bootstrap_difference(
            raw_results[baseline], raw_results[budget],
            arms[baseline]["entrant"], subject_b=arms[budget]["entrant"], seed=seed + 1)

    summary = {"budgets": arms, "comparisons": comparisons,
               "baseline_budget": baseline, "opening_seed": seed}

    (output_dir / "budget_sweep.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest["status"] = "completed"
    manifest["summary"] = {
        "budgets": {str(b): {k: v for k, v in arm.items() if k != "game_rows"}
                    for b, arm in arms.items()},
        "comparisons": comparisons,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\nBudget differences (paired, same openings):")
    for label, diff in comparisons.items():
        print(f"  {label}: {diff['difference'] * 100:+.1f} score points "
              f"(95% CI {diff['ci_low'] * 100:+.1f} to {diff['ci_high'] * 100:+.1f}, "
              f"{diff['pairs']} pairs)")
    print("Latency is reported as measured; this is not an equal-time comparison.")

    return summary


def run_grand_championship(checkpoint: str, output_dir: Path, games_per_matchup: int = 20,
                           simulations: int = 512, seed: int = 42, backend: str = "auto",
                           device: str = "cpu", leaf_batch: int = 16,
                           anchor: str = "cpp-alphabeta-d4") -> dict:
    """Round-robin across a fixed opponent pool, anchored for cross-run comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen, _ = freeze_checkpoint(checkpoint, output_dir / "evaluation-inputs" / "candidate")
    candidate_name = entrant_name(frozen, simulations)

    manifest = evaluation_manifest(
        kind="grand-championship",
        checkpoints={"candidate": frozen},
        config={"games_per_matchup": games_per_matchup, "simulations": simulations,
                "opening_plies": 2, "backend": backend, "device": device,
                "leaf_batch": leaf_batch, "anchor": anchor,
                "checkpoint_source": str(checkpoint)},
        seeds={"openings": seed},
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("=" * 90)
    print("           SUPER TIC-TAC-TOE: GRAND CHAMPIONSHIP TOURNAMENT")
    print("=" * 90)

    with ExitStack() as stack:
        bots = {
            candidate_name: open_bot(stack, str(frozen), simulations=simulations,
                                     leaf_batch=leaf_batch, device=device,
                                     backend=backend, name=candidate_name),
            "cpp-alphabeta-d6": open_bot(stack, "cpp-alphabeta:6", name="cpp-alphabeta-d6"),
            "cpp-alphabeta-d4": open_bot(stack, "cpp-alphabeta:4", name="cpp-alphabeta-d4"),
            "tactical": open_bot(stack, "tactical", name="tactical"),
            "threat-block": open_bot(stack, "threat-block", name="threat-block"),
            "openspiel-mcts": open_bot(stack, "openspiel", name="openspiel-mcts"),
            "utttai-128": open_bot(stack, "utttai", name="utttai-128"),
        }
        for name, bot in bots.items():
            print(f"  - {name}: {type(bot).__name__}")

        total_matchups = len(bots) * (len(bots) - 1) // 2
        total_games = total_matchups * games_per_matchup
        print(f"\nRound-robin: {len(bots)} entrants, {total_matchups} matchups, "
              f"{total_games} games ({total_games // 2} mirrored pairs)...")

        started = time.time()
        tourn = run_tournament(bots=bots, games_per_matchup=games_per_matchup,
                               opening_plies=2, seed=seed,
                               anchor_player=anchor if anchor in bots else None)
        elapsed = time.time() - started
        print(f"\nCompleted {len(tourn['results'])} games in {elapsed:.1f}s")
        print("\n" + tourn["scoreboard"])
        print("\nRatings are local to this pool, these budgets and this opening corpus.")

        (output_dir / "scoreboard.txt").write_text(tourn["scoreboard"] + "\n", encoding="utf-8")
        write_ratings_csv(tourn, output_dir / "ratings.csv")
        write_games_csv(tourn["results"], output_dir / "championship_games.csv")

        participants = tourn["participants"]
        with (output_dir / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
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

        summary = {
            "participants": participants,
            "games_per_matchup": games_per_matchup,
            "total_games": len(tourn["results"]),
            "elapsed_seconds": round(elapsed, 2),
            "side_stats": tourn["side_stats"],
            "anchor": anchor,
            "ratings": {
                p: {"elo": round(tourn["elo_ratings"][p].elo, 2),
                    "elo_ci95": round(tourn["elo_ratings"][p].error_margin, 2),
                    "glicko2": round(tourn["glicko_ratings"][p].rating, 2),
                    "glicko2_rd": round(tourn["glicko_ratings"][p].rd, 2),
                    "wins": tourn["elo_ratings"][p].wins,
                    "draws": tourn["elo_ratings"][p].draws,
                    "losses": tourn["elo_ratings"][p].losses,
                    "games": tourn["elo_ratings"][p].games}
                for p in participants
            },
            "matrix": tourn["matrix"],
        }
        (output_dir / "tournament.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        manifest["status"] = "completed"
        manifest["summary"] = {k: v for k, v in summary.items() if k != "matrix"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Artifacts saved to {output_dir}/")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Championship round-robin and search-budget sweep")
    parser.add_argument("--checkpoint", default="runs/big_run/latest.pt")
    parser.add_argument("--output", default="runs/tournaments/grand_championship")
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                        help="search budgets for the sweep; all share one opening corpus")
    parser.add_argument("--games", type=int, default=20, help="TOTAL games per budget (2 per pair)")
    parser.add_argument("--championship-games", type=int, default=20,
                        help="TOTAL games per round-robin matchup")
    parser.add_argument("--simulations", type=int, default=512,
                        help="search budget for the championship entrant")
    parser.add_argument("--seed", type=int, default=4242, help="opening corpus seed")
    parser.add_argument("--backend", choices=("auto", "python", "cpp"), default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--leaf-batch", type=int, default=16)
    parser.add_argument("--opponent-depth", type=int, default=10)
    parser.add_argument("--opponent-nodes", type=int, default=50_000_000)
    parser.add_argument("--skip-championship", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"Error: checkpoint not found: {checkpoint}")
        sys.exit(1)

    output = Path(args.output)
    if not args.skip_championship:
        run_grand_championship(str(checkpoint), output / "championship",
                               games_per_matchup=args.championship_games,
                               simulations=args.simulations, seed=args.seed,
                               backend=args.backend, device=args.device,
                               leaf_batch=args.leaf_batch)
    if not args.skip_sweep:
        run_budget_sweep(str(checkpoint), output / "budget_sweep", budgets=args.budgets,
                         games=args.games, seed=args.seed,
                         opponent_depth=args.opponent_depth,
                         opponent_nodes=args.opponent_nodes,
                         backend=args.backend, device=args.device,
                         leaf_batch=args.leaf_batch)


if __name__ == "__main__":
    main()
