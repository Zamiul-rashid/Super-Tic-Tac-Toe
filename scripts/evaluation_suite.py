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
from dataclasses import asdict
from itertools import combinations
import importlib.metadata
import json
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path

# Running as `python scripts/<name>.py` puts scripts/ on sys.path, not the repo
# root, so `import sttt` failed. check_training_ready.py already does this.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sttt.bots import create_bot
from sttt.evaluation import (
    candidate_move_seconds,
    entrant_name,
    game_record_for_reanalysis,
    game_rows,
    evaluation_manifest,
    freeze_checkpoint,
    move_disagreement,
    pair_bootstrap_difference,
    pair_bootstrap_score,
    write_games_csv,
    write_progress_json,
    write_ratings_csv,
    sha256_file,
)
from sttt.reanalysis import write_jsonl
from sttt.tournament import (
    generate_opening_corpus,
    load_opening_corpus,
    run_matched_budget_comparison,
    run_matchup,
    run_tournament,
    save_opening_corpus,
)

DEFAULT_BUDGETS = [512, 1024, 2000]
DEFAULT_CHAMPIONSHIP_OPPONENTS = {
    "cpp-alphabeta-d6": "cpp-alphabeta:6:50000000",
    "cpp-alphabeta-d4": "cpp-alphabeta:4:50000000",
    "tactical": "tactical",
    "threat-block": "threat-block",
    "openspiel-mcts": "openspiel-mcts:100",
    "utttai-128": "utttai:128",
}


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


def _is_external_spec(spec: str) -> bool:
    """Bot specs that resolve to an ExternalProcessBot, whose `fallback` kwarg
    controls whether an unreachable/broken engine silently degrades to a
    heuristic bot instead of raising. A matched-opening comparison against a
    named external engine must not silently substitute a different opponent."""
    return spec == "utttai" or spec.startswith(("utttai:", "cmd:", "external:"))


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


def _render_budget_comparison_markdown(summary: dict) -> str:
    native = summary["native_build"]
    lines = [
        "# Matched-opening search budget comparison",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Opponent: `{summary['opponent']}`",
        f"Opening pairs: {summary['opening_pairs']} (each played both colours; "
        f"{summary['total_games']} games total)",
        f"Openings file: `{summary['openings_file']}`",
        f"Native build: `{native.get('build_id')}` (source revision {native.get('source_revision')})",
        "",
        summary["note"],
        "",
        "## Per-arm results",
        "",
        "| Budget | Entrant | Score | 95% CI | W/D/L | Mean s/move (candidate) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for budget, arm in sorted(summary["budgets"].items(), key=lambda kv: int(kv[0])):
        rows = arm["game_rows"]
        wins = sum(1 for r in rows if r["winner"] == (1 if r["player_x"] == arm["entrant"] else -1))
        losses = sum(1 for r in rows if r["winner"] == (-1 if r["player_x"] == arm["entrant"] else 1))
        draws = len(rows) - wins - losses
        mspm = arm["mean_seconds_per_move"]
        mspm_text = f"{mspm:.3f}" if mspm is not None else "n/a"
        lines.append(f"| {budget} | {arm['entrant']} | {arm['score_rate']*100:.1f}% | "
                     f"{arm['ci_low']*100:.1f}-{arm['ci_high']*100:.1f}% | {wins}/{draws}/{losses} | "
                     f"{mspm_text} |")
    lines += ["", "## Paired score differences (bootstrapped over opening pairs)", "",
             "| Comparison | Difference | 95% CI | Pairs | Move disagreement |",
             "| --- | --- | --- | --- | --- |"]
    for label, diff in summary["comparisons"].items():
        tag = " (primary)" if diff.get("primary") else ""
        disagreement = (diff.get("move_disagreement") or {}).get("disagreement_rate")
        dtext = f"{disagreement*100:.1f}%" if disagreement is not None else "n/a"
        lines.append(f"| {label}{tag} | {diff['difference']*100:+.1f} pts | "
                     f"{diff['ci_low']*100:+.1f} to {diff['ci_high']*100:+.1f} | {diff['pairs']} | {dtext} |")
    lines.append("")
    return "\n".join(lines)


def run_budget_comparison(checkpoint: str, output_dir: Path, budgets=None, opponent: str = None,
                          pairs: int = 50, opening_plies: int = 2, seed: int = 4242,
                          openings_file: str | Path | None = None, save_moves: bool = True,
                          backend: str = "auto", device: str = "cpu", leaf_batch: int = 16) -> dict:
    """One checkpoint at several search budgets against one fixed opponent, on a
    pre-generated opening corpus SAVED to disk and shared identically by every
    budget, with the play order rotated per opening pair.

    This is stricter than `run_budget_sweep`: the corpus is written up front
    (not just seeded, so it is auditable and reusable), every game's full move
    sequence and per-move latency are recorded (`save_moves`), and each opening
    pair rotates which budget plays it first instead of running one block of
    games per budget -- a block schedule would confound budget with wherever
    machine load happened to be when that block ran.
    """
    if not opponent:
        raise ValueError("--opponent is required for a budget comparison")
    budgets = list(budgets or DEFAULT_BUDGETS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    openings_path = Path(openings_file) if openings_file else output_dir / "openings.json"
    if openings_path.is_file():
        corpus, meta = load_opening_corpus(openings_path)
        if len(corpus) != pairs or meta.get("opening_plies") != opening_plies:
            raise ValueError(
                f"{openings_path} holds {len(corpus)} pairs at opening_plies="
                f"{meta.get('opening_plies')}; requested {pairs} at {opening_plies}. Use a new "
                "--openings-file, or matching --pairs/--opening-plies to reuse this one.")
    else:
        corpus = generate_opening_corpus(seed, pairs, opening_plies)
        save_opening_corpus(openings_path, corpus, seed, opening_plies)

    frozen, _ = freeze_checkpoint(checkpoint, output_dir / "evaluation-inputs" / "candidate")

    manifest = evaluation_manifest(
        kind="matched-opening-budget-comparison",
        checkpoints={"candidate": frozen},
        config={"budgets": budgets, "pairs": len(corpus), "opening_plies": opening_plies,
                "opponent": opponent, "openings_file": str(openings_path), "backend": backend,
                "device": device, "leaf_batch": leaf_batch, "save_moves": save_moves,
                "checkpoint_source": str(checkpoint), "budget_order": "rotated per opening pair",
                "equal_time_comparison": False},
        seeds={"openings": seed, "game_rng": seed, "bootstrap": seed + 1},
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("=" * 90)
    print("     MATCHED-OPENING SEARCH BUDGET COMPARISON (shared, saved opening corpus)")
    print("=" * 90)
    print(f"Checkpoint: {frozen}")
    print(f"Opponent:   {opponent}")
    print(f"Budgets:    {budgets}   Opening pairs: {len(corpus)}   Openings file: {openings_path}")

    total_games = len(budgets) * len(corpus) * 2
    started = time.time()
    completed = 0
    progress_path = output_dir / "progress.json"
    entrant_names: dict[int, str] = {}

    with ExitStack() as stack:
        opp_kwargs = {"fallback": "raise"} if _is_external_spec(opponent) else {}
        opponent_bot = open_bot(stack, opponent, **opp_kwargs)

        candidate_bots = {}
        for budget in budgets:
            name = entrant_name(frozen, budget)
            entrant_names[budget] = name
            candidate_bots[budget] = open_bot(stack, str(frozen), simulations=budget,
                                              leaf_batch=leaf_batch, device=device,
                                              backend=backend, name=name)

        with (output_dir / "games.partial.jsonl").open("w", encoding="utf-8") as journal:
            def record_game(budget, result):
                nonlocal completed
                completed += 1
                journal.write(json.dumps({"sequence": completed, "budget": budget, **asdict(result)}) + "\n")
                journal.flush()
                write_progress_json(progress_path, status="running", completed_games=completed,
                                    total_games=total_games, started=started,
                                    last_game={"budget": budget, **asdict(result)})
            try:
                raw_results = run_matched_budget_comparison(
                    candidate_bots=candidate_bots, opponent=opponent_bot, openings=corpus,
                    seed=seed, save_moves=save_moves, on_game=record_game)
            except BaseException as exc:
                manifest["status"] = "failed"
                manifest["error"] = f"{type(exc).__name__}: {exc}"
                manifest["completed_games"] = completed
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                write_progress_json(progress_path, status="failed", completed_games=completed,
                                    total_games=total_games, started=started,
                                    error=f"{type(exc).__name__}: {exc}")
                raise
    elapsed = time.time() - started

    arms = {}
    for budget in budgets:
        results = raw_results[budget]
        stats = pair_bootstrap_score(results, entrant_names[budget], seed=seed + 1)
        write_games_csv(results, output_dir / f"budget_{budget}_games.csv")
        latencies = []
        if save_moves:
            for r in results:
                latencies.extend(candidate_move_seconds(r, entrant_names[budget], opening_plies))
        arms[budget] = {
            "entrant": entrant_names[budget],
            "simulations": budget,
            "game_rows": game_rows(results),
            "mean_seconds_per_move": (sum(latencies) / len(latencies)) if latencies else None,
            "moves_timed": len(latencies),
            **{k: v for k, v in stats.items() if k != "subject"},
        }
        print(f"budget {budget}: score {stats['score_rate']*100:.1f}% "
              f"(95% CI {stats['ci_low']*100:.1f}-{stats['ci_high']*100:.1f}%)")

    sorted_budgets = sorted(budgets)
    primary_pair = tuple(sorted_budgets[-2:]) if len(sorted_budgets) >= 2 else None
    comparisons = {}
    for lo, hi in combinations(sorted_budgets, 2):
        entry = pair_bootstrap_difference(
            raw_results[lo], raw_results[hi], entrant_names[lo], subject_b=entrant_names[hi], seed=seed + 1)
        entry["primary"] = (lo, hi) == primary_pair
        if save_moves:
            entry["move_disagreement"] = move_disagreement(raw_results[lo], raw_results[hi], opening_plies)
        comparisons[f"{hi}-{lo}"] = entry

    # Full move sequences, in the form `python -m sttt.ai reanalyse --records` reads.
    records = []
    for budget in budgets:
        for result in raw_results[budget]:
            if result.moves_played is None:
                continue
            colour = "x" if entrant_names[budget] == result.player_x else "o"
            record_id = f"budget{budget}-pair{result.pair_id}-{colour}"
            records.append(game_record_for_reanalysis(result, entrant_names[budget], opening_plies, record_id))
    if records:
        write_jsonl(output_dir / "games-budget-compare.jsonl", records)

    summary = {
        "checkpoint": str(frozen), "opponent": opponent, "budgets": arms,
        "comparisons": comparisons, "opening_pairs": len(corpus), "opening_plies": opening_plies,
        "openings_file": str(openings_path), "total_games": completed,
        "elapsed_seconds": round(elapsed, 2), "native_build": manifest["environment"]["native"],
        "note": "Timings are measured as observed, not controlled for other processes sharing "
                "the machine, and this is not an equal-time comparison across budgets.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_budget_comparison_markdown(summary), encoding="utf-8")

    manifest["status"] = "completed"
    manifest["summary"] = {
        "budgets": {str(b): {k: v for k, v in arm.items() if k != "game_rows"} for b, arm in arms.items()},
        "comparisons": comparisons,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_progress_json(progress_path, status="completed", completed_games=completed,
                        total_games=total_games, started=started)

    print("\nPaired score differences (bootstrapped over opening pairs):")
    for label, diff in comparisons.items():
        tag = " [PRIMARY]" if diff.get("primary") else ""
        print(f"  {label}{tag}: {diff['difference']*100:+.1f} pts "
              f"(95% CI {diff['ci_low']*100:+.1f} to {diff['ci_high']*100:+.1f}, {diff['pairs']} pairs)")
    print(f"Artifacts saved to {output_dir}/")
    return summary


def run_grand_championship(checkpoint: str, output_dir: Path, games_per_matchup: int = 20,
                           simulations: int = 512, seed: int = 42, backend: str = "auto",
                           device: str = "cpu", leaf_batch: int = 16,
                           anchor: str = "cpp-alphabeta-d4",
                           opponents: list[str] | None = None) -> dict:
    """Round-robin across an explicit opponent pool with durable game progress."""
    from sttt.tournament import _validate_sample_size
    games_per_matchup = _validate_sample_size(games_per_matchup)
    if opponents is not None and (not opponents or len(opponents) != len(set(opponents))):
        raise ValueError("Championship opponents must be a nonempty list without duplicates")
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
                "checkpoint_source": str(checkpoint),
                "opponent_specs": opponents if opponents is not None else DEFAULT_CHAMPIONSHIP_OPPONENTS,
                "equal_time_comparison": False},
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
        }
        roster_specs = {candidate_name: str(frozen)}
        entries = ([(None, spec) for spec in opponents] if opponents is not None
                   else list(DEFAULT_CHAMPIONSHIP_OPPONENTS.items()))
        for display_name, spec in entries:
            kwargs = {"name": display_name} if display_name else {}
            bot = open_bot(stack, spec, **kwargs)
            if bot.name in bots:
                raise ValueError(f"Duplicate entrant name: {bot.name}")
            bots[bot.name] = bot
            roster_specs[bot.name] = spec
        for name, bot in bots.items():
            print(f"  - {name}: {type(bot).__name__}")

        import torch
        manifest["config"]["resolved_entrants"] = {
            name: {"spec": roster_specs[name], "type": type(bot).__name__,
                   **{key: getattr(bot, key) for key in (
                       "depth", "node_budget", "simulations", "leaf_batch", "device",
                       "backend_info", "uct_c", "seed", "command", "protocol", "timeout")
                      if hasattr(bot, key)}}
            for name, bot in bots.items()
        }
        manifest["environment"]["torch_threads"] = torch.get_num_threads()
        manifest["environment"]["platform"] = platform.platform()
        manifest["environment"]["machine"] = platform.machine()
        manifest["environment"]["packages"] = {}
        for package in ("numpy", "open-spiel", "onnxruntime"):
            try:
                manifest["environment"]["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                manifest["environment"]["packages"][package] = None
        manifest["source_sha256"] = {
            str(path.relative_to(_REPO_ROOT)): sha256_file(path)
            for path in [Path(__file__).resolve(), *sorted((_REPO_ROOT / "sttt").glob("*.py"))]
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        total_matchups = len(bots) * (len(bots) - 1) // 2
        total_games = total_matchups * games_per_matchup
        print(f"\nRound-robin: {len(bots)} entrants, {total_matchups} matchups, "
              f"{total_games} games ({total_games // 2} mirrored pairs)...")

        started = time.time()
        completed = 0
        progress_path = output_dir / "progress.json"
        with (output_dir / "games.partial.jsonl").open("w", encoding="utf-8") as journal:
            def record_game(result):
                nonlocal completed
                completed += 1
                journal.write(json.dumps({"sequence": completed, **asdict(result)}) + "\n")
                journal.flush()
                write_progress_json(progress_path, status="running", completed_games=completed,
                                    total_games=total_games, started=started, last_game=asdict(result))
            try:
                tourn = run_tournament(bots=bots, games_per_matchup=games_per_matchup,
                                       opening_plies=2, seed=seed,
                                       anchor_player=anchor if anchor in bots else None,
                                       on_game=record_game)
            except BaseException as exc:
                manifest["status"] = "failed"
                manifest["error"] = f"{type(exc).__name__}: {exc}"
                manifest["completed_games"] = completed
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                # progress.json used to keep reporting "running" forever after a
                # failure, because only the manifest was updated on this path.
                write_progress_json(progress_path, status="failed", completed_games=completed,
                                    total_games=total_games, started=started,
                                    error=f"{type(exc).__name__}: {exc}")
                raise
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
        write_progress_json(progress_path, status="completed", completed_games=len(tourn["results"]),
                            total_games=total_games, started=started)

    print(f"Artifacts saved to {output_dir}/")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Championship round-robin and search-budget sweep")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                        help="search budgets for the sweep; all share one opening corpus")
    parser.add_argument("--games", type=int, default=20, help="TOTAL games per budget (2 per pair)")
    parser.add_argument("--championship-games", type=int, default=20,
                        help="TOTAL games per round-robin matchup")
    parser.add_argument("--championship-opponents", nargs="+", default=None,
                        help="explicit bot specs replacing the default championship opponents")
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
    parser.add_argument("--budget-compare", action="store_true",
                        help="run ONLY the matched-opening budget comparison "
                             "(shared saved corpus, rotated budget order), instead of "
                             "the championship/sweep")
    parser.add_argument("--opponent", default=None,
                        help="bot spec for --budget-compare, e.g. 'utttai:128'")
    parser.add_argument("--pairs", type=int, default=50,
                        help="opening pairs for --budget-compare (each played both colours)")
    parser.add_argument("--opening-plies", type=int, default=2,
                        help="random opening plies per pair for --budget-compare")
    parser.add_argument("--openings-file", default=None,
                        help="--budget-compare: path to save/reuse the opening corpus; "
                             "default OUTPUT/openings.json")
    parser.add_argument("--save-moves", dest="save_moves", action="store_true", default=True,
                        help="--budget-compare: record full move sequences and per-move "
                             "latency (default on)")
    parser.add_argument("--no-save-moves", dest="save_moves", action="store_false")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"Error: checkpoint not found: {checkpoint}")
        sys.exit(1)

    output = Path(args.output)
    if args.budget_compare:
        run_budget_comparison(str(checkpoint), output, budgets=args.budgets,
                              opponent=args.opponent, pairs=args.pairs,
                              opening_plies=args.opening_plies, seed=args.seed,
                              openings_file=args.openings_file, save_moves=args.save_moves,
                              backend=args.backend, device=args.device, leaf_batch=args.leaf_batch)
        return
    if not args.skip_championship:
        run_grand_championship(str(checkpoint), output / "championship",
                               games_per_matchup=args.championship_games,
                               simulations=args.simulations, seed=args.seed,
                               backend=args.backend, device=args.device,
                               leaf_batch=args.leaf_batch, opponents=args.championship_opponents)
    if not args.skip_sweep:
        run_budget_sweep(str(checkpoint), output / "budget_sweep", budgets=args.budgets,
                         games=args.games, seed=args.seed,
                         opponent_depth=args.opponent_depth,
                         opponent_nodes=args.opponent_nodes,
                         backend=args.backend, device=args.device,
                         leaf_batch=args.leaf_batch)


if __name__ == "__main__":
    main()
