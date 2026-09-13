"""Controlled comparison between a candidate checkpoint and the big_run reference.

Phases:
1. Head-to-head, candidate vs reference, on one shared opening corpus.
2. Round-robin against a fixed pool of reference opponents.
3. Reference challenge against a depth-limited Alpha-Beta search.

Every phase writes its per-game rows, not just a printed summary, plus a manifest
recording which checkpoint files actually produced them.

Ratings printed here are LOCAL to this pool, these budgets and this opening
corpus. They are not universal engine ratings, and the Alpha-Beta opponent is a
depth- and node-limited reference, not a solved-game oracle.
"""
import argparse
import json
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
    entrant_name,
    evaluation_manifest,
    freeze_checkpoint,
    pair_bootstrap_score,
    write_games_csv,
    write_ratings_csv,
)
from sttt.tournament import run_matchup, run_tournament

REFERENCE_CHECKPOINT = "runs/big_run/best.pt"


def _safe_close(bot):
    """Closing must never mask the error that triggered the unwind."""
    try:
        bot.close()
    except Exception as exc:                       # noqa: BLE001
        print(f"warning: failed to close {getattr(bot, 'name', bot)}: {exc}", file=sys.stderr)


def open_bot(stack: ExitStack, spec, **kwargs):
    """Create a bot and register its close on ANY exit path.

    An exception mid-tournament used to leave external engine subprocesses
    running with nothing left to reap them.
    """
    bot = create_bot(spec, **kwargs)
    stack.callback(_safe_close, bot)
    return bot


def summarize_matchup(results, subject: str, bootstrap_seed: int) -> dict:
    """Score rate with an opening-pair bootstrap interval.

    Mirrored games share an opening, so the pair is the resampling unit; see
    sttt.evaluation.pair_bootstrap_score.
    """
    # A phase that produced no games is a failed run, not a 0-0 result. Say so
    # here rather than dividing by zero three lines later.
    if not results:
        raise RuntimeError(
            f"Matchup for '{subject}' produced no games; the run is invalid and "
            "must not be recorded as evidence")
    stats = pair_bootstrap_score(results, subject, seed=bootstrap_seed)
    wins = sum(1 for r in results
               if (r.winner == 1 and r.player_x == subject) or (r.winner == -1 and r.player_o == subject))
    draws = sum(1 for r in results if r.winner == 0)
    return {**stats, "wins": wins, "draws": draws, "losses": len(results) - wins - draws}


def run_comparison(candidate_ckpt: str, output_dir: Path, h2h_games: int = 50,
                   round_robin_games: int = 20, simulations: int = 512,
                   seed: int = 42, reference_ckpt: str = REFERENCE_CHECKPOINT) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []

    def log(msg=""):
        print(msg, flush=True)
        report_lines.append(msg)

    # Freeze the evaluation inputs first: `latest.pt` of a live run can roll
    # forward mid-comparison, which would silently mix two candidates into one
    # set of results. Separate subdirectories because both files are named
    # best.pt. The originals are never renamed or removed.
    frozen_dir = output_dir / "evaluation-inputs"
    candidate_frozen, _ = freeze_checkpoint(candidate_ckpt, frozen_dir / "candidate")
    reference_frozen, _ = freeze_checkpoint(reference_ckpt, frozen_dir / "reference")

    # Names come from the checkpoints that actually loaded, so a row in these
    # artifacts can be traced back to a specific file.
    candidate_name = entrant_name(candidate_frozen, simulations)
    reference_name = entrant_name(reference_frozen, simulations)

    manifest = evaluation_manifest(
        kind="candidate-vs-reference",
        checkpoints={"candidate": candidate_frozen, "reference": reference_frozen},
        config={"h2h_games": h2h_games, "round_robin_games": round_robin_games,
                "simulations": simulations, "opening_plies": 2,
                "candidate_source": str(candidate_ckpt), "reference_source": str(reference_ckpt)},
        seeds={"h2h": seed, "round_robin": seed + 58, "oracle": seed, "bootstrap": seed + 1000},
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary: dict = {"manifest": manifest}

    with ExitStack() as stack:
        log("=" * 90)
        log("       SUPER TIC-TAC-TOE: CANDIDATE VS BIG_RUN REFERENCE")
        log("=" * 90)
        log(f"Candidate: {candidate_name}  <- {candidate_ckpt}")
        log(f"Reference: {reference_name}  <- {reference_ckpt}")
        log(f"Output Directory: {output_dir}")
        log("=" * 90)

        # -----------------------------------------------------
        # Phase 1: head-to-head on one shared opening corpus
        # -----------------------------------------------------
        # `h2h_games` is the TOTAL number of games; run_matchup forms
        # h2h_games/2 mirrored pairs from it. The old wording called 50 "100".
        log(f"\n[Phase 1/3] HEAD-TO-HEAD: {h2h_games} games "
            f"({h2h_games // 2} mirrored opening pairs)...")
        candidate = open_bot(stack, str(candidate_frozen), simulations=simulations, name=candidate_name)
        reference = open_bot(stack, str(reference_frozen), simulations=simulations, name=reference_name)

        t0 = time.time()
        h2h_results = run_matchup(bot_a=candidate, bot_b=reference, games=h2h_games,
                                  opening_plies=2, seed=seed)
        h2h_elapsed = time.time() - t0

        h2h = summarize_matchup(h2h_results, candidate_name, bootstrap_seed=seed + 1000)
        summary["head_to_head"] = {**h2h, "elapsed_seconds": round(h2h_elapsed, 2)}
        write_games_csv(h2h_results, output_dir / "head_to_head_games.csv")

        log(f"Completed {h2h['games']} games ({h2h['pairs']} pairs) in {h2h_elapsed:.1f}s "
            f"({h2h_elapsed / h2h['games']:.3f}s/game)")
        log(f"  Candidate W/D/L: {h2h['wins']}/{h2h['draws']}/{h2h['losses']}")
        log(f"  Candidate score rate: {h2h['score_rate'] * 100:.1f}% "
            f"(95% pair-bootstrap CI {h2h['ci_low'] * 100:.1f}%-{h2h['ci_high'] * 100:.1f}%)")
        # An interval spanning 50% does not establish a difference in either
        # direction, so the outcome is reported as undetermined rather than as a
        # win for whichever side happens to lead.
        if h2h["ci_low"] > 0.5:
            log("  >>> Candidate ahead; interval excludes parity. <<<")
        elif h2h["ci_high"] < 0.5:
            log("  >>> Reference ahead; interval excludes parity. <<<")
        else:
            log("  >>> Undetermined: the 95% interval includes 50%. <<<")

        # -----------------------------------------------------
        # Phase 2: round-robin against the fixed reference pool
        # -----------------------------------------------------
        log(f"\n[Phase 2/3] ROUND-ROBIN ({round_robin_games} games per matchup)...")
        bots = {
            candidate_name: candidate,
            reference_name: reference,
            "cpp-alphabeta-d6": open_bot(stack, "cpp-alphabeta:6", name="cpp-alphabeta-d6"),
            "cpp-alphabeta-d4": open_bot(stack, "cpp-alphabeta:4", name="cpp-alphabeta-d4"),
            "tactical": open_bot(stack, "tactical", name="tactical"),
            "threat-block": open_bot(stack, "threat-block", name="threat-block"),
            "openspiel-mcts": open_bot(stack, "openspiel", name="openspiel-mcts"),
            "utttai-128": open_bot(stack, "utttai", name="utttai-128"),
        }

        t1 = time.time()
        tourn = run_tournament(bots=bots, games_per_matchup=round_robin_games,
                               opening_plies=2, seed=seed + 58)
        tourn_elapsed = time.time() - t1

        log(f"\nTournament completed in {tourn_elapsed:.1f}s")
        log("\n" + tourn["scoreboard"])
        log("\nRatings are local to this pool, these budgets and this opening corpus.")
        write_games_csv(tourn["results"], output_dir / "round_robin_games.csv")
        summary["round_robin"] = {"elapsed_seconds": round(tourn_elapsed, 2),
                                  "participants": tourn["participants"]}

        # -----------------------------------------------------
        # Phase 3: Alpha-Beta reference challenge
        # -----------------------------------------------------
        oracle_games = 20
        log(f"\n[Phase 3/3] ALPHA-BETA DEPTH-10 CHALLENGE: {oracle_games} games "
            f"({oracle_games // 2} mirrored opening pairs)...")
        oracle = open_bot(stack, "cpp-alphabeta:10", name="cpp-alphabeta-d10")

        t2 = time.time()
        oracle_results = run_matchup(bot_a=candidate, bot_b=oracle, games=oracle_games,
                                     opening_plies=2, seed=seed)
        oracle_elapsed = time.time() - t2

        oracle_stats = summarize_matchup(oracle_results, candidate_name, bootstrap_seed=seed + 1000)
        summary["alphabeta_challenge"] = {**oracle_stats, "elapsed_seconds": round(oracle_elapsed, 2),
                                          "opponent": "cpp-alphabeta-d10"}
        write_games_csv(oracle_results, output_dir / "alphabeta_challenge_games.csv")

        log(f"Completed {oracle_stats['games']} games in {oracle_elapsed:.1f}s")
        log(f"  Candidate W/D/L: {oracle_stats['wins']}/{oracle_stats['draws']}/{oracle_stats['losses']} "
            f"(score {oracle_stats['score_rate'] * 100:.1f}%, 95% CI "
            f"{oracle_stats['ci_low'] * 100:.1f}%-{oracle_stats['ci_high'] * 100:.1f}%)")

        # -----------------------------------------------------
        # Artifacts
        # -----------------------------------------------------
        log("\nSaving artifacts...")
        (output_dir / "scoreboard.txt").write_text(tourn["scoreboard"] + "\n", encoding="utf-8")
        (output_dir / "comparison_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        write_ratings_csv(tourn, output_dir / "ratings.csv")

        manifest["status"] = "completed"
        manifest["summary"] = {k: v for k, v in summary.items() if k != "manifest"}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        log(f"All reports written to {output_dir.resolve()}")

    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Controlled comparison against the big_run reference")
    parser.add_argument("--checkpoint", default="runs/run_v2/best.pt", help="candidate checkpoint")
    parser.add_argument("--reference", default=REFERENCE_CHECKPOINT, help="reference checkpoint")
    parser.add_argument("--output", default="runs/tournaments/run_v2_vs_bigrun", help="output directory")
    parser.add_argument("--h2h-games", type=int, default=50, help="TOTAL head-to-head games (2 per opening pair)")
    parser.add_argument("--round-robin-games", type=int, default=20, help="games per round-robin matchup")
    parser.add_argument("--simulations", type=int, default=512, help="search budget for both neural entrants")
    parser.add_argument("--seed", type=int, default=42, help="opening corpus seed")
    args = parser.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        alt = ckpt_path.parent / "latest.pt"
        if alt.is_file():
            print(f"{ckpt_path} not found; falling back to {alt}")
            ckpt_path = alt
        else:
            print(f"Error: Neither {ckpt_path} nor {alt} found!")
            sys.exit(1)

    run_comparison(str(ckpt_path), Path(args.output), args.h2h_games,
                   args.round_robin_games, simulations=args.simulations,
                   seed=args.seed, reference_ckpt=args.reference)


if __name__ == "__main__":
    main()
