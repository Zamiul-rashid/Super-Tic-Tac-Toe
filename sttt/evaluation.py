"""M7: shared evaluation provenance, entrant identity and pair-level statistics.

Both comparison scripts previously carried their own copy of this logic and both
copies were wrong in the same way. The specific defects this module exists to
remove:

* **Entrant names were hardcoded labels.** `run_thorough_tournament.py` called
  `runs/big_run/latest.pt` "ckpt-iter6155" no matter which iteration was on disk,
  and the aggregate championship JSON kept no checkpoint hash, so no result in it
  can be replayed against a known file. `entrant_name()` derives the name from
  the checkpoint that actually loaded.
* **Rank came from dict insertion order.** Both scripts wrote a ratings CSV whose
  `rank` column could contradict the scoreboard printed beside it.
  `rank_ratings()` is the single ordering both now use.
* **Intervals treated mirrored games as independent.** Games 2k and 2k+1 share
  opening S_k by construction, so the resampling unit is the pair.

Nothing here plays games; it describes and summarizes games played elsewhere.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

SCHEMA_VERSION = 1

# A result is one game; these are the fields this module reads from it.
_WIN_X, _DRAW, _WIN_O = 1, 0, -1


def sha256_file(path: str | Path) -> str:
    """Streamed digest; checkpoints are large enough that read_bytes() is wasteful."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    """Identify a checkpoint by content, not by the name of the run that wrote it.

    Reads through the safe `weights_only=True` path, so this never executes
    pickled code from a checkpoint being evaluated.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")

    import torch                                   # local: keep import cost off module load
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)

    arch = checkpoint.get("arch")
    if arch is None:
        # Same inference rule as learning.load_model, for archless legacy files.
        state = checkpoint.get("model", {})
        arch = "resnet" if ("in_proj.0.weight" in state or "blocks.0.net.0.weight" in state) else "mlp"

    digest = sha256_file(resolved)
    return {
        "path": str(resolved),
        "sha256": digest,
        "sha8": digest[:8],
        "iteration": checkpoint.get("iteration"),
        "arch": arch,
    }


def entrant_name(path: str | Path, simulations: int, prefix: str = "ckpt") -> str:
    """`ckpt-iter5750-3f9a1c02-s512`: loaded iteration, content hash, search budget.

    The hash is what makes two checkpoints reporting the same iteration
    distinguishable, and what lets a later reader confirm which file produced a
    row. A checkpoint with no recorded iteration falls back to its file stem
    rather than printing `iterNone`.
    """
    identity = checkpoint_identity(path)
    iteration = identity["iteration"]
    label = f"iter{iteration}" if iteration is not None else Path(path).stem
    return f"{prefix}-{label}-{identity['sha8']}-s{int(simulations)}"


def freeze_checkpoint(src_path: str | Path, dest_dir: str | Path, max_retries: int = 3) -> tuple[Path, str]:
    """Snapshot a checkpoint, verifying the source did not roll forward mid-copy.

    Evaluation inputs must be immutable for the length of a comparison; `latest.pt`
    of a live run is not. The source is never renamed or deleted.
    """
    src = Path(src_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Checkpoint source does not exist: {src}")
    dest = Path(dest_dir).resolve() / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(max_retries):
        before = sha256_file(src)
        shutil.copy2(src, dest)
        after = sha256_file(src)
        copied = sha256_file(dest)
        if before == after == copied:
            return dest, copied
        time.sleep(0.5)
    raise RuntimeError(f"Source checkpoint '{src}' was concurrently modified during copy")


def rank_ratings(elo_ratings: Mapping[str, Any]) -> list[str]:
    """Participant names, strongest first. The one ordering every export uses."""
    return sorted(elo_ratings, key=lambda name: elo_ratings[name].elo, reverse=True)


def write_ratings_csv(tourn: Mapping[str, Any], path: str | Path) -> Path:
    """Ratings CSV whose `rank` column agrees with the printed scoreboard."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    elo_ratings, glicko_ratings = tourn["elo_ratings"], tourn["glicko_ratings"]

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "participant", "elo", "elo_ci95", "glicko2", "glicko2_rd",
                         "score_pct", "wins", "draws", "losses", "games", "win_pct"])
        for rank, name in enumerate(rank_ratings(elo_ratings), start=1):
            elo, glicko = elo_ratings[name], glicko_ratings[name]
            writer.writerow([
                rank, name, f"{elo.elo:.1f}", f"{elo.error_margin:.1f}",
                f"{glicko.rating:.1f}", f"{glicko.rd:.1f}",
                f"{(elo.wins + 0.5 * elo.draws) / elo.games * 100:.1f}",
                elo.wins, elo.draws, elo.losses, elo.games,
                f"{elo.wins / elo.games * 100:.1f}",
            ])
    return target


def game_score(result, subject: str) -> float:
    """Score rate contribution of one game for `subject`: win 1, draw 0.5, loss 0."""
    if subject == result.player_x:
        mine, theirs = _WIN_X, _WIN_O
    elif subject == result.player_o:
        mine, theirs = _WIN_O, _WIN_X
    else:
        raise ValueError(
            f"'{subject}' did not play game {result.game_id} "
            f"({result.player_x} vs {result.player_o})")
    if result.winner == mine:
        return 1.0
    if result.winner == theirs:
        return 0.0
    return 0.5


def _pair_scores(results: Sequence[Any], subject: str) -> tuple[list[int], list[float]]:
    """Mean score per opening pair. The pair is the independent sampling unit."""
    if not results:
        raise ValueError("No results to summarize")
    by_pair: dict[int, list[float]] = {}
    for result in results:
        by_pair.setdefault(result.pair_id, []).append(game_score(result, subject))
    pair_ids = sorted(by_pair)
    return pair_ids, [float(np.mean(by_pair[pid])) for pid in pair_ids]


def _percentile_ci(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0 * 100.0
    return float(np.percentile(samples, tail)), float(np.percentile(samples, 100.0 - tail))


def pair_bootstrap_score(results: Sequence[Any], subject: str, iterations: int = 10000,
                         seed: int = 12345, confidence: float = 0.95) -> dict[str, Any]:
    """Score rate for `subject` with an opening-pair bootstrap interval.

    Resampling is over opening pairs, not games: the two games of a pair share an
    opening and are played from both seats deliberately, so treating them as two
    independent draws understates the interval.
    """
    pair_ids, scores = _pair_scores(results, subject)
    values = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(int(iterations), len(values)))
    means = values[draws].mean(axis=1)
    low, high = _percentile_ci(means, confidence)
    return {
        "subject": subject,
        "score_rate": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "pairs": len(pair_ids),
        "games": len(results),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def pair_bootstrap_difference(results_a: Sequence[Any], results_b: Sequence[Any], subject: str,
                              iterations: int = 10000, seed: int = 12345,
                              confidence: float = 0.95) -> dict[str, Any]:
    """Paired difference (B minus A) in `subject`'s score rate on a shared corpus.

    Both arms must have been played from the *same* openings, which is the
    property `run_thorough_tournament.py` broke by seeding each search budget with
    `100 + simulations`. A mismatch is an error rather than a silently unpaired
    comparison.
    """
    ids_a, scores_a = _pair_scores(results_a, subject)
    ids_b, scores_b = _pair_scores(results_b, subject)
    if ids_a != ids_b:
        raise ValueError(f"Arms do not share opening pair ids: {ids_a} vs {ids_b}")

    openings_a = {r.pair_id: tuple(r.opening_moves) for r in results_a}
    openings_b = {r.pair_id: tuple(r.opening_moves) for r in results_b}
    mismatched = [pid for pid in ids_a if openings_a[pid] != openings_b[pid]]
    if mismatched:
        raise ValueError(
            "Arms were played from different opening positions for pair(s) "
            f"{mismatched}; a budget comparison requires one shared opening corpus")

    deltas = np.asarray(scores_b, dtype=np.float64) - np.asarray(scores_a, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(int(iterations), len(deltas)))
    means = deltas[draws].mean(axis=1)
    low, high = _percentile_ci(means, confidence)
    return {
        "subject": subject,
        "difference": float(deltas.mean()),
        "score_rate_a": float(np.mean(scores_a)),
        "score_rate_b": float(np.mean(scores_b)),
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
        "pairs": len(ids_a),
        "games": len(results_a) + len(results_b),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def native_build_info() -> dict[str, Any]:
    """Identity of the extension actually loaded, or why it is absent."""
    try:
        import sttt_cpp
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "path": getattr(sttt_cpp, "__file__", None),
        "version": getattr(sttt_cpp, "__version__", None),
        "build_id": getattr(sttt_cpp, "BUILD_ID", None),
        "source_revision": getattr(sttt_cpp, "SOURCE_REVISION", None),
    }


def environment_info() -> dict[str, Any]:
    import torch
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "native": native_build_info(),
    }


def evaluation_manifest(kind: str, checkpoints: Mapping[str, Any], config: Mapping[str, Any],
                        seeds: Mapping[str, Any], argv: Iterable[str] | None = None) -> dict[str, Any]:
    """Provenance record written beside every evaluation artifact.

    `status` starts at "running" so an aborted run leaves a record that is
    visibly incomplete instead of one that reads like a finished result.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "run_id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "checkpoints": {name: checkpoint_identity(path) for name, path in checkpoints.items()},
        "config": dict(config),
        "seeds": dict(seeds),
        "argv": list(argv) if argv is not None else list(sys.argv),
        "environment": environment_info(),
    }
