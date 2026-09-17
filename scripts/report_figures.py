"""Snapshot existing experiments and render publication figures; never run training.

Use --config to collect an evidence snapshot, or --snapshot to reproduce figures
without the original runs. All run paths belong in the JSON config, not here.
Requires only NumPy and Matplotlib. Run from any working directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BLUE, GREEN, ORANGE, RED = "#426a9c", "#537d62", "#c18b43", "#ad5959"
NAMES = {
    "cpp-alphabeta-d6": "AlphaBeta d6", "cpp-alphabeta-d4": "AlphaBeta d4",
    "tactical": "Tactical", "threat-block": "Threat-block",
    "openspiel-mcts": "OpenSpiel MCTS (100)", "utttai-128": "uttt.ai (128)",
}


def collect(config_path):
    config = json.loads(config_path.read_text())
    sources = []

    def read(path):
        p = ROOT / path
        raw = p.read_bytes()
        sources.append({"path": str(path), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        return raw.decode("utf-8")

    result = {"schema_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
              "code_revision": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "config": config, "sources": sources,
              "population": json.loads(read(config["population"])),
              "pretraining": [json.loads(s) for s in read(config["pretraining"]).splitlines() if s.strip()],
              "training": [], "experiments": []}
    fields = ("iteration", "loss", "q_loss", "seconds", "selfplay_seconds",
              "positions", "games", "simulations", "leaf_batch", "mean_inference_batch",
              "replay_sampling", "population_config_sha256")
    for path in config["training"]:
        records = [json.loads(s) for s in read(path).splitlines() if s.strip()]
        full, overhead, duplicates = {}, {}, 0
        requested, actual = Counter(), Counter()
        for r in records:
            if r["iteration"] > config["cutoff_iteration"]:
                continue
            if "loss" not in r:
                if "evaluation_overhead_seconds" in r:
                    overhead[r["iteration"]] = r["evaluation_overhead_seconds"]
                continue
            duplicates += int(r["iteration"] in full)
            full[r["iteration"]] = r
        for r in full.values():
            requested.update(r.get("population_requested_counts", {}))
            actual.update(r.get("population_match_counts", {}))
        result["training"].append({"source": path, "raw_records": len(records),
            "duplicate_full_records": duplicates, "overhead": overhead,
            "requested": dict(requested), "actual": dict(actual),
            "rows": [{k: r.get(k) for k in fields} for _, r in sorted(full.items())]})
    for item in config["experiments"]:
        exp = {"label": item["label"]}
        for key, filename in (("championship", "tournament.json"), ("sweep", "budget_sweep.json")):
            exp[key] = json.loads(read(str(Path(item[key]) / filename)))
            exp[key + "_manifest"] = json.loads(read(str(Path(item[key]) / "manifest.json")))
        exp["games"] = list(csv.DictReader(io.StringIO(
            read(str(Path(item["championship"]) / "championship_games.csv")))))
        for row in exp["games"]:
            for key in ("game_id", "pair_id", "winner", "moves", "opening_plies"):
                row[key] = int(row[key])
        result["experiments"].append(exp)
    for path in ("sttt/unet.py", "sttt/learning.py", "sttt/search.py", "sttt/selfplay.py",
                 "sttt/ai.py", "sttt/bootstrap.py", "sttt/bots.py", "sttt/tournament.py",
                 "sttt/evaluation.py", "scripts/evaluation_suite.py"):
        read(path)
    return result


def score(row, subject):
    if row["winner"] == 0:
        return .5
    return float(row["player_x" if row["winner"] == 1 else "player_o"] == subject)


def pair_scores(rows, subject):
    pairs = defaultdict(list)
    for r in rows:
        if subject in (r["player_x"], r["player_o"]):
            pairs[r["pair_id"]].append(score(r, subject))
    if not pairs or any(len(v) != 2 for v in pairs.values()):
        raise ValueError("Expected two games per opening pair")
    return np.array([np.mean(v) for _, v in sorted(pairs.items())])


def interval(values):
    rng = np.random.default_rng(4243)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [.025, .975])


def validate(data):
    """Cross-check aggregate artifacts against preserved per-game records."""
    for e in data["experiments"]:
        t = e["championship"]
        assert len(e["games"]) == t["total_games"]
        for a in t["participants"]:
            totals = Counter()
            for b in t["participants"]:
                if a == b:
                    continue
                rows = [r for r in e["games"] if {r["player_x"], r["player_o"]} == {a, b}]
                counts = Counter(score(r, a) for r in rows)
                observed = {"wins": counts[1.], "draws": counts[.5], "losses": counts[0.]}
                assert observed == t["matrix"][a][b], (e["label"], a, b)
                totals.update(observed)
                pair_scores(rows, a)
            for key, value in totals.items():
                assert value == t["ratings"][a][key]
        for arm in e["sweep"]["budgets"].values():
            p = pair_scores(arm["game_rows"], arm["entrant"])
            assert np.isclose(p.mean(), arm["score_rate"])
            assert np.allclose(interval(p), [arm["ci_low"], arm["ci_high"]])
    iterations = [r["iteration"] for p in data["training"] for r in p["rows"]]
    assert len(iterations) == len(set(iterations)), "Training segments overlap"


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.18, linewidth=.5)


def results_figures(data, output):
    latest = data["experiments"][-1]
    t = latest["championship"]
    candidate = t["participants"][0]
    opponents = t["participants"][1:]
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    left = np.zeros(len(opponents))
    for key, label, color in (("wins", "Win", GREEN), ("draws", "Draw", ORANGE), ("losses", "Loss", RED)):
        values = np.array([t["matrix"][candidate][o][key] for o in opponents])
        ax.barh(np.arange(len(opponents)), values, left=left, label=label, color=color, height=.65)
        for i, v in enumerate(values):
            if v:
                ax.text(left[i] + v/2, i, str(v), ha="center", va="center", color="white", fontsize=10)
        left += values
    ax.set(yticks=range(len(opponents)), yticklabels=[NAMES[o] for o in opponents],
           xlabel="Games (20 per opponent)", xlim=(0, 20), xticks=range(0, 21, 5),
           title=f"{latest['label']}: head-to-head outcomes")
    ax.invert_yaxis(); ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(.5, 1.08), frameon=False)
    save(fig, output, "matchup-outcomes")

    names = t["participants"]
    matrix = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                r = t["matrix"][a][b]; n = sum(r.values())
                matrix[i, j] = (r["wins"] + .5*r["draws"])/n
    labels = ["Our U-Net (512)" if n == candidate else NAMES[n] for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 6.7), layout="constrained")
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                ax.text(j, i, f"{matrix[i,j]:.1%}", ha="center", va="center",
                        color="white" if matrix[i,j] > .65 else "black", fontsize=10)
    ax.set(xticks=range(len(names)), yticks=range(len(names)), yticklabels=labels,
           title=f"{latest['label']}: row player's score against column player")
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.colorbar(im, ax=ax, fraction=.04, label="Score rate (win = 1, draw = ½)")
    save(fig, output, "championship-matrix")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.8, 1]})
    for k, e in enumerate(data["experiments"]):
        c = e["championship"]["participants"][0]
        pts, bounds = [], []
        for opp in opponents:
            rows = [r for r in e["games"] if {r["player_x"], r["player_o"]} == {c, opp}]
            p = pair_scores(rows, c); pts.append(p.mean()); bounds.append(interval(p))
        pts = np.array(pts); bounds = np.array(bounds)
        axes[0].errorbar(np.arange(len(opponents)) + (k-.5)*.15, pts*100,
                        yerr=np.maximum(0, np.array([pts-bounds[:,0], bounds[:,1]-pts]))*100,
                        fmt="o", capsize=3, color=[BLUE, RED][k % 2], label=e["label"])
        for budget, arm in e["sweep"]["budgets"].items():
            axes[1].errorbar(k, arm["score_rate"]*100,
                            yerr=[[100*(arm["score_rate"]-arm["ci_low"])],
                                  [100*(arm["ci_high"]-arm["score_rate"])]],
                            fmt="o", color=[BLUE, RED][k % 2], capsize=5)
            axes[1].annotate(f"{arm['score_rate']:.1%}", (k, arm["score_rate"]*100),
                             xytext=(10, 0), textcoords="offset points")
    axes[0].set(xticks=range(len(opponents)), ylim=(-3, 105), ylabel="Score (%)",
                title="(a) Championship matchups")
    axes[0].set_xticklabels([NAMES[o] for o in opponents], rotation=30, ha="right")
    axes[0].legend(frameon=False)
    axes[1].set(xticks=range(len(data["experiments"])),
                xticklabels=[e["label"] for e in data["experiments"]],
                xlim=(-.6, len(data["experiments"])-.3), ylim=(0, 100),
                title="(b) AlphaBeta d10 / 50M nodes", ylabel="Score (%)")
    for ax in axes:
        clean(ax); ax.axhline(50, color="gray", ls="--", lw=.8)
    save(fig, output, "checkpoint-comparison")


def training_figures(data, output):
    rows = [r for p in data["training"] for r in p["rows"]]
    x = np.array([r["iteration"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    for ax, key, title in zip(axes.flat, ("loss", "q_loss", "seconds", "mean_inference_batch"),
                             ("(a) Combined training loss", "(b) Auxiliary Q loss",
                              "(c) Seconds per training iteration", "(d) Mean inference batch size")):
        y = np.array([r[key] for r in rows], dtype=float)
        ax.plot(x, y, color=BLUE, alpha=.16, lw=.5)
        window = min(100, len(y))
        smooth = np.convolve(y, np.ones(window)/window, mode="valid")
        ax.plot(x[window-1:], smooth, color=BLUE, lw=1.3, label=f"{window}-iteration trailing mean")
        for p in data["training"][1:]:
            ax.axvline(p["rows"][0]["iteration"], color="gray", lw=.7, ls="--")
        ax.set(title=title, xlabel="Completed iteration"); clean(ax)
    axes[0,0].legend(frameon=False, fontsize=8)
    save(fig, output, "training-dynamics")

    pre = data["pretraining"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for ax, field, title in zip(axes, ("policy", "value", "q"),
                               ("(a) Policy cross-entropy", "(b) State-value MSE", "(c) Action-value MSE")):
        for prefix, label, color in (("train", "Training", BLUE), ("val", "Position holdout", RED)):
            ax.plot([r["epoch"] for r in pre], [r[f"{prefix}_{field}"] for r in pre],
                    "o-", markersize=3, color=color, label=label)
        ax.set(title=title, xlabel="Pretraining epoch"); clean(ax)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, output, "pretraining-losses")

    quotas = data["population"]["quotas"]
    fig, ax = plt.subplots(figsize=(8.8, 3.7), layout="constrained")
    requested, actual = Counter(), Counter()
    for phase in data["training"]:
        requested.update(phase["requested"]); actual.update(phase["actual"])
    keys = list(quotas); pos = np.arange(len(keys))
    ax.bar(pos-.24, [100*quotas[k]/sum(quotas.values()) for k in keys], width=.24,
           label="Configured quota", color="#adc2da")
    ax.bar(pos, [100*requested[k]/sum(requested.values()) for k in keys], width=.24,
           label="Recorded request", color=BLUE)
    ax.bar(pos+.24, [100*actual[k]/sum(actual.values()) for k in keys], width=.24,
           label="Recorded actual family", color=GREEN)
    ax.set(xticks=pos, xticklabels=keys, ylabel="Share of games (%)",
           title="Population composition in the included training segments")
    ax.legend(frameon=False, ncols=3, fontsize=8); clean(ax)
    save(fig, output, "population-composition")


def diagram_canvas(title, height=5):
    fig, ax = plt.subplots(figsize=(12, height))
    ax.set(xlim=(0, 12), ylim=(0, height)); ax.axis("off")
    ax.text(.15, height-.3, title, fontsize=14, weight="bold", va="top")
    return fig, ax


def box(ax, x, y, w, h, label, color="#e8edf4", size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.04,rounding_size=.07",
                              facecolor=color, edgecolor="#555555", lw=.8))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=size)


def arrow(ax, a, b, label=None, dashed=False, both=False):
    ax.annotate("", xy=b, xytext=a,
                arrowprops={"arrowstyle": "<->" if both else "->", "lw": 1, "color": "#333333",
                            "linestyle": "--" if dashed else "-"})
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.1, label, ha="center", fontsize=8)


def diagrams(output):
    fig, ax = diagram_canvas("CPU search workers and the inference owner", 4.7)
    for y, label in [(2.8, "Worker 1"), (1.8, "Worker 2"), (.8, "Worker N")]:
        box(ax, .2, y, 2.4, .6, label+" · native PUCT")
        arrow(ax, (2.65, y+.3), (4.1, 2.1), both=True)
    box(ax, 4.15, 1.65, 2.5, .95, "Inference request queue\nBatch across games", "#e3eadf")
    box(ax, 8, 1.65, 3.5, .95, "One model owner\nCPU or CUDA inference", "#f1e2dc")
    arrow(ax, (6.7, 2.35), (7.95, 2.35), "states")
    arrow(ax, (7.95, 1.9), (6.7, 1.9), dashed=True)
    ax.text(7.3, 1.35, "policy, value", ha="center", fontsize=9)
    ax.text(3.6, .8, "requests / replies", ha="center", fontsize=8)
    ax.text(6, .35, "Generate all games → assemble replay → optimize → save checkpoint → next iteration",
            ha="center", fontsize=10)
    ax.text(6, 3.75, "Workers are spawned; model weights remain fixed during each game-generation phase.",
            ha="center", fontsize=10)
    save(fig, output, "inference-flow")

    fig, ax = diagram_canvas("Paired evaluation protocol", 3.8)
    box(ax, .2, 1.35, 2.1, .8, "Freeze checkpoint\nSHA-256 + settings")
    box(ax, 3, 1.35, 2.1, .8, "Generate opening\nSeed + pair ID", "#e3eadf")
    box(ax, 6, 2, 2.5, .6, "Game 1: A = X, B = O")
    box(ax, 6, .85, 2.5, .6, "Game 2: B = X, A = O")
    box(ax, 9.4, 1.2, 2.25, 1.1, "Mean pair score\nBootstrap over pairs\nW / D / L", "#f1e2dc")
    arrow(ax, (2.35, 1.75), (2.95, 1.75))
    for y in (2.3, 1.15):
        arrow(ax, (5.15, 1.75), (5.95, y)); arrow(ax, (8.55, y), (9.35, 1.75))
    ax.text(6, .25, "Each pair reuses one board position. Swapping sides does not mean rotating the board.",
            ha="center", fontsize=10)
    save(fig, output, "evaluation-flow")


def write_summary(data, dest):
    rows = [r for p in data["training"] for r in p["rows"]]
    summary = {"training_rows": len(rows), "first_iteration": rows[0]["iteration"],
               "last_iteration": rows[-1]["iteration"],
               "recorded_training_games": sum(r["games"] for r in rows),
               "median_iteration_seconds": float(np.median([r["seconds"] for r in rows])),
               "evaluation_overhead_records": sum(len(p["overhead"]) for p in data["training"]),
               "experiments": []}
    for e in data["experiments"]:
        t = e["championship"]; c = t["participants"][0]
        item = {"label": e["label"], "candidate": c, "matchups": {}, "depth10": {}}
        for opp in t["participants"][1:]:
            rows_ = [r for r in e["games"] if {r["player_x"], r["player_o"]} == {c, opp}]
            p = pair_scores(rows_, c)
            item["matchups"][opp] = {**t["matrix"][c][opp], "score": float(p.mean()),
                                      "pair_bootstrap_ci95": interval(p).tolist()}
        for budget, arm in e["sweep"]["budgets"].items():
            counts = Counter(score(r, arm["entrant"]) for r in arm["game_rows"])
            item["depth10"][budget] = {"wins": counts[1.], "draws": counts[.5], "losses": counts[0.],
                                      "score": arm["score_rate"], "ci95": [arm["ci_low"], arm["ci_high"]]}
        summary["experiments"].append(item)
    dest.write_text(json.dumps(summary, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.config) if args.config else json.loads(args.snapshot.read_text())
    validate(data)
    evidence = args.output / "evidence"; figures = args.output / "figures"
    evidence.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    if args.config:
        (evidence / "snapshot.json").write_text(json.dumps(data, separators=(",", ":"))+"\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.linewidth": .7,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    results_figures(data, figures)
    training_figures(data, figures)
    diagrams(figures)
    write_summary(data, evidence / "summary.json")
    print(f"Verified game aggregates and paired intervals; wrote figures to {figures}")


if __name__ == "__main__":
    main()
