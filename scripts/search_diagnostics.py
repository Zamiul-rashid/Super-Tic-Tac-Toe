#!/usr/bin/env python3
"""Root-level search diagnostics for one checkpoint on fixed tactical and
opening positions. Records, per backend / budget / leaf batch: root policy,
visits, Q, model prior and value, completed simulations, network evaluations,
proof status and wall time. A diagnostic, not a strength measurement.

    python scripts/search_diagnostics.py --checkpoint PATH --output OUT.json \
        [--budgets 128 512 2000] [--leaf-batch 16] [--backends python cpp]
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sttt.env import State                                           # noqa: E402
from sttt.learning import load_model                                 # noqa: E402
from sttt.search import TreeSearch, SearchConfig, root_action_values  # noqa: E402
from sttt.cpp_env import CppTreeSearch, get_cpp_provenance           # noqa: E402
from tests.test_search_audit import (WIN, BLOCK, AVOID, LOSE, DRAWN,  # noqa: E402
                                     TERMINAL_OPTION, build, O_WON)


def mirror(state):
    """Same position with colours swapped (O to move)."""
    flip = lambda v: v if v == 2 else -v
    return State(tuple(-c for c in state.cells), tuple(flip(b) for b in state.boards),
                 -state.turn, state.forced)


def positions():
    opening = State()
    for a in (40, 36, 4, 44):
        opening = opening.play(a)
    # name, state, actions that avoid an immediate loss / take the win (None: any)
    return [
        ('global-win', WIN, [20]),
        ('global-win-O', mirror(WIN), [20]),
        ('global-win-free-choice', WIN.__class__(WIN.cells, WIN.boards, 1, -1), [20]),
        ('forced-block', BLOCK, [20]),
        ('forced-block-O', mirror(BLOCK), [20]),
        ('avoid-sending-to-threat', AVOID, [39]),
        ('avoid-sending-open-board', build({0: O_WON, 1: O_WON, 2: [-1, -1] + [0] * 7},
                                           (-1, -1, 0, 0, 0, 0, 0, 0, 0), 1, 4),
         [39, 40, 41, 42, 43, 44]),
        ('forced-loss', LOSE, None),
        ('proven-draw', DRAWN, None),
        ('terminal-draw-option', TERMINAL_OPTION, None),
        ('opening-empty', State(), None),
        ('opening-center', State().play(40), None),
        ('opening-4ply', opening, None),
    ]


def run(backend, model, state, sims, batch):
    cls = TreeSearch if backend == 'python' else CppTreeSearch
    tree = cls(model, config=SearchConfig())
    start = time.perf_counter()
    pi = tree.run(state, sims, batch_size=batch)
    seconds = time.perf_counter() - start
    q, visited = root_action_values(tree.root)
    children = sorted(((a, c.n) for a, c in tree.root.children.items()), key=lambda x: -x[1])
    return dict(
        chosen=int(pi.argmax()), seconds=round(seconds, 4), root_solved=tree.root.solved,
        stats={k: v for k, v in tree.stats.items() if not k.startswith('arena')},
        top=[dict(action=int(a), visits=int(n), policy=round(float(pi[a]), 4),
                  q=round(float(q[a]), 4) if visited[a] else None) for a, n in children[:6]],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[128, 512, 2000])
    ap.add_argument('--leaf-batch', type=int, nargs='+', default=[16])
    ap.add_argument('--backends', nargs='+', default=['python', 'cpp'], choices=['python', 'cpp'])
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, ckpt = load_model(args.checkpoint)
    rows = []
    for name, state, safe in positions():
        prior, value = model.evaluate(state)
        top_prior = np.argsort(-prior)[:5]
        for backend in args.backends:
            for batch in args.leaf_batch:
                for sims in args.budgets:
                    row = dict(position=name, turn=state.turn, backend=backend, leaf_batch=batch,
                               simulations=sims, expected=safe,
                               model_value=round(float(value), 4),
                               model_prior={int(a): round(float(prior[a]), 4) for a in top_prior},
                               **run(backend, model, state, sims, batch))
                    row['ok'] = None if safe is None else row['chosen'] in safe
                    rows.append(row)
                    print(name, backend, batch, sims, row['chosen'], row['ok'], row['stats'], row['seconds'])
    report = dict(
        checkpoint=str(args.checkpoint), iteration=ckpt.get('iteration'),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        native=get_cpp_provenance(), search_config=asdict(SearchConfig()), torch_threads=args.threads,
        rows=rows)
    Path(args.output).write_text(json.dumps(report, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
