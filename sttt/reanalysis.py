"""Loss review / reanalysis.

Game records keep every action, who played it and the learner's root search.
Review replays a record, re-searches chosen decision points from the SAME
parent position with a stronger budget and the current network, and compares
the played move with the alternatives from the player-to-move's view. The
refreshed search (policy, root Q, root value) becomes extra replay targets;
opponent moves are never imitated as one-hot labels.

A record needs only ``actions``. ``movers`` ('opening' | 'learner' |
'opponent'), ``learner_outcome``, ``result`` and ``id`` are optional, so
championship games can be reviewed once they store their move sequence.
"""
import json
from pathlib import Path
import numpy as np
from .env import State
from .search import TreeSearch, root_action_values, root_value

OUTCOMES = ('loss', 'draw', 'win')
SIDES = ('learner', 'opponent')
VALUE_TARGETS = ('outcome', 'search', 'mix')


def learner_outcome(result, match):
    if result == 0:
        return 'draw'
    # Self-play: both sides are the learner, so every decisive game is a loss
    # for one of them.
    if match.get('kind', 'self') == 'self':
        return 'loss'
    return 'win' if result * match['learner_side'] > 0 else 'loss'


def game_record(iteration, index, seed, trajectory, result, stats):
    """JSON-ready record of one self-play/population game."""
    record, match = stats['record'], stats['match']
    learner_plies = [i for i, mover in enumerate(record['movers']) if mover == 'learner']
    if len(learner_plies) != len(trajectory):
        raise ValueError('game record and trajectory disagree on learner moves')
    search = [{'ply': ply,
               'policy': [[int(a), round(float(pi[a]), 5)] for a in np.flatnonzero(pi)],
               'q': [[int(a), round(float(q[a]), 5)] for a in np.flatnonzero(q_mask)],
               'root_value': round(float(value), 5)}
              for ply, (_, pi, q, q_mask), value in zip(learner_plies, trajectory, record['root_values'])]
    return {'id': f'{iteration}-{index}', 'iteration': int(iteration), 'game': int(index),
            'seed': int(seed), 'match': match, 'learner_side': int(match['learner_side']),
            'result': int(result), 'learner_outcome': learner_outcome(result, match),
            'actions': [int(a) for a in record['actions']], 'movers': list(record['movers']),
            'learner_search': search}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        for row in rows:
            file.write(json.dumps(row, separators=(',', ':')) + '\n')


def load_records(path):
    """Records from one JSONL file or every games-*.jsonl in a directory."""
    path = Path(path)
    files = sorted(path.glob('games-*.jsonl')) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f'no games-*.jsonl under {path}')
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]


def replay_states(actions, use_cpp=False):
    """states[i] is the position before actions[i]; states[-1] is the final one."""
    if use_cpp:
        from .cpp_env import FastState
        state = FastState()
    else:
        state = State()
    states = [state]
    for action in actions:
        state = state.play(int(action))
        states.append(state)
    return states


def review_position(evaluator, parent, played, simulations, config, leaf_batch=1,
                    use_cpp=False, rng=None, margin=.3):
    """Stronger, noise-free search of `parent`, judged from its player to move.

    q_* values and `value` are from the parent's player-to-move view. A move is
    a suspected mistake when the best alternative's Q beats it by more than
    `margin`; it is `proven` only when both Q values are exact proof values.
    The best alternative is the most-visited other move.
    """
    def search(state):
        if use_cpp:
            from .cpp_env import CppTreeSearch
            tree = CppTreeSearch(evaluator, rng or np.random.default_rng(0), config)
        else:
            tree = TreeSearch(evaluator, rng or np.random.default_rng(0), config)
        return tree, tree.run(state, simulations, batch_size=leaf_batch, noise=False)
    tree, pi = search(parent)
    q, q_mask = root_action_values(tree.root)
    solved = {int(a): getattr(c, 'solved', None) for a, c in tree.root.children.items()}
    played = int(played)
    if q_mask[played]:
        q_played = float(q[played])
    else:
        # Never visited (e.g. the root was proven early): judge the played move
        # with its own search from the child, so a forced loss is still exact.
        child = parent.play(played)
        if child.result is not None:
            q_played = float(child.result * parent.turn)
            solved[played] = -q_played
        else:
            sub, _ = search(child)
            q_played = -root_value(sub.root)
            solved[played] = getattr(sub.root, 'solved', None)
    # The best alternative is the search's own preference (visit share, then Q):
    # the max-Q child is often a one-visit outlier and would flag noise.
    alternatives = [int(a) for a in np.flatnonzero(q_mask) if a != played]
    best = max(alternatives, key=lambda a: (pi[a], q[a])) if alternatives else None
    gap = float(q[best]) - q_played if best is not None else 0.
    mistake = best is not None and gap > margin
    return {'pi': pi, 'q': q, 'q_mask': q_mask, 'value': root_value(tree.root),
            'played': played, 'q_played': q_played, 'best': best,
            'q_best': float(q[best]) if best is not None else None, 'gap': gap,
            'q_played_exact': solved.get(played) is not None,
            'q_best_exact': best is not None and solved.get(best) is not None,
            'suspected_mistake': bool(mistake),
            'proven': bool(mistake and solved.get(played) is not None
                           and solved.get(best) is not None)}


def review_game(evaluator, task, simulations, config, leaf_batch=1, use_cpp=False):
    """Review task['plies'] of one game; each review also carries its parent state."""
    actions, movers = task['actions'], task.get('movers')
    states = replay_states(actions, use_cpp)
    rng = np.random.default_rng(task.get('seed', 0))
    reviews = []
    for ply in task['plies']:
        review = review_position(evaluator, states[ply], actions[ply], simulations, config,
                                 leaf_batch, use_cpp, rng, task.get('margin', .3))
        review.update(ply=int(ply), mover=movers[ply] if movers else 'unknown', state=states[ply])
        reviews.append(review)
    return reviews


def select_tasks(records, outcomes=('loss',), sides=SIDES, budget=None, rng=None, margin=.3):
    """Decision points to review, grouped per game.

    Opening plies are random, not decisions, and are skipped. Records without
    `learner_outcome` / `movers` (e.g. external games) are always eligible and
    every ply after their optional `opening_moves` is reviewed. With `budget`,
    a uniform sample of that many plies.
    """
    candidates = []
    for g, record in enumerate(records):
        if 'learner_outcome' in record and record['learner_outcome'] not in outcomes:
            continue
        opening = int(record.get('opening_moves', 0))
        movers = record.get('movers') or (['opening'] * opening
                                          + ['unknown'] * (len(record['actions']) - opening))
        candidates += [(g, ply) for ply, mover in enumerate(movers)
                       if mover != 'opening' and (mover == 'unknown' or mover in sides)]
    if budget is not None and len(candidates) > budget:
        rng = rng if rng is not None else np.random.default_rng(0)
        candidates = [candidates[i] for i in sorted(rng.choice(len(candidates), budget, replace=False))]
    tasks = {}
    for g, ply in candidates:
        if g not in tasks:
            record = records[g]
            result = record.get('result')
            tasks[g] = {'id': record.get('id', str(g)), 'actions': record['actions'],
                        'movers': record.get('movers'),
                        'result': replay_states(record['actions'])[-1].result if result is None else result,
                        'learner_outcome': record.get('learner_outcome'), 'plies': [], 'margin': margin,
                        'seed': int(rng.integers(2**31)) if rng is not None else g}
        tasks[g]['plies'].append(ply)
    return list(tasks.values())


def value_target(mode, outcome_z, search_value, lam=.5):
    """outcome: final result; search: reanalysis root value; mix: lam*outcome + (1-lam)*search.
    All from the player-to-move's view."""
    if mode == 'outcome':
        return float(outcome_z)
    if mode == 'search':
        return float(search_value)
    if mode == 'mix':
        return float(lam * outcome_z + (1. - lam) * search_value)
    raise ValueError(f'unknown value target {mode!r}')


def reanalysis_rows(tasks, reviews, mode='outcome', lam=.5):
    """Replay 6-tuples (x, pi, mask, z, q, q_mask) from reviewed positions. The
    policy target is always the reanalysis search, never the move played."""
    import torch
    from .learning import encode_states
    rows = []
    for task, game_reviews in zip(tasks, reviews):
        if not game_reviews:
            continue
        x, mask = encode_states([r['state'] for r in game_reviews])
        for i, r in enumerate(game_reviews):
            outcome_z = task['result'] * r['state'].turn
            rows.append((torch.from_numpy(x[i].copy()), torch.from_numpy(np.asarray(r['pi'], dtype=np.float32)),
                         torch.from_numpy(mask[i].copy()), value_target(mode, outcome_z, r['value'], lam),
                         torch.from_numpy(np.asarray(r['q'], dtype=np.float32)),
                         torch.from_numpy(np.asarray(r['q_mask'], dtype=bool))))
    return rows


def game_report(task, reviews):
    """Per-game findings: suspected mistakes with played move, best alternative and Q gap."""
    mistakes = [{'ply': r['ply'], 'mover': r['mover'], 'played': r['played'], 'best': r['best'],
                 'q_played': round(r['q_played'], 4), 'q_best': round(r['q_best'], 4),
                 'q_gap': round(r['gap'], 4), 'root_value': round(r['value'], 4),
                 'q_played_exact': r['q_played_exact'], 'q_best_exact': r['q_best_exact'],
                 'proven': r['proven']}
                for r in reviews if r['suspected_mistake']]
    return {'id': task['id'], 'result': task.get('result'), 'learner_outcome': task.get('learner_outcome'),
            'positions_reviewed': len(reviews), 'suspected_mistakes': mistakes}


def summarize(reviews):
    flat = [r for game in reviews for r in game]
    return {'positions': len(flat),
            'learner_positions': sum(r['mover'] == 'learner' for r in flat),
            'opponent_positions': sum(r['mover'] == 'opponent' for r in flat),
            'suspected_mistakes': sum(r['suspected_mistake'] for r in flat),
            'proven_mistakes': sum(r['proven'] for r in flat)}


def reanalyse_cmd(args):
    """Offline review of saved game records; writes one JSON report."""
    import time
    from .ai import resolve_device, search_config
    from .learning import load_model
    from .cpp_env import is_cpp_available
    use_cpp = args.backend == 'cpp' or (args.backend == 'auto' and is_cpp_available())
    if use_cpp and not is_cpp_available():
        raise RuntimeError('--backend cpp requested but the native extension is unavailable')
    model, checkpoint = load_model(args.checkpoint)
    model.to(resolve_device(args.device))
    records = load_records(args.records)
    tasks = select_tasks(records, args.outcomes, args.sides, margin=args.margin)
    started = time.monotonic()
    reviews = []
    for task in tasks:
        reviews.append(review_game(model, task, args.simulations, search_config(args),
                                   args.leaf_batch, use_cpp=use_cpp))
        print(f'{task["id"]}: {len(task["plies"])} positions reviewed', flush=True)
    report = {'records': str(args.records), 'checkpoint': str(args.checkpoint),
              'checkpoint_iteration': checkpoint.get('iteration'), 'simulations': args.simulations,
              'margin': args.margin, 'outcomes': list(args.outcomes), 'sides': list(args.sides),
              'backend': 'cpp' if use_cpp else 'python', 'games_in_records': len(records),
              'seconds': round(time.monotonic() - started, 2), **summarize(reviews),
              'games': [game_report(t, r) for t, r in zip(tasks, reviews)]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'games'}), flush=True)
    return report
