"""Versioned evaluation exports, including completed games from interrupted runs."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def summarize(games):
    counts = {key: sum(g['outcome'] == key for g in games)
              for key in ('wins', 'draws', 'losses')}
    n = len(games)
    return {**counts, 'games': n,
            'win_rate': counts['wins'] / n if n else None,
            'score_rate': (counts['wins'] + 0.5 * counts['draws']) / n if n else None,
            'mean_moves': sum(g['moves'] for g in games) / n if n else None,
            'mean_seconds': sum(g['seconds'] for g in games) / n if n else None}


def new_report(checkpoint, iteration, opponent, simulations, seed, requested_games):
    path = Path(checkpoint).resolve()
    hasher = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    return {'schema_version': 1, 'kind': 'sttt_evaluation',
            'run_id': uuid4().hex,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'checkpoint': str(path), 'checkpoint_sha256': digest,
            'iteration': iteration, 'opponent': opponent,
            'simulations': simulations, 'seed': seed,
            'requested_games': requested_games,
            'adaptation': False, 'status': 'running', 'games': []}


def write_report(report, directory):
    """All exports share a unique stem. CSV metadata repeats to support concatenation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report['summary'] = summarize(report['games'])
    report['by_side'] = {side: summarize([g for g in report['games'] if g['agent_side'] == side])
                         for side in ('X', 'O')}
    stem = directory / ('eval-' + report['run_id'])
    target = stem.with_suffix('.json')
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(target)
    metadata = {key: report[key] for key in (
        'schema_version', 'run_id', 'created_at', 'checkpoint', 'checkpoint_sha256',
        'iteration', 'opponent', 'simulations', 'seed', 'requested_games', 'adaptation', 'status')}
    for key in ('search_version', 'search_config', 'leaf_batch', 'device', 'opening_moves', 'opponent_depth',
                'opponent_nodes', 'opponent_simulations', 'opponent_checkpoint', 'opponent_checkpoint_sha256'):
        value = report.get(key)
        metadata[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
    game_columns = ['game', 'agent_side', 'opening_pair', 'opening_actions', 'outcome', 'score', 'moves', 'seconds',
                    'agent_moves', 'agent_search_seconds', 'completed_simulations', 'neural_positions',
                    'max_depth', 'hard_pruned_choices', 'soft_rechecks', 'retained_visits']
    exports = [('_games.csv', [*metadata, *game_columns],
                [{**metadata, **{key: json.dumps(value) if isinstance(value, list) else value
                                for key, value in g.items()}} for g in report['games']]),
               ('_summary.csv', [*metadata, *report['summary']],
                [{**metadata, **report['summary']}])]
    for suffix, fields, rows in exports:
        destination = Path(str(stem) + suffix)
        temporary = destination.with_suffix('.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
    return target
