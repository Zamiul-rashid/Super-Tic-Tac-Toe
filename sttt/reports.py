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


def write_tournament_report(tourn, directory):
    """Export structured tournament artifacts to target directory.

    Generates:
      - tournament.json: full structured JSON with participants, ratings, matchups
      - summary.csv: table with participant ratings and win/score metrics
      - matchups.csv: game-level match outcomes and opening plies
      - games.csv: alias to matchups.csv
      - scoreboard.txt: formatted ASCII leaderboard
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    participants = list(tourn.get("participants", []))
    elo_ratings = tourn.get("elo_ratings", {})
    glicko_ratings = tourn.get("glicko_ratings", {})
    results = tourn.get("results", [])
    matrix = tourn.get("matrix", {})
    side_stats = tourn.get("side_stats", {})
    scoreboard = tourn.get("scoreboard", "")
    games_per_matchup = tourn.get("games_per_matchup", 0)

    # 1. Prepare ratings dictionary & summary rows
    ratings_dict = {}
    summary_rows = []
    for p in participants:
        elo_r = elo_ratings.get(p)
        glicko_r = glicko_ratings.get(p)

        elo_val = round(float(elo_r.elo), 2) if elo_r is not None else 1500.0
        elo_ci = round(float(elo_r.error_margin), 2) if elo_r is not None else 0.0
        glicko_val = round(float(glicko_r.rating), 2) if glicko_r is not None else 1500.0
        glicko_rd = round(float(glicko_r.rd), 2) if glicko_r is not None else 350.0
        vol = round(float(glicko_r.volatility), 4) if glicko_r is not None else 0.06
        g_count = elo_r.games if elo_r is not None else (glicko_r.games if glicko_r is not None else 0)
        wins = elo_r.wins if elo_r is not None else 0
        draws = elo_r.draws if elo_r is not None else 0
        losses = elo_r.losses if elo_r is not None else 0
        win_rate = round(wins / g_count, 4) if g_count > 0 else 0.0
        score_rate = round((wins + 0.5 * draws) / g_count, 4) if g_count > 0 else 0.0

        ratings_dict[p] = {
            "elo": elo_val,
            "elo_ci95": elo_ci,
            "glicko2": glicko_val,
            "glicko2_rd": glicko_rd,
            "volatility": vol,
            "games": g_count,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "score_rate": score_rate,
        }
        summary_rows.append({
            "participant": p,
            "elo": elo_val,
            "elo_ci95": elo_ci,
            "glicko2": glicko_val,
            "glicko2_rd": glicko_rd,
            "volatility": vol,
            "games": g_count,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "score_rate": score_rate,
        })

    # Sort summary rows by Elo descending
    summary_rows.sort(key=lambda r: r["elo"], reverse=True)

    # 2. Prepare matchups list & rows
    matchups_list = []
    matchups_rows = []
    for r in results:
        m_dict = {
            "game_id": r.game_id,
            "pair_id": r.pair_id,
            "opening_plies": r.opening_plies,
            "opening_moves": list(r.opening_moves),
            "player_x": r.player_x,
            "player_o": r.player_o,
            "winner": r.winner,
            "moves": r.moves,
        }
        matchups_list.append(m_dict)
        matchups_rows.append({
            "game_id": r.game_id,
            "pair_id": r.pair_id,
            "opening_plies": r.opening_plies,
            "opening_moves": json.dumps(r.opening_moves),
            "player_x": r.player_x,
            "player_o": r.player_o,
            "winner": r.winner,
            "moves": r.moves,
        })

    # 3. Write tournament.json atomically
    tournament_json_path = directory / "tournament.json"
    temp_json = tournament_json_path.with_suffix(".tmp")
    json_payload = {
        "participants": participants,
        "games_per_matchup": games_per_matchup,
        "ratings": ratings_dict,
        "matchups": matchups_list,
        "matrix": matrix,
        "side_stats": side_stats,
    }
    temp_json.write_text(json.dumps(json_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp_json.replace(tournament_json_path)

    # 4. Write summary.csv atomically
    summary_csv_path = directory / "summary.csv"
    temp_summary = summary_csv_path.with_suffix(".tmp")
    summary_fields = ["participant", "elo", "elo_ci95", "glicko2", "glicko2_rd", "volatility", "games", "wins", "draws", "losses", "win_rate", "score_rate"]
    with temp_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    temp_summary.replace(summary_csv_path)

    # 5. Write matchups.csv and games.csv atomically
    matchups_fields = ["game_id", "pair_id", "opening_plies", "opening_moves", "player_x", "player_o", "winner", "moves"]
    for fname in ("matchups.csv", "games.csv"):
        target_csv = directory / fname
        temp_csv = target_csv.with_suffix(".tmp")
        with temp_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=matchups_fields)
            writer.writeheader()
            writer.writerows(matchups_rows)
        temp_csv.replace(target_csv)

    # 6. Write scoreboard.txt atomically
    scoreboard_txt_path = directory / "scoreboard.txt"
    temp_board = scoreboard_txt_path.with_suffix(".tmp")
    temp_board.write_text(scoreboard + "\n", encoding="utf-8")
    temp_board.replace(scoreboard_txt_path)

    return tournament_json_path
