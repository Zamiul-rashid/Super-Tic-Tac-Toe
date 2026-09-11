"""Run with python -m sttt.ai train|play|evaluate|tournament."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from .env import State
from .learning import ResNet, create_model, encode, load_model
from .opponent import Opponent, policies, NAMES
from .search import TreeSearch, SearchConfig
from .selfplay import SelfPlayPool
from .bots import TacticalBot, AlphaBetaBot, CheckpointBot, create_bot, Bot
from .reports import new_report, write_report, write_tournament_report
from .tournament import run_tournament, run_simulation_sweep

def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value

def search_config(args):
    return SearchConfig(soft_pruning=not getattr(args, 'no_soft_pruning', False),
                        proofs=not getattr(args, 'no_proofs', False),
                        reuse=not getattr(args, 'no_reuse', False))


def resolve_device(device):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Use --device cpu or check NVIDIA/PyTorch access.')
    return device

def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    if args.resume:
        model, saved = load_model(args.resume)
        arch = 'resnet' if isinstance(model, ResNet) else 'mlp'
    else:
        arch = getattr(args, 'arch', 'resnet')
        model = create_model(arch)
        saved = {}
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if 'optimizer' in saved:
        optimizer.load_state_dict(saved['optimizer'])
    replay = deque(saved.get('replay', []), maxlen=args.buffer)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f'{device}: {sum(p.numel() for p in model.parameters()):,} parameters ({arch}); workers={args.workers}; replay stays in RAM', flush=True)
    with SelfPlayPool(min(args.workers, args.games), args.inference_batch, args.inference_wait_ms) as pool:
        _train_loop(args, model, saved, arch, optimizer, replay, output, rng, device, pool)


def _train_loop(args, model, saved, arch, optimizer, replay, output, rng, device, pool):
    start_iteration = saved.get('iteration', 0)
    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        started = time.monotonic()
        seeds = rng.integers(0, 10**9, size=args.games)
        results, inference_stats = pool.run(model, seeds, args.simulations, search_config(args), args.leaf_batch)
        selfplay_seconds = time.monotonic() - started
        for game_idx, (trajectory, outcome, stats) in enumerate(results):
            for state, pi in trajectory:
                mask = np.zeros(81, dtype=bool)
                mask[state.legal_actions()] = True
                replay.append((torch.from_numpy(encode(state)), torch.from_numpy(pi),
                               torch.from_numpy(mask), float(outcome * state.turn)))
            print(f'iteration {iteration}: game {game_idx+1}/{args.games}, {len(trajectory)} moves', flush=True)

        model.train()
        losses = []
        for _ in range(args.steps):
            indices = rng.choice(len(replay), size=min(args.batch, len(replay)), replace=False)
            batch = [replay[int(i)] for i in indices]
            x, pi, mask = [torch.stack([row[k] for row in batch]).to(device) for k in range(3)]
            z = torch.tensor([row[3] for row in batch], device=device)
            logits, value = model(x)
            logits = logits.masked_fill(~mask, -1e9)
            loss = -(pi * logits.log_softmax(-1)).sum(-1).mean() + (value - z).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'iteration': iteration, 'replay': list(replay), 'arch': arch,
                      'training_config': vars(args), 'search_config': asdict(search_config(args))}
        temporary = output / 'latest.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(output / 'latest.pt')
        # Small historical weights retained separately for evaluation and milestones.
        save_snapshot = (getattr(args, 'save_every', 50) and iteration % args.save_every == 0) or (args.eval_every and iteration % args.eval_every == 0)
        if save_snapshot:
            torch.save({'model': model.state_dict(), 'iteration': iteration, 'arch': arch}, output / f'model-{iteration:04d}.pt')
        report = {'iteration': iteration, 'positions': len(replay), 'loss': float(np.mean(losses)),
                  'seconds': round(time.monotonic() - started, 2), 'selfplay_seconds': selfplay_seconds,
                  'games': args.games, 'simulations': args.simulations, 'leaf_batch': args.leaf_batch,
                  'search_config': asdict(search_config(args)), **inference_stats,
                  'completed_simulations': sum(r[2]['completed_simulations'] for r in results),
                  'max_search_depth': max(r[2]['max_depth'] for r in results),
                  'soft_rechecks': sum(r[2]['soft_rechecks'] for r in results),
                  'hard_pruned_choices': sum(r[2]['hard_pruned_choices'] for r in results),
                  'retained_visits': sum(r[2]['retained_visits'] for r in results)}
        if device.startswith('cuda'):
            report['peak_gpu_mb'] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        with (output / 'metrics.jsonl').open('a') as file:
            file.write(json.dumps(report) + '\n')
        print(json.dumps(report), flush=True)
        if args.eval_every and iteration % args.eval_every == 0:
            evaluation = argparse.Namespace(**vars(args))
            evaluation.checkpoint = str(output / f'model-{iteration:04d}.pt')
            evaluation.output = str(output / 'evaluations')
            evaluation.games, evaluation.simulations = args.eval_games, args.eval_simulations
            evaluation.opponent = 'alphabeta'
            evaluation.opponent_checkpoint = None
            evaluation.opponent_depth, evaluation.opponent_nodes = 3, 3000
            evaluation.opponent_simulations = 256
            evaluate(evaluation)

def play(args):
    model,_ = load_model(args.checkpoint)
    model.to(resolve_device(args.device))
    opponent = Opponent(args.profile)
    rng = np.random.default_rng(args.seed)
    state = State()
    human = 1 if args.side == 'X' else -1
    tree = TreeSearch(model, rng, search_config(args),
                      opponent=None if args.no_adapt else opponent, agent_side=-human)
    print('Enter board and cell numbers 1–9, e.g. 5 3. Enter q to quit.')
    while state.result is None:
        print('\n'+state.render())
        if state.turn == human:
            line = input('Your move: ').strip()
            if line.lower() == 'q':
                return
            try:
                board,cell = map(int,line.split())
                if not (1 <= board <= 9 and 1 <= cell <= 9):
                    raise ValueError()
                action = (board-1)*9+cell-1
                opponent.observe(state,action)
                tree.advance(action)
                if not args.no_adapt:
                    tree.reset()  # Human-policy update invalidates expected-response statistics.
                state = state.play(action)
            except ValueError:
                print('Enter a legal board and cell, both between 1 and 9.')
        else:
            pi = tree.run(state, args.simulations, batch_size=args.leaf_batch)
            action = int(pi.argmax())
            print(f'AI plays board {action//9+1}, cell {action%9+1}')
            print(f'Search: {tree.stats}')
            tree.advance(action)
            state = state.play(action)
    print(state.render())
    print('Draw' if state.result == 0 else ('You win' if state.result == human else 'AI wins'))

def evaluation_opening(seed, pair, plies):
    """Repeat an opening with swapped agent sides; keep it fixed across budgets."""
    if not 0 <= plies <= 8:
        raise ValueError('Opening moves must be between 0 and 8')
    rng = np.random.default_rng(np.random.SeedSequence([seed, pair, 1741]))
    state, actions = State(), []
    for _ in range(plies):
        action = int(rng.choice(state.legal_actions()))
        actions.append(action)
        state = state.play(action)
    return state, actions


def evaluate(args):
    model,checkpoint = load_model(args.checkpoint)
    model.to(resolve_device(getattr(args, 'device', 'cpu')))
    config = search_config(args)
    leaf_batch = getattr(args, 'leaf_batch', 1)
    opening_moves = getattr(args, 'opening_moves', 2)
    opponent_model = None
    bot = None
    if args.opponent == 'checkpoint':
        if not args.opponent_checkpoint:
            raise ValueError('--opponent checkpoint requires --opponent-checkpoint')
        opponent_model, _ = load_model(args.opponent_checkpoint)
        opponent_model.to(next(model.parameters()).device)
    elif args.opponent == 'tactical':
        bot = TacticalBot()
    elif args.opponent == 'alphabeta':
        bot = AlphaBetaBot(args.opponent_depth, args.opponent_nodes)
    opponent_name = {'tactical': 'tactical-v2', 'alphabeta': 'alphabeta-v1',
                     'legacy-tactical': 'legacy-tactical-v1'}.get(args.opponent, args.opponent)
    report = new_report(args.checkpoint, checkpoint.get('iteration'), opponent_name,
                        args.simulations, args.seed, args.games)
    report.update(search_config=asdict(config), leaf_batch=leaf_batch, opening_moves=opening_moves,
                  device=str(next(model.parameters()).device), search_version='batched-puct-v2',
                  opponent_depth=bot.depth if bot else None,
                  opponent_nodes=bot.node_budget if bot else None,
                  opponent_simulations=args.opponent_simulations if opponent_model is not None else None)
    if opponent_model is not None:
        identity = new_report(args.opponent_checkpoint, None, '', 0, 0, 0)
        report['opponent_checkpoint'] = identity['checkpoint']
        report['opponent_checkpoint_sha256'] = identity['checkpoint_sha256']
    output = args.output or str(Path(args.checkpoint).parent / 'evaluations')
    path = write_report(report, output)
    try:
        for game in range(args.games):
            streams = np.random.SeedSequence([args.seed, game]).spawn(2)
            rng, opponent_rng = [np.random.default_rng(stream) for stream in streams]
            tree = TreeSearch(model, rng, config)
            opponent_tree = TreeSearch(opponent_model, opponent_rng) if opponent_model is not None else None
            started = time.monotonic()
            state, opening_actions = evaluation_opening(args.seed, game // 2, opening_moves)
            agent = 1 if game%2 == 0 else -1
            moves, agent_moves, search_seconds = len(opening_actions), 0, 0.
            search_totals = dict(completed_simulations=0, neural_positions=0, max_depth=0,
                                 hard_pruned_choices=0, soft_rechecks=0, retained_visits=0)
            while state.result is None:
                if state.turn == agent:
                    search_start = time.monotonic()
                    action = int(tree.run(state, args.simulations, batch_size=leaf_batch).argmax())
                    for key in search_totals:
                        search_totals[key] = (max(search_totals[key], tree.stats[key]) if key == 'max_depth'
                                              else search_totals[key] + tree.stats[key])
                    search_seconds += time.monotonic() - search_start
                    agent_moves += 1
                elif opponent_tree is not None:
                    action = int(opponent_tree.run(state, args.opponent_simulations, batch_size=leaf_batch).argmax())
                elif bot is not None:
                    action = bot.choose(state, opponent_rng)
                else:
                    policy_idx = NAMES.index(args.opponent) if args.opponent in NAMES else 4
                    p = policies(state)[policy_idx]
                    action = int(opponent_rng.choice(81, p=p))
                tree.advance(action)
                if opponent_tree is not None:
                    opponent_tree.advance(action)
                state = state.play(action)
                moves += 1
            key = 'draws' if state.result == 0 else ('wins' if state.result == agent else 'losses')
            report['games'].append({'game': game+1, 'agent_side': 'X' if agent == 1 else 'O',
                                    'opening_pair': game // 2 + 1, 'opening_actions': opening_actions,
                                    'outcome': key, 'score': {'wins': 1., 'draws': .5, 'losses': 0.}[key],
                                    'moves': moves, 'seconds': time.monotonic()-started,
                                    'agent_moves': agent_moves, 'agent_search_seconds': search_seconds,
                                    **search_totals})
            write_report(report, output)
            print(f'game {game+1}/{args.games}: {key}',flush=True)
        report['status'] = 'complete'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        print('\nEvaluation interrupted; completed games have been saved.')
    except Exception:
        report['status'] = 'failed'
        raise
    finally:
        write_report(report, output)
        print(f'Reports: {path} (plus _games.csv and _summary.csv)', flush=True)
    print(json.dumps(report['summary']))
    return report


def tournament_cmd(args):
    device = resolve_device(getattr(args, 'device', 'cpu'))
    if getattr(args, 'output', None):
        output_dir = Path(args.output)
    elif getattr(args, 'checkpoint', None):
        output_dir = Path(args.checkpoint).parent / 'tournaments'
    else:
        output_dir = Path('runs/tournaments')
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, 'simulation_budgets', None):
        checkpoint_path = args.checkpoint if getattr(args, 'checkpoint', None) else ""
        opp_specs = args.opponents if getattr(args, 'opponents', None) else ["tactical", "alphabeta"]
        sweep_summary = run_simulation_sweep(
            checkpoint_path=checkpoint_path,
            opponent_specs=opp_specs,
            budgets=args.simulation_budgets,
            games=args.games,
            opening_plies=args.opening_plies,
            seed=args.seed,
            device=device,
            output_dir=output_dir,
        )
        lines = [
            "=" * 110,
            "                               SIMULATION BUDGET SCALING BENCHMARK",
            "=" * 110,
            f"{'Budget':<10} {'Bayesian Elo (95% CI)':<25} {'Glicko-2 (±RD)':<22} {'Win Rate':<10} {'Score Rate':<12} {'W':<5} {'D':<5} {'L':<5} {'Games':<6}",
            "-" * 110,
        ]
        for row in sweep_summary.get("scaling", []):
            b = row["budget"]
            elo_str = f"{row['elo']:.1f} [±{row['elo_ci95']:.1f}]"
            g_str = f"{row['glicko2']:.1f} (±{row['glicko2_rd']:.1f})"
            win_pct = f"{row['win_rate'] * 100:.1f}%"
            score_pct = f"{row['score_rate'] * 100:.1f}%"
            lines.append(
                f"{b:<10} {elo_str:<25} {g_str:<22} {win_pct:<10} {score_pct:<12} {row['wins']:<5} {row['draws']:<5} {row['losses']:<5} {row['games']:<6}"
            )
        lines.append("=" * 110)
        scoreboard_text = "\n".join(lines)
        (output_dir / "scoreboard.txt").write_text(scoreboard_text + "\n", encoding="utf-8")
        print(scoreboard_text)
        return sweep_summary

    participants = {}
    if getattr(args, 'checkpoint', None):
        ckpt_bot = CheckpointBot(
            path=args.checkpoint,
            simulations=args.simulations,
            device=device,
        )
        participants[ckpt_bot.name] = ckpt_bot

    opp_list = getattr(args, 'opponents', None) or []
    for opp in opp_list:
        bot = opp if isinstance(opp, Bot) else create_bot(opp)
        name = bot.name
        if name in participants:
            suffix = 2
            while f"{name}_{suffix}" in participants:
                suffix += 1
            name = f"{name}_{suffix}"
        participants[name] = bot

    if len(participants) < 2:
        for default_bot in ("alphabeta", "tactical"):
            if len(participants) >= 2:
                break
            b = create_bot(default_bot)
            if b.name not in participants:
                participants[b.name] = b

    tourn = run_tournament(
        bots=participants,
        games_per_matchup=args.games,
        opening_plies=args.opening_plies,
        seed=args.seed,
    )
    ratings_dict = {}
    for p in tourn["participants"]:
        elo_r = tourn["elo_ratings"].get(p)
        glicko_r = tourn["glicko_ratings"].get(p)
        ratings_dict[p] = {
            "elo": round(float(elo_r.elo), 2) if elo_r is not None else 1500.0,
            "elo_ci95": round(float(elo_r.error_margin), 2) if elo_r is not None else 0.0,
            "glicko2": round(float(glicko_r.rating), 2) if glicko_r is not None else 1500.0,
            "glicko2_rd": round(float(glicko_r.rd), 2) if glicko_r is not None else 350.0,
            "volatility": round(float(glicko_r.volatility), 4) if glicko_r is not None else 0.06,
            "games": elo_r.games if elo_r is not None else 0,
            "wins": elo_r.wins if elo_r is not None else 0,
            "draws": elo_r.draws if elo_r is not None else 0,
            "losses": elo_r.losses if elo_r is not None else 0,
            "win_rate": round(elo_r.wins / elo_r.games, 4) if elo_r and elo_r.games > 0 else 0.0,
            "score_rate": round((elo_r.wins + 0.5 * elo_r.draws) / elo_r.games, 4) if elo_r and elo_r.games > 0 else 0.0,
        }
    tourn["ratings"] = ratings_dict
    tourn["matchups"] = [
        {
            "game_id": r.game_id,
            "pair_id": r.pair_id,
            "opening_plies": r.opening_plies,
            "opening_moves": list(r.opening_moves),
            "player_x": r.player_x,
            "player_o": r.player_o,
            "winner": r.winner,
            "moves": r.moves,
        }
        for r in tourn["results"]
    ]
    write_tournament_report(tourn, output_dir)
    print(tourn["scoreboard"])
    return tourn


tournament = tournament_cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    t = commands.add_parser('train')
    for name,default in [('iterations',20),('games',8),('simulations',64),('steps',100),
                         ('batch',128),('buffer',50000)]:
        t.add_argument('--'+name,type=positive,default=default)
    t.add_argument('--arch', choices=['resnet', 'mlp'], default='resnet')
    t.add_argument('--workers', type=positive, default=8)
    t.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    t.add_argument('--output', default='runs/default')
    t.add_argument('--resume')
    t.add_argument('--inference-batch', type=positive, default=128,
                   help='Maximum positions evaluated together by the inference owner')
    t.add_argument('--inference-wait-ms', type=float, default=2.,
                   help='Maximum wait for other workers to fill an inference batch')
    t.add_argument('--save-every', type=int, default=100, help='Save snapshot model checkpoint every N iterations; 0 disables')
    t.add_argument('--eval-every', type=int, default=0, help='Evaluate every N iterations; 0 disables')
    t.add_argument('--eval-games', type=positive, default=20)
    t.add_argument('--eval-simulations', type=positive, default=512)
    p = commands.add_parser('play')
    p.add_argument('--profile',default='profiles/player.json')
    p.add_argument('--side',choices=['X','O'],default='X')
    p.add_argument('--no-adapt',action='store_true')
    e = commands.add_parser('evaluate')
    e.add_argument('--games',type=positive,default=20)
    e.add_argument('--opponent', choices=['random', 'center', 'corners', 'local-win', 'global-win', 'legacy-tactical', 'tactical', 'alphabeta', 'checkpoint'], default='alphabeta')
    e.add_argument('--opponent-checkpoint')
    e.add_argument('--opponent-depth', type=positive, default=3)
    e.add_argument('--opponent-nodes', type=positive, default=3000)
    e.add_argument('--opponent-simulations', type=positive, default=256)
    e.add_argument('--opening-moves', type=int, default=2,
                   help='0–8 seeded opening plies, paired with sides swapped; 0 starts empty')
    e.add_argument('--seeds', nargs='+', type=int, help='Run separate reports for each seed')
    e.add_argument('--simulation-budgets', nargs='+', type=positive,
                   help='Run separate reports for each search budget')
    e.add_argument('--output',help='Report directory (default: checkpoint directory/evaluations)')
    tourn = commands.add_parser('tournament', help='Execute automated round-robin tournament or simulation sweep')
    tourn.add_argument('--checkpoint', help='Path to checkpoint model (optional)')
    tourn.add_argument('--opponents', nargs='+', help='Bot identifiers (e.g. alphabeta, tactical, corners, etc.)')
    tourn.add_argument('--games', type=positive, default=50, help='Games per matchup in [20, 500] (default: 50)')
    tourn.add_argument('--simulations', type=positive, default=512, help='MCTS simulation budget (default: 512)')
    tourn.add_argument('--simulation-budgets', nargs='+', type=positive, help='Simulation budgets for scaling sweep')
    tourn.add_argument('--opening-plies', type=int, default=2, help='Opening plies in [0, 4] (default: 2)')
    tourn.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    tourn.add_argument('--output', help='Report directory (default: runs/<run>/tournaments/ or runs/tournaments/)')
    tourn.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    for sub in (p,e):
        sub.add_argument('--device', choices=['auto','cpu','cuda'], default='cpu')
        sub.add_argument('--checkpoint',required=True)
        sub.add_argument('--simulations',type=positive,default=128)
    for sub in (t,p,e):
        sub.add_argument('--seed',type=int,default=0)
        sub.add_argument('--leaf-batch', type=positive, default=16,
                         help='Pending leaf positions per tree before an inference request')
        sub.add_argument('--no-soft-pruning', action='store_true')
        sub.add_argument('--no-proofs', action='store_true')
        sub.add_argument('--no-reuse', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.seed < 0 or (args.command == 'evaluate' and args.seeds and min(args.seeds) < 0):
        parser.error('Seeds must be nonnegative')
    if args.command == 'train' and (args.eval_every < 0 or not np.isfinite(args.inference_wait_ms) or args.inference_wait_ms < 0):
        parser.error('Evaluation interval and inference wait must be nonnegative and finite')
    if args.command == 'evaluate' and not 0 <= args.opening_moves <= 8:
        parser.error('Opening moves must be between 0 and 8')
    if args.command == 'tournament':
        if not 20 <= args.games <= 500:
            parser.error('Games per matchup must be between 20 and 500')
        if not 0 <= args.opening_plies <= 4:
            parser.error('Opening plies must be between 0 and 4')
    try:
        if args.command == 'evaluate':
            for seed in args.seeds or [args.seed]:
                for budget in args.simulation_budgets or [args.simulations]:
                    args.seed, args.simulations = seed, budget
                    if evaluate(args)['status'] != 'complete':
                        return  # Ctrl+C ends the whole sweep, not just one budget.
        else:
            {'train': train, 'play': play, 'evaluate': evaluate, 'tournament': tournament_cmd}[args.command](args)
    except (KeyboardInterrupt,EOFError):
        print('\nStopped. Completed training iterations and observed human moves are saved.')

if __name__ == '__main__':
    main()
