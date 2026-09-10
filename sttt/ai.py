"""Run with python -m sttt.ai train|play|evaluate."""
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from .env import State
from .learning import Network, encode, load_model
from .opponent import Opponent, policies
from .search import search

def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value

def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, saved = load_model(args.resume) if args.resume else (Network(), {})
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    if 'optimizer' in saved:
        optimizer.load_state_dict(saved['optimizer'])
    replay = deque(saved.get('replay', []),maxlen=args.buffer)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    print(f'{device}: {sum(p.numel() for p in model.parameters()):,} parameters; replay stays in RAM',flush=True)
    start_iteration = saved.get('iteration',0)
    for iteration in range(start_iteration+1,start_iteration+args.iterations+1):
        started = time.monotonic()
        actor = copy.deepcopy(model).cpu().eval()
        for game in range(args.games):
            state, trajectory = State(), []
            while state.result is None:
                pi = search(state,actor,args.simulations,rng,noise=True)
                legal_mask = np.zeros(81,dtype=bool)
                legal_mask[state.legal_actions()] = True
                trajectory.append((torch.from_numpy(encode(state)),torch.from_numpy(pi),
                                   torch.from_numpy(legal_mask),state.turn))
                action = rng.choice(81,p=pi) if len(trajectory)<16 else int(pi.argmax())
                state = state.play(int(action))
            for x,pi,mask,turn in trajectory:
                replay.append((x,pi,mask,float(state.result*turn)))
            print(f'iteration {iteration}: game {game+1}/{args.games}, {len(trajectory)} moves',flush=True)
        model.train()
        losses = []
        for _ in range(args.steps):
            indices = rng.choice(len(replay),size=min(args.batch,len(replay)),replace=False)
            batch = [replay[int(i)] for i in indices]
            x,pi,mask = [torch.stack([row[k] for row in batch]).to(device) for k in range(3)]
            z = torch.tensor([row[3] for row in batch],device=device)
            logits,value = model(x)
            logits = logits.masked_fill(~mask,-1e9)
            loss = -(pi*logits.log_softmax(-1)).sum(-1).mean() + (value-z).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(loss.item())
        checkpoint = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                      'iteration':iteration,'replay':list(replay)}
        temporary = output/'latest.tmp'
        torch.save(checkpoint,temporary)
        temporary.replace(output/'latest.pt')
        # Small historical weights retained separately for evaluation.
        torch.save({'model':model.state_dict(),'iteration':iteration},output/f'model-{iteration:04d}.pt')
        report = {'iteration':iteration,'positions':len(replay),'loss':float(np.mean(losses)),
                  'seconds':round(time.monotonic()-started,2)}
        if device.startswith('cuda'):
            report['peak_gpu_mb'] = round(torch.cuda.max_memory_allocated()/1024**2,1)
        with (output/'metrics.jsonl').open('a') as file:
            file.write(json.dumps(report)+'\n')
        print(json.dumps(report),flush=True)

def play(args):
    model,_ = load_model(args.checkpoint)
    opponent = Opponent(args.profile)
    rng = np.random.default_rng(args.seed)
    state = State()
    human = 1 if args.side == 'X' else -1
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
                state = state.play(action)
            except ValueError:
                print('Enter a legal board and cell, both between 1 and 9.')
        else:
            pi = search(state,model,args.simulations,rng,
                        opponent=None if args.no_adapt else opponent)
            action = int(pi.argmax())
            print(f'AI plays board {action//9+1}, cell {action%9+1}')
            state = state.play(action)
    print(state.render())
    print('Draw' if state.result == 0 else ('You win' if state.result == human else 'AI wins'))

def evaluate(args):
    model,_ = load_model(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    results = {'wins':0,'draws':0,'losses':0}
    for game in range(args.games):
        state,agent = State(), (1 if game%2 == 0 else -1)
        while state.result is None:
            if state.turn == agent:
                action = int(search(state,model,args.simulations,rng).argmax())
            else:
                p = policies(state)[0 if args.opponent == 'random' else 4]
                action = int(rng.choice(81,p=p))
            state = state.play(action)
        key = 'draws' if state.result == 0 else ('wins' if state.result == agent else 'losses')
        results[key] += 1
        print(f'game {game+1}/{args.games}: {key}',flush=True)
    print(json.dumps(results))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    t = commands.add_parser('train')
    for name,default in [('iterations',20),('games',8),('simulations',64),('steps',100),
                         ('batch',128),('buffer',50000)]:
        t.add_argument('--'+name,type=positive,default=default)
    t.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    t.add_argument('--output',default='runs/default')
    t.add_argument('--resume')
    p = commands.add_parser('play')
    p.add_argument('--profile',default='profiles/player.json')
    p.add_argument('--side',choices=['X','O'],default='X')
    p.add_argument('--no-adapt',action='store_true')
    e = commands.add_parser('evaluate')
    e.add_argument('--games',type=positive,default=20)
    e.add_argument('--opponent',choices=['random','tactical'],default='tactical')
    for sub in (p,e):
        sub.add_argument('--checkpoint',required=True)
        sub.add_argument('--simulations',type=positive,default=128)
    for sub in (t,p,e):
        sub.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        {'train':train,'play':play,'evaluate':evaluate}[args.command](args)
    except (KeyboardInterrupt,EOFError):
        print('\nStopped. Completed training iterations and observed human moves are saved.')

if __name__ == '__main__':
    main()
