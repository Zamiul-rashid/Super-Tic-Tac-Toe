"""Plot saved evaluations and training logs without a server or GPU.

python -m sttt.visualize runs/starter --output runs/charts
"""
import argparse
import json
from pathlib import Path

COLORS = {'wins': '#197d63', 'draws': '#b28a28', 'losses': '#bc4b51'}


def discover(inputs):
    reports, logs, seen = [], [], set()
    for item in inputs:
        path = Path(item)
        if not path.exists():
            raise ValueError(f'Input does not exist: {path}')
        candidates = sorted(path.rglob('*.json')) + sorted(path.rglob('metrics.jsonl')) if path.is_dir() else [path]
        for source in candidates:
            if source.resolve() in seen:
                continue
            seen.add(source.resolve())
            try:
                if source.name == 'metrics.jsonl':
                    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
                    if rows:
                        logs.append((str(source.parent), rows))
                else:
                    data = json.loads(source.read_text())
                    if isinstance(data, dict) and data.get('kind') == 'sttt_evaluation':
                        if data.get('schema_version') != 1:
                            raise ValueError(f'Unsupported evaluation schema in {source}')
                        if data.get('summary', {}).get('games', 0):
                            reports.append(data)
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f'Cannot read {source}: {error}') from error
    reports.sort(key=lambda r: r['created_at'])
    return reports, logs


def draw_evaluations(reports, plt):
    import numpy as np
    # Reports remain separate; results with different opponents/budgets aren't pooled.
    height = max(8, len(reports)*.6 + 4)
    fig, axes = plt.subplots(2, 2, figsize=(15, height), layout='constrained')
    fig.suptitle('Ultimate Tic-Tac-Toe evaluation', fontsize=18, weight='bold')
    x = np.arange(len(reports))
    labels = [f'{i+1}: iter {r["iteration"]} · {r["opponent"]}\n'
              f'{r["simulations"]} sims · n={r["summary"]["games"]} · seed {r["seed"]}'
              + (f' · {r["status"]}' if r['status'] != 'complete' else '')
              for i,r in enumerate(reports)]
    ax = axes[0,0]
    bottom = np.zeros(len(reports))
    for outcome in COLORS:
        values = np.array([r['summary'][outcome]/r['summary']['games']*100 for r in reports])
        ax.barh(x, values, left=bottom, label=outcome.capitalize(), color=COLORS[outcome])
        for index, value in enumerate(values):
            if value >= 10:
                ax.text(bottom[index]+value/2,index,f'{value:.0f}%',ha='center',va='center',color='white')
        bottom += values
    ax.set(yticks=x, yticklabels=labels, xlim=(0,100), xlabel='Games (%)', title='Outcome distribution')
    ax.invert_yaxis()
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3)

    ax = axes[0,1]
    for side, offset, color in [('X',-.18,'#3975ac'),('O',.18,'#9964a6')]:
        values = [100*r['by_side'][side]['score_rate']
                  if r['by_side'][side]['games'] else np.nan for r in reports]
        bars = ax.barh(x+offset,values,height=.34,label=f'Agent plays {side}',color=color)
        for bar, r in zip(bars,reports):
            if r['by_side'][side]['games']:
                ax.text(1,bar.get_y()+bar.get_height()/2,
                        f'n={r["by_side"][side]["games"]}',va='center',fontsize=9)
    ax.set(yticks=x,yticklabels=[str(i+1) for i in x],xlim=(0,100),xlabel='Score (%)',
           ylabel='Report number',title='Score by starting side (win=1, draw=½)')
    ax.invert_yaxis()
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2)

    ax = axes[1,0]
    for i,r in enumerate(reports):
        scores = np.array([g['score'] for g in r['games']])
        games = np.arange(1,len(scores)+1)
        ax.plot(games,100*np.cumsum(scores)/games,label=f'Report {i+1}')
    ax.set(xlabel='Completed evaluation games',ylabel='Cumulative score (%)',ylim=(0,100),
           title='Results as the test progresses')
    ax.legend(fontsize=9)

    ax = axes[1,1]
    for i,r in enumerate(reports):
        moves = [g['moves'] for g in r['games']]
        ax.scatter([i+1]*len(moves),moves,s=18,alpha=.35,color='#3975ac')
        ax.scatter(i+1,r['summary']['mean_moves'],marker='_',s=280,color='#202020')
    ax.set(xticks=np.arange(1,len(reports)+1),xlabel='Report number',ylabel='Moves per game',
           title='Game length (black marks show averages)')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x' if ax in axes[0] else 'y',alpha=.15)
        ax.set_axisbelow(True)
    return fig


def draw_training(logs, plt):
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    fig.suptitle('Training history', fontsize=18, weight='bold')
    for name, rows in logs:
        for ax,field in zip(axes,('loss','positions','seconds')):
            ax.plot([r['iteration'] for r in rows],[r[field] for r in rows],label=name)
    for ax,title,ylabel in zip(axes,('Training loss','Replay buffer','Iteration time'),
                               ('Combined policy/value loss','Positions','Seconds')):
        ax.set(title=title,xlabel='Iteration',ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',help='Run directories, evaluation JSON, or metrics.jsonl')
    parser.add_argument('--output',default='runs/charts',help='PNG output directory')
    parser.add_argument('--show',action='store_true',help='Also open interactive Matplotlib windows')
    args = parser.parse_args()
    try:
        reports, logs = discover(args.inputs)
    except (ValueError,OSError) as error:
        parser.error(str(error))
    if not reports and not logs:
        parser.error('No completed game data or training logs found. Run an evaluation first.')
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'savefig.facecolor': 'white'})
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    for name,figure in [('evaluation',draw_evaluations(reports,plt) if reports else None),
                        ('training',draw_training(logs,plt) if logs else None)]:
        if figure is not None:
            destination = output/f'{name}.png'
            figure.savefig(destination,dpi=150)
            print(f'Saved {destination}')
    if reports:
        print('Report key (chronological; settings differ, so compare like-for-like):')
        for i,r in enumerate(reports,1):
            print(f'{i}: {r["checkpoint"]} | {r["created_at"]} | {r["run_id"]}')
    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
