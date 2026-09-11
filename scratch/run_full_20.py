"""Run tests and all built-in 20-game benchmarks against one frozen checkpoint."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
output = ROOT / 'runs/big_run/benchmarks' / stamp
output.mkdir(parents=True)
source = ROOT / 'runs/big_run/latest.pt'
# The trainer replaces latest.pt atomically. This open/load sees one inode.
saved = torch.load(source, map_location='cpu', weights_only=True)
checkpoint = output / 'checkpoint.pt'
torch.save({k: saved[k] for k in ('model', 'arch', 'iteration')}, checkpoint)
iteration = saved['iteration']
del saved
opponents = ['random', 'center', 'corners', 'local-win', 'global-win',
             'legacy-tactical', 'tactical', 'alphabeta']
manifest = dict(source=str(source), iteration=iteration, checkpoint=str(checkpoint),
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                games_per_opponent=20, simulations=512, seed=42, jobs={})
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(f'OUTPUT={output}\nITERATION={iteration}', flush=True)
base = [sys.executable, '-u', '-m', 'sttt.ai']
jobs = [('tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])]
for opponent in opponents:
    jobs.append((opponent, base + ['evaluate', '--checkpoint', str(checkpoint),
                '--games', '20', '--opponent', opponent, '--simulations', '512',
                '--seed', '42', '--device', 'cpu', '--output', str(output / opponent)]))
# Factory supports the five style bots, tactical and alpha-beta.
jobs.append(('tournament', base + ['tournament', '--checkpoint', str(checkpoint),
             '--opponents', 'random', 'center', 'corners', 'local-win', 'global-win',
             'tactical', 'alphabeta', '--games', '20', '--simulations', '512',
             '--seed', '42', '--device', 'cpu', '--output', str(output / 'tournament')]))

def run(job):
    name, command = job
    print(f'START {name}', flush=True)
    with (output / f'{name}.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    print(f'END {name}: exit={result.returncode}', flush=True)
    return name, dict(command=command, exit_code=result.returncode)

with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed([pool.submit(run, job) for job in jobs]):
        name, result = future.result()
        manifest['jobs'][name] = result
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('COMPLETE', flush=True)
sys.exit(int(any(j['exit_code'] != 0 for j in manifest['jobs'].values())))
