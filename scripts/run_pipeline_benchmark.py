"""M8: manifest-driven end-to-end pipeline benchmark.

Runs the real training entry point at a fixed configuration and reports where
the time goes, per completed iteration. Every arm shares one frozen checkpoint,
one config and one seed; only the backend differs.

Two things this deliberately does not do:

* **Claim a speedup from a single ordering.** Arms alternate (cpp, python,
  python, cpp, ...) so a machine warming up or another job starting cannot be
  read as a backend difference.
* **Sum worker time into elapsed time.** Four workers busy for ten seconds are
  forty worker-seconds and ten seconds of runtime; the stage report keeps those
  separate.

ETA comes from the measured per-iteration mean and spread plus the observed
checkpoint and evaluation overhead, and is recomputed whenever budgets, mix,
hardware or worker count change.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sttt.benchmarks.harness import summarize_repeats
from sttt.evaluation import evaluation_manifest, freeze_checkpoint, native_build_info

#: Unexplained regression beyond this blocks a speedup claim and triggers
#: profiling, per the plan. There is no required speedup factor.
REGRESSION_THRESHOLD = 0.05


def read_metrics(output_dir: Path) -> list[dict]:
    path = Path(output_dir) / 'metrics.jsonl'
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if 'throughput' in r]


def evaluation_overhead(output_dir: Path) -> float:
    path = Path(output_dir) / 'metrics.jsonl'
    if not path.is_file():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += float(row.get('evaluation_overhead_seconds', 0.0))
    return total


def run_arm(backend: str, checkpoint: Path | None, output_dir: Path, config: dict,
            python_executable: str = sys.executable) -> dict:
    """One measured run of the real training CLI. Returns its parsed metrics."""
    output_dir = Path(output_dir)
    command = [python_executable, '-m', 'sttt.ai', 'train',
               '--backend', backend,
               '--device', config['device'],
               '--arch', config['arch'],
               '--output', str(output_dir),
               '--iterations', str(config['warmup_iterations'] + config['measured_iterations']),
               '--games', str(config['games']),
               '--workers', str(config['workers']),
               '--simulations', str(config['simulations']),
               '--leaf-batch', str(config['leaf_batch']),
               '--inference-batch', str(config['inference_batch']),
               '--steps', str(config['steps']),
               '--batch', str(config['batch']),
               '--buffer', str(config['buffer']),
               '--seed', str(config['seed']),
               '--eval-every', '0', '--save-every', '0', '--keep-checkpoint-window', '0']
    if config.get('population'):
        command.append('--population')
        if config.get('population_config'):
            command += ['--population-config', str(config['population_config'])]
    if checkpoint is not None:
        command += ['--resume', str(checkpoint)]

    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(_REPO_ROOT))
    wall = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f'{backend} arm failed (exit {completed.returncode}):\n'
                           f'{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}')

    rows = read_metrics(output_dir)
    # Warm-up iterations are discarded: the first iterations pay process start,
    # worker spawn, extension load and an empty replay buffer.
    measured = rows[config['warmup_iterations']:]
    if len(measured) < config['measured_iterations']:
        raise RuntimeError(f'{backend} arm produced {len(measured)} measured iterations, '
                           f'expected {config["measured_iterations"]}')

    per_iteration = [r['throughput']['iteration_seconds'] for r in measured]
    stage_totals: dict[str, float] = {}
    for row in measured:
        for stage, seconds in row['stage_seconds']['stages'].items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + seconds

    return {
        'backend': backend,
        'wall_seconds': wall,
        'measured_iterations': len(measured),
        'iteration_seconds': summarize_repeats(per_iteration),
        'stage_seconds_total': stage_totals,
        'stage_share': {k: v / sum(stage_totals.values()) for k, v in stage_totals.items()},
        'evaluation_overhead_seconds': evaluation_overhead(output_dir),
        'throughput': {
            key: statistics.fmean([r['throughput'][key] for r in measured
                                   if r['throughput'].get(key) is not None])
            for key in ('selfplay_games_per_sec', 'learner_positions_per_sec',
                        'completed_simulations_per_sec', 'inference_batch_occupancy')
        },
        'output_dir': str(output_dir),
    }


def compute_eta(arm: dict, total_iterations: int, eval_every: int = 0,
                eval_overhead_seconds: float = 0.0) -> dict:
    """Projected runtime from the measured mean and observed spread.

    The band uses the measured min/max per-iteration time, so it reports what
    was actually observed rather than a confidence claim the sample cannot
    support.
    """
    stats = arm['iteration_seconds']
    evaluations = (total_iterations // eval_every) if eval_every else 0
    overhead = evaluations * eval_overhead_seconds
    return {
        'total_iterations': total_iterations,
        'mean_seconds_per_iteration': stats['mean'],
        'hours_mean': (stats['mean'] * total_iterations + overhead) / 3600,
        'hours_low': (stats['min'] * total_iterations + overhead) / 3600,
        'hours_high': (stats['max'] * total_iterations + overhead) / 3600,
        'evaluation_overhead_hours': overhead / 3600,
        'evaluations': evaluations,
        'basis': f"{arm['measured_iterations']} measured iterations, backend={arm['backend']}",
    }


def compare_backends(arms: dict) -> dict:
    """Relative per-iteration cost. A regression blocks a speedup claim."""
    if 'python' not in arms or 'cpp' not in arms:
        return {}
    py = arms['python']['iteration_seconds']['median']
    cp = arms['cpp']['iteration_seconds']['median']
    speedup = py / cp if cp else None
    regressed = speedup is not None and speedup < (1.0 - REGRESSION_THRESHOLD)
    return {
        'python_median_seconds': py,
        'cpp_median_seconds': cp,
        'cpp_speedup_vs_python': speedup,
        'regression_threshold': REGRESSION_THRESHOLD,
        'regression_blocks_claim': regressed,
        'note': ('cpp is slower than python beyond the threshold; profile before '
                 'claiming any speedup' if regressed else
                 'measured on one machine, this configuration and these budgets only'),
    }


def run_pipeline_benchmark(output_dir: Path, config: dict, checkpoint=None,
                           backends=('cpp', 'python'), repeats: int = 1) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = None
    if checkpoint is not None:
        frozen, _ = freeze_checkpoint(checkpoint, output_dir / 'evaluation-inputs' / 'candidate')

    manifest = evaluation_manifest(
        kind='pipeline-benchmark',
        checkpoints={'candidate': frozen} if frozen else {},
        config={**config, 'backends': list(backends), 'repeats': repeats},
        seeds={'training': config['seed']},
    )
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    build = native_build_info()
    if 'cpp' in backends and not build['available']:
        raise RuntimeError(f'cpp arm requested but the extension is unavailable: {build["reason"]}')
    print(f'native build: {build.get("build_id")} ({build.get("path")})')

    # Alternate the order every repeat so drift cannot masquerade as a backend
    # difference.
    order = []
    for r in range(repeats):
        sequence = list(backends) if r % 2 == 0 else list(reversed(backends))
        order.extend((r, backend) for backend in sequence)

    runs: dict[str, list] = {backend: [] for backend in backends}
    for repeat, backend in order:
        arm_dir = output_dir / f'{backend}-r{repeat}'
        print(f'\n--- {backend} (repeat {repeat}) ---', flush=True)
        arm = run_arm(backend, frozen, arm_dir, config)
        print(f"  median {arm['iteration_seconds']['median']:.3f}s/iteration "
              f"(min {arm['iteration_seconds']['min']:.3f}, max {arm['iteration_seconds']['max']:.3f})")
        for stage, share in sorted(arm['stage_share'].items(), key=lambda kv: -kv[1]):
            print(f"    {stage:<20} {share * 100:5.1f}%")
        runs[backend].append(arm)

    arms = {}
    for backend, results in runs.items():
        if not results:
            continue
        merged = dict(results[0])
        samples = [s for arm in results for s in [arm['iteration_seconds']['median']] * arm['measured_iterations']]
        merged['iteration_seconds'] = summarize_repeats(
            [arm['iteration_seconds']['median'] for arm in results]) if len(results) > 1 else results[0]['iteration_seconds']
        merged['repeats'] = len(results)
        arms[backend] = merged

    summary = {
        'arms': arms,
        'comparison': compare_backends(arms),
        'eta': {backend: compute_eta(arm, config['eta_iterations'],
                                     config.get('eta_eval_every', 0),
                                     arm['evaluation_overhead_seconds'])
                for backend, arm in arms.items()},
        'native_build': build,
    }
    (output_dir / 'pipeline_benchmark.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    manifest['status'] = 'completed'
    manifest['summary'] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    print('\n=== ETA ===')
    for backend, eta in summary['eta'].items():
        print(f"  {backend}: {eta['hours_mean']:.2f} h for {eta['total_iterations']} iterations "
              f"(observed band {eta['hours_low']:.2f}-{eta['hours_high']:.2f} h), "
              f"basis: {eta['basis']}")
    if summary['comparison']:
        print(f"  cpp vs python: {summary['comparison']['cpp_speedup_vs_python']:.2f}x "
              f"({summary['comparison']['note']})")
    return summary


DEFAULT_CONFIG = {
    'device': 'cpu', 'arch': 'mlp', 'games': 4, 'workers': 2, 'simulations': 32,
    'leaf_batch': 4, 'inference_batch': 8, 'steps': 4, 'batch': 16, 'buffer': 1000,
    'seed': 11, 'warmup_iterations': 2, 'measured_iterations': 20,
    'eta_iterations': 5000, 'eta_eval_every': 0, 'population': False,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Manifest-driven end-to-end pipeline benchmark')
    parser.add_argument('--manifest', help='JSON file overriding the default configuration')
    parser.add_argument('--output', default='runs/benchmarks/pipeline')
    parser.add_argument('--checkpoint', help='Frozen start checkpoint (optional)')
    parser.add_argument('--backends', nargs='+', default=['cpp', 'python'], choices=['cpp', 'python'])
    parser.add_argument('--repeats', type=int, default=1, help='Alternating passes over the backends')
    parser.add_argument('--measured-iterations', type=int, default=None,
                        help='Completed iterations to measure after warm-up (plan asks for >= 20)')
    parser.add_argument('--eta-iterations', type=int, default=None)
    args = parser.parse_args(argv)

    config = dict(DEFAULT_CONFIG)
    if args.manifest:
        config.update(json.loads(Path(args.manifest).read_text()))
    if args.measured_iterations is not None:
        config['measured_iterations'] = args.measured_iterations
    if args.eta_iterations is not None:
        config['eta_iterations'] = args.eta_iterations

    run_pipeline_benchmark(Path(args.output), config, checkpoint=args.checkpoint,
                           backends=tuple(args.backends), repeats=args.repeats)


if __name__ == '__main__':
    main()
