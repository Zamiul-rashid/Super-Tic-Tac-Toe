# Superhuman Engine Blueprint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the training-side blueprint from `handover/handover.md` §5–§6 — game-phase-stratified replay sampling, a dense action-value (Q) head, a hierarchical conv U-Net, and a two-stage C++-bootstrapped offline pretraining pipeline — so a fresh network starts online self-play with a strong prior instead of learning from scratch.

**Architecture:** Every change is additive and behaviour-preserving by default: new CLI flags default to the current behaviour, new checkpoint fields load legacy checkpoints unchanged, and the U-Net consumes the existing 289-float canonical encoding (reshaping it into `(8, 9, 9)` planes inside `forward`) so the replay buffer, the native `encode_batch`, and `augment_batch` are untouched. The Q-head reads root child statistics that both the Python `Node` and the native `FastNode` already expose (`n`, `total`, `children`), so no C++ change is needed. Bootstrap generation reuses `selfplay._play_game` with a model-free `CppTreeSearch` (heuristic leaf evaluation) and `CppAlphaBetaBot` opponents, writing shards in the replay's packed format so `pretrain` and `train --resume` load them with the same code.

**Tech Stack:** Python 3.14 (conda `sttt`), PyTorch 2.14 + CUDA, numpy, `unittest`, the `sttt_cpp` CPython extension built by `cpp/Makefile`.

**Spec:** `handover/handover.md` §5 (architectural deconstruction of `uttt.ai`) and §6 Steps 1–3 (U-Net, C++ bootstrap, dense action-value loss). The vendored reference implementation the design is checked against is `engines/vendor/utttai/utttpy/selfplay/{policy_value_network,training,datatools}.py`. §6 Step 4 (distillation from `uttt.ai`) and initialising unvisited MCTS leaves with predicted Q are **out of scope** for this plan (see "Deferred" at the end).

## Global Constraints

- Interpreter: `/home/mt/miniconda3/envs/sttt/bin/python` (Python 3.14.6, torch 2.14.0+cu130, CUDA available). Never `/usr/bin/python3`. Define `PY=/home/mt/miniconda3/envs/sttt/bin/python` in every shell.
- CPU: the machine hangs at 32×100 %. Prefix every long command with `nice -n 19` and `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`; never pass `-j` to `make`; cap generator workers at 10; check `uptime` before anything heavy.
- Native extension: `cpp/*.so` is tracked and goes stale silently. Before any native test or benchmark: `make -C cpp clean && make -C cpp PYTHON=$PY`, then confirm `python -c "import sttt_cpp; print(sttt_cpp.SOURCE_REVISION)"` matches `git rev-parse --short HEAD`.
- Tests: `unittest`, one class per behaviour, in `tests/test_<module>.py`. Run the targeted module while iterating: `nice -n 19 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m unittest tests.test_x -v`. Run the full suite (`$PY -m unittest discover -s tests`, ~190 s, 1 pre-existing known failure: branch isolation) before every merge to `main`. A skipped native or GPU test does not satisfy a gate.
- Repository rules (`TRAINING_READINESS_PLAN.md` §4): one coherent change per commit; regression test before fix; never weaken a test to pass; each part on its own worktree branched from current `main`; never write into `runs/run_v2/` or `runs/big_run/`; planning does not authorise starting a multi-day training run.
- Defaults preserve current behaviour: `--replay-sampling uniform`, `--arch resnet`, legacy checkpoints load with zero Q targets. `./train.sh` is the only place new defaults are switched on.
- Commit trailer (per session attribution rule): end every commit message with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Encoding layout (from `sttt/learning.py:6-12`), used by every part — do not guess it:
  - `[0:81)` empty-cell plane, `[81:162)` current player's stones, `[162:243)` opponent's stones (cells are multiplied by `state.turn`, so plane 1 is always "mine");
  - `[243:252)` boards open, `[252:261)` boards mine, `[261:270)` boards theirs, `[270:279)` boards drawn;
  - `[279:289)` one-hot of `forced+1`: index 0 = any board, index `k` = board `k-1`.
  - Cell index `= board*9 + cell`; board `b` sits at macro `(b//3, b%3)`, cell `c` at micro `(c//3, c%3)`; 9×9 grid row `= (b//3)*3 + c//3`, col `= (b%3)*3 + c%3`.

## Part map and order

| Part | Deliverable | Depends on | Worktree branch |
| --- | --- | --- | --- |
| 1 | Prove the existing 8-fold symmetry augmentation on random positions (no production change) | — | `feat/p1-symmetry-proof` |
| 2 | Game-phase-stratified replay sampling behind `--replay-sampling` | — | `feat/p2-stratified-replay` |
| 3 | One shared `policy_value_loss` used by the trainer (refactor, no behaviour change) | — | `feat/p3-shared-loss` |
| 4 | Q-head data path: root action-values recorded in trajectories, replay `packed-v2`, masked Q loss | 3 | `feat/p4-q-head` |
| 5 | `UNet` architecture (`--arch unet`) with policy, value and Q outputs | 4 | `feat/p5-unet` |
| 6 | Bootstrap: `generate-dataset` (C++ heuristic MCTS + alpha-beta) and `pretrain` → resumable checkpoint | 2, 4, 5 | `feat/p6-bootstrap` |
| 7 | Launcher, docs, handover banner, readiness ledger | 1–6 | `feat/p7-docs` |

Parts 1, 2 and 3 are independent and may run in parallel worktrees. Merge each part to `main` when its gate passes; rebase the next part onto the merged `main`.

---

## Part 1 — Prove the existing symmetry augmentation

`--augment-symmetry` already exists (`sttt/ai.py:394`, `sttt/population.py:182-196`, `train.sh:98`) and is covered by one fixed-position test (`tests/test_population.py:69-90`). Rule §4.1: an existing requirement is proved with an acceptance test, not re-implemented. This part adds the missing property tests.

### Task 1.1: Symmetry table is the dihedral group and commutes with `encode` on random games

**Files:**
- Test: `tests/test_population.py` (add a class after `test_symmetry_respects_rules_and_forced_board`)

**Interfaces:**
- Consumes: `sttt.population.SYMMETRIES` (list of 8 `(inputs, cells)` index arrays: `inputs` are destination indices over the 289 features, `cells` destination indices over the 81 actions), `sttt.population.augment_batch(x, pi, mask, rng)`, `sttt.learning.encode`, `sttt.env.State`.
- Produces: nothing new; a gate for Parts 4–6, which rely on `augment_batch` being a correct group action.

- [ ] **Step 1: Write the failing tests**

```python
class SymmetryGroupTests(unittest.TestCase):
    """augment_batch is only valid if SYMMETRIES is a faithful action of the
    dihedral group on both the features and the action space."""

    def test_eight_distinct_permutations_on_features_and_actions(self):
        feature_maps = {tuple(inputs) for inputs, _ in SYMMETRIES}
        action_maps = {tuple(cells) for _, cells in SYMMETRIES}
        self.assertEqual(len(SYMMETRIES), 8)
        self.assertEqual(len(feature_maps), 8)
        self.assertEqual(len(action_maps), 8)
        for inputs, cells in SYMMETRIES:
            self.assertEqual(sorted(inputs.tolist()), list(range(289)))
            self.assertEqual(sorted(cells.tolist()), list(range(81)))
            self.assertEqual(int(inputs[279]), 279)  # "any board" one-hot slot is fixed

    def test_action_maps_are_whole_board_dihedral_transforms(self):
        # Applying the same 3x3 transform to the macro index and the micro
        # index must equal applying it to the whole 9x9 grid; this is what makes
        # the flat cell permutation usable by a conv net over the 9x9 board.
        idx = np.arange(81)
        b, c = idx // 9, idx % 9
        grid_of_cell = ((b // 3) * 3 + c // 3) * 9 + ((b % 3) * 3 + c % 3)
        cell_of_grid = np.argsort(grid_of_cell)
        grid = np.arange(81).reshape(9, 9)
        expected = set()
        for mirror in (False, True):
            base = np.fliplr(grid) if mirror else grid
            for k in range(4):
                expected.add(tuple(np.rot90(base, k).ravel().tolist()))
        for _, cells in SYMMETRIES:
            # grid position g holds cell cell_of_grid[g]; after the transform it
            # holds cells[cell_of_grid[g]]; read it back in grid coordinates.
            as_grid_perm = tuple(grid_of_cell[cells[cell_of_grid]].tolist())
            self.assertIn(as_grid_perm, expected)

    def test_transform_commutes_with_encode_and_legality_on_random_games(self):
        rng = np.random.default_rng(2026)
        checked = 0
        for _ in range(60):
            actions, state = [], State()
            plies = int(rng.integers(1, 60))
            for _ in range(plies):
                if state.result is not None:
                    break
                a = int(rng.choice(state.legal_actions()))
                actions.append(a)
                state = state.play(a)
            if state.result is not None:
                continue
            for inputs, cells in SYMMETRIES:
                mirrored = State()
                for a in actions:
                    mirrored = mirrored.play(int(cells[a]))
                expected = np.empty(289, dtype=np.float32)
                expected[inputs] = encode(state)
                np.testing.assert_array_equal(expected, encode(mirrored))
                self.assertEqual(sorted(int(cells[a]) for a in state.legal_actions()),
                                 sorted(mirrored.legal_actions()))
                checked += 1
        self.assertGreater(checked, 200)

    def test_augment_batch_keeps_policy_mass_on_legal_moves_of_the_transformed_state(self):
        rng = np.random.default_rng(7)
        state = State()
        for a in [40, 4, 36, 0, 8, 72, 80, 79]:
            state = state.play(a)
        legal = state.legal_actions()
        pi = np.zeros(81, dtype=np.float32)
        pi[legal] = rng.random(len(legal)).astype(np.float32)
        pi /= pi.sum()
        x = torch.tensor(np.stack([encode(state)] * 64))
        pi_t = torch.tensor(np.stack([pi] * 64))
        mask = torch.tensor(np.stack([np.isin(np.arange(81), legal)] * 64))
        xx, pp, mm = augment_batch(x, pi_t, mask, rng)
        self.assertTrue(torch.equal(pp > 0, mm))
        self.assertTrue(torch.allclose(pp.sum(1), torch.ones(64)))
        # every row is one of the eight exact transforms, never a mixture
        originals = {tuple(np.flatnonzero(cells[legal]).tolist()) for _, cells in SYMMETRIES}
        for row in mm:
            self.assertIn(tuple(torch.nonzero(row).flatten().tolist()), originals)
```

- [ ] **Step 2: Run the tests to verify they run (they should pass — this is a proof, not a repair)**

Run: `nice -n 19 $PY -m unittest tests.test_population.SymmetryGroupTests -v`
Expected: 4 tests, all `ok`. If `test_action_maps_are_whole_board_dihedral_transforms` fails, STOP and report: it means `augment_batch` is not a valid transform for a conv architecture and Part 5 must not proceed on it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_population.py
git commit -m "Prove the symmetry table is the dihedral group and commutes with encode

Property tests over random games for the existing --augment-symmetry path,
which was covered by one fixed position. Also proves the macro+micro cell
permutation equals a whole-board 9x9 transform, which the U-Net relies on.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Part 2 — Game-phase-stratified replay sampling

`uttt.ai` samples round-robin across per-depth files (`training.py:113-125`), so every batch is balanced across opening/midgame/endgame regardless of how many positions each phase produced. Our sampler is `rng.choice(len(replay), ...)` at `sttt/ai.py:385`, uniform over a FIFO. Ply depth is derivable from an encoded row (`81 − Σ empty-plane`), so no replay schema change is needed and legacy checkpoints get depth for free.

### Task 2.1: `replay_sampling` module — ply from a row, bins, stratified sampler

**Files:**
- Create: `sttt/replay_sampling.py`
- Test: `tests/test_replay_sampling.py`

**Interfaces:**
- Produces:
  - `row_ply(x) -> int` — plies played, from a `(289,)` tensor or ndarray.
  - `ply_bin(ply: int) -> int` — bin id in `[0, PLY_BINS)`; `PLY_BINS = 8` (9 plies each, last bin 63–80).
  - `BIN_LABELS: tuple[str, ...]` — `('0-8', '9-17', ..., '63-80')`.
  - `class StratifiedSampler(bins: np.ndarray)` with `.sample(batch: int, rng) -> np.ndarray` (int indices, length `min(batch, len(bins))`, no repeats) and `.histogram() -> dict[str, int]`.
  - `sample_bin_fractions(bins: np.ndarray, indices: np.ndarray) -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

```python
import unittest
import numpy as np
import torch
from sttt.env import State
from sttt.learning import encode
from sttt.replay_sampling import (BIN_LABELS, PLY_BINS, StratifiedSampler, ply_bin,
                                  row_ply, sample_bin_fractions)


class RowPlyTests(unittest.TestCase):
    def test_ply_counts_moves_played_on_random_games(self):
        rng = np.random.default_rng(3)
        for _ in range(30):
            state, plies = State(), 0
            while state.result is None and plies < 70:
                state = state.play(int(rng.choice(state.legal_actions())))
                plies += 1
                self.assertEqual(row_ply(encode(state)), plies)
                self.assertEqual(row_ply(torch.from_numpy(encode(state))), plies)

    def test_bins_cover_all_plies_and_labels_match(self):
        self.assertEqual(len(BIN_LABELS), PLY_BINS)
        self.assertEqual(ply_bin(0), 0)
        self.assertEqual(ply_bin(8), 0)
        self.assertEqual(ply_bin(9), 1)
        self.assertEqual(ply_bin(62), 6)
        self.assertEqual(ply_bin(63), 7)
        self.assertEqual(ply_bin(80), 7)
        self.assertEqual(BIN_LABELS[0], '0-8')
        self.assertEqual(BIN_LABELS[-1], '63-80')


class StratifiedSamplerTests(unittest.TestCase):
    def test_equal_share_per_bin_when_every_bin_is_deep(self):
        bins = np.repeat(np.arange(PLY_BINS), 1000)
        picks = StratifiedSampler(bins).sample(512, np.random.default_rng(0))
        self.assertEqual(len(picks), 512)
        self.assertEqual(len(set(picks.tolist())), 512)
        counts = np.bincount(bins[picks], minlength=PLY_BINS)
        self.assertTrue((counts == 64).all(), counts)

    def test_thin_bins_are_exhausted_and_their_share_redistributed(self):
        # 5 endgame rows, 2000 rows in each of two midgame bins, nothing else.
        bins = np.concatenate([np.full(5, 7), np.full(2000, 3), np.full(2000, 4)])
        picks = StratifiedSampler(bins).sample(300, np.random.default_rng(1))
        self.assertEqual(len(picks), 300)
        self.assertEqual(len(set(picks.tolist())), 300)
        counts = np.bincount(bins[picks], minlength=PLY_BINS)
        self.assertEqual(counts[7], 5)
        self.assertEqual(counts[3] + counts[4], 295)
        self.assertLessEqual(abs(counts[3] - counts[4]), 1)

    def test_batch_larger_than_buffer_returns_every_row_once(self):
        bins = np.array([0, 0, 3, 7, 7, 7])
        picks = StratifiedSampler(bins).sample(50, np.random.default_rng(2))
        self.assertEqual(sorted(picks.tolist()), [0, 1, 2, 3, 4, 5])

    def test_single_bin_degenerates_to_uniform_without_replacement(self):
        bins = np.zeros(100, dtype=np.int8)
        picks = StratifiedSampler(bins).sample(40, np.random.default_rng(3))
        self.assertEqual(len(set(picks.tolist())), 40)

    def test_empty_buffer_returns_empty(self):
        picks = StratifiedSampler(np.zeros(0, dtype=np.int8)).sample(16, np.random.default_rng(4))
        self.assertEqual(len(picks), 0)

    def test_histogram_and_fractions_are_labelled_by_bin(self):
        bins = np.array([0, 0, 0, 3, 7])
        sampler = StratifiedSampler(bins)
        self.assertEqual(sampler.histogram(), {'0-8': 3, '27-35': 1, '63-80': 1})
        fractions = sample_bin_fractions(bins, np.array([0, 3, 4, 1]))
        self.assertAlmostEqual(fractions['0-8'], 0.5)
        self.assertAlmostEqual(fractions['27-35'], 0.25)
        self.assertAlmostEqual(fractions['63-80'], 0.25)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_replay_sampling -v`
Expected: `ModuleNotFoundError: No module named 'sttt.replay_sampling'`

- [ ] **Step 3: Implement the module**

```python
"""Game-phase-stratified sampling from the replay buffer.

uttt.ai trains from per-depth files and round-robins across them, so every
batch is balanced over opening, midgame and endgame no matter how many
positions each phase produced. Our buffer is one FIFO; this module recovers
the same balance at sampling time. Depth is read off the encoded row itself
(81 minus the empty-cell plane), so legacy checkpoints need no migration.
"""
import numpy as np

EMPTY_PLANE = slice(0, 81)
PLY_BINS = 8
_EDGES = [(9 * i, 9 * i + 8) for i in range(PLY_BINS - 1)] + [(63, 80)]
BIN_LABELS = tuple(f'{lo}-{hi}' for lo, hi in _EDGES)


def row_ply(x):
    """Plies played so far in the encoded position `x` (shape (289,))."""
    return 81 - int(round(float(x[EMPTY_PLANE].sum())))


def ply_bin(ply):
    return min(int(ply) // 9, PLY_BINS - 1)


class StratifiedSampler:
    """Equal share per non-empty ply bin, without replacement.

    Built once per iteration from the buffer's bin ids; `sample` is then cheap
    enough to call once per optimizer step. A bin thinner than its share is
    exhausted and the leftover share is spread over the remaining bins, so the
    batch is always min(batch, len(bins)) rows.
    """

    def __init__(self, bins):
        self.bins = np.asarray(bins)
        self.members = [np.flatnonzero(self.bins == b) for b in range(PLY_BINS)]

    def __len__(self):
        return len(self.bins)

    def histogram(self):
        return {BIN_LABELS[b]: int(len(m)) for b, m in enumerate(self.members) if len(m)}

    def sample(self, batch, rng):
        batch = min(int(batch), len(self.bins))
        if batch <= 0:
            return np.empty(0, dtype=np.int64)
        quota = np.zeros(PLY_BINS, dtype=np.int64)
        open_bins = [b for b in range(PLY_BINS) if len(self.members[b])]
        remaining = batch
        while remaining > 0 and open_bins:
            share = max(1, remaining // len(open_bins))
            for b in list(open_bins):
                take = min(share, len(self.members[b]) - quota[b], remaining)
                quota[b] += take
                remaining -= take
                if quota[b] == len(self.members[b]):
                    open_bins.remove(b)
                if remaining == 0:
                    break
        picks = [rng.choice(self.members[b], size=int(quota[b]), replace=False)
                 for b in range(PLY_BINS) if quota[b]]
        return np.concatenate(picks).astype(np.int64)


def sample_bin_fractions(bins, indices):
    """Fraction of a sampled batch that came from each bin, by label."""
    indices = np.asarray(indices)
    if len(indices) == 0:
        return {}
    counts = np.bincount(np.asarray(bins)[indices], minlength=PLY_BINS)
    return {BIN_LABELS[b]: float(counts[b]) / len(indices) for b in range(PLY_BINS) if counts[b]}
```

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m unittest tests.test_replay_sampling -v`
Expected: 8 tests `ok`.

- [ ] **Step 5: Commit**

```bash
git add sttt/replay_sampling.py tests/test_replay_sampling.py
git commit -m "Add game-phase-stratified replay sampler

Ply depth is read off the encoded row, so no schema change; equal share per
nine-ply bin with leftover share redistributed from thin bins.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

### Task 2.2: Wire `--replay-sampling` into the trainer with observable metrics

**Files:**
- Modify: `sttt/ai.py:150-157` (replay load), `sttt/ai.py:351-369` (append), `sttt/ai.py:383-386` (sampler), `sttt/ai.py:508-528` (report), `sttt/ai.py:853-854` (argparse, next to `--augment-symmetry`)
- Test: `tests/test_replay_sampling.py` (add `TrainerWiringTests`)

**Interfaces:**
- Consumes: `StratifiedSampler`, `row_ply`, `ply_bin`, `sample_bin_fractions` from Task 2.1.
- Produces: `--replay-sampling {uniform,stratified}` (default `uniform`); `metrics.jsonl` keys `replay_sampling`, `replay_depth_histogram` (buffer composition by bin) and `replay_sample_depth_fractions` (what the optimizer saw).

- [ ] **Step 1: Write the failing wiring test (subprocess, tiny budgets, Python backend so no native build is needed)**

```python
import json, subprocess, sys, tempfile
from pathlib import Path


class TrainerWiringTests(unittest.TestCase):
    def _run(self, sampling, output):
        cmd = [sys.executable, '-m', 'sttt.ai', 'train', '--backend', 'python', '--device', 'cpu',
               '--iterations', '1', '--games', '2', '--simulations', '4', '--steps', '3',
               '--batch', '8', '--workers', '1', '--leaf-batch', '2', '--save-every', '0',
               '--replay-sampling', sampling, '--output', output, '--seed', '5']
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        rows = [json.loads(l) for l in (Path(output) / 'metrics.jsonl').read_text().splitlines()]
        return [r for r in rows if 'replay_sampling' in r][-1]

    def test_stratified_run_reports_buffer_histogram_and_sampled_fractions(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run('stratified', tmp)
        self.assertEqual(report['replay_sampling'], 'stratified')
        self.assertEqual(sum(report['replay_depth_histogram'].values()), report['positions'])
        self.assertAlmostEqual(sum(report['replay_sample_depth_fractions'].values()), 1.0, places=6)
        self.assertGreaterEqual(len(report['replay_sample_depth_fractions']), 2)

    def test_uniform_is_the_default_and_still_reports_composition(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, '-m', 'sttt.ai', 'train', '--backend', 'python', '--device', 'cpu',
                   '--iterations', '1', '--games', '1', '--simulations', '4', '--steps', '2',
                   '--batch', '8', '--workers', '1', '--leaf-batch', '2', '--save-every', '0',
                   '--output', tmp, '--seed', '5']
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            rows = [json.loads(l) for l in (Path(tmp) / 'metrics.jsonl').read_text().splitlines()]
        report = [r for r in rows if 'replay_sampling' in r][-1]
        self.assertEqual(report['replay_sampling'], 'uniform')
        self.assertEqual(sum(report['replay_depth_histogram'].values()), report['positions'])
```

- [ ] **Step 2: Run to verify failure**

Run: `nice -n 19 $PY -m unittest tests.test_replay_sampling.TrainerWiringTests -v`
Expected: `CalledProcessError` — `error: unrecognized arguments: --replay-sampling`.

- [ ] **Step 3: Implement — import, load-time bins, append-time bins, sampler, report, flag**

In `sttt/ai.py` imports (after `from .population import ...`):

```python
from .replay_sampling import StratifiedSampler, ply_bin, row_ply, sample_bin_fractions
```

At `sttt/ai.py:150-157`, after `replay_meta` is built:

```python
    # Game-phase bins for stratified sampling; derived from each row, so legacy
    # checkpoints need no migration. Kept parallel to `replay` like replay_meta.
    replay_bins = deque((ply_bin(row_ply(row[0])) for row in replay), maxlen=args.buffer)
```

Pass `replay_bins=replay_bins` into `_train_loop(...)` (add a `replay_bins=None` keyword; when `None`, build it the same way from `replay`).

At both append sites (`sttt/ai.py:360-362` native path and `sttt/ai.py:367-369` Python path), directly after `replay_meta.append(origin)`:

```python
                    replay_bins.append(ply_bin(row_ply(replay[-1][0])))
```

Replace the sampler at `sttt/ai.py:384-386`:

```python
        sampler = StratifiedSampler(np.fromiter(replay_bins, dtype=np.int8, count=len(replay_bins)))
        sampled_bins = []
        for _ in range(args.steps):
            if getattr(args, 'replay_sampling', 'uniform') == 'stratified':
                indices = sampler.sample(min(args.batch, len(replay)), rng)
            else:
                indices = rng.choice(len(replay), size=min(args.batch, len(replay)), replace=False)
            sampled_bins.append(indices)
            batch = [replay[int(i)] for i in indices]
```

In the report dict (`sttt/ai.py:508-528`), after `'replay_sample_age'`:

```python
                  'replay_sampling': getattr(args, 'replay_sampling', 'uniform'),
                  'replay_depth_histogram': sampler.histogram(),
                  'replay_sample_depth_fractions': sample_bin_fractions(
                      sampler.bins, np.concatenate(sampled_bins) if sampled_bins else np.empty(0, dtype=int)),
```

Argparse, next to `--augment-symmetry` (`sttt/ai.py:854`):

```python
    t.add_argument('--replay-sampling', choices=['uniform', 'stratified'], default='uniform',
                   help='stratified: equal share of each nine-ply game phase per batch (uttt.ai-style '
                        'depth balancing); uniform: the historical FIFO sampler')
```

- [ ] **Step 4: Run to verify pass, then the neighbouring suites**

Run: `nice -n 19 $PY -m unittest tests.test_replay_sampling tests.test_training_state tests.test_ai -v`
Expected: all `ok` (the AST checkpoint-schema test is unaffected: the checkpoint dict is unchanged).

- [ ] **Step 5: Commit**

```bash
git add sttt/ai.py tests/test_replay_sampling.py
git commit -m "Trainer: --replay-sampling stratified with depth histogram in metrics

Default stays uniform. metrics.jsonl now records the buffer's ply-bin
composition and the fraction of each batch drawn from each bin.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

**Part 2 gate:** full suite green (minus the known failure); `metrics.jsonl` from Step 1 shows non-degenerate fractions. Merge to `main`. The `train.sh` switch happens in Part 7.

---

## Part 3 — One shared `policy_value_loss`

The loss lives inline at `sttt/ai.py:398-406`. `pretrain` (Part 6) needs the identical objective, and the Q term (Part 4) must land in exactly one place. Pure refactor: bit-identical numbers.

### Task 3.1: Extract `policy_value_loss` into `sttt/learning.py`

**Files:**
- Modify: `sttt/learning.py` (add function after `encode_states`), `sttt/ai.py:398-406`
- Test: `tests/test_learning_loss.py`

**Interfaces:**
- Produces: `policy_value_loss(model, x, pi, mask, z, *, use_fp16=False) -> tuple[torch.Tensor, dict]`; the dict has keys `'policy'`, `'value'` (0-d tensors). Part 4 adds `q`/`q_mask` keyword arguments and a `'q'` key.

- [ ] **Step 1: Write the failing tests**

```python
import ast, inspect, unittest
import numpy as np
import torch
from sttt import ai
from sttt.learning import Network, policy_value_loss


def _batch(n=6, seed=0):
    """Random but legal-shaped training batch shared by every loss test."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 289, generator=g)
    mask = torch.rand(n, 81, generator=g) > 0.6
    mask[:, 0] = True
    pi = torch.rand(n, 81, generator=g) * mask
    pi = pi / pi.sum(-1, keepdim=True)
    z = torch.rand(n, generator=g) * 2 - 1
    return x, pi, mask, z


class SharedLossTests(unittest.TestCase):
    def test_matches_the_reference_formula(self):
        torch.manual_seed(0)
        model = Network()
        x, pi, mask, z = _batch()
        loss, parts = policy_value_loss(model, x, pi, mask, z)
        logits, value = model(x)
        log_p = logits.masked_fill(~mask, -1e9).log_softmax(-1)
        log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
        ref_policy = -(pi * log_p).sum(-1).mean()
        ref_value = (value - z).square().mean()
        self.assertTrue(torch.allclose(parts['policy'], ref_policy))
        self.assertTrue(torch.allclose(parts['value'], ref_value))
        self.assertTrue(torch.allclose(loss, ref_policy + ref_value))

    def test_illegal_logits_do_not_leak_into_the_policy_term(self):
        torch.manual_seed(1)
        model = Network()
        x, pi, mask, z = _batch()
        loss_a, _ = policy_value_loss(model, x, pi, mask, z)
        # Poison illegal logits by changing inputs' effect only on masked-out
        # actions is not possible directly; instead verify the gradient of the
        # policy term w.r.t. illegal logits is exactly zero.
        logits, _ = model(x)
        logits = logits.detach().requires_grad_(True)
        log_p = logits.masked_fill(~mask, -1e9).log_softmax(-1)
        log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
        (-(pi * log_p).sum(-1).mean()).backward()
        self.assertTrue(torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask])))
        self.assertTrue(torch.isfinite(loss_a))

    def test_train_loop_has_a_single_loss_implementation(self):
        source = inspect.getsource(ai._train_loop)
        self.assertNotIn('log_softmax', source)
        self.assertIn('policy_value_loss(', source)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_learning_loss -v`
Expected: `ImportError: cannot import name 'policy_value_loss'`.

- [ ] **Step 3: Implement**

In `sttt/learning.py`, after `encode_states`:

```python
def policy_value_loss(model, x, pi, mask, z, *, use_fp16=False):
    """AlphaZero objective for one batch: masked policy cross-entropy + value MSE.

    The only implementation; the trainer and the offline pretrainer both call
    it. `-1e4` under fp16 because `-inf`/`-1e9` overflow to NaN in half
    precision; both saturate the softmax identically.
    """
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_fp16):
        logits, value = model(x)
        mask_val = -1e4 if use_fp16 else -1e9
        logits = logits.masked_fill(~mask, mask_val)
        log_p = logits.log_softmax(-1)
        log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
        policy_loss = -(pi * log_p).sum(-1).mean()
        value_loss = (value - z).square().mean()
        loss = policy_loss + value_loss
    return loss, {'policy': policy_loss, 'value': value_loss}
```

Replace `sttt/ai.py:398-406` with:

```python
            loss, parts = policy_value_loss(model, x, pi, mask, z, use_fp16=use_fp16)
            policy_loss, value_loss = parts['policy'], parts['value']
```

and add `policy_value_loss` to the `from .learning import ...` line at `sttt/ai.py:13`.

- [ ] **Step 4: Run to verify pass**

Run: `nice -n 19 $PY -m unittest tests.test_learning_loss tests.test_training_state tests.test_encoding -v`
Expected: all `ok`. `tests.test_encoding.TestNoDeadCodeInEvaluateMany` is unaffected (it inspects `evaluate_many`, not the loss).

- [ ] **Step 5: Commit**

```bash
git add sttt/learning.py sttt/ai.py tests/test_learning_loss.py
git commit -m "Extract policy_value_loss so the trainer and pretrainer share one objective

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Part 4 — Dense action-value (Q) head data path

`uttt.ai` regresses a Q value for every legal action (`policy_value_loss.py`, spec §5.2), giving dense gradients even in lost games. The MCTS target is `Q(s,a) = W(s,a)/N(s,a)` at the root. Both `sttt.search.Node` and native `FastNode` expose `n`, `total`, `children`, and after `run()` no visits are in flight, so the target is read from Python for both backends. Sign convention: `Node.total` accumulates values from the child's mover's perspective, so from the root player's view `Q = -total/n` (this is what `TreeSearch._q` does at `sttt/search.py:86-90`).

### Task 4.1: `root_action_values` for Python and native trees

**Files:**
- Modify: `sttt/search.py` (module-level function after `class Node`)
- Test: `tests/test_search.py` (add class), `tests/test_cpp_mcts.py` (add differential test)

**Interfaces:**
- Produces: `root_action_values(root) -> tuple[np.ndarray, np.ndarray]` — `q` float32 `[81]` from the root player's perspective, `visited` bool `[81]`; unvisited/illegal entries are `0.0`/`False`. Proven children use their exact proof value.

- [ ] **Step 1: Write the failing tests**

In `tests/test_search.py`:

```python
from sttt.search import root_action_values


class RootActionValueTests(unittest.TestCase):
    def test_mean_backed_up_value_from_root_perspective_and_visited_mask(self):
        tree = TreeSearch(FakeEvaluator(value=0.3), np.random.default_rng(0),
                          SearchConfig(proofs=False, reuse=False))
        state = State()
        tree.run(state, 64, batch_size=4)
        q, visited = root_action_values(tree.root)
        self.assertEqual(q.shape, (81,))
        self.assertEqual(q.dtype, np.float32)
        self.assertTrue(visited.any())
        for action, child in tree.root.children.items():
            if child.n:
                self.assertTrue(visited[action])
                self.assertAlmostEqual(float(q[action]), -child.total / child.n, places=6)
            else:
                self.assertFalse(visited[action])
                self.assertEqual(float(q[action]), 0.0)
        illegal = np.setdiff1d(np.arange(81), state.legal_actions())
        self.assertFalse(visited[illegal].any())

    def test_proven_children_report_their_proof_value_even_when_unvisited(self):
        # Contract test on a hand-built root: proofs win over visit statistics,
        # an unvisited unproven child is neither valued nor marked, and the
        # sign is flipped into the root player's perspective.
        root = Node(State())
        proven_win = Node(State().play(0)); proven_win.solved = -1      # mover of that child loses
        proven_loss = Node(State().play(1)); proven_loss.solved = 1
        visited = Node(State().play(2)); visited.n, visited.total = 4, 1.0
        untouched = Node(State().play(3))
        root.children = {0: proven_win, 1: proven_loss, 2: visited, 3: untouched}
        q, mask = root_action_values(root)
        self.assertEqual(float(q[0]), 1.0); self.assertTrue(mask[0])
        self.assertEqual(float(q[1]), -1.0); self.assertTrue(mask[1])
        self.assertAlmostEqual(float(q[2]), -0.25, places=6); self.assertTrue(mask[2])
        self.assertEqual(float(q[3]), 0.0); self.assertFalse(mask[3])
        self.assertEqual(int(mask.sum()), 3)
```

(`FakeEvaluator`, `Node`, `TreeSearch`, `SearchConfig` and `State` are already imported/defined in `tests/test_search.py`; add `Node` and `root_action_values` to the `from sttt.search import ...` line.)

In `tests/test_cpp_mcts.py`, inside the existing native test class:

```python
    def test_root_action_values_match_python_tree(self):
        from sttt.search import root_action_values
        evaluator = DeterministicEvaluator(value=0.2)
        cfg = SearchConfig(soft_pruning=False, proofs=False, reuse=False, c_puct=1.5)
        s = State().play(40).play(4)
        py_tree, cpp_tree = PyTreeSearch(evaluator, config=cfg), CppTreeSearch(evaluator, config=cfg)
        py_tree.run(s, simulations=96, batch_size=8)
        cpp_tree.run(s, simulations=96, batch_size=8)
        py_q, py_v = root_action_values(py_tree.root)
        cpp_q, cpp_v = root_action_values(cpp_tree.root)
        self.assertTrue((py_v == cpp_v).all())
        np.testing.assert_allclose(py_q, cpp_q, atol=1e-5)
        self.assertEqual(cpp_tree.root.in_flight, 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_search.RootActionValueTests tests.test_cpp_mcts -v`
Expected: `ImportError: cannot import name 'root_action_values'`.

- [ ] **Step 3: Implement** in `sttt/search.py` after `class Node`:

```python
def root_action_values(root):
    """Q(s, a) targets from a finished search: mean backed-up value of each
    visited root child from the root player's view, and which actions were
    visited. Works for both `Node` and the native `FastNode` (same `n`,
    `total`, `solved`, `children` attributes). Proven children report their
    exact proof value. Unvisited and illegal actions are 0 / False.
    """
    q = np.zeros(81, dtype=np.float32)
    visited = np.zeros(81, dtype=bool)
    for action, child in root.children.items():
        solved = getattr(child, 'solved', None)
        if solved is not None:
            q[action] = -float(solved)
            visited[action] = True
        elif child.n > 0:
            q[action] = -float(child.total) / child.n
            visited[action] = True
    return q, visited
```

- [ ] **Step 4: Rebuild native, run to verify pass**

Run: `make -C cpp clean && make -C cpp PYTHON=$PY && nice -n 19 $PY -m unittest tests.test_search tests.test_cpp_mcts -v`
Expected: all `ok`, native tests executed (not skipped).

- [ ] **Step 5: Commit**

```bash
git add sttt/search.py tests/test_search.py tests/test_cpp_mcts.py
git commit -m "search: root_action_values reads Q(s,a) targets from either tree backend

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

### Task 4.2: Record Q in trajectories; replay rows become 6-tuples; `packed-v2`

**Files:**
- Modify: `sttt/selfplay.py:99` (trajectory append), `sttt/ai.py:236-275` (pack/unpack), `sttt/ai.py:351-369` (append), `sttt/ai.py:393-396` (batch build + augment), `sttt/population.py:182-196` (`augment_batch` extra tensors)
- Modify tests that unpack 2-tuples: `tests/test_population.py:65-67`, `tests/test_selfplay.py:20-22`
- Test: `tests/test_training_state.py` (`TestReplayPacking` additions), `tests/test_population.py` (augment extras)

**Interfaces:**
- Trajectory entry: `(state, pi, q, q_mask)` — `q` float32 `[81]`, `q_mask` bool `[81]`.
- Replay row: `(x, pi, mask, z, q, q_mask)` tensors (z is a float).
- `pack_replay(rows)` writes `{'format': 'packed-v2', 'count', 'x', 'pi', 'mask', 'z', 'q', 'q_mask'}`; `unpack_replay` accepts `packed-v1`, `packed-v2`, legacy lists of 4- or 6-tuples, and fills `q = zeros`, `q_mask = False` when absent. `replay_length` accepts both formats.
- `augment_batch(x, pi, mask, rng, *cell_tensors)` returns `(x, pi, mask, *transformed_cell_tensors)`; each extra tensor is `[N, 81]` and is permuted with `cells`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_training_state.py`, inside `TestReplayPacking` (adjust the existing `_rows` helper to produce 6-tuples with `q = torch.rand(81)` and `q_mask = torch.rand(81) > 0.5`):

```python
    def test_packed_v2_roundtrip_carries_q_and_mask(self):
        rows = self._rows(9)
        packed = pack_replay(rows)
        self.assertEqual(packed['format'], 'packed-v2')
        self.assertEqual(packed['q'].shape, (9, 81))
        self.assertEqual(packed['q_mask'].dtype, torch.bool)
        back = unpack_replay(packed)
        for a, b in zip(rows, back):
            self.assertEqual(len(b), 6)
            self.assertTrue(torch.equal(a[4], b[4]))
            self.assertTrue(torch.equal(a[5], b[5]))

    def test_packed_v1_loads_with_zero_q_and_empty_mask(self):
        rows = self._rows(4)
        v1 = {'format': 'packed-v1', 'count': 4,
              'x': torch.stack([r[0] for r in rows]), 'pi': torch.stack([r[1] for r in rows]),
              'mask': torch.stack([r[2] for r in rows]),
              'z': torch.tensor([r[3] for r in rows], dtype=torch.float64)}
        back = unpack_replay(v1)
        self.assertEqual(replay_length(v1), 4)
        for row in back:
            self.assertEqual(len(row), 6)
            self.assertTrue(torch.equal(row[4], torch.zeros(81)))
            self.assertFalse(row[5].any())

    def test_legacy_four_tuple_list_is_widened(self):
        rows = [r[:4] for r in self._rows(3)]
        back = unpack_replay(rows)
        self.assertEqual([len(r) for r in back], [6, 6, 6])
        self.assertTrue(torch.equal(back[0][0], rows[0][0]))
```

In `tests/test_population.py`, add to `SymmetryGroupTests`:

```python
    def test_augment_batch_permutes_extra_cell_tensors_with_the_policy(self):
        rng = np.random.default_rng(11)
        state = State().play(40).play(4)
        legal = state.legal_actions()
        x = torch.tensor(np.stack([encode(state)] * 32))
        pi = torch.zeros(32, 81); pi[:, legal] = 1 / len(legal)
        mask = pi > 0
        q = torch.rand(32, 81) * mask
        xx, pp, mm, qq, qm = augment_batch(x, pi, mask, rng, q, mask.clone())
        self.assertTrue(torch.equal(qm, mm))                      # extra mask moved with the legal mask
        self.assertTrue(torch.equal(qq != 0, mm))                 # q values sit exactly on legal cells
        # the multiset of q values is preserved per row (a permutation, not a mixture)
        self.assertTrue(torch.allclose(qq.sort(1).values, q.sort(1).values))
```

Update the existing unpackers:
- `tests/test_population.py:65-67`: `for state, pi, q, q_mask in trajectory:` and add `self.assertTrue(set(np.flatnonzero(q_mask)).issubset(state.legal_actions()))`.
- `tests/test_selfplay.py:20-22`: `for state, pi, q, q_mask in trajectory:` and add `self.assertEqual(q.shape, (81,)); self.assertTrue(q_mask.any())`.

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_training_state.TestReplayPacking tests.test_population tests.test_selfplay -v`
Expected: failures on `packed-v2`, `ValueError: not enough values to unpack (expected 4, got 2)`, and `augment_batch() takes 4 positional arguments`.

- [ ] **Step 3: Implement**

`sttt/selfplay.py` — import and record:

```python
from .search import TreeSearch, root_action_values
```

Replace `sttt/selfplay.py:99` `trajectory.append((state, pi))` with:

```python
        q, q_mask = root_action_values(tree.root)
        trajectory.append((state, pi, q, q_mask))
```

`sttt/population.py:182-196`:

```python
def augment_batch(x, pi, mask, rng, *cell_tensors):
    """Transform inputs, forced board, legal mask, policy and any further
    per-action tensors (Q targets and their mask) together."""
    import torch
    out_x, out_pi, out_mask = x.clone(), pi.clone(), mask.clone()
    extras = [t.clone() for t in cell_tensors]
    choices = rng.integers(8, size=len(x))
    for sym, (inputs, cells) in enumerate(SYMMETRIES):
        rows = torch.as_tensor(np.flatnonzero(choices == sym), device=x.device)
        if not len(rows):
            continue
        ii = torch.as_tensor(inputs, device=x.device)
        cc = torch.as_tensor(cells, device=x.device)
        out_x[rows[:, None], ii] = x[rows]
        out_pi[rows[:, None], cc] = pi[rows]
        out_mask[rows[:, None], cc] = mask[rows]
        for out, src in zip(extras, cell_tensors):
            out[rows[:, None], cc] = src[rows]
    return (out_x, out_pi, out_mask, *extras)
```

`sttt/ai.py:236-275`:

```python
REPLAY_PACKED_FORMAT = 'packed-v2'
REPLAY_PACKED_FORMATS = ('packed-v1', 'packed-v2')


def widen_row(row):
    """Legacy (x, pi, mask, z) -> (x, pi, mask, z, q, q_mask) with no Q targets."""
    if len(row) == 6:
        return row
    x, pi, mask, z = row
    return (x, pi, mask, z, torch.zeros(81, dtype=torch.float32), torch.zeros(81, dtype=torch.bool))


def pack_replay(replay):
    """Stack the replay's (x, pi, mask, z, q, q_mask) rows into tensors for saving."""
    rows = [widen_row(r) for r in replay]
    if not rows:
        return {'format': REPLAY_PACKED_FORMAT, 'count': 0}
    return {'format': REPLAY_PACKED_FORMAT, 'count': len(rows),
            'x': torch.stack([r[0] for r in rows]),
            'pi': torch.stack([r[1] for r in rows]),
            'mask': torch.stack([r[2] for r in rows]),
            'z': torch.tensor([r[3] for r in rows], dtype=torch.float64),
            'q': torch.stack([r[4] for r in rows]),
            'q_mask': torch.stack([r[5] for r in rows])}


def unpack_replay(saved):
    """Rows from packed-v1, packed-v2 or a legacy list; always 6-tuples."""
    if saved is None:
        return []
    if isinstance(saved, dict) and saved.get('format') in REPLAY_PACKED_FORMATS:
        n = int(saved['count'])
        if n == 0:
            return []
        q = saved.get('q') if saved.get('format') == 'packed-v2' else None
        q_mask = saved.get('q_mask') if saved.get('format') == 'packed-v2' else None
        if q is None:
            q = torch.zeros(n, 81, dtype=torch.float32)
            q_mask = torch.zeros(n, 81, dtype=torch.bool)
        return [(x.clone(), pi.clone(), mask.clone(), z, qq.clone(), qm.clone())
                for x, pi, mask, z, qq, qm in
                zip(saved['x'].unbind(0), saved['pi'].unbind(0), saved['mask'].unbind(0),
                    saved['z'].tolist(), q.unbind(0), q_mask.unbind(0))]
    return [widen_row(r) for r in saved]


def replay_length(saved):
    if isinstance(saved, dict) and saved.get('format') in REPLAY_PACKED_FORMATS:
        return int(saved['count'])
    return len(saved or [])
```

`sttt/ai.py:351-369` — both append sites become:

```python
                for i, (state, pi, q, q_mask) in enumerate(trajectory):
                    mask = np.zeros(81, dtype=bool)
                    mask[state.legal_actions()] = True
                    replay.append((torch.from_numpy(encoded_batch[i].copy()), torch.from_numpy(pi),
                                   torch.from_numpy(mask), float(outcome * state.turn),
                                   torch.from_numpy(q), torch.from_numpy(q_mask)))
```

(and the Python-encode path identically with `torch.from_numpy(encode(state))`; `traj_states = [s for s, *_ in trajectory]`).

`sttt/ai.py:393-396` batch build:

```python
            x, pi, mask, q, q_mask = [torch.stack([row[k] for row in batch]).to(device) for k in (0, 1, 2, 4, 5)]
            if getattr(args, 'augment_symmetry', False):
                x, pi, mask, q, q_mask = augment_batch(x, pi, mask, rng, q, q_mask)
```

- [ ] **Step 4: Confirm no other trajectory unpackers remain**

Run: `grep -rnE "for (state|s), ?pi in trajectory|\(state, pi\) in trajectory" sttt scripts tests`
Expected: no output. (`scripts/check_training_ready.py:1243,1260` only `extend`s and slices the list — unaffected.)

- [ ] **Step 5: Run to verify pass**

Run: `nice -n 19 $PY -m unittest tests.test_training_state tests.test_population tests.test_selfplay tests.test_replay_sampling -v`
Expected: all `ok`.

- [ ] **Step 6: Commit**

```bash
git add sttt/selfplay.py sttt/population.py sttt/ai.py tests/test_training_state.py tests/test_population.py tests/test_selfplay.py
git commit -m "Record root Q(s,a) targets in trajectories; replay packed-v2 with legacy load

Rows carry q and q_mask; packed-v1 and legacy lists load with zero targets so
existing checkpoints resume unchanged. augment_batch permutes the new per-
action tensors with the policy.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

### Task 4.3: `forward_all` and the masked Q loss

**Files:**
- Modify: `sttt/learning.py` (`BasePolicyValue.forward_all`, `policy_value_loss`), `sttt/ai.py` (call site, metrics print, checkpoint `metrics`, report)
- Test: `tests/test_learning_loss.py`

**Interfaces:**
- `BasePolicyValue.forward_all(x) -> (logits, value, q_or_None)`; default implementation returns `(*self(x), None)`. `UNet` (Part 5) overrides it.
- `policy_value_loss(model, x, pi, mask, z, *, use_fp16=False, q=None, q_mask=None)`; `parts['q']` is a 0-d tensor when the model has a Q output and `q_mask.any()`, else `None`.
- Metrics: `q_loss` in the iteration print, `checkpoint['metrics']['q_loss']` and the `metrics.jsonl` row (`None` for models without a Q head).

- [ ] **Step 1: Write the failing tests**

```python
class QHeadLossTests(unittest.TestCase):
    class WithQ(Network):
        def __init__(self):
            super().__init__()
            self.qhead = torch.nn.Linear(256, 81)
        def forward_all(self, x):
            h = self.trunk(x)
            return self.policy(h), self.value(h).tanh().squeeze(-1), self.qhead(h).tanh()

    def test_models_without_q_output_ignore_q_targets(self):
        torch.manual_seed(0)
        model = Network()
        x, pi, mask, z = _batch()
        loss_plain, parts_plain = policy_value_loss(model, x, pi, mask, z)
        loss_q, parts_q = policy_value_loss(model, x, pi, mask, z, q=torch.rand(6, 81), q_mask=mask)
        self.assertIsNone(parts_q['q'])
        self.assertTrue(torch.equal(loss_plain, loss_q))

    def test_q_term_is_masked_mse_over_visited_actions_only(self):
        torch.manual_seed(0)
        model = self.WithQ()
        x, pi, mask, z = _batch()
        q = torch.rand(6, 81) * 2 - 1
        q_mask = mask.clone(); q_mask[0] = False       # one row with no targets
        loss, parts = policy_value_loss(model, x, pi, mask, z, q=q, q_mask=q_mask)
        _, _, q_pred = model.forward_all(x)
        ref = ((q_pred - q).square() * q_mask).sum() / q_mask.sum()
        self.assertTrue(torch.allclose(parts['q'], ref))
        self.assertTrue(torch.allclose(loss, parts['policy'] + parts['value'] + parts['q']))

    def test_all_false_mask_contributes_nothing_and_no_nan(self):
        torch.manual_seed(0)
        model = self.WithQ()
        x, pi, mask, z = _batch()
        loss, parts = policy_value_loss(model, x, pi, mask, z, q=torch.zeros(6, 81),
                                        q_mask=torch.zeros(6, 81, dtype=torch.bool))
        self.assertIsNone(parts['q'])
        self.assertTrue(torch.isfinite(loss))

    def test_trainer_passes_q_targets_and_reports_q_loss(self):
        source = inspect.getsource(ai._train_loop)
        self.assertIn('q=q, q_mask=q_mask', source)
        self.assertIn("'q_loss'", source)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_learning_loss.QHeadLossTests -v`
Expected: `TypeError: policy_value_loss() got an unexpected keyword argument 'q'`.

- [ ] **Step 3: Implement**

In `BasePolicyValue` (`sttt/learning.py`, before `evaluate_many`):

```python
    def forward_all(self, x):
        """(logits, value, q) — q is None for architectures without an action-value head."""
        logits, value = self(x)
        return logits, value, None
```

Replace `policy_value_loss`:

```python
def policy_value_loss(model, x, pi, mask, z, *, use_fp16=False, q=None, q_mask=None):
    """AlphaZero objective plus, when the model has an action-value head and the
    batch carries visited-action targets, uttt.ai's dense Q regression: masked
    MSE over visited legal actions, averaged over the number of targets so rows
    without targets (legacy replay) contribute nothing.
    """
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_fp16):
        logits, value, q_pred = model.forward_all(x)
        mask_val = -1e4 if use_fp16 else -1e9
        logits = logits.masked_fill(~mask, mask_val)
        log_p = logits.log_softmax(-1)
        log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
        policy_loss = -(pi * log_p).sum(-1).mean()
        value_loss = (value - z).square().mean()
        loss = policy_loss + value_loss
        q_loss = None
        if q_pred is not None and q is not None and q_mask is not None and bool(q_mask.any()):
            q_loss = ((q_pred.float() - q).square() * q_mask).sum() / q_mask.sum()
            loss = loss + q_loss
    return loss, {'policy': policy_loss, 'value': value_loss, 'q': q_loss}
```

In `sttt/ai.py` `_train_loop`: call `policy_value_loss(model, x, pi, mask, z, use_fp16=use_fp16, q=q, q_mask=q_mask)`; keep a `q_losses = []` list, append `float(parts['q'])` when not `None`; add `q_loss={np.mean(q_losses):.4f}` to the print when the list is non-empty; add `'q_loss': float(np.mean(q_losses)) if q_losses else None` to `checkpoint['metrics']` and to the `report` dict.

- [ ] **Step 4: Run to verify pass**

Run: `nice -n 19 $PY -m unittest tests.test_learning_loss tests.test_training_state tests.test_replay_sampling -v`
Expected: all `ok`.

- [ ] **Step 5: Commit**

```bash
git add sttt/learning.py sttt/ai.py tests/test_learning_loss.py
git commit -m "Dense action-value loss: forward_all + masked Q MSE, reported as q_loss

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

**Part 4 gate:** full suite; one native 2-iteration smoke on CUDA (`--backend cpp --device cuda --fp16 --games 2 --simulations 32 --iterations 2`) shows `q_loss` is `None` for `resnet` and the checkpoint reloads via `train --resume`. Merge.

---

## Part 5 — Hierarchical conv U-Net (`--arch unet`)

Spec §5.1/§6 Step 1. Design choices fixed here:
- Input is the existing 289-float row, reshaped inside the model to `(8, 9, 9)`: 3 cell planes in grid order, 4 board planes tiled 3×3 to 9×9, 1 forced-board plane (ones over the forced board's 9 cells, ones everywhere when any board is allowed). Nothing upstream changes.
- `GroupNorm`, not `BatchNorm`: the same module object serves self-play inference (`model.eval()` in `SelfPlayPool.run`) and training; GroupNorm has no running statistics to drift and behaves identically at batch size 1 and 1024.
- Heads: policy `Conv2d(·,1,1)` → 81 logits in **cell order**; Q `Conv2d(·,1,1)` → 81 `tanh`; value from macro features `Flatten → Linear(1152,256) → ReLU → Linear(256,1) → tanh`.

### Task 5.1: Plane geometry — `planes_from_flat` and index tables

**Files:**
- Create: `sttt/unet.py`
- Test: `tests/test_unet.py`

**Interfaces:**
- Produces: `CELL_TO_GRID: torch.LongTensor[81]` (grid flat index for each cell index), `GRID_TO_CELL` (its inverse), `planes_from_flat(x: Tensor[N,289]) -> Tensor[N,8,9,9]`.

- [ ] **Step 1: Write the failing tests**

```python
import unittest
import numpy as np
import torch
from sttt.env import State
from sttt.learning import encode
from sttt.population import SYMMETRIES
from sttt.unet import CELL_TO_GRID, GRID_TO_CELL, planes_from_flat


class PlaneGeometryTests(unittest.TestCase):
    def test_index_tables_are_inverse_permutations(self):
        self.assertEqual(sorted(CELL_TO_GRID.tolist()), list(range(81)))
        self.assertTrue(torch.equal(GRID_TO_CELL[CELL_TO_GRID], torch.arange(81)))

    def test_stone_lands_at_the_geometric_cell(self):
        for action in [0, 8, 40, 72, 80, 13, 67]:
            state = State().play(action)                      # X played; now O to move
            planes = planes_from_flat(torch.from_numpy(encode(state))[None])
            b, c = action // 9, action % 9
            row, col = (b // 3) * 3 + c // 3, (b % 3) * 3 + c % 3
            self.assertEqual(planes.shape, (1, 8, 9, 9))
            self.assertEqual(float(planes[0, 2, row, col]), 1.0)   # opponent-of-mover plane
            self.assertEqual(float(planes[0, 2].sum()), 1.0)
            self.assertEqual(float(planes[0, 0].sum()), 80.0)     # empties

    def test_board_and_forced_planes_are_tiled_over_their_mini_board(self):
        state = State().play(40)                              # forces board 4 (centre)
        planes = planes_from_flat(torch.from_numpy(encode(state))[None])
        forced = planes[0, 7]
        self.assertEqual(float(forced.sum()), 9.0)
        self.assertTrue(torch.equal(forced[3:6, 3:6], torch.ones(3, 3)))
        boards_open = planes[0, 3]
        self.assertEqual(float(boards_open.sum()), 81.0)      # all nine boards still open

    def test_free_move_forced_plane_is_all_ones(self):
        state = State()
        planes = planes_from_flat(torch.from_numpy(encode(state))[None])
        self.assertTrue(torch.equal(planes[0, 7], torch.ones(9, 9)))

    def test_symmetry_table_acts_on_planes_as_a_whole_board_transform(self):
        rng = np.random.default_rng(5)
        state = State()
        for _ in range(17):
            state = state.play(int(rng.choice(state.legal_actions())))
        x = torch.from_numpy(encode(state))
        base = planes_from_flat(x[None])[0]
        grid = np.arange(81).reshape(9, 9)
        expected_ops = []
        for mirror in (False, True):
            b = np.fliplr(grid) if mirror else grid
            for k in range(4):
                expected_ops.append(torch.as_tensor(np.rot90(b, k).copy().ravel()))
        for inputs, _ in SYMMETRIES:
            xt = torch.empty_like(x); xt[torch.as_tensor(inputs)] = x
            transformed = planes_from_flat(xt[None])[0]
            matches = any(torch.equal(transformed.flatten(1)[:, op], base.flatten(1)) for op in expected_ops)
            self.assertTrue(matches)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_unet -v`
Expected: `ModuleNotFoundError: No module named 'sttt.unet'`.

- [ ] **Step 3: Implement geometry in `sttt/unet.py`**

```python
"""Hierarchical conv U-Net over the canonical 289-float encoding.

uttt.ai's network pools each 3x3 mini-board into one macro cell with a
stride-3 convolution, reasons on the 3x3 macro grid, and upsamples back with a
stride-3 transposed convolution plus skip connections. Our flat MLP has no idea
the board is 3x3-of-3x3. This module keeps the encoding, replay buffer, native
encoder and symmetry augmentation untouched by reshaping the flat row into
(8, 9, 9) planes inside forward().
"""
import numpy as np
import torch
from torch import nn
from .learning import BasePolicyValue


def _cell_to_grid():
    idx = np.arange(81)
    b, c = idx // 9, idx % 9
    return ((b // 3) * 3 + c // 3) * 9 + ((b % 3) * 3 + c % 3)


CELL_TO_GRID = torch.as_tensor(_cell_to_grid(), dtype=torch.long)
GRID_TO_CELL = torch.argsort(CELL_TO_GRID)


def planes_from_flat(x):
    """float[N,289] -> float[N,8,9,9].

    Planes: 0 empty, 1 mine, 2 theirs (cells in grid order); 3 open, 4 mine,
    5 theirs, 6 drawn (boards, each tiled over its 3x3 cells); 7 forced board
    (ones over the forced mini-board, ones everywhere when any board is legal).
    """
    n = x.shape[0]
    cells = x[:, :243].reshape(n, 3, 81)
    grid = torch.empty_like(cells)
    grid[:, :, CELL_TO_GRID.to(x.device)] = cells
    cells9 = grid.reshape(n, 3, 9, 9)
    boards9 = x[:, 243:279].reshape(n, 4, 3, 3).repeat_interleave(3, 2).repeat_interleave(3, 3)
    forced = x[:, 279:289]
    forced3 = forced[:, 1:].reshape(n, 1, 3, 3) + forced[:, :1].reshape(n, 1, 1, 1)
    forced9 = forced3.repeat_interleave(3, 2).repeat_interleave(3, 3)
    return torch.cat([cells9, boards9, forced9], dim=1)
```

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m unittest tests.test_unet -v`
Expected: 5 tests `ok`.

- [ ] **Step 5: Commit**

```bash
git add sttt/unet.py tests/test_unet.py
git commit -m "unet: plane geometry from the flat encoding, proven against State and SYMMETRIES

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

### Task 5.2: The `UNet` module and its registration in `create_model` / `load_model` / `--arch`

**Files:**
- Modify: `sttt/unet.py`, `sttt/learning.py:138-163` (`create_model`, `load_model`), `sttt/ai.py:113` (arch detection on resume), `sttt/ai.py:835` (`--arch` choices)
- Test: `tests/test_unet.py`

**Interfaces:**
- `class UNet(BasePolicyValue)` with `forward(x) -> (logits[N,81], value[N])` and `forward_all(x) -> (logits, value, q[N,81])`, constructor `UNet(micro=64, macro=128, micro_blocks=2, macro_blocks=3)`.
- `create_model('unet')`, `load_model` recognises `arch == 'unet'` and state dicts containing `'stem.0.weight'`.
- `sttt.learning.arch_name(model) -> str` (`'unet' | 'resnet' | 'mlp'`) replacing the `isinstance` at `sttt/ai.py:113`.

- [ ] **Step 1: Write the failing tests**

```python
from sttt.learning import create_model, load_model, arch_name
from sttt.unet import UNet


class UNetModuleTests(unittest.TestCase):
    def test_output_shapes_and_ranges(self):
        torch.manual_seed(0)
        model = UNet()
        x = torch.from_numpy(np.stack([encode(State()), encode(State().play(40))]))
        logits, value, q = model.forward_all(x)
        self.assertEqual(logits.shape, (2, 81))
        self.assertEqual(value.shape, (2,))
        self.assertEqual(q.shape, (2, 81))
        self.assertTrue((value.abs() <= 1).all() and (q.abs() <= 1).all())
        l2, v2 = model(x)
        self.assertTrue(torch.equal(l2, logits) and torch.equal(v2, value))

    def test_parameter_count_is_in_the_resnet_class(self):
        n = sum(p.numel() for p in UNet().parameters())
        self.assertGreater(n, 1_000_000)
        self.assertLess(n, 3_000_000)

    def test_policy_logits_are_in_cell_order(self):
        # Zero every weight, then bias the policy conv so grid position (row 4,
        # col 4) is hot: that is board 4, cell 4, i.e. action index 40.
        model = UNet()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        x = torch.from_numpy(encode(State()))[None]
        # Inject a spatial signal through the fuse conv's raw-plane skip: the
        # forced plane is all ones at the empty board, so use plane 0 (empties).
        # Simpler and exact: call the head on a synthetic feature map.
        feats = torch.zeros(1, 64, 9, 9); feats[0, 0, 4, 4] = 1.0
        with torch.no_grad():
            model.policy.weight[0, 0] = 1.0
        logits = model.policy(feats).flatten(1)[:, CELL_TO_GRID]
        self.assertEqual(int(logits.argmax()), 40)

    def test_evaluate_many_works_and_masks_illegal_moves(self):
        torch.manual_seed(0)
        model = UNet().eval()
        state = State().play(40)
        (probs, value), = model.evaluate_many([state])
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        illegal = np.setdiff1d(np.arange(81), state.legal_actions())
        self.assertEqual(float(probs[illegal].sum()), 0.0)

    def test_registration_and_checkpoint_roundtrip(self):
        self.assertIsInstance(create_model('unet'), UNet)
        self.assertEqual(arch_name(create_model('unet')), 'unet')
        self.assertEqual(arch_name(create_model('resnet')), 'resnet')
        self.assertEqual(arch_name(create_model('mlp')), 'mlp')
        import tempfile
        from pathlib import Path
        model = UNet()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'm.pt')
            torch.save({'model': model.state_dict(), 'arch': 'unet', 'iteration': 0}, path)
            loaded, ckpt = load_model(path)
            self.assertIsInstance(loaded, UNet)
            torch.save({'model': model.state_dict(), 'iteration': 0}, path)   # no arch field
            loaded, _ = load_model(path)
            self.assertIsInstance(loaded, UNet)

    def test_fp16_autocast_forward_is_finite(self):
        if not torch.cuda.is_available():
            self.fail('CUDA required for this gate; do not skip on the training box')
        model = UNet().cuda()
        x = torch.from_numpy(np.stack([encode(State())] * 16)).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            logits, value, q = model.forward_all(x)
        self.assertTrue(torch.isfinite(logits).all() and torch.isfinite(value).all() and torch.isfinite(q).all())
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_unet.UNetModuleTests -v`
Expected: `ImportError: cannot import name 'UNet'`.

- [ ] **Step 3: Implement the module** (append to `sttt/unet.py`)

```python
def _conv_block(cin, cout, groups=8):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(groups, cout), nn.ReLU())


class ConvResBlock(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(groups, ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(groups, ch))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class UNet(BasePolicyValue):
    """Micro (9x9) -> macro (3x3) -> micro U-Net with policy, value and Q heads.

    GroupNorm rather than BatchNorm: one module object serves self-play
    inference and training, and GroupNorm has no running statistics to drift
    between the two modes or between batch sizes 1 and 1024.
    """

    def __init__(self, micro=64, macro=128, micro_blocks=2, macro_blocks=3):
        super().__init__()
        self.stem = _conv_block(8, micro)
        self.micro = nn.Sequential(*[ConvResBlock(micro) for _ in range(micro_blocks)])
        # stride-3 3x3 conv: each mini-board becomes one macro cell
        self.pool = nn.Sequential(nn.Conv2d(micro, macro, 3, stride=3, bias=False),
                                  nn.GroupNorm(8, macro), nn.ReLU())
        self.macro = nn.Sequential(*[ConvResBlock(macro) for _ in range(macro_blocks)])
        # stride-3 transposed conv: each macro cell is broadcast back to its 3x3 cells
        self.up = nn.Sequential(nn.ConvTranspose2d(macro, micro, 3, stride=3, bias=False),
                                nn.GroupNorm(8, micro), nn.ReLU())
        self.fuse = _conv_block(micro * 2 + 8, micro)
        self.policy = nn.Conv2d(micro, 1, 1)
        self.q = nn.Conv2d(micro, 1, 1)
        self.value = nn.Sequential(nn.Flatten(), nn.Linear(macro * 9, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward_all(self, x):
        planes = planes_from_flat(x)
        h = self.micro(self.stem(planes))
        m = self.macro(self.pool(h))
        f = self.fuse(torch.cat([h, self.up(m), planes], dim=1))
        order = CELL_TO_GRID.to(x.device)
        logits = self.policy(f).flatten(1)[:, order]
        q = self.q(f).flatten(1)[:, order].tanh()
        value = self.value(m).squeeze(-1).tanh()
        return logits, value, q

    def forward(self, x):
        logits, value, _ = self.forward_all(x)
        return logits, value
```

In `sttt/learning.py`:

```python
def arch_name(model):
    from .unet import UNet          # local import: unet imports BasePolicyValue from here
    if isinstance(model, UNet):
        return 'unet'
    return 'resnet' if isinstance(model, ResNet) else 'mlp'


def create_model(arch='resnet'):
    if arch == 'resnet':
        return ResNet()
    if arch == 'mlp':
        return Network()
    if arch == 'unet':
        from .unet import UNet
        return UNet()
    raise ValueError(f"Unknown architecture: {arch}")


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    arch = checkpoint.get('arch')
    state = checkpoint['model']
    if arch is None:
        if 'stem.0.weight' in state:
            arch = 'unet'
        elif 'in_proj.0.weight' in state or 'blocks.0.net.0.weight' in state:
            arch = 'resnet'
        else:
            arch = 'mlp'
    model = create_model(arch)
    model.load_state_dict(state)
    return model.eval(), checkpoint
```

In `sttt/ai.py:113`: `arch = arch_name(model)` (import `arch_name` from `.learning`). At `sttt/ai.py:835`: `choices=['resnet', 'mlp', 'unet']`.

- [ ] **Step 4: Run to verify pass**

Run: `nice -n 19 $PY -m unittest tests.test_unet tests.test_ai tests.test_encoding -v`
Expected: all `ok`, including the CUDA fp16 test (it must run, not skip).

- [ ] **Step 5: Commit**

```bash
git add sttt/unet.py sttt/learning.py sttt/ai.py tests/test_unet.py
git commit -m "Add the hierarchical conv U-Net (--arch unet) with policy, value and Q heads

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

### Task 5.3: End-to-end: the trainer runs a U-Net with a real Q loss

**Files:**
- Test: `tests/test_unet.py` (subprocess smoke)

- [ ] **Step 1: Write the test**

```python
class UNetTrainerSmokeTests(unittest.TestCase):
    def test_one_iteration_reports_q_loss_and_resumes(self):
        import json, subprocess, sys, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            base = [sys.executable, '-m', 'sttt.ai', 'train', '--backend', 'python', '--device', 'cpu',
                    '--arch', 'unet', '--games', '2', '--simulations', '6', '--steps', '3', '--batch', '8',
                    '--workers', '1', '--leaf-batch', '2', '--save-every', '0', '--augment-symmetry',
                    '--replay-sampling', 'stratified', '--output', tmp, '--seed', '3']
            subprocess.run(base + ['--iterations', '1'], check=True, capture_output=True, text=True, timeout=600)
            subprocess.run(base + ['--iterations', '1', '--resume', str(Path(tmp, 'latest.pt'))],
                           check=True, capture_output=True, text=True, timeout=600)
            rows = [json.loads(l) for l in Path(tmp, 'metrics.jsonl').read_text().splitlines()]
        rows = [r for r in rows if 'q_loss' in r]
        self.assertEqual([r['iteration'] for r in rows], [1, 2])
        for r in rows:
            self.assertIsNotNone(r['q_loss'])
            self.assertTrue(np.isfinite(r['q_loss']))
        ckpt = torch.load(Path(tmp, 'latest.pt'), map_location='cpu', weights_only=False)
        self.assertEqual(ckpt['arch'], 'unet')
        self.assertEqual(ckpt['replay']['format'], 'packed-v2')
```

- [ ] **Step 2: Run**

Run: `nice -n 19 $PY -m unittest tests.test_unet.UNetTrainerSmokeTests -v`
Expected: `ok`. If `q_loss` is `None`, `forward_all` is not being dispatched — check `policy_value_loss` is the call site used by `_train_loop`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_unet.py
git commit -m "unet: trainer smoke — q_loss reported, packed-v2 replay, resume works

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

**Part 5 gate:** full suite; native CUDA smoke `--backend cpp --device cuda --fp16 --arch unet --games 4 --simulations 64 --iterations 2 --workers 4` completes with finite `q_loss` and reports `peak_gpu_mb`. Record the per-iteration seconds against the resnet figure in the ledger (Part 7). Merge.

---

## Part 6 — Two-stage bootstrap: `generate-dataset` and `pretrain`

Spec §5.3 / §6 Step 2. Stage 1 is network-free: `CppTreeSearch(model=None)` runs native MCTS with the C++ heuristic leaf evaluator (`test_heuristic_search_pure_cpp` proves this path), and `CppAlphaBetaBot` supplies strong opponents. Games are played by `selfplay._play_game`, so trajectories carry `pi`, `q`, `q_mask` exactly as online self-play does. Shards are written in the replay's `packed-v2` format. Stage 2 (`pretrain`) trains a fresh network on the shards with the shared loss, stratified sampling, and symmetry augmentation, then writes a checkpoint `train --resume` accepts, with a warm replay buffer sampled from the dataset.

### Task 6.1: Game generation worker and shard writer

**Files:**
- Create: `sttt/bootstrap.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- `sample_bootstrap_match(seed: int, rng_stream: int, alphabeta_share: float, depths: Sequence[int]) -> MatchSpec` — `kind` in `{'self', 'alphabeta'}`; for alpha-beta: random `learner_side`, `depth` from `depths`, `nodes=500000`, `epsilon` in `[0, .05]`; `opening_moves` in `[0, 4]`.
- `generate_rows(seed: int, games: int, simulations: int, leaf_batch: int, alphabeta_share: float, depths) -> dict` — numpy arrays `x[N,289] f32`, `pi[N,81] f32`, `mask[N,81] bool`, `z[N] f64`, `q[N,81] f32`, `q_mask[N,81] bool`, `ply[N] i16`, `kind` list of per-game family strings, `games` int. Runs in a worker; torch-free.
- `pack_arrays(rows: dict) -> dict` — the `packed-v2` dict (`format`, `count`, six tensors).
- `write_shard(path, rows, provenance: dict) -> dict` — saves `{'format': 'bootstrap-v1', 'replay': pack_arrays(rows), 'ply': tensor, 'kind': list, 'provenance': dict}`; returns a summary `{'positions', 'games', 'kinds': Counter, 'ply_histogram'}`.
- `load_shards(dataset_dir) -> dict` of concatenated tensors `x, pi, mask, z, q, q_mask, ply`.

- [ ] **Step 1: Write the failing tests**

```python
import tempfile, unittest
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sttt.cpp_env import is_cpp_available
from sttt.bootstrap import (generate_rows, load_shards, pack_arrays, sample_bootstrap_match,
                            write_shard)
from sttt.ai import unpack_replay, replay_length
from sttt.replay_sampling import row_ply


class MatchSamplingTests(unittest.TestCase):
    def test_share_and_ranges_are_respected(self):
        kinds = Counter()
        for i in range(400):
            spec = sample_bootstrap_match(1, i, alphabeta_share=0.5, depths=(4, 5, 6))
            kinds[spec.kind] += 1
            self.assertIn(spec.kind, ('self', 'alphabeta'))
            self.assertIn(spec.opening_moves, range(0, 5))
            if spec.kind == 'alphabeta':
                self.assertIn(spec.depth, (4, 5, 6))
                self.assertIn(spec.learner_side, (1, -1))
                self.assertTrue(0. <= spec.epsilon <= .05)
        self.assertGreater(kinds['alphabeta'], 150)
        self.assertGreater(kinds['self'], 150)

    def test_is_deterministic_in_seed_and_stream(self):
        self.assertEqual(sample_bootstrap_match(3, 7, 0.5, (4,)), sample_bootstrap_match(3, 7, 0.5, (4,)))
        self.assertNotEqual(sample_bootstrap_match(3, 7, 0.5, (4, 5, 6)), sample_bootstrap_match(3, 8, 0.5, (4, 5, 6)))


class GenerationTests(unittest.TestCase):
    def setUp(self):
        if not is_cpp_available():
            self.fail('native extension required: make -C cpp clean && make -C cpp PYTHON=$PY')

    def test_rows_are_consistent_training_targets(self):
        rows = generate_rows(seed=11, games=3, simulations=16, leaf_batch=4, alphabeta_share=0.5, depths=(2,))
        n = rows['x'].shape[0]
        self.assertGreater(n, 20)
        self.assertEqual(rows['games'], 3)
        self.assertEqual(rows['x'].shape, (n, 289)); self.assertEqual(rows['x'].dtype, np.float32)
        self.assertEqual(rows['pi'].shape, (n, 81)); self.assertEqual(rows['mask'].dtype, bool)
        self.assertEqual(rows['q'].shape, (n, 81)); self.assertEqual(rows['q_mask'].dtype, bool)
        self.assertEqual(rows['z'].shape, (n,)); self.assertTrue(np.isin(rows['z'], [-1., 0., 1.]).all())
        np.testing.assert_allclose(rows['pi'].sum(1), 1., atol=1e-5)
        self.assertTrue((rows['pi'] * ~rows['mask'] == 0).all())
        self.assertTrue((rows['q_mask'] <= rows['mask']).all())
        self.assertTrue(rows['q_mask'].any(1).all())
        for i in range(n):
            self.assertEqual(int(rows['ply'][i]), row_ply(rows['x'][i]))

    def test_shard_roundtrip_is_loadable_by_the_replay_code(self):
        rows = generate_rows(seed=2, games=2, simulations=8, leaf_batch=4, alphabeta_share=0., depths=(2,))
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_shard(Path(tmp, 'shard-0000.pt'), rows, {'note': 'test'})
            self.assertEqual(summary['positions'], rows['x'].shape[0])
            shard = torch.load(Path(tmp, 'shard-0000.pt'), map_location='cpu', weights_only=False)
            self.assertEqual(shard['format'], 'bootstrap-v1')
            self.assertEqual(replay_length(shard['replay']), rows['x'].shape[0])
            back = unpack_replay(shard['replay'])
            self.assertEqual(len(back[0]), 6)
            data = load_shards(tmp)
            self.assertEqual(data['x'].shape[0], rows['x'].shape[0])
            self.assertEqual(data['ply'].dtype, torch.int16)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_bootstrap -v`
Expected: `ModuleNotFoundError: No module named 'sttt.bootstrap'`.

- [ ] **Step 3: Implement**

```python
"""Two-stage bootstrap: network-free C++ search generates an offline dataset;
`pretrain` fits a fresh network to it and emits a checkpoint `train --resume`
accepts. uttt.ai spent weeks on stage 1 with Python MCTS; the native engine
does it in hours.
"""
from collections import Counter
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from .population import MatchSpec, make_opponent
from .search import SearchConfig

SHARD_FORMAT = 'bootstrap-v1'
DEFAULT_DEPTHS = (4, 5, 6)


def sample_bootstrap_match(seed, stream, alphabeta_share, depths):
    """Self-play (both sides recorded) or heuristic-MCTS vs C++ alpha-beta."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream), 4409]))
    opening = int(rng.integers(0, 5))
    if rng.random() < alphabeta_share:
        return MatchSpec(kind='alphabeta', learner_side=int(rng.choice([1, -1])),
                         depth=int(rng.choice(list(depths))), nodes=500000,
                         epsilon=float(rng.uniform(0., .05)), opening_moves=opening,
                         requested_kind='alphabeta')
    return MatchSpec(kind='self', opening_moves=opening, requested_kind='self')


def generate_rows(seed, games, simulations, leaf_batch, alphabeta_share, depths):
    """Play `games` games with the model-free native tree; return numpy arrays.

    Runs inside a worker process. Imports the native pieces lazily so the
    parent can spawn torch-free workers.
    """
    from .cpp_env import CppTreeSearch, encode_batch, is_cpp_available
    from .selfplay import _play_game
    if not is_cpp_available() or CppTreeSearch is None:
        raise RuntimeError('bootstrap generation requires the native extension; run make -C cpp')
    xs, pis, masks, zs, qs, qms, plies, kinds = [], [], [], [], [], [], [], []
    for g in range(games):
        game_seed = int(np.random.SeedSequence([int(seed), g]).generate_state(1)[0])
        spec = sample_bootstrap_match(seed, g, alphabeta_share, depths)
        rng = np.random.default_rng(game_seed)
        tree = CppTreeSearch(None, rng, SearchConfig())
        opponent = make_opponent(spec)
        try:
            trajectory, outcome, _ = _play_game(tree, rng, simulations, game_seed, leaf_batch,
                                                spec, opponent, use_cpp=True)
        finally:
            if opponent is not None:
                opponent.close()
        if not trajectory:
            continue
        states = [s for s, *_ in trajectory]
        encoded = np.ascontiguousarray(encode_batch(states), dtype=np.float32)
        for i, (state, pi, q, q_mask) in enumerate(trajectory):
            mask = np.zeros(81, dtype=bool)
            mask[state.legal_actions()] = True
            xs.append(encoded[i].copy()); pis.append(pi.astype(np.float32)); masks.append(mask)
            zs.append(float(outcome * state.turn)); qs.append(q); qms.append(q_mask)
            plies.append(81 - int(round(float(encoded[i][:81].sum()))))
        kinds.append(spec.kind)
    return {'x': np.stack(xs), 'pi': np.stack(pis), 'mask': np.stack(masks),
            'z': np.asarray(zs, dtype=np.float64), 'q': np.stack(qs), 'q_mask': np.stack(qms),
            'ply': np.asarray(plies, dtype=np.int16), 'kind': kinds, 'games': games}


def pack_arrays(rows):
    import torch
    return {'format': 'packed-v2', 'count': int(rows['x'].shape[0]),
            'x': torch.from_numpy(rows['x']), 'pi': torch.from_numpy(rows['pi']),
            'mask': torch.from_numpy(rows['mask']), 'z': torch.from_numpy(rows['z']),
            'q': torch.from_numpy(rows['q']), 'q_mask': torch.from_numpy(rows['q_mask'])}


def write_shard(path, rows, provenance):
    import torch
    from .replay_sampling import BIN_LABELS, ply_bin
    histogram = Counter(BIN_LABELS[ply_bin(p)] for p in rows['ply'].tolist())
    payload = {'format': SHARD_FORMAT, 'replay': pack_arrays(rows),
               'ply': torch.from_numpy(rows['ply']), 'kind': list(rows['kind']),
               'provenance': dict(provenance)}
    path = Path(path)
    torch.save(payload, path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)
    return {'positions': int(rows['x'].shape[0]), 'games': int(rows['games']),
            'kinds': dict(Counter(rows['kind'])), 'ply_histogram': dict(histogram)}


def load_shards(dataset_dir):
    import torch
    paths = sorted(Path(dataset_dir).glob('shard-*.pt'))
    if not paths:
        raise FileNotFoundError(f'no shard-*.pt under {dataset_dir}')
    parts = {k: [] for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask', 'ply')}
    for p in paths:
        shard = torch.load(p, map_location='cpu', weights_only=False)
        if shard.get('format') != SHARD_FORMAT:
            raise ValueError(f'{p}: unexpected shard format {shard.get("format")!r}')
        r = shard['replay']
        for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask'):
            parts[k].append(r[k])
        parts['ply'].append(shard['ply'])
    return {k: torch.cat(v) for k, v in parts.items()}
```

- [ ] **Step 4: Run to verify pass**

Run: `make -C cpp clean && make -C cpp PYTHON=$PY && nice -n 19 $PY -m unittest tests.test_bootstrap -v`
Expected: all `ok`. (If `CppTreeSearch(None, rng, SearchConfig())` rejects a `None` model positionally, use the keyword form `CppTreeSearch(model=None, rng=rng, config=SearchConfig())` — the constructor kwlist is `model, rng, config, opponent, agent_side`.)

- [ ] **Step 5: Commit**

```bash
git add sttt/bootstrap.py tests/test_bootstrap.py
git commit -m "bootstrap: network-free native game generation and packed shards

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

### Task 6.2: `generate-dataset` command — worker pool, manifest, throughput

**Files:**
- Modify: `sttt/bootstrap.py` (add `generate_dataset(args)`), `sttt/ai.py` `main()` (subparser + dispatch)
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- CLI: `python -m sttt.ai generate-dataset --output DIR --games N --workers W --simulations S --leaf-batch B --shard-games G --alphabeta-share F --depths 4 5 6 --seed K`
- Writes `DIR/shard-NNNN.pt` and `DIR/manifest.json` with keys `games, positions, kinds, ply_histogram, simulations, leaf_batch, alphabeta_share, depths, seed, workers, elapsed_seconds, positions_per_second, native_build, created`.
- Refuses to run without the native extension; refuses `--workers > 10`.

- [ ] **Step 1: Write the failing test**

```python
class GenerateDatasetCommandTests(unittest.TestCase):
    def test_writes_shards_and_a_manifest(self):
        import json, subprocess, sys
        if not is_cpp_available():
            self.fail('native extension required')
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, '-m', 'sttt.ai', 'generate-dataset', '--output', tmp, '--games', '6',
                   '--workers', '2', '--simulations', '8', '--leaf-batch', '4', '--shard-games', '3',
                   '--alphabeta-share', '0.5', '--depths', '2', '--seed', '9']
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            shards = sorted(Path(tmp).glob('shard-*.pt'))
            manifest = json.loads(Path(tmp, 'manifest.json').read_text())
        self.assertEqual(len(shards), 2)
        self.assertEqual(manifest['games'], 6)
        self.assertGreater(manifest['positions'], 40)
        self.assertEqual(sum(manifest['ply_histogram'].values()), manifest['positions'])
        self.assertGreater(manifest['positions_per_second'], 0)
        self.assertIn('native_build', manifest)

    def test_refuses_more_than_ten_workers(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([sys.executable, '-m', 'sttt.ai', 'generate-dataset', '--output', tmp,
                                   '--games', '1', '--workers', '11'], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('workers', proc.stderr)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_bootstrap.GenerateDatasetCommandTests -v`
Expected: `invalid choice: 'generate-dataset'`.

- [ ] **Step 3: Implement**

Append to `sttt/bootstrap.py`:

```python
def _worker_init():
    # The box hangs at 32 x 100 %; workers are polite by construction.
    os.nice(10)
    os.environ.setdefault('OMP_NUM_THREADS', '1')


def _generate_task(task):
    seed, games, simulations, leaf_batch, share, depths = task
    return generate_rows(seed, games, simulations, leaf_batch, share, depths)


def generate_dataset(args):
    from .cpp_env import is_cpp_available
    from .evaluation import native_build_info
    if not is_cpp_available():
        raise RuntimeError('generate-dataset requires the native extension; run make -C cpp')
    if args.workers > 10:
        raise ValueError('--workers is capped at 10 on this machine (see CPU-load rule)')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    remaining, shard = args.games, 0
    while remaining > 0:
        n = min(args.shard_games, remaining)
        tasks.append((int(args.seed) * 100003 + shard, n, args.simulations, args.leaf_batch,
                      args.alphabeta_share, tuple(args.depths)))
        remaining -= n
        shard += 1
    started = time.monotonic()
    totals = Counter(); kinds = Counter(); histogram = Counter(); positions = 0
    provenance = {'simulations': args.simulations, 'leaf_batch': args.leaf_batch,
                  'alphabeta_share': args.alphabeta_share, 'depths': list(args.depths),
                  'seed': args.seed, 'native_build': native_build_info()}
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(args.workers, len(tasks)), initializer=_worker_init) as pool:
        for index, rows in enumerate(pool.imap(_generate_task, tasks)):
            summary = write_shard(output / f'shard-{index:04d}.pt', rows, provenance)
            positions += summary['positions']; kinds.update(summary['kinds'])
            histogram.update(summary['ply_histogram']); totals['games'] += summary['games']
            elapsed = time.monotonic() - started
            print(f'shard {index + 1}/{len(tasks)}: {summary["positions"]} positions, '
                  f'{positions} total, {positions / elapsed:.0f} positions/s', flush=True)
    elapsed = time.monotonic() - started
    manifest = {'games': int(totals['games']), 'positions': positions, 'kinds': dict(kinds),
                'ply_histogram': dict(histogram), 'workers': args.workers,
                'elapsed_seconds': round(elapsed, 2),
                'positions_per_second': positions / elapsed if elapsed else None,
                'created': time.strftime('%Y-%m-%dT%H:%M:%S'), **provenance}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    return manifest
```

In `sttt/ai.py` `main()`, after the `tournament` subparser:

```python
    gen = commands.add_parser('generate-dataset', help='Stage-1 bootstrap: network-free native search games')
    gen.add_argument('--output', required=True)
    gen.add_argument('--games', type=positive, default=20000)
    gen.add_argument('--workers', type=positive, default=8)
    gen.add_argument('--simulations', type=positive, default=512)
    gen.add_argument('--leaf-batch', type=positive, default=64)
    gen.add_argument('--shard-games', type=positive, default=500)
    gen.add_argument('--alphabeta-share', type=float, default=0.5)
    gen.add_argument('--depths', nargs='+', type=positive, default=[4, 5, 6])
    gen.add_argument('--seed', type=int, default=1)
```

and in the dispatch dict add `'generate-dataset': generate_dataset` (import `from .bootstrap import generate_dataset`). Guard: `if args.command == 'generate-dataset' and not 0. <= args.alphabeta_share <= 1.: parser.error(...)`. Note `--seed`/`--leaf-batch` are added on `(t, p, e)` at `sttt/ai.py:910-916`; do not add `gen` to that loop (it defines its own).

- [ ] **Step 4: Run to verify pass**

Run: `nice -n 19 $PY -m unittest tests.test_bootstrap -v`
Expected: all `ok`.

- [ ] **Step 5: Measure, do not forecast — a 60-second throughput probe**

Run: `uptime && nice -n 19 env OMP_NUM_THREADS=1 $PY -m sttt.ai generate-dataset --output scratch/bootstrap-probe --games 200 --workers 8 --simulations 512 --leaf-batch 64 --shard-games 25 --seed 1`
Record `positions_per_second` from the manifest. The production run size is `--games` chosen so that `positions ≈ 2,000,000` at the measured rate within the wall-clock the user accepts; write the chosen number into the Part 7 runbook, not here.

- [ ] **Step 6: Commit**

```bash
git add sttt/bootstrap.py sttt/ai.py tests/test_bootstrap.py
git commit -m "generate-dataset: spawn-pool native bootstrap generator with manifest and throughput

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

### Task 6.3: `pretrain` command — supervised fit, held-out metrics, resumable checkpoint

**Files:**
- Modify: `sttt/bootstrap.py` (add `pretrain(args)`), `sttt/ai.py` `main()`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- CLI: `python -m sttt.ai pretrain --dataset DIR --output RUN --arch unet --epochs E --batch B --lr 1e-3 --lr-min 1e-5 --weight-decay 1e-4 --holdout 0.02 --replay-buffer 200000 --device auto --fp16 --seed 0`
- Writes `RUN/pretrain_metrics.jsonl` (one row per epoch: `epoch, lr, train_loss, train_policy, train_value, train_q, val_policy, val_value, val_q, seconds`), `RUN/model-0000.pt` (`{'model','iteration':0,'arch'}`), `RUN/latest.pt` with: `model, optimizer, iteration=0, arch, replay` (packed-v2 random subset of size `--replay-buffer`), `replay_meta=[(0,'bootstrap')]*n`, `lr_schedule=None`, `scaler`, `precision`, `training_config=vars(args)`, `search_config=None`, `backend_info=None`, `population_games=0`, `metrics` (last epoch), `pretrain={'dataset','manifest','epochs','positions'}`.
- Uses `policy_value_loss`, `StratifiedSampler` over `ply`, `augment_batch` with `q, q_mask`, `LRSchedule(kind='cosine')` stepped per optimizer step with `horizon=total_steps`.

- [ ] **Step 1: Write the failing test**

```python
class PretrainCommandTests(unittest.TestCase):
    def test_pretrain_writes_a_checkpoint_that_train_resumes(self):
        import json, subprocess, sys
        if not is_cpp_available():
            self.fail('native extension required')
        with tempfile.TemporaryDirectory() as tmp:
            data, run, cont = Path(tmp, 'data'), Path(tmp, 'boot'), Path(tmp, 'cont')
            subprocess.run([sys.executable, '-m', 'sttt.ai', 'generate-dataset', '--output', str(data),
                            '--games', '6', '--workers', '2', '--simulations', '8', '--leaf-batch', '4',
                            '--shard-games', '3', '--depths', '2', '--seed', '4'],
                           check=True, capture_output=True, text=True, timeout=600)
            subprocess.run([sys.executable, '-m', 'sttt.ai', 'pretrain', '--dataset', str(data),
                            '--output', str(run), '--arch', 'unet', '--epochs', '2', '--batch', '16',
                            '--holdout', '0.2', '--replay-buffer', '50', '--device', 'cpu', '--seed', '0'],
                           check=True, capture_output=True, text=True, timeout=600)
            rows = [json.loads(l) for l in Path(run, 'pretrain_metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in rows], [1, 2])
            for r in rows:
                for k in ('train_loss', 'train_policy', 'train_value', 'train_q', 'val_policy', 'val_value', 'val_q'):
                    self.assertTrue(np.isfinite(r[k]), k)
            self.assertLess(rows[1]['lr'], rows[0]['lr'])
            ckpt = torch.load(Path(run, 'latest.pt'), map_location='cpu', weights_only=False)
            self.assertEqual(ckpt['arch'], 'unet'); self.assertEqual(ckpt['iteration'], 0)
            self.assertEqual(replay_length(ckpt['replay']), 50)
            self.assertEqual(len(ckpt['replay_meta']), 50)
            self.assertIsNone(ckpt['lr_schedule'])
            self.assertTrue(Path(run, 'model-0000.pt').is_file())
            # The online trainer resumes it and attaches a fresh cosine phase.
            subprocess.run([sys.executable, '-m', 'sttt.ai', 'train', '--resume', str(run / 'latest.pt'),
                            '--output', str(cont), '--backend', 'python', '--device', 'cpu', '--iterations', '1',
                            '--games', '1', '--simulations', '4', '--steps', '2', '--batch', '8', '--workers', '1',
                            '--leaf-batch', '2', '--save-every', '0', '--lr-schedule', 'cosine', '--lr', '0.001',
                            '--lr-min', '0.00001', '--lr-iterations', '10', '--replay-sampling', 'stratified'],
                           check=True, capture_output=True, text=True, timeout=600)
            out = [json.loads(l) for l in Path(cont, 'metrics.jsonl').read_text().splitlines()]
        out = [r for r in out if 'q_loss' in r]
        self.assertEqual(out[-1]['iteration'], 1)
        self.assertGreater(out[-1]['positions'], 50)          # warm buffer plus the new game
        self.assertIsNotNone(out[-1]['q_loss'])
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m unittest tests.test_bootstrap.PretrainCommandTests -v`
Expected: `invalid choice: 'pretrain'`.

- [ ] **Step 3: Implement** (append to `sttt/bootstrap.py`)

```python
def pretrain(args):
    import torch
    from .ai import resolve_device, pack_replay, unpack_replay
    from .learning import create_model, policy_value_loss, arch_name
    from .population import augment_batch
    from .replay_sampling import StratifiedSampler, ply_bin
    from .training_schedule import LRSchedule, apply_lr
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    data = load_shards(args.dataset)
    manifest_path = Path(args.dataset) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    n = data['x'].shape[0]
    order = rng.permutation(n)
    holdout = int(round(n * args.holdout))
    val_idx, train_idx = order[:holdout], order[holdout:]
    if len(train_idx) < args.batch:
        raise ValueError(f'{len(train_idx)} training rows is fewer than one batch of {args.batch}')
    bins = np.fromiter((ply_bin(int(p)) for p in data['ply'][train_idx].tolist()), dtype=np.int8,
                       count=len(train_idx))
    sampler = StratifiedSampler(bins)
    steps_per_epoch = max(1, len(train_idx) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    model = create_model(args.arch).to(device)
    use_fp16 = bool(args.fp16) and str(device) == 'cuda'
    model.use_fp16 = use_fp16
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    schedule = LRSchedule(kind='cosine', lr_start=args.lr, lr_min=args.lr_min, horizon=total_steps,
                          completed=0, phase=0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    def to_device(idx):
        idx = torch.as_tensor(idx)
        return [data[k][idx].to(device) for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask')]

    def evaluate_holdout():
        if holdout == 0:
            return {'val_policy': None, 'val_value': None, 'val_q': None}
        model.eval()
        sums, count, q_sum, q_n = np.zeros(2), 0, 0., 0
        with torch.no_grad():
            for start in range(0, holdout, 1024):
                x, pi, mask, z, q, qm = to_device(val_idx[start:start + 1024])
                _, parts = policy_value_loss(model, x, pi, mask, z.float(), use_fp16=use_fp16, q=q, q_mask=qm)
                sums += np.array([float(parts['policy']), float(parts['value'])]) * len(x)
                count += len(x)
                if parts['q'] is not None:
                    q_sum += float(parts['q']) * len(x); q_n += len(x)
        model.train()
        return {'val_policy': sums[0] / count, 'val_value': sums[1] / count,
                'val_q': (q_sum / q_n) if q_n else None}

    model.train()
    last = None
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        losses, p_l, v_l, q_l = [], [], [], []
        for _ in range(steps_per_epoch):
            apply_lr(optimizer, schedule.current_lr)
            picks = train_idx[sampler.sample(args.batch, rng)]
            x, pi, mask, z, q, qm = to_device(picks)
            x, pi, mask, q, qm = augment_batch(x, pi, mask, rng, q, qm)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = policy_value_loss(model, x, pi, mask, z.float(), use_fp16=use_fp16, q=q, q_mask=qm)
            if not torch.isfinite(loss):
                raise RuntimeError(f'epoch {epoch}: nonfinite loss; refusing to continue')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update()
            schedule.advance()
            losses.append(float(loss)); p_l.append(float(parts['policy'])); v_l.append(float(parts['value']))
            if parts['q'] is not None:
                q_l.append(float(parts['q']))
        row = {'epoch': epoch, 'lr': float(optimizer.param_groups[0]['lr']),
               'train_loss': float(np.mean(losses)), 'train_policy': float(np.mean(p_l)),
               'train_value': float(np.mean(v_l)), 'train_q': float(np.mean(q_l)) if q_l else None,
               **evaluate_holdout(), 'seconds': round(time.monotonic() - started, 2)}
        with (output / 'pretrain_metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        last = row

    # Warm replay: a random subset of the dataset in the trainer's row format.
    warm = rng.choice(n, size=min(args.replay_buffer, n), replace=False)
    packed_source = {'format': 'packed-v2', 'count': len(warm),
                     **{k: data[k][torch.as_tensor(warm)] for k in ('x', 'pi', 'mask', 'z', 'q', 'q_mask')}}
    rows = unpack_replay(packed_source)
    arch = arch_name(model)
    torch.save({'model': model.state_dict(), 'iteration': 0, 'arch': arch}, output / 'model-0000.pt')
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': 0,
                  'arch': arch, 'replay': pack_replay(rows), 'replay_meta': [(0, 'bootstrap')] * len(rows),
                  'lr_schedule': None, 'scaler': scaler.state_dict() if use_fp16 else None,
                  'precision': 'fp16' if use_fp16 else 'fp32', 'training_config': vars(args),
                  'search_config': None, 'backend_info': None, 'population_games': 0,
                  'metrics': last, 'pretrain': {'dataset': str(args.dataset), 'manifest': manifest,
                                                 'epochs': args.epochs, 'positions': n, 'holdout': holdout}}
    torch.save(checkpoint, output / 'latest.tmp')
    (output / 'latest.tmp').replace(output / 'latest.pt')
    print(f'wrote {output / "latest.pt"} ({arch}, {len(rows)} warm replay rows)', flush=True)
    return checkpoint
```

`sttt/ai.py` `main()`:

```python
    pre = commands.add_parser('pretrain', help='Stage-2 bootstrap: supervised fit to a generated dataset')
    pre.add_argument('--dataset', required=True)
    pre.add_argument('--output', required=True)
    pre.add_argument('--arch', choices=['resnet', 'mlp', 'unet'], default='unet')
    pre.add_argument('--epochs', type=positive, default=10)
    pre.add_argument('--batch', type=positive, default=1024)
    pre.add_argument('--lr', type=float, default=1e-3)
    pre.add_argument('--lr-min', type=float, default=1e-5)
    pre.add_argument('--weight-decay', type=float, default=1e-4)
    pre.add_argument('--holdout', type=float, default=0.02)
    pre.add_argument('--replay-buffer', type=positive, default=200000)
    pre.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    pre.add_argument('--fp16', action='store_true')
    pre.add_argument('--seed', type=int, default=0)
```

Dispatch: `'pretrain': pretrain` (import from `.bootstrap`). Guard `0 <= holdout < 1`.

`train()` compatibility check (`sttt/ai.py:111-157`): every `saved.get(...)` used there tolerates the fields above; `legacy_meta` is not needed because `replay_meta` is present and its length equals the replay's. Confirm no `KeyError` path — `resolve_population_config` uses `.get` throughout.

- [ ] **Step 4: Run to verify pass**

Run: `nice -n 19 $PY -m unittest tests.test_bootstrap -v`
Expected: all `ok`.

- [ ] **Step 5: Commit**

```bash
git add sttt/bootstrap.py sttt/ai.py tests/test_bootstrap.py
git commit -m "pretrain: supervised bootstrap fit emitting a warm, resumable checkpoint

Stratified over ply bins, symmetry-augmented, cosine per step, held-out
policy/value/Q losses per epoch; latest.pt resumes under train.sh with a
fresh cosine phase and a dataset-sampled replay buffer.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

**Part 6 gate:** full suite; a real small-scale run on CUDA — `generate-dataset --games 2000 --workers 8` (from the Task 6.2 probe rate), `pretrain --arch unet --epochs 3 --fp16 --device cuda`, then `./train.sh runs/bootstrap-smoke/latest.pt runs/bootstrap-smoke-online` with `ITERATIONS=3`. Record dataset size, positions/s, val losses per epoch, and `q_loss`/iteration seconds in the ledger. Then `python -m sttt.ai evaluate --checkpoint runs/bootstrap-smoke/model-0000.pt --opponent alphabeta --opponent-depth 3 --games 20` — the pretrained-only U-Net should not lose every game to depth-3 alpha-beta; record the score, whatever it is.

---

## Part 7 — Launcher, docs, handover reconciliation, ledger

### Task 7.1: `train.sh` — switch on stratified sampling, fix the workers comment, document new flows

**Files:**
- Modify: `train.sh:23` (comment says default 16; code at `train.sh:44` says 8 — make the comment say 8), `train.sh:98` (add `--replay-sampling stratified` next to `--augment-symmetry`), header comment (document `ARCH`-independent resume: the launcher resumes any arch the checkpoint carries).
- Modify: `README.md` (new commands under the training section), `handover/handover.md` (top banner), `TRAINING_READINESS_PLAN.md` §7 ledger.

- [ ] **Step 1: Edit `train.sh`**

```bash
#   WORKERS     self-play worker processes (default 8; the CPU-load rule caps this box at 10)
...
  --augment-symmetry \
  --replay-sampling stratified \
```

Add to the header comment:

```bash
# Bootstrapped start: generate a dataset and pretrain first, then resume here:
#   nice -n 19 $PY -m sttt.ai generate-dataset --output data/bootstrap --games <N> --workers 8
#   nice -n 19 $PY -m sttt.ai pretrain --dataset data/bootstrap --output runs/bootstrap --arch unet --fp16
#   ./train.sh runs/bootstrap/latest.pt runs/run_v4 configs/population/baseline.json
```

- [ ] **Step 2: Verify the launcher still parses and the trainer accepts the flag set**

Run: `bash -n train.sh && $PY -m sttt.ai train --help | grep -E "replay-sampling|arch"`
Expected: no syntax error; both flags listed with `unet` among arch choices.

- [ ] **Step 3: Handover banner** — prepend to `handover/handover.md`:

```markdown
> **Status, 2026-09-14 (post-blueprint reconciliation).** Paths below are from
> the laptop (`/home/entropy/...`, Python 3.14.4); on this machine the
> interpreter is `/home/mt/miniconda3/envs/sttt/bin/python` and the repo is
> `/home/mt/Zami/Super-Tic-Tac-Toe`. Corrections: the input width is 289
> floats, not 172 (`sttt/learning.py: INPUTS`); 8-fold symmetry augmentation
> already existed (`--augment-symmetry`, on in `train.sh`) and is proved in
> `tests/test_population.py::SymmetryGroupTests`; `WORKERS` defaults to 8.
> The implementation of §5–§6 Steps 1–3 is tracked in `handover/plan.md`.
```

- [ ] **Step 4: README** — add under the training commands:

```markdown
### Bootstrapped training (two-stage)

    # Stage 1: network-free native search games (C++ heuristic MCTS vs itself and alpha-beta d4–d6)
    nice -n 19 python -m sttt.ai generate-dataset --output data/bootstrap --games 20000 --workers 8
    # Stage 2: supervised fit; writes runs/bootstrap/latest.pt with a warm replay buffer
    nice -n 19 python -m sttt.ai pretrain --dataset data/bootstrap --output runs/bootstrap --arch unet --fp16
    # Online: the canonical launcher resumes the bootstrapped checkpoint
    ./train.sh runs/bootstrap/latest.pt runs/run_v4 configs/population/baseline.json

Architectures: `--arch resnet` (1.8M MLP-ResNet, default), `--arch unet` (hierarchical conv
U-Net with a dense action-value head). Replay sampling: `--replay-sampling stratified`
balances each batch across nine-ply game phases (`train.sh` enables it).
```

- [ ] **Step 5: Ledger** — append to `TRAINING_READINESS_PLAN.md` §7 a section `### Blueprint evidence (2026-09-14)` with one row per part: commit hash, test command and result, and the measured numbers from each gate (Part 5 U-Net iteration seconds vs resnet; Part 6 positions/s, dataset size, val losses, evaluate score). Numbers come from the runs; do not write forecasts.

- [ ] **Step 6: Full suite, commit, merge**

Run: `nice -n 19 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m unittest discover -s tests 2>&1 | tail -5`
Expected: only the one pre-existing known failure (branch isolation).

```bash
git add train.sh README.md handover/handover.md TRAINING_READINESS_PLAN.md
git commit -m "Docs and launcher for the blueprint: stratified sampling on, bootstrap runbook, handover banner

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Deferred (explicitly not in this plan)

- **Q-initialised MCTS leaves** (spec §5.2, second half): initialising unvisited children with the network's predicted Q instead of 0 changes `_select` in both `sttt/search.py` and `cpp/src/mcts.cpp`, and the differential Python/C++ tests. It is the step that turns the Q-head into search strength; do it as its own plan once Part 5's Q predictions have a measurable held-out error from Part 6.
- **Distillation from `uttt.ai`** (spec §6 Step 4): needs a teacher-policy data path (`TRAINING_READINESS_PLAN.md` §3 notes opponent turns are excluded from learner trajectories today).
- **Readiness runner stage** for bootstrap (`scripts/check_training_ready.py`): add a `bootstrap` stage running Task 6.3's smoke once the commands have settled.

## Self-review notes

- Spec coverage: §5.1 → Part 5; §5.2 (dense Q loss) → Part 4; §5.3 stage 1 → Task 6.1–6.2, stage 2 → Task 6.3; depth-file balancing → Part 2; dihedral augmentation → already present, proved in Part 1; §6 Step 4 and Q-leaf initialisation → Deferred.
- Names used across tasks: `policy_value_loss` (3.1, 4.3, 6.3), `forward_all` (4.3, 5.2), `root_action_values` (4.1, 4.2), `augment_batch(x, pi, mask, rng, *cell_tensors)` (4.2, 6.3), `StratifiedSampler.sample/histogram` and `ply_bin/row_ply` (2.1, 2.2, 6.1, 6.3), `pack_replay/unpack_replay/replay_length` with `packed-v2` (4.2, 6.1, 6.3), `arch_name/create_model/load_model` (5.2, 6.3), `CELL_TO_GRID/planes_from_flat` (5.1, 5.2).
- Behaviour preservation: defaults `uniform`/`resnet`; legacy checkpoints widen to zero-Q rows; `augment_batch` with three arguments returns three values.
