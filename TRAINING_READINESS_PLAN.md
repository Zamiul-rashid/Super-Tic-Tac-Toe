# Training readiness and improvement plan

Date: 2026-09-13. Planning baseline: `main` at `bea16e650c73c4f50787278a817a130716dc899f`.

Status: **implementation plan, not a training-readiness certificate**. Every unchecked gate below remains pending until its evidence is recorded. This document is written as an execution specification for a coding agent that needs explicit instructions. Do not replace the checks with a verbal claim that the implementation looks correct.

## 1. Objective and boundaries

Make the existing learner reliable for long training runs with C++ board/search work and Python/PyTorch GPU inference and optimization. Preserve old checkpoints, Python reference search, population opponents, and adaptive human play. Establish trustworthy speed and strength measurements. Add resumable learning-rate scheduling and configurable curriculum experiments.

The immediate deliverable is this plan on `main`. Implement the milestones later in order. Do not start a 5,000-iteration run as part of writing or reviewing this plan. An implementation milestone is complete only when its code, tests, evidence, and focused commit are complete.

Do not promise a particular Elo, an iteration-to-Elo curve, a fixed training speedup, or a 22-hour completion time. Do not introduce a new network architecture, LibTorch, multiple CUDA owners, shared-memory IPC, teacher distillation, or a new replay algorithm in the initial readiness work. Those are separate experiments after the baseline is reliable.

### Definitions used throughout

- **Iteration:** collect the configured number of games, append learner targets, perform `--steps` optimization attempts, and save the completed iteration.
- **Simulation:** one completed MCTS backup; `--simulations` is the number of new simulations requested per move. Exact proofs may finish sooner. Reused visits are separate.
- **Game versus pair:** `games=20` means 20 games / 10 mirrored opening pairs. It does not mean 40 games.
- **Training search budget:** learner search used to produce replay policies. **Evaluation search budget:** search at test time; changing it does not change network weights.
- **Score rate:** `(wins + 0.5 * draws) / games`. It is different from win rate.
- **Local Elo:** relative rating estimated against a particular opponent pool and settings. It is not a universal human or engine rating.
- **CPU-certified:** native behavior, CPU smoke/resume, and memory checks passed. **GPU-certified:** actual CUDA inference, AMP/resume, representative soak, and GPU performance gates also passed.

## 2. Refresh the baseline before changing anything

The September 12 audit was against `cpp` at `9ce0a29`. It is historical: commits through September 13 have already integrated C++ training and fixed several audit findings. Do not reimplement completed work or carry historical findings forward without checking them.

| Component | September 13 baseline | Required treatment |
| --- | --- | --- |
| Learner training backend | `ai.train` accepts `--backend`; `selfplay.play_game` can create native trees and states | Harden selection/worker verification; integration is already present |
| Neural/replay encoding | Native batch encoding is used, with copies and Python mask construction | Make backend choice explicit and test ownership/parity |
| Checkpoint opponents | Native branch exists in `CheckpointBot.choose` | Centralize backend behavior; strict requests must not silently fall back |
| Native RNG | Constructor now draws a seed from Python RNG | Verify independent streams and error propagation; do not call it entirely ignored |
| Native cleanup and stats | Batch validation and decref helper were added | Reproduce old failures and verify remaining malformed-policy/OOM paths |
| Adaptive human play | Still Python search | Preserve it deliberately; reject unsupported native adaptation |
| Learning rate | AdamW initialized at `1e-3`; latest run_v2 optimizer stores `0.001`; no scheduler state | Add an explicit, resumable schedule |
| Mixed precision | FP16 training/inference exists; GradScaler is currently created inside each iteration | Persist scaler across iterations and checkpoints |
| Curriculum | 30% self, 25% uttt.ai, 25% AlphaBeta, 10% history, 5% best, 5% other combined | Treat as the current baseline, not the older population README mix |
| Latest checkpoint observed | run_v2 iteration 2007; big_run metrics end at 6155 | Re-read metadata; rolling files may advance |
| `best.pt` observed | run_v2 iteration 1600; big_run iteration 5750; both weights-only | Do not confuse these with latest or with a full optimizer/replay resume |
| Current GPU access | CUDA unavailable in the planning session | GPU gates remain pending; do not certify via CPU fallback |

Read these files first: `sttt/ai.py`, `selfplay.py`, `learning.py`, `cpp_env.py`, `search.py`, `population.py`, `bots.py`, `reports.py`, `tournament.py`, `cpp/src/mcts.cpp`, `cpp/src/python_module.cpp`, `cpp/include/sttt_mcts.hpp`, build files, relevant tests, and both handovers. Use symbol searches because line numbers will move.

Record `git status --short`, branch, commit, worktrees, Python executable/version, Torch/CUDA versions, extension path/build identity, GPU availability, and checkpoint metadata in a new audit manifest. Preserve user changes. Never reset, stash, delete checkpoints, rebuild a loaded extension in place, or stop a training process merely to prepare the workspace.

## 3. Review of the proposed strength improvements

| Claim | Evidence and interpretation | Action |
| --- | --- | --- |
| Flat learning rate exists | Confirmed in `ai.train` and run_v2 latest optimizer | Add schedule support and compare it with a constant-rate control |
| Flat `1e-3` caused model-1600 to outperform model-2000 | Plausible hypothesis, not an isolated causal result; opponent distribution, replay, seeds, exploration and test noise also vary | Run controlled continuations; record policy/value loss, gradients, replay mix and held-out strength |
| 512 to 2,000 simulations improved score 22.5% to 32.5% | `grand_championship/oracle_challenge.json` records 3W/3D/14L versus 5W/3D/12L, 20 games each | Repeat with identical frozen checkpoint and opening pairs |
| That search test proves a +100–150 Elo gain at 1,024 simulations | Not established: it never tested 1,024; `run_thorough_tournament.py` uses `seed=100+sims`, changing openings across budgets | Sweep 512/1024/2000 with shared openings and measured latency |
| big_run is 1671 Elo | The eight-player report estimates 1671.11 with reported CI half-width 81.41 for the named iteration-6155 entrant | Keep the pool, hash and budgets attached to the number; one point does not prove a plateau |
| run_v2 is reliably better | An earlier 1600-iteration 1650-Elo/8-to-4 result was not located in the inspected artifacts; the saved 2000-iteration report has 20W/5D/25L against big_run-best and 1585.9 versus 1626.1 in that pool | Evaluate immutable 1600, 2000, latest and old references together before selecting a continuation |
| 25% uttt.ai and 25% AlphaBeta is the current mix | Confirmed in `POPULATION_WEIGHTS`; uttt.ai budgets are 64/128/256 | Expose the mix/budgets as configuration; compare 25% with 35% before increasing again |
| More strong-opponent games directly teach its policy | Not what this code does: opponent turns are excluded from learner trajectories; targets are learner MCTS policies and final outcomes | Keep this distinction explicit; teacher-policy training would be a new objective/data path |
| 5,000 iterations take about 22 hours | Last 100 observed run_v2 iterations average 29.2735 s, approximately 40.66 h for 5,000, excluding later periodic evaluation overhead | Estimate from a representative pilot for each configuration; harder opponents can increase runtime |
| LR decay + stronger curriculum guarantees 1820–1880 or 3x efficiency | No controlled evidence establishes either forecast | Report actual score improvement per hour and uncertainty; do not put forecast Elo into acceptance criteria |

Reports inspected: `runs/tournaments/grand_championship/tournament.json`, its `oracle_challenge.json`, and `runs/tournaments/run_v2_iter2000_vs_bigrun/comparison_report.txt` / `ratings.csv`. These are local artifacts and may not exist in a fresh clone. The aggregate grand-championship JSON lacks the full checkpoint hashes and per-game evidence needed for a controlled replay. Treat it as historical observation.

The comparison script also has concrete consistency problems: it passes unsupported `pairs=` to `run_matchup`, labels 50 total games as 100, and writes CSV rank from unsorted rating-dictionary order. Repair the evaluation harness before using it to select an LR or curriculum. A depth-limited, node-capped AlphaBeta opponent is a reference opponent, not a solved-game oracle.

## 4. Implementation rules for every milestone

1. Re-read the current milestone and its dependencies. Search for the named symbols. If a requirement already exists, prove it with the acceptance test and avoid duplicating it.
2. Work on an isolated feature branch/worktree based on current `main`. Put build products and test runs in that worktree or a temporary directory; never in active run directories.
3. Make one coherent change. Add a regression test for each reproduced defect before fixing it. Prefer behavior assertions over assertions about private implementation layout.
4. Run the milestone tests. Record commands, interpreter/build identity, result, skips, and artifacts. A skipped native/GPU test does not satisfy that gate.
5. Commit a completed, tested change immediately. Keep fixes independently revertible. Split a milestone if its patch becomes difficult to review; do not bundle an 800–1000-line feature commit.
6. Merge the tested feature into `main`, resolving conflicts against current behavior and rerunning affected checks. Never weaken tests to make a merge pass.
7. Update the milestone ledger with commit, evidence and remaining limitations. On a failed gate, stop only dependent work; do not silently reduce search budgets, change opponents, disable population, or switch CPU/GPU to obtain a pass.

Planning does not authorize sending messages, pushing to remotes, or starting a multi-day training run. Ordinary local implementation, tests, focused commits and main merges follow the user's existing repository instructions. A new launch must identify its exact source checkpoint, configuration, output directory and run owner.

## 5. Milestones and dependency order

All milestones below are initially **not completed by this plan**. Existing code can satisfy individual items after verification.

| ID | Focused commit target | Dependencies | Principal files |
| --- | --- | --- | --- |
| M0 | Freeze provenance and establish fresh-build baseline | None | build files, new `scripts/check_training_ready.py`, evidence manifest |
| M1 | Complete native boundary/error/state robustness | M0 | C++ bindings/core, `cpp_env.py`, native tests |
| M2 | Centralize explicit backend contracts and worker verification | M1 | new `sttt/backends.py`, search/selfplay/bots/ai |
| M3 | Validate native batches and owned tensor/mask encoding | M2 | learning/cpp_env/bindings/selfplay/ai |
| M4 | Make training state, AMP and output ownership resumable | M2–M3 | ai/learning, new checkpoint helpers, training tests |
| M5 | Add resumable constant/cosine learning-rate policy | M4 | new `sttt/training_schedule.py`, ai, schedule tests |
| M6 | Make population configuration and coverage measurable | M4 | population/ai, config files, tests |
| M7 | Repair controlled evaluation and provenance exports | M2 | comparison scripts/tournament/reports/bots, tests |
| M8 | Correct benchmarks and profile real pipeline stages | M3, M7 | benchmark scripts/native benchmark, reports |
| M9 | Run CPU/GPU smoke, resume and memory gates | M1–M8 | readiness runner and artifacts |
| M10 | Run bounded LR/curriculum/search ablations | M5–M9 | experiment manifests and held-out reports |
| M11 | Reconcile docs, release configuration and launch runbook | M9–M10 | README, handovers, population docs, launch script |

M7 can proceed independently of training-state work once M2 is complete. Do not wait for research experiments to fix a known crash or misleading report.

### M0 — Provenance, isolation and reproducible builds

Required work:

- Make Make/CMake/setuptools use the selected interpreter's include and ABI information. Remove the unused hardcoded interpreter/include mismatch. Keep warning-as-error builds.
- Prefer the correctly installed extension over an arbitrary stale `cpp/sttt_cpp.so`. Expose extension version, source revision/build ID, compiler flags and loaded path. A tracked binary is not proof it matches source.
- Build native artifacts from source in the implementation worktree. Test wheel/install import from outside the repository as well as editable development import. Keep the source-only Python backend usable without a native build.
- Add `scripts/check_training_ready.py` with explicit `--backend python|cpp` and `--device cpu|cuda`. Its initial stage prints a machine-readable capability manifest and exits nonzero when a required capability is missing. It must not create training workers or modify existing runs just to inspect capabilities.
- Freeze evaluation inputs by copying selected checkpoint files to a new evaluation-input directory and hashing the copies. Verify a rolling source did not change during the copy; retry if it did. Do not delete or rename the originals.

Acceptance: freshly built extension imports in a fresh subprocess; its path/ABI/build ID match the intended build; `--backend cpp` fails with actionable text when deliberately unavailable; `--device cuda` fails when CUDA is unavailable. Native tests are actually executed, not skipped. Save the current discovered test count rather than assuming the September 12 count of 258.

### M1 — Finish native correctness and lifetime handling

Required work, each backed by a regression test:

- Validate every public action/budget/config input before integer narrowing or allocation. Reject out-of-range and in-range illegal moves, terminal advances, nonpositive budgets, nonfinite PUCT parameters and zero revisit interval appropriately. No C++ exception may escape the Python boundary. Translate invalid input to `ValueError`, allocation failure to `MemoryError`, unexpected native errors to `RuntimeError`.
- Ensure GIL-released paths restore the GIL before setting Python errors. Catch exceptions inside benchmark worker threads and propagate after joining. `benchmark_mcts(1, 1, 2)` must either distribute one simulation safely or reject the configuration; it must not abort.
- Apply identical evaluator validation at root and leaves: exactly 81 numeric finite nonnegative policy entries; a numeric finite value in [-1,1] within documented tolerance; correct tuple/batch shape. Check every `PyFloat_AsDouble` error. No Python allocation result may be dereferenced without checking it.
- Validate the full inference batch before expanding/backing up any member. Release each pending path once on every failure. On failure, all pending flags and in-flight counters must be clear/nonnegative and the tree must be reusable or explicitly reset with a documented error; never silently keep corrupt statistics.
- Recheck the fixed statistics decref helper; propagate dictionary insertion failures. Repeat 20,000 discarded `.stats` reads with garbage collection and confirm no proportional retained-object growth.
- Make mutable `FastState` and `CppState` unhashable. Add `state_key(state)` returning an immutable canonical tuple for callers needing dictionary/set keys. Preserve cross-backend equality. Add `clone()` or correct every documented example; prefer implementing an independent-copy `clone()` to preserve the intended API.
- Document native node-view lifetime. Any debug view surviving reset/arena compaction must fail clearly rather than silently refer to another node. Add a tree generation ID if exposing views requires it.
- Measure arena growth across complete games. Implement sibling reclamation/compaction at safe move boundaries if required by the memory gate. Maintain child contiguity and repair indices; preserve retained statistics. No compaction while inference paths are outstanding. At reset/game end, reclaim unreachable nodes and release excessive capacity according to a documented limit.

Acceptance fixtures: empty board; forced board; move into closed board; local/macro wins and draws; 2,000 seeded random games against Python rules; first-valid/second-NaN batch; malformed root policy; negative/infinite policy entry; evaluator exception after root expansion; allocation/invalid-call subprocess tests. Same-backend clone/equality/pickle round trips must preserve all state fields. Run sanitizers on a bounded native harness where available; record an unavailable sanitizer separately from functional results.

### M2 — One explicit search backend contract

Create `sttt/backends.py`; keep `sttt.search.TreeSearch` as the Python reference. Do not globally alias it to native search.

Expose a factory such as `create_search(model, *, backend, seed, config, opponent=None, agent_side=None)` and an adapter with:

```python
run(state, simulations, batch_size=1, noise=False)  # float32[81]
advance(action)                                  # legal move only
reset()                                         # forget tree, keep configured RNG stream
root_state                                      # read-only state or None
stats                                           # snapshot dictionary, caller may cache once
backend_info                                    # requested, actual, version, fallback reason
```

Caller rules: never write `.root` or `.rng`; reset when model weights or adaptive profile change; call `advance` exactly once for every played move, including openings/opponent/noise moves as appropriate. `run` must reconcile a mismatched state safely. A bot's own `choose`/advance convention must remain consistent with `tournament._play_single_game` to avoid double advances.

Backend selection rules:

- `python`: Python search AND reference encoding for controlled comparisons.
- `cpp`: require the actual native search API and the agreed encoding path in the owner and every worker. Missing capabilities fail before a game; never fall back silently.
- `auto`: choose a supported backend once per role/run; log the actual choice and reason. Adaptive human play explicitly chooses Python. A direct native call with `opponent` must raise `NotImplementedError` until native adaptation is implemented and tested.
- Apply `--backend` consistently to train, play, evaluate, tournament and checkpoint entrants. Record opponent search/encoding backend separately from learner backend. Preserve independent opponent CPU inference unless a later project explicitly changes it.

RNG rule: derive independent search, move-sampling and opponent streams from each game seed using fixed named numeric stream IDs. Initialize a native search seed explicitly; never default all games to 42 and never swallow an invalid RNG error. Do not reseed every move. Promise reproducibility within the same backend/build/configuration; exact noise equality across NumPy and C++ is not required.

Worker handshake must return actual backend and native version before games are dispatched. Fetch `tree.stats` once after each search, then aggregate that snapshot. Preserve current trajectory ownership and both-side/self versus learner-only/opponent semantics.

Acceptance: CLI/backend matrix for all commands; strict missing-native test in spawned workers; same seed reproduces policies/noise, different seeds produce different sampled noise across a fixed multi-seed fixture; no root/rng setter errors; adaptive prediction calls still occur in Python; no native claim when Python fallback was used; no stale tree after model update.

### M3 — Correct encoding and batch boundaries

Keep the existing bounded Pipe IPC for the first release. Do not redesign transport before profiling it.

- Add a shared `encode_states(states, backend)` that returns owned, C-contiguous `float32[N,289]` features and `bool[N,81]` legal masks. Native encoding must preserve the existing feature layout exactly; implement batch mask generation in native code if it is part of the selected native path.
- The returned storage must remain valid after source states/buffers are released and must be safe for `torch.from_numpy`. A NumPy view of immutable Python bytes is not writable zero-copy tensor ownership; keep an explicit copy until a correct owner-backed buffer exists.
- Eliminate hidden native encoding from a requested Python reference run. Do not catch arbitrary native errors and substitute Python encoding.
- Keep one model owner and one CUDA context for learner inference. Workers send plain picklable states and receive outputs in exactly the request order. Never send CUDA tensors/models through the pipes.
- Preserve `inference_batch` as an upper bound; split oversized requests, handle underfilled batches, and bound waits. `leaf_batch` remains a separate search parameter.
- Replay targets must snapshot states/policies: later mutable state changes cannot alter old samples. Check legal masks, normalized finite policies, value sign (`outcome * state.turn`), and symmetry transforms together.

Acceptance: exact feature/mask parity on random, terminal, forced/free-choice and color-swapped fixtures; empty and mixed state-type batches; lifetime tests after garbage collection; two native spawned workers across two model updates; owner inference order and size bounds; all complete-game replay targets valid. Preserve both MLP and ResNet evaluation APIs.

### M4 — Resume, AMP, checkpoint retention and run ownership

- Move GradScaler creation outside the iteration loop; create once per training process. Save and restore its state with optimizer state. Preserve the correct sequence: zero gradients, autocast forward, scaled backward, unscale, clip once, optimizer/scaler step once, scaler update. Do not reintroduce the historical double-optimizer-step bug.
- Log policy loss and value loss separately, LR, gradient norm before clipping, scaler scale and skipped optimizer updates. Reject nonfinite losses/targets before committing an iteration. If all optimizer updates are skipped, fail the readiness gate rather than claiming training occurred.
- Define checkpoint schema fields for model/arch, optimizer, scaler, schedule, completed iteration, replay, population cursor/config, effective training config, NumPy/Torch RNG states, requested/actual backends, precision and build identity. Use values compatible with the existing safe `weights_only=True` loading path. Never pickle a scheduler instance or callable.
- Support `--resume-mode continue|weights` (default continue). Continue requires optimizer/replay for a full resume; missing optional legacy scaler/scheduler fields get documented defaults. A weights-only best/history checkpoint requires explicit weights mode, fresh optimizer/replay, and a recorded parent checkpoint identity. Do not pretend best.pt contains the optimizer from latest.pt.
- Omitted algorithm options on resume inherit saved effective configuration; explicit CLI options override and are recorded. Distinguish omitted Boolean flags from explicit false. Runtime output/device/worker choices are resolved explicitly. Old checkpoints without fields retain compatible defaults; print a concise effective-config diff.
- Continue mode preserves optimizer moments, replay content, population cursor and schedule phase. Restore Torch RNG state when present. Do not promise byte-identical full training trajectories across GPU kernels/process scheduling; test deterministic schedule/checkpoint semantics separately.
- Acquire a per-output-directory exclusive OS lock for the whole run. Reject a second writer before spawning workers. Keep save-to-temp-and-rename behavior; incomplete iterations never replace a valid completed checkpoint.
- Add a pinned checkpoint manifest for experiment starts, fixed evaluation references, manual best and history anchors. Retention may remove only eligible unpinned snapshots inside this run's output directory; it must not delete another run's checkpoints or active experiment inputs. Test the 500-iteration retention policy against those pins.

Acceptance: full-checkpoint CPU resume and weights-only initialization; arch-tagged and archless legacy MLP/ResNet; two writers rejected; interrupted partial iteration leaves last checkpoint loadable; exactly one optimizer update per minibatch; AMP scaler survives iteration and resume boundaries on CUDA; pinned checkpoints survive cleanup. Keep old production files byte-for-byte unchanged by tests.

### M5 — Learning-rate scheduling with explicit units

Add `sttt/training_schedule.py` and CLI options `--lr`, `--lr-schedule constant|cosine`, `--lr-min`, `--lr-iterations`, `--reset-lr-schedule`.

Concrete behavior:

1. Default for a fresh legacy-compatible run is constant `1e-3`; a legacy resume with no schedule continues its restored optimizer LR unless explicitly overridden.
2. For cosine, the schedule unit is **completed training iterations**, not minibatches or games. A 5,000-iteration horizon with 100 optimizer updates per iteration spans approximately 500,000 update attempts. Do not set `T_max=5000` and call it 100 times per iteration.
3. Let `k` be completed iterations in this schedule phase and `H` its configured horizon. LR for the next iteration is `lr_min + 0.5*(lr_start-lr_min)*(1+cos(pi*min(k,H)/H))`. Use start LR at `k=0`, midpoint at `H/2`, floor at/after `H`. Clamp after the horizon; never restart or climb automatically.
4. Advance once after a successful iteration's optimization block, before saving its completed checkpoint. Store the phase start, horizon, completed count, start/min LR and next LR. Log both LR used in the completed iteration and next LR.
5. Resume an existing schedule without restarting it or recomputing its horizon from additional `--iterations`. Changing `--steps` does not change the iteration-based schedule but must be logged. Explicit schedule changes require `--reset-lr-schedule`, a recorded phase change, and preferably a new output directory.
6. Apply explicit LR overrides after optimizer-state restoration without discarding optimizer moments. For a scheduler-backed implementation, initialize/load scheduler and optimizer in a tested order that preserves the saved next LR. Validate finite positive start LR, `0 <= lr_min <= lr_start`, and positive integer horizon.

Tests: constant rate unchanged; cosine start/mid/end/floor; two uninterrupted iterations versus one + save/load + one have identical LR sequence; no scheduler reset on each outer iteration; legacy missing-state behavior; invalid CLI values; an explicit phase reset is visible; an interrupted partial iteration does not advance the persisted schedule.

Experiment policy: the proposed `1e-3 -> 1e-5` over 5,000 iterations is a candidate, not a guaranteed fix. A short pilot with that long horizon barely changes LR early on, so it cannot establish its full benefit. Include a lower constant-rate control, such as `3e-4`, or a separately labeled shorter cosine pilot. Never compare unlike schedule fractions as if they tested the same hypothesis.

### M6 — Configurable curriculum with observable coverage

- Add a versioned JSON population configuration, loaded by `--population-config`, containing integer quotas summing to 100 plus per-family budget/noise ranges. Keep `POPULATION_WEIGHTS` only as the backward-compatible baseline default.
- Save the resolved configuration and hash in checkpoints/metrics. On resume, inherit it unless explicitly changed; record a phase change and keep a clearly defined 100-game quota cursor. Reject unknown families, negative quotas, invalid budgets and sums other than 100.
- Ship two initial presets. Baseline: self 30, uttt.ai 25, AlphaBeta 25, history 10, best 5, tactical 2, OpenSpiel 1, threat 1, style 1. Candidate: self 30, uttt.ai 35, AlphaBeta 15, with all other quotas unchanged. Keep uttt.ai budgets at 64/128/256 for the mix-only comparison.
- Test stronger uttt.ai budgets separately (for example 128/256/512) after the mix experiment. Do not simultaneously change LR, opponent fraction, opponent budget, learner budget and exploration when attributing gains.
- Record games, learner positions, wall time, score rate and requested/actual opponent budgets by family; replay sampled-position fractions and age distribution should be measurable without changing the target representation. Game quota does not equal replay quota or compute quota.
- Keep strong opponents as challenging references while retaining self/history coverage. Persistent near-zero score is a reason to inspect target quality and budget mix, not automatically to allocate still more games to that opponent.
- Preserve strict external-engine failures and seeded openings. Missing history fallback to self must remain explicit in actual counts. Do not label learner policies as teacher policies; distillation is out of scope.

Acceptance: exact counts over a 100-game cycle sliced across arbitrary iteration sizes and resumes; deterministic configuration resolution; all families exercised with fixed explicit match specs; mixed real-engine workers finish/close; zero silent tactical substitution; budget/mix changes reproducible from checkpoint metadata.

### M7 — Evaluation that can support a decision

- Fix `scripts/run_v2_vs_bigrun_comparison.py`: remove unsupported `pairs`, use total games consistently, sort CSV rank by the same rating ordering as the scoreboard, and close bots on every exception. Preserve per-game results, not only printed summaries.
- Fix `scripts/run_thorough_tournament.py`: use the same held-out opening seeds for every search budget and immutable checkpoint copies. Names derive from loaded iteration plus short hash, not hardcoded labels. Do not assume `big_run/best.pt` is iteration 6155; it was observed as 5750.
- Use one comparison harness for 512/1024/2000 simulations with explicit leaf batch, backend, precision, opponent depth/node cap and device. Add elapsed-time budgets only through a documented search deadline API with completed-simulation reporting; do not claim equal-time tests by merely changing simulation counts.
- Export checkpoint SHA-256, iteration, architecture, native build/version, CLI/effective config, opponent model/source checksum, seeds/opening actions, result, search timings, actual simulations and errors. Snapshot mutable sources once at evaluation start. An engine's depth cap alone does not prove that depth was reached.
- For cross-run ratings, retain a fixed immutable anchor/opponent pool and budgets. Prefer direct paired score differences. Keep local Elo labels and reported uncertainty; do not call uttt.ai the strongest engine in existence based on one local tournament.
- Bootstrap at the opening-pair level for head-to-head and budget-difference intervals. Mirrored games share an opening and are not independent. Preserve raw pairs so the interval can be recomputed. Do not conclude significance from overlapping/non-overlapping individual Elo intervals alone.

Acceptance: 20 requested games produces exactly 20 rows / 10 pairs; both seats use identical opening actions; budgets share the identical opening corpus; candidate hashes constant throughout; one result schema feeds JSON/CSV/scoreboard; mocked CLI smoke catches unsupported arguments without an expensive tournament; a real 20-game match validates the final wiring. Invalid runs are recorded as failed, not scored as normal completed evidence.

### M8 — Correct performance benchmarks

- Fix native primitive loops so outputs affect a checked result and the compiler cannot remove the measured work. Vary pre-generated positions; do not time input generation. Warm up and repeat timed blocks; report median and spread.
- Label dummy-evaluator traversal, heuristic native search, actual neural inference, complete self-play, mixed league and full training iteration separately. Parallel independent-tree throughput is not single-move parallel latency.
- Add a manifest-driven end-to-end runner with frozen checkpoint/config/openings, explicit backend, same device/batches/workers, and alternating Python/C++ run order. Fresh worktrees must load the intended extension, not a stale binary.
- Measure self-play games/s, learner positions/s, actual sims/s, inference batch occupancy, encoding/mask time, owner inference time, IPC/wait, optimization, replay preparation, checkpoint writes and periodic evaluation overhead. Do not sum overlapping worker times as elapsed runtime. CUDA timings must synchronize only at measurement boundaries or use CUDA events correctly.
- For a representative pilot, record at least 20 completed measured iterations after warm-up; separate pure self-play from the actual league. Preserve per-iteration samples and hardware load. Compare throughput only when match mix/precision/search settings match.
- Compute ETA from the pilot mean and observed spread plus evaluation/checkpoint overhead. Recompute when budgets, mix, hardware or worker count change. There is no release requirement for an arbitrary 15x speedup; unexplained regression greater than 5% on repeated matched measurements blocks a speedup claim and triggers profiling.

Acceptance: no impossible zero-work primitive timings; neural benchmark actually loads/forwards a model; requested and actual backend recorded; all timing denominators use completed work; controlled comparison produces reproducible artifacts without modifying source checkpoints.

### M9 — Readiness gates and bounded launch tests

Implement `scripts/check_training_ready.py` as a staged runner. Each stage writes a result with pass/fail/skipped/pending, command, duration and artifact paths. Use fresh output directories under `runs/readiness/<unique-id>/`. Never use big_run or run_v2 as a test output.

1. **Build gate:** M0 fresh-build/import checks and current full suite. Explicitly require native availability first so native unittest skips cannot masquerade as success.
2. **Native integration gate:** two spawned workers, four complete games, 32 simulations, leaf batch 4, owner batch 8; run twice with a model update in between. Verify worker identity reuse, actual native types/backend, legal targets and cleanup. Also exercise Python and both MLP/ResNet models.
3. **CPU training gate:** two iterations, four games, 32 simulations, four optimizer steps, small replay, no periodic evaluation. Resume for one additional iteration. Assert iteration advancement, nonzero model change, finite losses, optimizer/replay/schedule restoration and no leftover workers.
4. **Mixed-opponent gate:** explicit small-budget match specs cover self, history/best, AlphaBeta/tactical/threat/style, official OpenSpiel and uttt.ai, both seats and injected openings. Then verify a full 100-game configured quota cycle at low test budgets; distinguish those test budgets from the production preset.
5. **Failure gate:** malformed inference, worker exit, engine timeout, Ctrl+C, duplicate writer and invalid backend. Each must fail promptly, close owned children and preserve the last completed checkpoint. Fix the worker timeout contract if a measured legitimate stronger-opponent move can exceed it; do not mask deadlocks with unlimited waits.
6. **GPU correctness gate:** explicit CUDA; FP32 and FP16 forward/train, finite masked policy loss, at least one real optimizer update, scaler persistence, CUDA checkpoint/resume, one learner CUDA owner. A CPU run cannot satisfy this stage.
7. **Memory gate:** after fixed workload/replay warm-up, run 100 complete games and at least 20 training iterations. Record owner/workers/external processes separately, aggregate RSS (noting shared-page double counting), native arena live/capacity, and allocated/reserved CUDA peaks. Fail known reference leaks, orphan processes, unbounded post-warm-up growth, OOM, or a configured resource cap breach. Investigate >10% growth from the warm baseline instead of treating replay fill as a leak.
8. **Representative pilot gate:** on the target GPU, run the exact proposed workers/games/replay/AMP/mix/budgets for 20 measured iterations; preserve checkpoints and compute ETA. Require GPU memory headroom and no CPU fallback. If the full replay is expensive to populate, resume a frozen copied full checkpoint and record its provenance.

Do not apply a blanket 17-GiB `RLIMIT_AS` to CUDA and call it total-memory protection. CUDA virtual mappings and aggregate worker memory need separate treatment. A configured launch resource budget must report what it measures; fail capability checks if a required limit cannot be enforced. CPU-only certification can finish independently while GPU stages remain pending.

### M10 — Bounded experiments, then select the long run

Freeze a **full** start checkpoint for comparable optimizer/replay continuations. A weights-only 1600 checkpoint can start a separate experiment family; do not compare it against a full latest continuation without labeling the reset difference.

Use identical initial model/optimizer/replay, algorithm code, precision, learner budget, output isolation and evaluation corpus across arms. Preserve each arm's seed list and population configuration. Run a short functional pilot first, then an explicitly budgeted learning pilot; pilots are experiments, not mandatory long production launches.

Suggested sequential arms:

| Arm | LR | Curriculum | Purpose |
| --- | --- | --- | --- |
| A | Constant restored `1e-3` | Baseline 25% uttt.ai | Control |
| B | Constant `3e-4` | Same baseline | Isolate a lower late-training LR |
| C | Cosine candidate to `1e-5`, declared start/horizon | Same baseline | Test a schedule; record fraction of horizon observed |
| D | Best-supported A/B/C policy | 35% uttt.ai preset, unchanged opponent budgets | Test mix only |
| E, only if warranted | Same as selected D/control | Same mix, stronger uttt.ai budgets | Test opponent compute cost separately |

Default learning screen: 300 completed iterations per arm per seed, seeds 101 and 202, maximum three hours per arm/seed. Finish A/B/C before deciding whether D is warranted; do not run E automatically. Stop only at an iteration boundary when the wall-time cap is reached. Compare the last saved checkpoints within the same elapsed-time budget across arms, and report the exact iterations, positions and optimizer updates completed. One seed remains explicitly exploratory if the second is not run. Evaluate every 100 pilot iterations at a fixed inference budget. A 300-iteration screen covers only 6% of a 5,000-iteration cosine horizon, so an inconclusive C result cannot establish that the full schedule helps or fails. Do not select the winner from training loss alone or automatically extend an inconclusive screen into a 5,000-iteration run.

Evaluate selected candidates against frozen big_run and run_v2 references plus fixed AlphaBeta/uttt.ai budgets. Initial screening uses 100 paired-seat games (50 opening pairs); confirmation uses a separate held-out set of 200 games and pair-bootstrap intervals. These sample sizes are a practical starting point, not a promise to resolve a small Elo difference. If uncertainty remains, report inconclusive and predeclare any extension instead of repeatedly testing until a favorable result appears.

Promote a new `best` only with recorded held-out evidence, acceptable score against the reference pool, and no severe family-specific regression. Store a promotion record and preserve the former best as a pinned rollback. A latest checkpoint is not automatically strongest. Final inference-budget choice comes from a same-checkpoint 512/1024/2000 sweep and a documented latency budget; no retraining is needed to run that sweep.

### M11 — Documentation and production runbook

- Make this plan the canonical checklist; update its milestone ledger with actual results. Reconcile README, `POPULATION_TRAINING.md`, `engines/README.md`, benchmark reports, test docs and both handovers. Mark old results historical rather than silently rewriting them as fresh tests.
- Remove universal Elo forecasts, unverified OpenSpiel claims, phantom APIs, incorrect population percentages, game/pair errors, and broad memory/zero-copy claims. Explain current native support and the adaptive Python exception.
- Update `train_v2.sh` or add a separate continuation launcher that accepts checkpoint/output/config explicitly. Avoid hardcoded changing Elo labels. Print effective config and refuse duplicate output ownership before workers start.
- The long-run command must identify an immutable full checkpoint, new output directory, selected LR phase, population config, explicit CUDA/native/precision settings, worker/batch limits, checkpoint retention/pins, evaluation schedule and a measured ETA.
- Rollback: stop the new run gracefully, retain its last completed checkpoint/evidence, and resume a pinned original in a separate output directory with its saved optimizer/config. Revert a code regression via its focused commit. Never overwrite the source run to undo an experiment.

## 6. Command templates for the implementer

These templates describe the intended interface after the relevant milestone exists. Do not run a command with a not-yet-implemented flag and count argparse failure as a successful readiness test. Set the interpreter to the environment verified at M0; the current launcher uses `.venv`, while older docs use Conda.

```bash
STTT_PY=/home/entropy/miniconda3/envs/sttt/bin/python

# Existing validation entry points, using the isolated implementation checkout.
"$STTT_PY" -m unittest tests.test_cpp_engine tests.test_cpp_mcts -v
"$STTT_PY" -m unittest discover -s tests -v

# New readiness runner required by M0/M9; each stage saves its own manifest.
"$STTT_PY" scripts/check_training_ready.py --backend cpp --device cpu --stage cpu
"$STTT_PY" scripts/check_training_ready.py --backend cpp --device cuda --stage gpu

# New scheduling flags required by M5; deliberately tiny functional smoke.
"$STTT_PY" -m sttt.ai train --backend cpp --device cpu --arch mlp \
  --output runs/readiness/example-cpu --iterations 2 --games 4 --workers 2 \
  --simulations 32 --leaf-batch 4 --inference-batch 8 --steps 4 --batch 16 \
  --buffer 1000 --eval-every 0 --keep-checkpoint-window 0 \
  --lr-schedule cosine --lr 0.001 --lr-min 0.00001 --lr-iterations 10

"$STTT_PY" -m sttt.ai train --resume runs/readiness/example-cpu/latest.pt \
  --resume-mode continue --output runs/readiness/example-cpu \
  --backend cpp --device cpu --iterations 1
```

Use unique real output directories; `example-cpu` is illustrative. On resume, M4's saved effective config must prevent accidental replacement of small smoke budgets by production defaults. A continuation from full iteration 2007 with 5,000 additional iterations ends at 7007 unless an explicit cumulative cap says otherwise. A weights-only initialization is a new training run with recorded ancestry, not a full-state continuation.

The chosen 5,000-iteration production command is intentionally not preselected here: start checkpoint, LR arm and curriculum must come from M10 evidence. Proposed knobs are not evidence of improvement.

## 7. Completion ledger

For each row, replace Pending with the actual result only after recording the commit and artifacts. Keep CPU readiness, GPU readiness, speed measurement and strength improvement as separate results.

| Gate | Status | Commit / artifact |
| --- | --- | --- |
| M0 fresh build/provenance | **Done** (verify-only) | `a837fa2` predates this ledger and already ships `scripts/check_training_ready.py` + build-provenance fixes. Verified 2026-09-13: fresh `make -C cpp` uses the *selected* interpreter's headers; extension imports in a fresh subprocess; discovered test count is **311**, not the assumed 258. |
| M1 native robustness | **Done** | `0bd9d8a`, `b58ffcd`, `997c42a`, `9dd3387`. 42 new regression tests; suite 356, 1 pre-existing skip. See notes below. |
| M2 backend/worker contract | Pending | — |
| M3 encoding/IPC integration | Pending | — |
| M4 AMP/resume/ownership | Pending | — |
| M5 LR policy and resume tests | Pending | — |
| M6 curriculum configuration | Pending | — |
| M7 controlled evaluation | Pending | — |
| M8 honest benchmarks | Pending | — |
| M9 CPU certification | Pending | — |
| M9 GPU certification and soak | Pending | Requires real CUDA access |
| M10 bounded strength experiments | Pending | No Elo outcome assumed |
| M11 docs/launch/rollback | Pending | — |

### M1 evidence (2026-09-13)

Four focused commits on `feat/m1-native-robustness`. Defects found and fixed:

1. **`0bd9d8a` — unchecked `PyFloat_AsDouble` at every float boundary.** Policy
   priors and `SearchConfig` floats were converted without checking the error
   return, so NaN/inf/negative/non-numeric entries became real priors;
   `expand()` normalizes, so one bad entry corrupted the whole distribution. A
   nonfinite `c_puct` made every PUCT score NaN. **A string policy entry
   segfaulted the pre-patch build (exit 139).** 16 tests; against the old build
   they give 11 failures and one SIGSEGV.
2. **`b58ffcd` — mutable states were hashable.** `FastState`/`CppState` are
   mutated by `play_inplace`, so a dict key silently relocated. Both are now
   unhashable; `state_key()` gives an immutable canonical tuple that compares
   equal across all three backends, and `clone()` gives an independent copy.
3. **`997c42a` — node views had no lifetime marker.** A view holds an arena
   index; `reset()`/`init_root()`/re-rooting rebuild the arena, after which the
   index names a *different* node and the getters returned its statistics.
   Generation counter added; stale views now raise `RuntimeError`. Also:
   `advance()` accepted in-range **illegal** actions and re-rooted onto
   impossible boards; `sttt_benchmark_rollouts` let worker-thread exceptions
   call `std::terminate`.
4. **`9dd3387` — no exception handling in `mcts.cpp` at all**, so `std::bad_alloc`
   would unwind through the CPython C boundary. All entry points now translate
   to `MemoryError`/`ValueError`/`RuntimeError`. Stats dict insertion failures
   propagate. **Arena reclamation:** measured one 60-move game at 1024 sims/move
   — 437,343 nodes (40 MiB) of which **212 reachable, 99.9% garbage**, growing
   with game length and multiplied per worker. `advance()` now compacts to the
   live subtree; the same game ends at 212 nodes. `stats` exposes `arena_nodes`
   and `arena_capacity` for the M9 memory gate.

**Known limitation:** `test_f20_git_branch_isolation` hard-codes a branch
allowlist (`testing`/`cpp`/`main`) and therefore fails on any feature branch,
which §4.2 mandates. Not weakened (§4.6); it passes once merged to `main`.

**Environment note:** `engines/runtime/` is gitignored, so a fresh worktree or
clone must have it copied in or 8 OpenSpiel/uttt.ai tests error spuriously.

## 8. Research and API references

- [PyTorch CosineAnnealingLR documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html): defines schedule stepping and serializable state. The schedule unit/horizon and clamping rules above are this project's explicit design choices; the API alone does not choose a useful LR.
- [Scaling Laws for a Multi-Agent Reinforcement Learning Model](https://arxiv.org/abs/2210.00849): studies AlphaZero scaling on Connect Four and Pentago. It supports investigating compute/model scaling, not transferring a fixed UTTT iteration-to-Elo table or architectural threshold.
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815): reference for the self-play/search learning approach. Opponent-game collection in this project is not automatically expert-policy imitation.
- [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565): useful future reference for efficiency experiments; do not import its speed/strength results as measurements of this repository.
