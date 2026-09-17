# Conversation handover — Super Tic-Tac-Toe

Recorded on 2026-09-17. This is a comprehensive technical handover of the available conversation, not a verbatim transcript. Earlier requests whose execution is not established are explicitly marked. Runtime observations are snapshots, not promises that jobs remain active.

## Immediate context and authority

The latest request is to write this handover. It does **not** authorize starting the proposed search audit, changing training targets, restarting training, downloading datasets, or launching additional experiments.

The central research question is why our iteration-4530 model remains weaker than the integrated uttt.ai stage-2 model, and how to learn more effectively from mistakes and opponent moves. No causal explanation has yet been established.

Read [AGENTS.md](../AGENTS.md) before acting. Workspace: `/home/entropy/Code/Super-Tic-Tac-Toe`. Python environment: `/home/entropy/miniconda3/envs/sttt/bin/python`. Default shell: zsh. Host processes and tmux may be invisible from the sandbox; use appropriately approved host-side read-only checks rather than interpreting an empty sandbox process list as evidence that nothing is running.

## User requests and preferences, in order

1. Resume training from the last checkpoint in tmux named `train`.
2. Stop training and run a full championship including AlphaBeta depth 10; retrieve numbers.
3. Continue training.
4. Clean the repository: consolidate documentation, improve folder structure, move operational scripts under `scripts/`, and use generalized training/evaluation launchers with configurable paths instead of creating a new script for every run.
5. Add an `AGENTS.md` describing rules and where everything belongs.
6. Ask how much usage a fully working Docker frontend with automatic CPU/GPU detection and local U-Net inference would take. The available conversation does not establish an estimate or implementation; do not claim this frontend was built.
7. Write a paper in a README under `docs/report`, covering the problem, methodology, literature, current solutions including uttt.ai and Google OpenSpiel, comparisons, and figures.
8. Explicitly request the imagegen skill for paper illustrations. Reject decorative/generic figures and provide academic figure references. Desired style: dense, structured multi-panel methods diagram, clear arrows, restrained colors, feature blocks, labels, and publication-like layout. The supplied examples illustrate unrelated methods; use their style, not their scientific content.
9. Request a roughly 2,000-game championship; briefly select iteration 4500, then explicitly override that with “use the latest one.” The latest at launch was iteration 4530, frozen for the whole experiment.
10. Request progress updates and a separate parallel 20-game test of the model at 2,000 simulations versus uttt.ai-128.
11. Ask why nothing was running. The separate test had finished; training had crashed; the large championship was still active.
12. Discuss whether uttt.ai is the best model, why ours loses, and how to audit search and compare budgets fairly.
13. Ask how uttt.ai trained, whether replay/final-outcome targets limit us, and why we do not explicitly learn from opponent moves and analyse losses.
14. Request this handover.

Preserve existing work, checkpoints, datasets, and results. Never stop training without explicit authorization. Resume into a new output directory. Do not silently substitute a weaker external engine after failures. Do not create per-run scripts. Do not claim that suggestions were implemented.

## Current artifact snapshot

At handover creation, the saved large-championship progress file reported **872/2,016 games**, status `running`, elapsed **8,759.17 seconds**. Its latest recorded pairing was AlphaBeta d10 versus uttt.ai-128. This is a file observation; check fresh progress and host processes before reporting live status.

The separate 20-game experiment is complete. The phase-two training log ends in an external-opponent timeout and exit status 1. No restart was performed in this conversation after diagnosing that failure.

### Large championship

- tmux session: `report-championship`.
- Output: `runs/report-championship-2016-20260917/`.
- Original source: `runs/unet-pipeline-phase2/latest.pt`.
- Frozen input: `runs/report-championship-2016-20260917/championship/evaluation-inputs/candidate/latest.pt`.
- Iteration: **4530**; architecture: U-Net.
- SHA-256: `cb04e705a015be6fc1e3c244ffd4f0f23dc277fa99fb2798bfcc104a3f671c6d`.
- Candidate: 512 simulations, CPU, C++ backend, leaf batch 16.
- Seed: **17092026**; two opening plies; mirrored player swaps.
- Eight entrants, 28 pairings, 72 games per pairing: **2,016 games / 1,008 mirrored pairs**.
- Roster: candidate; `cpp-alphabeta:10:50000000`; `cpp-alphabeta:6:50000000`; `cpp-alphabeta:4:50000000`; `tactical`; `threat-block`; `openspiel-mcts:100`; `utttai:128`.
- CPU thread environment: OMP, MKL, and OpenBLAS each 1; launched with `nice -n 10` to coexist with training.
- The budgets are **not equal-time or equal-compute comparisons**.

All 504 candidate games completed before the baseline-versus-baseline portion:

| Opponent | Wins | Draws | Losses | Score, draws worth half |
| --- | ---: | ---: | ---: | ---: |
| AlphaBeta d10 | 46 | 11 | 15 | 71.5% |
| AlphaBeta d6 | 63 | 5 | 4 | 91.0% |
| AlphaBeta d4 | 70 | 1 | 1 | 97.9% |
| Tactical | 71 | 1 | 0 | 99.3% |
| Threat-block | 72 | 0 | 0 | 100% |
| OpenSpiel MCTS-100 | 72 | 0 | 0 | 100% |
| uttt.ai-128 | 9 | 41 | 22 | 41.0% |
| **Total** | **403** | **59** | **42** | **85.8%** |

Earlier progress reports: 69 games after about 23 minutes (candidate versus d10 then 44/10/15); 541 games after about 91 minutes; later 792 games. These were partial snapshots, superseded by subsequent rows. Final tournament rankings require the remaining baseline games.

Progress files under `championship/`: `progress.json`, `games.partial.jsonl`, `manifest.json`. Final artifacts include `scoreboard.txt`, `ratings.csv`, `championship_games.csv`, `matrix.csv`, and `tournament.json`. The journal preserves completed games but is **not automatic resume support**.

### Separate 2,000-simulation test

- tmux session was `utttai-s2000`; it closed normally when the command completed.
- Output: `runs/iter4530-s2000-vs-utttai128-20/`.
- Same frozen iteration 4530 and hash as the championship, not a newer model.
- Candidate 2,000 simulations versus uttt.ai-128; CPU/C++/leaf batch 16.
- 20 games, 10 mirrored pairs; seed **17098026**.
- That seed was selected to match the championship's candidate–uttt.ai matchup seed (base plus matchup index 6 × 1000). Explicitly verify opening rows before claiming a matched-subset statistical comparison.
- Completed in **481.52 seconds**, approximately eight minutes.
- Candidate: **4 wins, 8 draws, 8 losses**, **20% win rate**, **40% score**.
- Opponent: 8 wins, 8 draws, 4 losses, 60% score.

Do not infer that 2,000 simulations is worse than 512 from 40% over 20 games versus 41% over 72 games. Compare matched opening pairs, quantify uncertainty, and distinguish simulation counts from compute.

### Training failure

Run directory: `runs/unet-pipeline-phase2/`. The last inspected training command used CUDA, FP16, C++ backend, eight workers, 16 games per iteration, 512 simulations, leaf batch 64, inference batch 1024, inference wait 2 ms, 100 optimizer steps per iteration, batch size 512, replay capacity 200,000, cosine learning rate 0.0002 down to 0.00001 over 2,500 iterations, population baseline config, and symmetry augmentation. Evaluation was every 300 iterations, 20 games, 512 simulations. Checkpoint saves every 50 iterations, retaining a 500-iteration window and 10 checkpoints.

Resume source for that run was `runs/unet-pipeline-phase2/start-checkpoint/phase2_seed.pt`. Earlier host PID was 3479437; it is historical and must not be used as a current process target without revalidation.

The actual failure message in `train.log`:

```text
RuntimeError: External bot 'utttai' failed (exception: External bot 'utttai' failed (timeout after 20.065s (pid=3788020, alive=True)))
Training exited with status 1.
```

The `train` tmux session was absent at diagnosis. The championship process was still active. The timeout's deeper cause was **not diagnosed**; do not assert that concurrent evaluation caused it. Do not silently restart, increase timeouts, or enable fallback based only on this handover.

## Repository and evaluation work already performed

Canonical operational entry points: [scripts/train.sh](../scripts/train.sh), [scripts/evaluate.sh](../scripts/evaluate.sh). Read [scripts/README.md](../scripts/README.md), [docs/training.md](../docs/training.md), and [docs/testing.md](../docs/testing.md).

For the larger championship, the existing harness was extended:

- `scripts/evaluation_suite.py`: configurable championship opponent list, explicit resolved entrant settings, package/environment metadata, source hashes, durable per-game JSONL journal, progress JSON, and failed-run manifest records.
- `scripts/evaluate.sh`: `--championship-opponents "SPECS"` forwarding.
- `sttt/tournament.py`: optional per-game observer, called for both players' mirrored games without intentionally changing RNG behavior.
- Tests in `tests/test_budget_sweep_harness.py` and `tests/test_tournament.py`: custom roster, duplicate rejection, partial journal preservation after failure, observer invariance.
- Documentation in `scripts/README.md` updated.

The focused run of `tests.test_budget_sweep_harness`, `tests.test_script_launchers`, and `tests.test_tournament` passed **45 tests**. Shell syntax and `git diff --check` were also checked during that work. This is not a claim that the full suite or proposed search audit passed.

Known harness caveats: a failure before tournament execution can leave an incomplete running manifest; failure progress JSON may retain its last running status even when the manifest records failure. Inspect logs, manifest, timestamps, and processes together. Custom checkpoint opponents are not guaranteed individually frozen by this extension; the executed roster used only one checkpoint candidate.

At the start of this handover turn, `git status --short` was empty. Earlier edits may have been incorporated externally; do not assume they are still uncommitted or attribute a commit to this assistant.

## Paper and figures

Main manuscript: [docs/report/README.md](../docs/report/README.md). Supporting files:

- [EVIDENCE.md](../docs/report/EVIDENCE.md): sources, validation, limitations.
- [figure-prompts.md](../docs/report/figure-prompts.md): image-generation provenance and prompts.
- [report-config.json](../docs/report/report-config.json): selected historical runs.
- `docs/report/evidence/snapshot.json` and `summary.json`: frozen evidence and computed summaries.
- [scripts/report_figures.py](../scripts/report_figures.py): generalized evidence collector/renderer, not a per-run script.

The manuscript was approximately 6,433 words, 11 sections, nine figures, and seven tables when checked. It covers rules, problem statement, literature, architecture, search, bootstrap/online training, population, native/inference pipeline, checkpoints, results, limitations, and reproduction. It is an engineering research draft, not a peer-reviewed paper; no author identities were invented.

Figure inventory: methodology; population composition; inference flow; evaluation flow; pretraining losses; training dynamics; matchup outcomes; championship matrix; checkpoint comparison. Measured charts/flowcharts have PNG and SVG versions. The methodology illustration is imagegen raster, styled after the user's reference; quantitative charts were rendered from actual evidence, not invented by image generation.

Imagegen skill used: `/home/entropy/.codex/skills/.system/imagegen/SKILL.md`. Final selected source image: `/home/entropy/.codex/generated_images/01a0a97d-5231-79c2-a1f5-fec71112c7d6/exec-cd24852e-26f2-4feb-aec8-dea5da180f6a.png`, copied to `docs/report/figures/methodology.png`. Rejected decorative/generic drafts were not included. The caption/table clarify exact concatenation semantics where the schematic arrow is approximate.

Previously verified: 84 local manuscript links resolved; figure rendering from config and independently from snapshot succeeded; model instantiated with 1,560,835 parameters and policy/value/Q output shapes `(2,81)`, `(2,)`, `(2,81)`; archived game counts and reported results were audited. No PDF was requested or generated.

The existing manuscript primarily describes **historical checkpoints 1806 and 4193**, not the new 4530 championship. Do not claim new results are already integrated. The renderer assumes the older experiment structure (championship plus sweep) and may need extension for a third checkpoint/eight-player pool. Preserve the old evidence; identify new experiments separately.

### Historical evidence retained in the report

- Iteration 1806 hash: `228830ff79b24f5f42e6f34b7393bce077d96a42434292fc3d452a9cffc5ee24`.
- Iteration 4193 hash: `562510ece3cb9a859f6c1aee50c6da7486f27688decc4cc3e14f39ab75dcb4a6`.
- Two historical championships: 420 games each; two separate d10 evaluations: 20 games each; total 880 evaluation rows.
- Iteration 4193 candidate championship: 97 wins, 14 draws, 9 losses / 120 games; 86.7% score. Against uttt.ai: 1/12/7, 35% score. Against d6: 19/0/1; d4: 18/2/0; tactical: 19/0/1; threat and OpenSpiel: 20 wins each.
- Iteration 1806 versus uttt.ai: 4/3/13, 27.5% score. Versus d6: 15/2/3; d4: 17/2/1; tactical: 19/1/0; threat and OpenSpiel: 20 wins each.
- Separate d10: iteration 1806 11/5/4 (67.5% score); iteration 4193 9/8/3 (65%). Pair-bootstrap intervals were broad: 50–82.5% and 50–80%, respectively.
- Archived full training rows: 4,007, iterations 187–4193; 64,112 scheduled games within that range, not total lifetime games. Thirteen additional evaluation-overhead rows are not full training iterations.
- Bootstrap log: ten epochs, 9,570 optimizer steps. Directory `unet-bootstrap-100k` is **not proof of exactly 100,000 source games**; complete early lineage was not established.
- Position-level validation splitting may leak correlated positions from the same game; future splits should be by complete game.

## What differs between uttt.ai and our system

Primary upstream source: [uttt.ai repository](https://github.com/arnowaczynski/utttai). Architecture: [policy_value_network.py](https://github.com/arnowaczynski/utttai/blob/main/utttpy/selfplay/policy_value_network.py).

| Aspect | uttt.ai documentation | Our inspected system |
| --- | --- | --- |
| Architecture | Custom policy-value CNN, about 5 million parameters; down/up sampling and skip connections | Hierarchical U-Net-style network, 1,560,835 parameters |
| Data generation | Initial pure-MCTS dataset, then neural-MCTS dataset | Bootstrap followed by online population games |
| Dataset sizes | 8 million evaluated positions per stage, two stages | Exact comparable lifetime position total not established; online replay capacity 200,000 |
| Training schedule | Generation and training separated into offline stages | Alternating game generation and optimizer updates |
| Value supervision | Root mean search value | Final outcome from the state's player perspective |
| Policy supervision | Search-derived policy, documented masked KL loss | Learner search policy, masked cross-entropy |
| Auxiliary supervision | Action-value prediction | Masked root-child Q prediction |
| Search | Sequential synchronous simulations | Batched neural search, native backend, subtree reuse/proof mechanisms |

Do not equate positions with games: many positions come from one game. Do not equate replay capacity with total data seen. Do not assume KL versus cross-entropy causes a strength difference: for fixed target distributions their parameter gradients can be equivalent apart from implementation/weighting details. Do not call their architecture identical to ours merely because both have U-Net-like features.

Our configured game quotas: self 30%; uttt.ai 25%; AlphaBeta 25%; historical 10%; best 5%; tactical 2%; OpenSpiel 1%; threat 1%; style 1%. Actual historical/best choices can fall back according to available snapshots. These are game shares, not replay-position shares. Online logs used uniform replay despite optional stratification support.

Our encoder uses 289 canonical features mapped to eight 9×9 planes. Network: 64-channel micro representation with two residual blocks; stride-three aggregation to 128-channel macro representation with three residual blocks; upsampling and skip/input fusion; policy 81, Q 81, scalar value. Value uses the macro representation. Q is auxiliary, not direct move selection by maximum predicted Q.

## Scientific conclusions and corrections from the discussion

### Strength claims

uttt.ai is stronger in these tested matchups. This does **not** establish it as the world's best available UTTT model. Our model wins individual games but has not beaten it consistently. Results against a 128-simulation integrated engine are not results against every upstream setting or browser configuration.

The 2,000-simulation experiment did not demonstrate an advantage over 512, but it was too small and differently sized to establish that more search cannot help. The hypothesis that training-target quality limits us is plausible, not proven. Architecture, search correctness/settings, data, and optimization remain confounded.

### Replay and final outcomes are not inherently flawed

The assistant initially emphasized limitations of outcome targets; this was subsequently qualified. AlphaZero itself trains state values on final outcomes and policies on search distributions. It achieves strong play without requiring opponent imitation. See [AlphaZero paper](https://arxiv.org/html/1712.01815v1).

Final outcome is a legitimate sample of the result under the continuation that was played. It is not a certificate that every earlier move was bad or that the position was objectively lost. Search-value targets are more local but can inherit model/search bias. Neither target is automatically superior.

Replay supports repeated learning and is not an obstacle by itself. Possible limitations worth measuring: eviction of important mistakes, excessive representation of easy/common positions, stale search labels, and repeated fitting to low-quality targets. Fixed offline datasets can also be biased or stale.

### What “learning from the opponent” currently means

Opponent choices change encountered positions and outcomes, so the learner already learns **indirectly** from opponent games. But opponent games currently record learner training positions, and policy labels come from the learner's own search, not the opponent's full search distribution or evaluations.

In `sttt/ai.py`, replay entries include encoded state, learner policy, legal mask, `outcome * state.turn`, Q targets, and Q mask. A loss pushes the value target toward −1 for that player; it does **not** directly tell the policy head to avoid the chosen move. The policy still learns the original search distribution. This distinction explains why “I lost” is weaker feedback than “this particular alternative would have avoided the loss.”

Do not blindly imitate every opponent move or every move in a winning game. Strong opponents can make mistakes, and a single observed move is not a complete target distribution. Evaluate alternatives and preserve player perspective correctly.

## Proposed search audit — not executed

Use frozen iteration 4530. Inspect existing tests, add controlled cases with known outcomes, and instrument root diagnostics.

| Check | Method / expected result |
| --- | --- |
| Value perspective | Trace alternating turns and backup signs; correct player-relative values |
| Terminal handling | Forced wins, losses, draws; exact outcomes propagated |
| Legal actions | Forced-board, closed-board, free-choice cases; no illegal selection/visits |
| Simulation accounting | Budgets 128/512/2000; count actual work under documented conventions, reuse, and early proof termination |
| Python/C++ agreement | Identical states and controlled network outputs; explain legitimate tie/ordering differences |
| Sequential/batched search | Controlled positions; check pending visits, stale statistics, update loss, and tactical behavior |

Record root policy probabilities, visits, action values, model value, completed simulations, and network evaluations. If a correctness failure appears, report it and propose a fix before the strength comparison; do not mix pre-fix and post-fix results.

## Proposed matched-opening budget comparison — not executed

- Same frozen model and fixed uttt.ai-128 engine/model.
- Budgets 128, 512, and 2,000.
- Same 50 opening pairs for each budget, both player assignments: 100 games per arm, **300 games total**.
- Pre-generate/save opening corpus and separate opening RNG from search RNG.
- Keep backend, batch size, exploration, and hardware conditions fixed.
- Rotate budget order across pairs to mitigate machine-load timing bias.
- Primary contrast: 2,000 versus 512 simulations.
- Report W/D/L, score, paired score differences, and confidence intervals bootstrapped by opening pair; also latency, changed move choices, and reviewed tactical mistakes.
- Existing workloads remain untouched unless explicitly authorized. Timings under concurrent load are approximate.
- One hundred games per arm may still be inconclusive. Report uncertainty rather than forcing a conclusion.
- Extend existing generalized launchers, not new per-run scripts.

Decision sequence: audit → cheap smoke test → matched comparison → evidence-based training decision. Bigger search gains suggest a compute/strength tradeoff; no clear gain does not by itself prove model weakness. Provably wrong decisions warrant targeted diagnosis.

## Proposed loss-review / reanalysis learning — not implemented

User's key intuition: after losing, identify **why** rather than merely replaying the result. Practical proposal:

1. Save full game action sequences, opponent moves, and learner search statistics. Current championship rows record outcomes and opening moves, not necessarily the full action sequence; inspect availability before promising reconstruction of old games.
2. Reanalyse selected positions from both sides using stronger search or a stronger teacher.
3. Compare the played action against alternatives from the **same parent position**, with correct perspective. Do not identify a mistake merely from a value change after swapping players.
4. Identify suspected tactical/strategic errors; reserve “proven” for exact solvable outcomes.
5. Produce refreshed policy distributions and evaluated action/value targets; test outcome-only, search-only, and mixed value supervision as separate experiments.
6. Retain useful mistakes alongside wins, draws, ordinary positions, and fresh games. Avoid training only on losses or only one opponent.
7. Hold out complete games/openings for evaluation. Once championship positions are used for training, they are not untouched evaluation data anymore.

Reanalysis is supported as a general method by [Online and Offline Reinforcement Learning by Planning with a Learned Model](https://arxiv.org/abs/2104.06294), but no benefit has yet been demonstrated in this repository.

A separate proposed data-quality experiment: keep our architecture fixed, train a separate checkpoint on a fixed stronger-search-labelled dataset, then compare against the unchanged baseline. This isolates teaching quality more cleanly than simultaneously enlarging the network and changing replay, targets, and search. Dataset size, teacher budget, compute allocation, and exact mixing weights have not been chosen or authorized.

## Recommended next handoff actions

1. Recheck large-championship progress and completion; derive final rankings only from completed artifacts.
2. Keep the finished 20-game experiment and frozen model intact.
3. If asked to restore training, diagnose the external-engine timeout and validate the latest intact checkpoint, then resume into a new directory with explicit authority.
4. Ask for execution direction before implementing the proposed audit/training changes; the latest request only created documentation.
5. When updating the paper, distinguish archived 1806/4193 evidence, the 4530 championship, the 20-game test, and any later training experiments.
6. Continue using reusable scripts, exact provenance, held-out evaluations, and appropriately qualified claims.

No new experiment, model modification, training restart, or search fix was performed while writing this handover.
