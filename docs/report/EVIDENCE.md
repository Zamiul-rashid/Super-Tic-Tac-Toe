# Evidence and figure reproduction

This folder is a report package, not a new training run. Its measurements come from completed local artifacts selected in [report-config.json](report-config.json). All paper assets are stored here; the reusable renderer is [scripts/report_figures.py](../../scripts/report_figures.py).

## Package contents

| Path | Purpose |
| --- | --- |
| [README.md](README.md) | Main manuscript, equations, tables, limitations, and references |
| [evidence/snapshot.json](evidence/snapshot.json) | Frozen normalized data, original evaluation manifests, game rows, input paths and SHA-256 hashes |
| [evidence/summary.json](evidence/summary.json) | Recomputed counts, scores, intervals, and log-selection summary |
| [report-config.json](report-config.json) | Input paths and iteration cutoff for refreshing the evidence |
| [figures/](figures/) | Eleven numbered paper figures; measured plots and deterministic flowcharts have both PNG and SVG versions |
| [figure-prompts.md](figure-prompts.md) | Built-in imagegen prompt, correction prompt, style-reference role, and final image path |
| [evidence/iter4530-championship/](evidence/iter4530-championship/) | Checked-in small summary files (`manifest.json`, `tournament.json`, `championship_games.csv`) for the **new** iteration-4530 championship; not the raw run directory |
| [evidence/iter4530-s2000-utttai128/](evidence/iter4530-s2000-utttai128/) | Same three small summary files for the **new** iteration-4530 vs. uttt.ai-128, 2,000-simulation spot check |

The snapshot excludes checkpoint tensors, raw training replay, and the full training logs. Its manifests preserve historical absolute paths as provenance; those paths do not need to exist to regenerate the figures. The two archived completed tournaments contain 840 raw game rows; the two archived depth-10 match sets add 40, for 880 archived preserved evaluation games. The new iteration-4530 championship adds 2,016 raw game rows, and the new 2,000-simulation spot check adds 20, for 2,036 new preserved evaluation games — 2,916 in total across archived and new evidence.

## Selection and interpretation

- The cutoff is iteration **4193**, not whichever checkpoint is latest when the report is opened.
- Training segments are `runs/unet-pipeline/metrics.jsonl` and `runs/unet-pipeline-resume-20260916/metrics.jsonl`.
- Only records containing `loss` are full training observations. Separate `evaluation_overhead_seconds` rows share some iteration numbers and are retained separately.
- The selected logs contain 4,007 full training observations, 13 overhead records, and no duplicate full training records. They cover iterations 187–4193 and 64,112 scheduled games.
- If a future input contains repeated full records, the renderer retains the last record per iteration within a segment and records the count. Overlapping iterations across segments are rejected.
- The pretraining curves come from `runs/unet-bootstrap-100k/pretrain_metrics.jsonl`. A directory name is not a verified dataset-size measurement or complete checkpoint-lineage certificate.
- All plotted numbers are derived from evidence. The imagegen figure is schematic and supplies no measured values.
- **The archived cutoff above is unchanged.** `report-config.json`'s `experiments` list (iterations 1806 and 4193) and the corresponding `snapshot.json` fields (`population`, `pretraining`, `training`, `experiments`) are byte-identical to the original collection; this update only adds new top-level `championships` and `spot_checks` lists next to them. See [New evidence: iteration 4530](#new-evidence-iteration-4530-evaluation) below.

## Figure inventory

| Paper figure | Files (PNG and SVG unless noted) | Source and transformation |
| --- | --- | --- |
| 1 | `methodology.png` (raster only) | Imagegen, user-supplied style reference, architecture and workflow checked against code; Table 2 is the exact tensor specification |
| 2 | `population-composition.*` | Configured quotas versus sums of recorded requested/actual family counts; converted to game percentages |
| 3 | `inference-flow.*` | Deterministic explanatory diagram of `SelfPlayPool` and the model owner |
| 4 | `evaluation-flow.*` | Deterministic diagram of frozen inputs and paired seat swapping |
| 5 | `pretraining-losses.*` | Ten saved pretraining epochs; raw training and position-holdout losses |
| 6 | `training-dynamics.*` | Full training rows, raw values plus trailing mean of 100 iterations; continuation marked at 1807 |
| 7 | `matchup-outcomes.*` | Later championship matrix, checked against raw games; bars are counts out of 20 |
| 8 | `championship-matrix.*` | Later championship score `(wins + draws/2) / games`, diagonal omitted |
| 9 | `checkpoint-comparison.*` | Two championships and two depth-10 match sets; 10,000 opening-pair bootstrap replicates, seed 4243 |
| 10 *(new)* | `championship-matrix-iter4530.*` | Iteration-4530 championship score `(wins + draws/2) / games`, diagonal omitted; same renderer as figure 8, generalized to any roster size |
| 11 *(new)* | `championship-ratings-iter4530.*` | Iteration-4530 championship saved Bayesian-Elo ratings with 95% CI, one point per entrant |

Figures 1–9 are the archived evidence and are unchanged in content (regenerating them only changes embedded Matplotlib metadata such as the render timestamp and internal SVG clip-path IDs, not any plotted number). Figures 10–11 are new. Numbers in parentheses on opponent labels are simulation settings, not ratings. OpenSpiel's budget is per board/cell decision. Chart axes, legends, and intervals are produced with Matplotlib, not image generation. The PNGs are rendered at 220 dpi; SVG files retain editable text. Both formats are retained so Markdown viewers and publication workflows can use the same evidence.

## Reproduce without original runs

From the repository root, in an environment containing NumPy and Matplotlib:

```bash
python scripts/report_figures.py \
  --snapshot docs/report/evidence/snapshot.json \
  --output docs/report
```

To produce a separate copy, change `--output` to a new directory. Snapshot mode only reads the supplied JSON: it does not inspect live runs, load neural networks, or require CUDA, OpenSpiel, ONNX Runtime, or the native extension. It also does not regenerate `methodology.png`.

## Refresh from selected source artifacts

Edit the paths and cutoff in a config file, then run:

```bash
python scripts/report_figures.py \
  --config docs/report/report-config.json \
  --output docs/report
```

Config paths are resolved against the repository root. Refreshing rewrites the evidence snapshot and generated charts, so the paper's prose and tables must be reviewed afterward. It does not automatically rewrite scientific conclusions. Original run artifacts are opened read-only. The evidence snapshot records the source bytes' hashes and the repository revision at collection time; future working-tree changes are not retroactively part of those experiments.

The archived `experiments` entries (iterations 1806 and 4193) point at `runs/...` directories that are gitignored working output; they are no longer present on every machine that holds this repository, including the one that added the iteration-4530 evidence below. A full `--config` collection therefore is not always reproducible end to end for the archived portion. This is why the iteration-4530 evidence copies its small summary files into `docs/report/evidence/` instead of pointing at a `runs/...` path: the `championships` and `spot_checks` config entries stay collectible from a plain checkout even after the original remote run directories are cleaned up.

## Integrity checks and uncertainty

The renderer verifies championship record counts, every pairwise W/D/L count, aggregate W/D/L totals in the saved ratings, and two games per opening pair. It recomputes depth-10 score means and percentile intervals and compares them with the saved sweep artifacts. It does not re-fit or certify the historical Elo/Glicko implementation.

During report preparation, both collection mode and independent snapshot rendering completed successfully. All local Markdown links resolved. A CPU model check confirmed 1,560,835 parameters and policy/value/Q output shapes of `(2, 81)`, `(2,)`, and `(2, 81)` for a two-position batch. Corresponding championship opening sequences and candidate seat assignments matched across checkpoints. All eleven final figures were visually inspected. No new training or tournament was launched to produce this report.

Verification for the iteration-4530 addition: `validate()` in `scripts/report_figures.py` reran its per-championship matrix/ratings/pair-count checks over the merged snapshot (archived plus new), and the archived `experiments`, `training`, `population`, and `pretraining` fields were confirmed byte-identical before and after. `--snapshot` rendered all eleven figures from that merged snapshot; `--config` was separately exercised against the new `championships`/`spot_checks` entries (the archived entries cannot currently be collected end to end — see above).

Scores are 1/0.5/0 for win/draw/loss. Two games with the same opening are averaged before bootstrap resampling. Resampling individual games would discard this grouping. The bootstrap intervals describe uncertainty within this small observed corpus; all-win samples yield degenerate intervals, and repeated use of development openings can bias broader performance claims.

The championship uses a base seed plus matchup-specific offsets. A depth-10 sweep uses one shared corpus across its arms, but the selected artifacts contain only one arm each (512 simulations). There is no measured multi-budget scaling curve in this package. No equal-time or equal-hardware superiority is claimed.

## New evidence: iteration 4530 evaluation

This section documents provenance for the two runs added after the archived report: an 8-entrant championship and a small spot check, both against checkpoint iteration 4530. Numbers below are the exact figures inspected in the copied manifests and `tournament.json` files, not independently re-derived.

**Checkpoint.** SHA-256 `cb04e705a015be6fc1e3c244ffd4f0f23dc277fa99fb2798bfcc104a3f671c6d` (`sha8` `cb04e705`), iteration 4530, U-Net architecture, same as both entrants below (512 simulations in the championship, 2,000 in the spot check).

**Source paths on the remote host.** Both runs were produced on `entropy@100.68.226.55`, under `/home/entropy/Code/Super-Tic-Tac-Toe/runs/`:

| Run | Remote path | Local copy |
| --- | --- | --- |
| Championship | `runs/report-championship-2016-20260917/` | [evidence/iter4530-championship/](evidence/iter4530-championship/) |
| 2,000-sim spot check | `runs/iter4530-s2000-vs-utttai128-20/` | [evidence/iter4530-s2000-utttai128/](evidence/iter4530-s2000-utttai128/) |

Only `manifest.json`, `tournament.json`, and `championship_games.csv` were copied from each remote run directory into this repository — small summary files, not the raw run directory (no `games.partial.jsonl`, `progress.json`, `matrix.csv`, `ratings.csv`, or `scoreboard.txt`, and no checkpoint tensors). `championship_games.csv` preserves per-game openings, seats, and outcomes; **it does not preserve full move sequences**, so a game cannot be replayed move-by-move from this evidence alone.

**SHA-256 of the copied files** (also recorded in [evidence/snapshot.json](evidence/snapshot.json)'s `sources` list):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `iter4530-championship/tournament.json` | 8,708 | `dcf4b5562d995fc507239a2b98c81b400890293a00cb9aff1f9af74bd48a91d7` |
| `iter4530-championship/manifest.json` | 8,885 | `d5b854626d55d29a09082628217a308e049c32c4c3ba05708f618d6933a36e3f` |
| `iter4530-championship/championship_games.csv` | 106,417 | `703bf2c60114237f0e016023c6b4e105764a7496d4861e52c39723231c375bdf` |
| `iter4530-s2000-utttai128/tournament.json` | 1,274 | `5856085567bd057d4ed9c16e1a76bce0a0533dd637fb75ee27051f41ca687472` |
| `iter4530-s2000-utttai128/manifest.json` | 6,118 | `31ed8efba40e4edae95849a1169617ea8deb14c148af922ac48ba6e6d971425f` |
| `iter4530-s2000-utttai128/championship_games.csv` | 1,242 | `bba0e1270ca8d6c2e3d8d89461a0e3071f918fb5b927135808426f25ddefadf0` |

**Championship settings.** Round robin, 8 entrants (candidate plus `cpp-alphabeta-d10`/`d6`/`d4` at a 50,000,000-node cap, `tactical`, `threat-block`, `openspiel-mcts-100`, `utttai-128`), 72 games per pairing, 2,016 games total, two-ply random openings, mirrored seats, opening seed `17092026`, candidate at 512 simulations, CPU inference, C++ backend, leaf batch 16. Manifest-reported environment: Python 3.14.6, PyTorch 2.14.0+cu130, native build `20260916222904_f770ec7` (source revision `f770ec7`) — an **earlier** commit than this report's `feb5982` code-inspection point (see [Current versus historical evidence](#current-versus-historical-evidence)); the discrepancy is recorded, not resolved, consistent with how this package already treats the archived manifests.

**Spot-check settings.** Same candidate checkpoint at 2,000 simulations vs. `utttai-128`, 20 games (10 opening pairs, mirrored), opening seed `17098026`, otherwise identical harness settings to the championship.

**Results (from the copied `tournament.json`/`ratings.csv`, not re-derived):**

- Championship leaderboard: `utttai-128` 1st (91.3% score, Bayesian Elo 2303.0), candidate 2nd (85.8% score, Elo 2201.0), `cpp-alphabeta-d10` 3rd (71.2% score, Elo 1959.5). Candidate record vs. `utttai-128`: 9 wins, 41 draws, 22 losses (72 games; pair-score 41.0%, 95% pair-bootstrap CI 35.4–46.5%, seed 4243, computed by `scripts/report_figures.py`).
- Spot check: candidate 4 wins, 8 draws, 8 losses vs. `utttai-128` at 2,000 simulations (20 games; score 40.0%, 95% pair-bootstrap CI 25.0–55.0%).

**Caveats that apply to every number in this section:**

1. **Not equal-time or equal-compute.** Simulation counts (512 or 2,000 for the candidate; fixed depth/node caps for AlphaBeta; fixed simulation counts for OpenSpiel and uttt.ai) are not equalized wall-clock or FLOP budgets across entrants. No equal-time or equal-hardware superiority is claimed anywhere in this section.
2. **The championship ran concurrently with training.** It executed under `nice -n 10` alongside an active training job on the remote host, not on otherwise-idle hardware. Per-game and per-matchup timing in the manifest reflects that contention and should not be read as a clean latency benchmark.
3. **The 2,000-simulation spot check is too small to support a budget comparison.** Twenty games (ten opening pairs) is far too small a sample to conclude that 2,000 simulations perform no better than 512; its 40.0% score should not be compared against the championship's 41.0% candidate-vs-`utttai-128` pair-score as if the two experiments were matched in openings, sample size, or concurrent load — they were not the same run.
4. **A loss to uttt.ai here is not a claim about the field.** `utttai-128` outscoring the candidate in these matchups says only that this specific integration, at this budget, in this pool, scored higher; it is not evidence that uttt.ai is the strongest Ultimate Tic-Tac-Toe engine in existence, and no such claim is made.
5. **Saved game records omit move sequences.** As noted above, `championship_games.csv` records openings, seats, and outcomes, not full move-by-move trajectories, for either run in this section.

## Current versus historical evidence

The paper describes code inspected at `feb5982c4d40a0500f0e1496d365b2b2b920a011`; saved manifests report their own runtime and native-build information. They refer to an older launcher filename that was later consolidated. This discrepancy is documented, not silently edited away. Original reports are historical observations; rendering their graphs today does not turn them into newly executed experiments.

The final methodology illustration is a simplified visual overview. In particular, both dashed skip paths belong to the concatenation-plus-convolution fusion stage; the exact implementation concatenates all three inputs **before** its 3×3 convolution. The value branch originates at the macro representation. Diagram glyphs are explanatory, not tensor samples or observed policy distributions.
