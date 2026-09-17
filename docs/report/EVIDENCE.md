# Evidence and figure reproduction

This folder is a report package, not a new training run. Its measurements come from completed local artifacts selected in [report-config.json](report-config.json). All paper assets are stored here; the reusable renderer is [scripts/report_figures.py](../../scripts/report_figures.py).

## Package contents

| Path | Purpose |
| --- | --- |
| [README.md](README.md) | Main manuscript, equations, tables, limitations, and references |
| [evidence/snapshot.json](evidence/snapshot.json) | Frozen normalized data, original evaluation manifests, game rows, input paths and SHA-256 hashes |
| [evidence/summary.json](evidence/summary.json) | Recomputed counts, scores, intervals, and log-selection summary |
| [report-config.json](report-config.json) | Input paths and iteration cutoff for refreshing the evidence |
| [figures/](figures/) | Nine numbered paper figures; measured plots and deterministic flowcharts have both PNG and SVG versions |
| [figure-prompts.md](figure-prompts.md) | Built-in imagegen prompt, correction prompt, style-reference role, and final image path |

The snapshot excludes checkpoint tensors, raw training replay, and the full training logs. Its manifests preserve historical absolute paths as provenance; those paths do not need to exist to regenerate the figures. The two completed tournaments contain 840 raw game rows; the two depth-10 match sets add 40, for 880 preserved evaluation games in total.

## Selection and interpretation

- The cutoff is iteration **4193**, not whichever checkpoint is latest when the report is opened.
- Training segments are `runs/unet-pipeline/metrics.jsonl` and `runs/unet-pipeline-resume-20260916/metrics.jsonl`.
- Only records containing `loss` are full training observations. Separate `evaluation_overhead_seconds` rows share some iteration numbers and are retained separately.
- The selected logs contain 4,007 full training observations, 13 overhead records, and no duplicate full training records. They cover iterations 187–4193 and 64,112 scheduled games.
- If a future input contains repeated full records, the renderer retains the last record per iteration within a segment and records the count. Overlapping iterations across segments are rejected.
- The pretraining curves come from `runs/unet-bootstrap-100k/pretrain_metrics.jsonl`. A directory name is not a verified dataset-size measurement or complete checkpoint-lineage certificate.
- All plotted numbers are derived from evidence. The imagegen figure is schematic and supplies no measured values.

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

Numbers in parentheses on opponent labels are simulation settings, not ratings. OpenSpiel's budget is per board/cell decision. Chart axes, legends, and intervals are produced with Matplotlib, not image generation. The PNGs are rendered at 220 dpi; SVG files retain editable text. Both formats are retained so Markdown viewers and publication workflows can use the same evidence.

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

## Integrity checks and uncertainty

The renderer verifies championship record counts, every pairwise W/D/L count, aggregate W/D/L totals in the saved ratings, and two games per opening pair. It recomputes depth-10 score means and percentile intervals and compares them with the saved sweep artifacts. It does not re-fit or certify the historical Elo/Glicko implementation.

During report preparation, both collection mode and independent snapshot rendering completed successfully. All 84 local Markdown links resolved. A CPU model check confirmed 1,560,835 parameters and policy/value/Q output shapes of `(2, 81)`, `(2,)`, and `(2, 81)` for a two-position batch. Corresponding championship opening sequences and candidate seat assignments matched across checkpoints. All nine final figures were visually inspected. No new training or tournament was launched to produce this report.

Scores are 1/0.5/0 for win/draw/loss. Two games with the same opening are averaged before bootstrap resampling. Resampling individual games would discard this grouping. The bootstrap intervals describe uncertainty within this small observed corpus; all-win samples yield degenerate intervals, and repeated use of development openings can bias broader performance claims.

The championship uses a base seed plus matchup-specific offsets. A depth-10 sweep uses one shared corpus across its arms, but the selected artifacts contain only one arm each (512 simulations). There is no measured multi-budget scaling curve in this package. No equal-time or equal-hardware superiority is claimed.

## Current versus historical evidence

The paper describes code inspected at `feb5982c4d40a0500f0e1496d365b2b2b920a011`; saved manifests report their own runtime and native-build information. They refer to an older launcher filename that was later consolidated. This discrepancy is documented, not silently edited away. Original reports are historical observations; rendering their graphs today does not turn them into newly executed experiments.

The final methodology illustration is a simplified visual overview. In particular, both dashed skip paths belong to the concatenation-plus-convolution fusion stage; the exact implementation concatenates all three inputs **before** its 3×3 convolution. The value branch originates at the macro representation. Diagram glyphs are explanatory, not tensor samples or observed policy distributions.
