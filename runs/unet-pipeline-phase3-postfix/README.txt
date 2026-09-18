Resumed 2026-09-18 on this machine (tmux: train) from
  entropy@100.68.226.55:runs/unet-pipeline-phase2/latest.pt  (iteration 4785)
  sha256 307f084bd727834fa9060dd28c515da579229202f205eefff942afd8aebcf85c
  local copy: runs/unet-pipeline-phase2-import/latest.pt (+ model-4300..4750.pt as population history)
Code: branch handover-fixes @ 63a72ea, native BUILD_ID 20260918084249_63a72ea.
Settings mirror phase2 (8 workers, 16 games, 512 sims, leaf batch 64, cosine 2e-4 -> 1e-5,
horizon 2500 continuing from 585), --iterations 1915.
NOTE: search behaviour changed (batched-search fix, ba150bf) and external-engine timeouts are
no longer capped at 20 s, with one retry per failed game. Metrics before and after iteration
4785 are not directly comparable. Reanalysis (--reanalyse) is OFF in this run.
