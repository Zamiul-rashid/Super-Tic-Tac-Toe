# Self-hosted web frontend

Play Ultimate Tic-Tac-Toe in a browser against the iteration-6000 self-play
network. One container, no GPU, no PyTorch.

```bash
docker run -p 8000:8000 --cpus 4 ghcr.io/zamiul-rashid/moja:latest
```

Then open <http://localhost:8000>.

To build it yourself:

```bash
docker compose -f docker/docker-compose.yml up --build
```

## What is in the image

| | |
| --- | --- |
| Size | **385 MB** (measured) |
| Python | 3.13-slim |
| Inference | ONNX Runtime 1.30, CPU execution provider |
| Search | the native C++ engine, built in-image |
| Network | `models/model-6000.onnx`, 6.2 MB, converted from the championship checkpoint |
| PyTorch | **absent** |

PyTorch is not installed. `site-packages/torch` measures 1.2 GB in the
development environment, of which 989 MB is `torch/lib`; serving through ONNX
Runtime instead is most of the difference between this image and an ~800 MB one.
The `.onnx` is converted ahead of time by `scripts/export_onnx.py`, outside the
image — see [Using your own checkpoint](#using-your-own-checkpoint).

## Difficulty

All four tiers run on CPU. There is no GPU build, because there is nothing for a
GPU to do here that matters: the network is 1.56 M parameters, and even the top
tier answers in about half a second on four threads.

| Tier | Simulations | `leaf_batch` | Median per move |
| --- | --- | --- | --- |
| Casual | 128 | 8 | 28 ms |
| Standard | 512 | 16 | 100 ms |
| Strong | 2048 | 64 | 354 ms |
| Championship | 4096 | 128 | 582 ms |

Measured on four threads with the native backend, excluding HTTP. A full
round-trip through the container at Standard measured 88 ms median, 110 ms max.

**Standard is the configuration the network actually won its championship at.**
`runs/grand-championship-6000/championship/manifest.json` records
`simulations: 512, leaf_batch: 16`, and the published championships in
`docs/report/evidence/snapshot.json` ran on CPU. A player on Standard is facing
the configuration behind this project's headline results, not a reduced one.

Tiers live in `configs/web/difficulty.json`. Changing an opponent's strength is a
config edit.

### ONNX Runtime against PyTorch, per tier

Measured on the same host, four threads, native backend:

| Tier | ONNX | PyTorch | ONNX speedup |
| --- | --- | --- | --- |
| Casual | 28 ms | 59 ms | 2.13× |
| Standard | 100 ms | 117 ms | 1.17× |
| Strong | 354 ms | 313 ms | 0.88× |
| Championship | 582 ms | 588 ms | 1.01× |

ONNX Runtime wins clearly at small leaf batches and loses slightly at large
ones. The reason to use it here is image size, not speed; the speed result is
reported because it is mixed and should not be oversold.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `STTT_WEB_MODEL` | `/app/models/model-6000.onnx` | Path to the `.onnx`. Not a `.pt`. |
| `STTT_WEB_THREADS` | `4` | ONNX Runtime `intra_op_num_threads`. |
| `STTT_WEB_MAX_SESSIONS` | `8` | Concurrent games. |
| `STTT_WEB_SESSION_TTL` | `1800` | Idle game expiry, seconds. |
| `STTT_WEB_PORT` | `8000` | Listen port. |
| `STTT_WEB_BACKEND` | `auto` | `auto`, `cpp` or `python`. `cpp` refuses to fall back. |
| `STTT_WEB_TIERS` | `configs/web/difficulty.json` | Tier table. |

`STTT_WEB_THREADS` is set deliberately. Left unset, ONNX Runtime takes every core
on the host. Pair it with `--cpus`.

## HTTP API

The server is authoritative about the rules. Every state payload carries `legal`
straight from `State.legal_actions()`; the browser renders what it is told and
never decides legality itself.

```
GET  /api/health
POST /api/game                 {difficulty, human_side}  -> {session, state, engine_action}
GET  /api/game/{session}          -> {state, difficulty, human_side, history, backend}
POST /api/game/{session}/move  {action}                  -> {state, engine_action}
DELETE /api/game/{session}
```

| Condition | Status |
| --- | --- |
| Action outside 0–80 | 422 |
| Action not currently legal | 400 |
| Unknown or expired session | 404 |
| Move in a finished game | 409 |
| Body over 4 KiB | 413 |
| Concurrent-game cap reached | 429 |

The page remembers its game in `sessionStorage`, so a refresh resumes it; an
expired game quietly starts a fresh one. Two query parameters exist for
previews: `?session=<id>` opens an existing game, and `?theme=light|dark`
forces a surface instead of following the OS.

## Using your own checkpoint

The image cannot read a `.pt` — that needs PyTorch. Convert outside the
container, then mount the result:

```bash
python scripts/export_onnx.py \
    --checkpoint runs/my-run/model-9000.pt \
    --output models/model-9000.onnx

docker run -p 8000:8000 \
    -v "$PWD/models:/models:ro" \
    -e STTT_WEB_MODEL=/models/model-9000.onnx \
    ghcr.io/zamiul-rashid/moja:latest
```

`export_onnx.py` refuses to write anything if the exported network disagrees with
the checkpoint, and records the source SHA-256, opset and measured parity deltas
in a `.onnx.json` sidecar next to the model. For `models/model-6000.onnx` those
deltas are ≤ 1.26e-05 on logits and ≤ 4.77e-06 on values, over 512 sampled
positions at five batch sizes.

Requires `torch` and `onnxscript`, neither of which the runtime needs.

## Limitations

- **linux/amd64 only.** The native search is compiled with `-march=x86-64-v2`,
  a portable x86 baseline. There is no arm64 build.
- **No authentication.** This is meant for your own machine or a trusted LAN. Do
  not expose it to the open internet without a reverse proxy in front of it.
- **Games live in memory.** Restarting the container ends every game in progress.
- **One player per game.** There is no human-versus-human mode and no lobby.
