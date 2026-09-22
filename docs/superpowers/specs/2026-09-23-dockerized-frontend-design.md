# Dockerized self-hostable frontend — design and implementation plan

Date: 2026-09-23
Branch: `frontend` (cut from `main`)
Status: plan. No code written yet.
Revision: 2 — CPU-only, ONNX Runtime at inference. See §2 for what changed and why.

## 1. What this is

A stranger runs one `docker run` and plays Ultimate Tic-Tac-Toe in their browser
against the iteration-6000 self-play network.

```bash
docker run -p 8000:8000 ghcr.io/…/sttt:latest
```

One image. No GPU, no NVIDIA driver, no container toolkit, no `--gpus` flag.
PyTorch is not installed in the image at all — inference runs on ONNX Runtime
against a `.onnx` converted from the championship checkpoint.

### Out of scope

No training, evaluation, or dashboards in the container. No human-vs-human and no
multiplayer lobby. No authentication — this is something you run on your own
machine or LAN. linux/amd64 only; arm64 is deliberately excluded and documented.

## 2. Decisions, and two that were reversed

| Decision | Choice | Why |
| --- | --- | --- |
| Server | FastAPI + uvicorn | Request validation, body-size limits, threadpool offload at a boundary strangers expose on their LAN. |
| Client | React + Vite, built to static files | Served by the same Python process; no nginx, no second container. |
| Rules | Server-authoritative, `legal` sent to client | A third rules implementation in JavaScript is the thing that silently drifts. |
| Game state | Server-side sessions | The MCTS subtree is reused across turns; a stateless API throws that away every move. |
| Inference | **ONNX Runtime, no torch in the image** | §2.2. |
| Variants | **CPU only** | §2.1. |
| Weights | `models/model-6000.onnx` baked in | 6 MB. Nothing to download or mount. |
| Distribution | GHCR, built by GitHub Actions | `docker run` and play. |
| Visual direction | Graph-paper duel | The game's origin is paper and pencil; the forced-board rule is the hero. |

### 2.1 The GPU image was dropped

The original plan shipped a second `gpu` image on the premise that a GPU would
unlock higher simulation tiers. Measurement killed the premise. Running the real
checkpoint on CPU at four torch threads:

| Tier | Simulations | `leaf_batch` | Median per move | Max |
| --- | --- | --- | --- | --- |
| Casual | 128 | 8 | 40 ms | 55 ms |
| Standard | 512 | 16 | 115 ms | 118 ms |
| Strong | 2048 | 64 | 264 ms | 315 ms |

The tier that was to be GPU-only answers in a quarter second on four CPU threads.
Championship at 4096 lands near half a second. Neither is something a human
notices between moves, and even a weak two-core host at 3–4× slower keeps Strong
near one second.

Against that, the GPU image cost a 5–6 GB download (`site-packages/nvidia` alone
measures 3.2 GB, plus 1.2 GB of torch), an NVIDIA container toolkit prerequisite
that fails at the Docker daemon with an opaque error, and a whole class of "I
forgot `--gpus all` and did not notice" support questions.

**All four tiers now ship in the single CPU image.** Do not reintroduce a GPU
variant for strength. The only defensible GPU use here is batched inference
across many concurrent games, which is a different feature with a different
design.

### 2.2 Inference moved from PyTorch to ONNX Runtime

Not for speed — 115 ms was never the problem. For image size.

`site-packages/torch` measures **1.2 GB**, of which **989 MB is `torch/lib`**.
The `+cpu` wheel is smaller but `libtorch_cpu.so` still dominates; call it
400–500 MB installed. `onnxruntime` is roughly 50 MB.

| | With torch | With onnxruntime |
| --- | --- | --- |
| Final image | ~700–800 MB | **~250 MB** |

Three-fold cut on the artifact people actually pull, which is the single biggest
thing available to a self-hosted container. A speed gain is likely as well, since
ONNX Runtime avoids per-call Python and dispatch overhead on a small graph, but
it has not been measured and no claim is made.

TensorRT was considered and rejected outright: it is an NVIDIA GPU-only runtime
with no CPU execution provider.

## 3. Why the swap is a drop-in

The search never depends on torch. The native MCTS calls the evaluator purely by
name — `cpp/src/mcts.cpp:772` is
`PyObject_CallMethod(self->model, "evaluate_many", "O", py_states_list)`, with an
`evaluate` fallback at line 777. The Python `TreeSearch` does the same at
`sttt/search.py:218`. Anything exposing those two methods works on both backends.

`sttt/backends.py` imports only numpy and `.search`; `sttt/search.py` imports only
the standard library and numpy. So `create_search()` returns a `SearchAdapter`
with `run`, `advance` and `reset` — everything `CheckpointBot` was doing for
subtree reuse — with no torch in the import graph.

`CheckpointBot` itself is **not** used by the server: it lives in `sttt/bots.py`,
which imports torch at module level and pulls in `load_model`. The session layer
constructs the search directly instead.

### The one blocker

`encode_states` lives in `sttt/learning.py:23`, which imports torch at module
level. Its body is pure numpy plus the optional C++ encoder — there is no torch
in it — but importing it drags torch back into the image.

Fix: move `INPUTS`, `encode`, `legal_masks` and `encode_states` into a new
torch-free `sttt/encoding.py`, and re-export them from `learning.py` so every
existing caller keeps working. `tests/test_encoding.py` already covers these.

### The U-Net exports cleanly

`sttt/unet.py` uses only `Conv2d`, `ConvTranspose2d`, `GroupNorm`, `ReLU`,
`Linear` and `Flatten`. No control flow, no `.item()`, no custom operators.

- Opset **18 or higher**, required by `GroupNorm`.
- **Dynamic batch axis**, because `leaf_batch` varies by tier and the final batch
  of a search is partial.
- The graph outputs **raw logits and value**. Masking and softmax stay in Python.

## 4. The conversion step

Conversion happens once, up front, as its own step — not inside the Docker build.
The image then contains no torch in any stage, builder included.

```bash
python scripts/export_onnx.py \
  --checkpoint runs/grand-championship-6000/championship/evaluation-inputs/candidate/model-6000.pt \
  --output models/model-6000.onnx \
  --opset 18
```

Generalized, taking `--checkpoint` and `--output`, per the AGENTS.md rule against
per-checkpoint scripts. It writes the `.onnx` **and** a provenance sidecar
`models/model-6000.onnx.json`:

```json
{
  "source_checkpoint": "runs/grand-championship-6000/.../model-6000.pt",
  "source_sha256": "…",
  "iteration": 6000,
  "arch": "unet",
  "opset": 18,
  "torch_version": "2.14.0",
  "exported_at": "2026-09-23T…Z",
  "parity": {"max_policy_abs_diff": 0.0, "max_value_abs_diff": 0.0, "positions": 512}
}
```

Both files are committed. The sidecar is what makes a committed derived artifact
defensible: anyone can check which checkpoint it came from and whether it still
matches.

**Bringing your own checkpoint.** The image cannot load a `.pt` — that needs
torch. Run `scripts/export_onnx.py` outside the container, then mount the
resulting `.onnx` and point `STTT_WEB_MODEL` at it.

## 5. The evaluator

`sttt/web/evaluator.py` — roughly thirty lines, mirroring
`BasePolicyValue.evaluate_many` in `sttt/learning.py:92`:

```python
class OnnxEvaluator:
    def evaluate_many(self, states):
        if not states:
            return []
        if any(s.result is not None for s in states):
            raise ValueError('Neural evaluation expects nonterminal positions')
        features, mask = encode_states(states, backend=self.encode_backend)
        logits, values = self.session.run(None, {self.input_name: features})
        logits = np.where(mask, logits, -np.inf)
        probabilities = softmax(logits, axis=-1).astype(np.float64)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return list(zip(probabilities, values.astype(float).ravel().tolist()))

    def evaluate(self, state):
        return self.evaluate_many([state])[0]
```

Two details that must match the torch path exactly, or the agent changes:

- Masked logits are filled with `-inf`, not `-1e4`. The `-1e4` guard in
  `learning.py` exists only for the fp16 autocast path, which is CUDA-only and
  therefore never applies here.
- Probabilities are cast to float64 and renormalized after the softmax, exactly
  as the torch path does.

One `InferenceSession` is created at startup and shared by every session — it is
thread-safe, so there is no per-session model copy. `intra_op_num_threads` is set
from `STTT_WEB_THREADS`; left alone, ONNX Runtime grabs every core, which is the
failure mode that hangs a busy machine.

## 6. Repository layout

| Path | Purpose |
| --- | --- |
| `sttt/encoding.py` | **New.** Torch-free `INPUTS`, `encode`, `legal_masks`, `encode_states`, moved out of `learning.py`. |
| `sttt/web/__init__.py` | Package marker. |
| `sttt/web/evaluator.py` | `OnnxEvaluator`: `evaluate` / `evaluate_many`. |
| `sttt/web/sessions.py` | Session store: creation, lookup, TTL sweep, cap, per-session lock. |
| `sttt/web/difficulty.py` | Loads `configs/web/difficulty.json`. |
| `sttt/web/server.py` | FastAPI app, routes, request/response models. |
| `sttt/web/static/` | Vite build output target (gitignored). |
| `frontend/` | Vite + React + TypeScript source. |
| `scripts/export_onnx.py` | Generalized `.pt` → `.onnx` exporter with provenance and parity check. |
| `configs/web/difficulty.json` | Tier table. Versioned config, per AGENTS.md. |
| `models/model-6000.onnx` | Converted champion network, 6 MB, committed. |
| `models/model-6000.onnx.json` | Provenance sidecar. |
| `docker/Dockerfile` | The image. |
| `docker/docker-compose.yml` | One service. |
| `.dockerignore` | Repository root. Mandatory — see §10. |
| `.github/workflows/docker.yml` | Build, smoke-test, push. |
| `tests/test_web_api.py` | API tests against a stub evaluator. |
| `tests/test_onnx_parity.py` | Torch-vs-ONNX agreement. Skipped when torch is absent. |
| `docs/web.md` | Operator documentation. |

`sttt/web/` lives inside the existing package so it ships with
`pip install -e .` and needs no separate packaging story.

## 7. HTTP API

```
POST /api/game                  {difficulty, human_side}  -> {session, state}
POST /api/game/{session}/move   {action}                  -> {state, ai_action}
GET  /api/game/{session}                                  -> {state}
GET  /api/health   -> {backend, iteration, opset, threads, tiers}
```

### State payload

Serialized straight off `env.State`, plus the legal-action list:

```json
{
  "cells":  [0, 0, 1, -1, "…"],
  "boards": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "turn":   1,
  "forced": -1,
  "result": null,
  "legal":  [0, 1, 2, "…"]
}
```

`legal` comes from `State.legal_actions()`. The client renders what it is told and
never computes legality itself.

### Error handling at the trust boundary

- Action not an integer in range, or not in `state.legal_actions()` → **400**.
  `env.State.play` already raises `ValueError`; the route maps it rather than
  trusting the client's filtering.
- Unknown or expired session → **404**.
- Session cap reached → **429**, with a message naming the cap.
- Request body capped; `difficulty` must be a key in the tier table.

### Concurrency

Move routes are `def`, not `async def`, so FastAPI runs the blocking search in its
threadpool. Each session carries a `threading.Lock`, held for the whole move, so
two concurrent requests for one session cannot corrupt one tree.

## 8. Sessions

A dict `session_id -> {search, state, lock, last_seen}`, where `search` is the
`SearchAdapter` from `backends.create_search(evaluator, backend="auto", seed=…)`.

- Cap: `STTT_WEB_MAX_SESSIONS`, default 8. Unbounded sessions means unbounded RAM
  and unbounded CPU from anyone who can reach the port. This is a real trust
  boundary and the cap is not optional.
- TTL: `STTT_WEB_SESSION_TTL`, default 1800 s. Swept lazily on each request — no
  background thread.
- Per-move flow: `search.run(state, sims, batch_size=leaf_batch)` → `argmax` →
  `search.advance(action)` for the engine's own move, then `search.advance(…)`
  again for the human's reply. Advancing exactly once per played move is what
  preserves the subtree; advancing twice for one move raises
  `ValueError: Illegal action … for the current root state`.
- Sessions share the one `InferenceSession`, so a session costs a tree, not a
  model.

## 9. Difficulty tiers

`configs/web/difficulty.json`, all available in the single image:

| Tier | Simulations | `leaf_batch` | Measured median |
| --- | --- | --- | --- |
| Casual | 128 | 8 | 40 ms |
| Standard | 512 | 16 | 115 ms |
| Strong | 2048 | 64 | 264 ms |
| Championship | 4096 | 128 | ~500 ms (extrapolated) |

Standard is the configuration the network actually won its championship at —
`runs/grand-championship-6000/championship/manifest.json` records
`simulations: 512, leaf_batch: 16`, and the published championships in
`docs/report/evidence/snapshot.json` ran at `device: cpu`. A player on Standard
is facing the exact configuration behind the project's headline results, and
`docs/web.md` should say so.

Championship's figure is extrapolated, not measured. Measure it in Phase 3 and
replace the number here.

## 10. Docker

Two stages: `node:22-alpine` builds the Vite bundle; the Python stage compiles
`cpp/`, installs `onnxruntime`, `numpy`, `fastapi` and `uvicorn`, and copies the
bundle plus `models/`. Non-root user. `HEALTHCHECK` polls `/api/health`.

No PyTorch, in any stage. The `.onnx` arrives pre-converted from the repository.

### Two things that will bite otherwise

**`CXXFLAGS` override.** Both `cpp/Makefile:3` and `setup.py:20` hardcode
`-march=native`. An image built on one machine and pulled onto an older CPU dies
with SIGILL. The Dockerfile passes an explicit portable baseline:

```
CXXFLAGS="-O3 -march=x86-64-v2 -Wall -Wextra -Werror -std=c++17 -fPIC -pthread -I./include"
```

`x86-64-v2` covers SSE4.2 and POPCNT — safe on anything from roughly 2009 on. Do
not let it revert to `native`; the CI assertion in Phase 6 is what prevents a
silent regression, since the Makefile already bakes `STTT_COMPILER_FLAGS` into the
module.

**`.dockerignore`.** `runs/` is 7.8 GB and `.git` is 264 MB. Without an ignore
file the build context alone kills the build. Exclude `runs/`, `data/`, `.git`,
`docs/report/`, `engines/runtime/`, `__pycache__`, `*.so`, `node_modules`,
`build*`. The model is read from `models/`, which is why it is copied there.

## 11. Visual design — graph-paper duel

Worked against the official `frontend-design` brief. That plugin is **not
installed** in this environment; install it with
`claude plugin install frontend-design@claude-plugins-official` to have it
callable during implementation. Its two-pass process was followed: a token plan,
then a review against the brief. §11.7 records what the review changed.

### 11.1 Subject, audience, job

An Ultimate Tic-Tac-Toe board you play against a network that taught itself the
game over 6000 self-play iterations. The audience is whoever self-hosts it and
their friends. The primary job is making the **forced-board rule legible** — your
move dictates where your opponent must play next, and that constraint is the
whole game.

So the hero is not the board. It is **the moment the constraint moves**: the
instant the network's reply redirects you to a different board. Everything else
on the page is built to stay quiet so that moment reads.

### 11.2 Colour

Two grounds, because graph paper has a real-world negative: the engineering
blueprint. Light mode is the notebook; dark mode is the blueprint. This is not an
invented dark theme, it is the same subject seen the other way round.

| Token | Light (paper) | Dark (blueprint) | Role |
| --- | --- | --- | --- |
| `--ground` | `#FBFAF6` | `#0E2233` | The sheet. |
| `--rule` | `#D3E4EA` | `#1C3C55` | Printed grid rule. Deliberately near-invisible. |
| `--rule-major` | `#A8CBD8` | `#2E5A79` | Every third rule; the local-board divisions. |
| `--graphite` | `#3A3F46` | `#C8D6DF` | The player's marks, and body text. |
| `--ink` | `#17365D` | `#7FC4E8` | The network's marks. |
| `--circle` | `#C2352B` | `#FF6B5A` | The forced-board circle. Nothing else is ever this colour. |
| `--dead` | `#E4E6E7` | `#16304A` | Won or drawn local boards, flattened. |

Seven values. `--circle` is the entire boldness budget and is spent in exactly one
place, on one element, once per turn.

### 11.3 Type

One family: **Atkinson Hyperlegible Next**, chosen for a reason specific to this
brief rather than as a safe accessible default. Its whole design premise is
making letterforms unmistakable from one another — and this interface's core job
is telling someone unmistakably *which* board they may play in. The typeface
argument and the product argument are the same argument.

Hierarchy comes from weight and size, not a second face. Board indices are set in
tabular lining numerals so coordinates never shift width between turns.

The X and O marks are **not type at all**. They are SVG paths with slightly
irregular stroke geometry and open, non-meeting terminals, so they read as drawn
rather than set. That is why one family suffices, and it is where the page's
character actually comes from.

### 11.4 Layout

The page is a sheet, edge to edge, not a card floating on a background. The board
sits on the sheet with a wide left margin — the way you would actually draw it in
an exercise book — and that margin carries the move log, running down the page as
a record of the game. Left-aligned throughout. Body copy under 80 characters.

No header banner. The board is the first thing on the page; the title sits small
beneath it.

```
  ┌─ the sheet: faint grid, edge to edge ───────────────────────────┐
  │        │                                                        │
  │  1. e5 │      ╭──────────────╮                                  │
  │  2. c3 │      │  X  ·  O     │    ·  ·  ·      ·  ·  ·          │
  │  3. a7 │      │  ·  X  ·     │    ·  O  ·      ·  ·  ·          │
  │  4. …  │      │  ·  ·  ·     │    ·  ·  ·      ·  ·  ·          │
  │        │      ╰──────────────╯                                  │
  │        │       red pencil circle: you must play here            │
  │  move  │                                                        │
  │  log   │         ·  ·  ·       ▒▒▒▒▒▒▒▒       ·  ·  ·           │
  │  in    │         ·  ·  ·       ▒  dead  ▒     ·  ·  ·           │
  │  the   │         ·  ·  ·       ▒▒▒▒▒▒▒▒       ·  ·  ·           │
  │ margin │                                                        │
  │        │                                                        │
  │        │   Your move, top-left board.                           │
  │        │   Playing Standard against iteration 6000.             │
  └────────┴────────────────────────────────────────────────────────┘
```

Below roughly 640px the margin folds under the board and the move log collapses
to the last three moves.

### 11.5 The difficulty selector

Four options rendered as a row of strokes of increasing weight — a visibly heavier
pencil line for a stronger opponent. The *visual* is grounded in the world; the
*labels* stay plain words (Casual, Standard, Strong, Championship), because a
person choosing a difficulty should not have to decode a metaphor.

### 11.6 Motion

Exactly one orchestrated moment on the whole page: the red circle drawing itself
around the new forced board (SVG `stroke-dashoffset`, roughly 320 ms) when the
network's reply redirects you. Nothing animates on page load. Nothing animates on
hover. `prefers-reduced-motion` renders the circle already closed.

### 11.7 Review pass: what changed and why

Four things in the first draft were generic defaults rather than choices made for
this brief.

**Centred single column → sheet with a margin.** "Single column, centred,
constrained max-width" is the layout any web page gets by default. A notebook page
is not centred; it has a margin, and that margin is where you write things down.
Putting the move log there makes the layout subject-derived and gives the log a
home it would otherwise have had to earn as a panel.

**`Standard · iteration 6000 · native search` → a plain sentence.** The
frontend-design brief names middle-dot meta strings as one of the commonest tells
of a generated page, and I wrote one anyway. It is now
`Playing Standard against iteration 6000.` The backend is diagnostic information
and belongs in `/api/health`, not under the board.

**Red used twice → red used once.** The exercise-book margin rule is red in real
life, and the forced-board circle is red. Using it for both would have split the
boldness budget and weakened the only element that has to be unmissable. The
margin rule is now the same cyan family as the grid, and `--circle` appears
nowhere but the circle.

**No dark mode → paper and blueprint.** An inverted graph paper is not a
recoloured theme, it is a blueprint, which is the same artefact in its other
real-world form. That made a dark mode subject-derived instead of obligatory.

### 11.8 Explicitly avoided

Cards. All-caps eyebrow labels. Gradient washes. A big-number hero. Numbered
`01 / 02 / 03` markers. `→` appended to buttons. The warm-cream-plus-terracotta
palette. And the specific risk of this direction: skeuomorphic paper kitsch —
**no** paper texture or noise, no drop shadow under the sheet, no torn or curled
edges, no fake spiral binding. The paper is conveyed by grid geometry and nothing
else.

## 12. Tests

### `tests/test_onnx_parity.py` — the gate that matters

The exported network must pick the same moves as the `.pt` that won the
championship, or the container quietly ships a different agent than the project's
evidence describes. Requires torch, so it is skipped when torch is unavailable and
is always run in CI.

1. Across at least 512 sampled positions (fixed seed, drawn from random legal
   playouts), the torch and ONNX policies agree within `1e-4` and the values
   within `1e-4`.
2. Masked-out actions have probability exactly zero in both.
3. A fixed-seed scripted game produces an **identical action sequence** under
   both evaluators.
4. The sidecar's `source_sha256` matches the checkpoint it names.

### `tests/test_web_api.py`

FastAPI `TestClient` against a **stub evaluator**, not the real network, so the
suite stays fast and needs no model file. Deterministic seed, per the AGENTS.md
reproducibility rule.

1. New game returns a session and an empty board with 81 legal actions.
2. A legal move advances the state and returns the network's reply.
3. An illegal action returns 400 and does not mutate the session.
4. The forced-board constraint holds: after a move to cell `c`, `legal` contains
   only actions in board `c` while that board is open.
5. A scripted game plays to a terminal `result`.
6. Session cap returns 429; unknown session returns 404.
7. **Anti-divergence check:** the served `legal` list equals
   `State.legal_actions()` across sampled positions.

### `tests/test_encoding.py`

Already exists. It must pass unchanged after the `sttt/encoding.py` move — that is
the regression signal for the refactor.

## 13. Implementation phases

Each phase ends with its verification command actually run, not assumed.

### Phase 0 — branch and groundwork

- `git checkout -b frontend` from `main`.
- Write `.dockerignore` **first**, before any build is attempted.
- Add `configs/web/difficulty.json`.

Verify: `git status` shows only intended additions; the build context with the
ignore file applied is under roughly 50 MB.

### Phase 1 — torch-free encoding

Move `INPUTS`, `encode`, `legal_masks`, `encode_states` to `sttt/encoding.py`;
re-export from `learning.py`.

Verify: `python -m pytest tests/test_encoding.py tests/test_learning_loss.py`,
then the full `python -m pytest tests`. Then confirm the point of the exercise:
`python -c "import sttt.encoding, sys; assert 'torch' not in sys.modules"`.

### Phase 2 — exporter and parity gate (TDD)

Write `tests/test_onnx_parity.py` first, then `scripts/export_onnx.py` until it
passes. Commit `models/model-6000.onnx` and its sidecar.

Verify: `python -m pytest tests/test_onnx_parity.py -v`. Record the measured
maximum policy and value deltas in the sidecar.

### Phase 3 — evaluator and server (TDD)

Write `tests/test_web_api.py` first, then `sttt/web/evaluator.py`, `sessions.py`,
`difficulty.py`, `server.py`.

Verify: `python -m pytest tests/test_web_api.py -v`, then the full suite. Then
measure the Championship tier's real latency and replace the extrapolated figure
in §9.

### Phase 4 — React client

Board, difficulty selector, status line, SVG marks, pencil circle. Build output
lands in `sttt/web/static/`.

Verify: `npm run build` succeeds; `uvicorn sttt.web.server:app` serves the page;
play a full game in a browser against the real `models/model-6000.onnx`.

### Phase 5 — image

`docker/Dockerfile` and `docker/docker-compose.yml`.

Verify: `docker build`, `docker run -p 8000:8000`, `curl localhost:8000/api/health`
reports `backend: cpp`. Then assert the point of the rewrite:
`docker run --rm IMAGE python -c "import torch"` **must fail**. Play a scripted
game to completion over the API. Record the final image size.

### Phase 6 — CI and GHCR

`.github/workflows/docker.yml`: build on tag push, smoke-test, push to GHCR. The
smoke test asserts the extension's reported compiler flags contain `x86-64-v2`
and not `native`, and runs the parity test.

Verify: a tagged dry run; `docker pull` and play from a pruned local Docker.

### Phase 7 — documentation

- `docs/web.md`: run command, port, every environment variable, the tier table
  with **measured** latencies, the "Standard is the championship configuration"
  note, how to bring your own checkpoint, the arm64 exclusion, and the security
  note from §15.
- Link it from `docs/README.md` under "Active guides".
- Add a README section.
- Add `sttt/web/`, `sttt/encoding.py`, `frontend/`, `docker/`, `models/` and
  `scripts/export_onnx.py` rows to the AGENTS.md repository map, per its own
  closing instruction.

Verify: `bash -n` on any shell script; `git diff --check`; review the final diff
for generated files.

## 14. Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `STTT_WEB_MODEL` | `/app/models/model-6000.onnx` | ONNX model path. Not a `.pt`; see §4. |
| `STTT_WEB_THREADS` | `4` | ONNX Runtime `intra_op_num_threads`. |
| `STTT_WEB_MAX_SESSIONS` | `8` | Concurrent game cap. |
| `STTT_WEB_SESSION_TTL` | `1800` | Idle session expiry, seconds. |
| `STTT_WEB_PORT` | `8000` | Listen port. |
| `STTT_WEB_BACKEND` | `auto` | Search backend. `cpp` refuses to fall back; see `sttt/backends.py`. |

## 15. Risks and open items

- **Parity is the whole risk.** If the export is subtly wrong, the container plays
  a different agent than the evidence in `docs/report/` describes, and nothing
  else in the system will notice. `tests/test_onnx_parity.py` is the only thing
  standing between those two outcomes; it runs in CI on every build, not just at
  conversion time.
- **A committed derived artifact can drift.** `models/model-6000.onnx` is
  generated from a `.pt`. The sidecar's `source_sha256` and the CI parity run are
  what keep that honest. If the checkpoint is ever replaced, both files are
  regenerated together.
- **Opset support.** `GroupNorm` needs opset ≥ 18. Confirm at Phase 2 that the
  pinned `onnxruntime` supports the opset the exporter emits; pin both versions
  in `requirements` and record them in the sidecar.
- **GHCR package visibility** must be set to public after the first push, or
  `docker pull` fails for everyone else with an unhelpful auth error.
- **`-march=native` regression.** If someone rebuilds the extension inside the
  image without the override, the published image starts crashing on older CPUs,
  for some users only. The Phase 6 CI assertion is the guard.
- **No authentication, by design.** `docs/web.md` must say plainly that this is
  meant for a personal machine or a trusted LAN, and should not face the open
  internet without a reverse proxy in front of it.

---

## 16. As built (2026-09-23)

The plan above is preserved as written. This section records where reality
differed, so the estimates in it are not mistaken for measurements.

### Numbers that changed

| | Planned | Measured |
| --- | --- | --- |
| Image size | ~250 MB | **385 MB** |
| Championship latency | ~500 ms (extrapolated) | **582 ms** |
| ONNX speed vs torch | "likely faster, unmeasured" | 2.13× Casual, 1.17× Standard, **0.88× Strong**, 1.01× Championship |
| Test runner | `pytest` | `unittest` — the repository has no pytest, and `python -m unittest discover -s tests` is the documented command |

ONNX Runtime beats PyTorch clearly at small leaf batches and loses slightly at
large ones. §2.2 claimed image size as the reason and declined to claim speed;
that was the right call, and `docs/web.md` reports the mixed result rather than
the flattering half of it.

### Three things the plan did not anticipate

**`planes_from_flat` pinned the batch size.** It read `n = x.shape[0]` and
reshaped to `(n, ...)`. Under ONNX tracing that `n` becomes a constant, so the
exported graph accepted only the batch it was exported with and failed at
inference with "cannot be reshaped to the requested shape". Fixed by reshaping
to `-1`, which is identical arithmetic. `sttt/unet.py` therefore changed —
production model code the plan did not expect to touch.

**A batch-1 example input is degenerate.** Even after the `-1` fix, exporting
with `torch.zeros(1, 289)` let the exporter fold the batch dimension away. The
exporter now uses a batch-4 example, and the parity check runs at five batch
sizes including 1 — a graph that folds its batch dimension passes a single
fixed-size check and then fails on the partial final batch of a real search.

**`SessionStore` defines `__len__`, so an empty store is falsy.**
`store = store or SessionStore(...)` in `create_app` silently discarded the
injected store and built a default-capped replacement, which the session-cap
tests caught as `201 != 429`. All three injection points now use `is None`.

`torch.onnx.export` in torch 2.14 also requires `onnxscript`, an export-time
dependency the runtime image does not carry.

### Deviations from the plan

- The `.onnx` exports as **one self-contained file** via `external_data=False`.
  The default splits weights into a sibling `.onnx.data` that the image and any
  mounted-model setup would both have to know to carry.
- `DELETE /api/game/{session}` was added so finishing a game frees a slot
  against the cap. The plan listed four endpoints; there are five.
- Body-size capping is a middleware returning **413**, and out-of-range actions
  are rejected by the Pydantic model as **422** before reaching the engine. The
  plan only described the 400 case.

### Verified

- `python -m unittest tests.test_onnx_parity` — 5 tests, including the shipped
  model against its source checkpoint.
- `python -m unittest tests.test_web_api` — 15 tests.
- `docker build` → 385 MB; `docker run --rm sttt:latest python -c "import torch"`
  fails, as intended; `sttt_cpp.COMPILER_FLAGS` reports `-march=x86-64-v2`.
- A full game played against the real network inside the container: native `cpp`
  backend, 88 ms median round-trip, healthcheck `healthy`.

### Still open

- Nothing has been pushed to GHCR. The workflow exists and has never run.
  Package visibility must be set to public after the first push.
- The UI has been exercised through its API, not clicked through in a browser.

## 17. Design revision after the first real screenshot (2026-09-23)

The first deployed build was screenshotted in a real browser and it was bad.
The board had no drawn lines of its own -- §11 relied on the printed
graph-paper rule to imply the grid, but the board's cells did not align with
that rule, so the 3x3-of-3x3 structure was invisible: marks floating on a grid,
and a red ring around nothing. The full-page grid also dominated in dark mode
and related to nothing on the page. This is precisely the failure the
frontend-design brief's "take screenshots to review" step exists to catch, and
it was skipped because no screenshot path was set up. It is set up now (headless
Chrome against the running container) and §11's direction has been replaced:

- **The printed grid is gone.** The surface is a flat colour. The paper is
  conveyed by the drawing on it, not by a texture under it.
- **The board is drawn as a #, never a box.** A light # inside each small board
  and a heavy # for the big one, built with `:nth-child` borders on the right
  and bottom edges only. This is how the game is actually drawn by hand, and it
  is unmistakable at a glance.
- **Dark mode is chalk on slate**, not blueprint. Without the grid, "blueprint"
  had nothing left to say; slate is the other surface the game is really played
  on and uses the same drawn-by-hand language.
- **Colour means sides, not symbols.** Your marks are pencil (or white chalk),
  the network's are ink (or blue chalk), whichever symbol each of you holds.
  The margin log uses the same two colours.
- **A decided board fades and the winner is drawn large over it** -- what you
  do on paper -- rather than a flat fill.
- **The title moved into the margin** above the move log, where you would write
  it. On phones it sits above the board.
- **The page loads a game.** No start button, and a refresh resumes the game in
  progress via `sessionStorage` (server: `GET /api/game/{id}` now returns
  `human_side` and `history`).

Kept from §11: one typeface, hand-drawn SVG marks, the red pencil loop as the
sole bold element and the page's only non-user-triggered motion, the marginal
move log, and the ban on paper kitsch.
