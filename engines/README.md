# External tournament engines

The tournament framework supports named external engines through a JSON
registry. Third-party engines are not vendored. Each entry must point to a
verified executable or wrapper that speaks one of the supported protocols.

Copy `registry.example.json` to `registry.json`, set the real commands, and run:

```bash
cp engines/registry.example.json engines/registry.json
python -m sttt.ai tournament --checkpoint runs/big_run/latest.pt \
  --engine-config engines/registry.json \
  --opponents codingame-legend utttai --games 20 --simulations 512 \
  --device cpu --output runs/big_run/tournaments/top-engines-20
```

The `codingame` protocol receives the previous move as `row col`, then the
number of legal moves and each legal coordinate. The `action_index` protocol
receives the previous action index, then the number of legal moves and each
legal action index. The engine prints one move per turn. Registry entries
default to `fallback: raise`, so missing binaries, timeouts, crashes, and
illegal moves are reported as failures instead of silently becoming Tactical.

`utttai` is an AlphaZero-like browser project and needs a local wrapper before
it can participate as a subprocess. The registry records the reference and
does not pretend the browser source is already an executable engine.
