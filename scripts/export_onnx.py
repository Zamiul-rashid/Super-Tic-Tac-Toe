"""Convert a policy/value checkpoint to ONNX, and prove it plays the same.

    python scripts/export_onnx.py \
        --checkpoint runs/.../model-6000.pt \
        --output models/model-6000.onnx

The container serves ONNX Runtime and has no torch, so this conversion happens
once, here, outside the image. That makes the ``.onnx`` a committed derived
artifact, which is only defensible if anyone can check what it came from -- so
this writes a provenance sidecar next to it recording the source checkpoint's
SHA-256, the opset, and the measured parity deltas, and it refuses to write
anything at all if parity fails.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Running as `python scripts/<name>.py` puts scripts/ on sys.path, not the repo
# root. Same shim as compare_checkpoints.py and evaluation_suite.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from sttt.encoding import encode_states
from sttt.env import State
from sttt.learning import load_model

# Opset 18 is the floor: GroupNorm, which every block of the U-Net uses, is not
# expressible before it.
MIN_OPSET = 18


def sample_positions(count, seed=0):
    """Distinct nonterminal states from random playouts, for the parity check."""
    rng = np.random.default_rng(seed)
    seen, states = set(), []
    while len(states) < count:
        state = State()
        while state.result is None:
            legal = state.legal_actions()
            key = (state.cells, state.boards, state.turn, state.forced)
            if key not in seen:
                seen.add(key)
                states.append(state)
                if len(states) >= count:
                    break
            state = state.play(int(rng.choice(legal)))
    return states


def torch_outputs(model, states):
    with torch.inference_mode():
        features, _ = encode_states(states)
        logits, value = model(torch.from_numpy(features))
    return logits.numpy(), value.numpy()


def onnx_outputs(path, states):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    features, _ = encode_states(states)
    logits, value = session.run(None, {session.get_inputs()[0].name: features})
    return np.asarray(logits), np.asarray(value)


def export(checkpoint, output, opset=MIN_OPSET, positions=512, tolerance=1e-4, seed=0):
    """Export, verify, then write. Returns the provenance record."""
    if opset < MIN_OPSET:
        raise ValueError(f"opset {opset} is below {MIN_OPSET}; GroupNorm needs {MIN_OPSET}+")

    checkpoint, output = Path(checkpoint), Path(output)
    model, data = load_model(str(checkpoint))
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    # Batch is dynamic because leaf_batch varies per difficulty tier and the
    # final batch of any search is partial.
    #
    # The example batch is 4, not 1. With a batch-1 example the exporter
    # constant-folds the batch dimension out of the reshapes in
    # planes_from_flat and the graph silently accepts only batch 1, failing at
    # inference with "cannot be reshaped to the requested shape". Batch 1 is
    # degenerate; do not reduce it.
    torch.onnx.export(
        model,
        (torch.zeros(4, 289, dtype=torch.float32),),
        str(output),
        input_names=["features"],
        output_names=["logits", "value"],
        dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
        opset_version=opset,
        dynamo=True,
        # One self-contained file. The default splits weights into a sibling
        # .onnx.data, which the image and any mounted-model setup would both
        # have to know to carry. 6 MB is nowhere near the protobuf limit.
        external_data=False,
    )

    states = sample_positions(positions, seed=seed)
    # Several batch sizes, not one. A graph that folded its batch dimension
    # passes a single fixed-size check and then fails on the partial final
    # batch of a real search, so the sizes below deliberately include 1.
    logit_delta = value_delta = 0.0
    for size in (1, 3, 16, 64, len(states)):
        batch = states[:size]
        t_logits, t_value = torch_outputs(model, batch)
        o_logits, o_value = onnx_outputs(output, batch)
        logit_delta = max(logit_delta, float(np.abs(t_logits - o_logits).max()))
        value_delta = max(value_delta, float(np.abs(t_value - o_value).max()))

    if logit_delta > tolerance or value_delta > tolerance:
        output.unlink(missing_ok=True)
        raise SystemExit(
            f"parity failed: logits differ by {logit_delta:.3g}, values by "
            f"{value_delta:.3g} (tolerance {tolerance:g}). Nothing written -- a "
            f"silently different network is worse than no network.")

    record = {
        "source_checkpoint": str(checkpoint),
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "iteration": data.get("iteration"),
        "arch": data.get("arch"),
        "opset": opset,
        "torch_version": torch.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "parity": {"positions": len(states), "tolerance": tolerance,
                   "max_logit_abs_diff": logit_delta,
                   "max_value_abs_diff": value_delta},
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="source .pt")
    parser.add_argument("--output", required=True, help="destination .onnx")
    parser.add_argument("--opset", type=int, default=MIN_OPSET)
    parser.add_argument("--positions", type=int, default=512,
                        help="positions sampled for the parity check")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    record = export(args.checkpoint, args.output, opset=args.opset,
                    positions=args.positions, tolerance=args.tolerance, seed=args.seed)
    parity = record["parity"]
    print(f"wrote {args.output} (iteration {record['iteration']}, arch {record['arch']}, "
          f"opset {record['opset']})")
    print(f"parity over {parity['positions']} positions: "
          f"logits <= {parity['max_logit_abs_diff']:.3g}, "
          f"values <= {parity['max_value_abs_diff']:.3g}")


if __name__ == "__main__":
    main()
