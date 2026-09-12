"""Paired benchmark of the latest local checkpoint against deployed uttt.ai.

The uttt.ai model is loaded from its public ONNX deployment, while its
published Python NMCTS implementation supplies the browser-equivalent search.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--utttai-root", required=True)
    p.add_argument("--utttai-model", required=True)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--local-simulations", type=int, default=512)
    p.add_argument("--utttai-simulations", type=int, default=1000)
    p.add_argument("--opening-plies", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p.parse_args()


class ONNXPolicyValueNet:
    """Adapter matching the return shape expected by uttt.ai's NMCTS code."""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.device = torch.device("cpu")

    def __call__(self, x: torch.Tensor):
        outputs = self.session.run(None, {"input": x.detach().cpu().numpy()})
        policy_logits = torch.from_numpy(outputs[0])
        state_value = torch.from_numpy(outputs[1]).reshape(-1)
        action_values = torch.zeros_like(policy_logits)
        return policy_logits, action_values, state_value


def make_uttt_state(uttt_module, local_state, opening_moves):
    from utttpy.game.action import Action

    state = uttt_module.UltimateTicTacToe()
    for action in opening_moves:
        symbol = 1 if state.is_next_symbol_X() else 2
        state.execute(Action(symbol=symbol, index=int(action)))
    assert bytes(state.state[:81]) == bytes(
        1 if c == 1 else 2 if c == -1 else 0 for c in local_state.cells
    )
    return state


def run_game(local_bot, uttt_bot, opening_moves, local_is_x, seed):
    from sttt.env import State
    from sttt.search import SearchConfig, TreeSearch

    import utttpy.game.ultimate_tic_tac_toe as ug
    from utttpy.game.action import Action

    local_state = State()
    for action in opening_moves:
        local_state = local_state.play(int(action))
    uttt_state = make_uttt_state(ug, local_state, opening_moves)

    local_rng = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    local_tree = TreeSearch(local_bot.model, local_rng, SearchConfig())
    local_side = 1 if local_is_x else -1
    local_actions = []
    started = time.monotonic()
    random.seed(seed + 101)

    while local_state.result is None:
        if local_state.turn == local_side:
            probs = local_tree.run(local_state, local_bot.simulations)
            action = int(probs.argmax())
            local_actions.append(action)
        else:
            uttt_bot.tree = uttt_bot.tree.__class__(uttt_bot.tree.root.__class__(uttt_state))
            uttt_bot.run(progress_bar=False)
            evaluated = uttt_bot.get_evaluated_actions()
            action = uttt_bot.select_action(evaluated, "argmax").index
        local_state = local_state.play(action)
        uttt_state.execute(Action(symbol=1 if uttt_state.is_next_symbol_X() else 2, index=action))
        local_tree.advance(action)
        uttt_bot.synchronize(uttt_state)

    elapsed = time.monotonic() - started
    return {
        "winner": local_state.result,
        "local_side": "X" if local_is_x else "O",
        "moves": 81 - local_state.cells.count(0),
        "seconds": elapsed,
        "opening_moves": list(map(int, opening_moves)),
        "local_actions": local_actions,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.utttai_root)
    sys.path.insert(0, str(root))

    from sttt.bots import CheckpointBot
    from sttt.tournament import paired_opening
    from utttpy.selfplay.neural_monte_carlo_tree_search import NeuralMonteCarloTreeSearch

    class LocalBot(CheckpointBot):
        def __init__(self):
            super().__init__(args.checkpoint, simulations=args.local_simulations, device="cpu")
            self.simulations = args.local_simulations

    official_net = ONNXPolicyValueNet(args.utttai_model)
    local_bot = LocalBot()
    results = []
    for game in range(args.games):
        pair = game // 2
        _, opening = paired_opening(args.seed, pair, args.opening_plies)
        local_is_x = game % 2 == 0
        uttt_bot = NeuralMonteCarloTreeSearch(
            uttt=__import__("utttpy.game.ultimate_tic_tac_toe", fromlist=["UltimateTicTacToe"]).UltimateTicTacToe(),
            num_simulations=args.utttai_simulations,
            exploration_strength=2.0,
            policy_value_net=official_net,
        )
        # run_game recreates the official root at each turn to keep the
        # reference search independent of the local search tree.
        result = run_game(local_bot, uttt_bot, opening, local_is_x, args.seed + game)
        result["game"] = game + 1
        result["result_for_local"] = (
            "win" if result["winner"] == (1 if local_is_x else -1)
            else "draw" if result["winner"] == 0 else "loss"
        )
        results.append(result)
        print(json.dumps({k: result[k] for k in ("game", "local_side", "result_for_local", "moves", "seconds")}), flush=True)

    local_bot.close()

    counts = {key: sum(r["result_for_local"] == key for r in results) for key in ("win", "draw", "loss")}
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "utttai_model": str(Path(args.utttai_model).resolve()),
        "games": args.games,
        "local_simulations": args.local_simulations,
        "utttai_simulations": args.utttai_simulations,
        "opening_plies": args.opening_plies,
        "seed": args.seed,
        "counts": counts,
        "score_rate": (counts["win"] + 0.5 * counts["draw"]) / args.games,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": counts, "score_rate": report["score_rate"]}))


if __name__ == "__main__":
    main()
