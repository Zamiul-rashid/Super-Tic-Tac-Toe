"""Official uttt.ai NMCTS/ONNX adapter and persistent line-protocol worker.

Run ``python -m sttt.utttai_wrapper --simulations 128 --protocol state_json``.
Each JSON request contains cells, boards, turn, forced and result. The response
is one action index. The action_index protocol is also supported for empty-board
external clients. Errors go to stderr and terminate the worker; no substitute bot.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from .engine_runtime import ROOT, activate_utttai
from .env import State

MODEL_SHA256 = '81161d2e262c9904dbb9a14961a2fcd043009fab4fd4f76b41451587f89c60d2'
DEFAULT_MODEL = ROOT / 'engines/models/utttai_stage2.onnx'


@lru_cache(maxsize=2)
def load_network(path):
    activate_utttai()
    import onnxruntime as ort
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError('uttt.ai model checksum mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=['CPUExecutionProvider'])

    class Network:
        device = torch.device('cpu')

        def __call__(self, x):
            outputs = session.run(None, {'input': x.detach().cpu().numpy()})
            policy = torch.from_numpy(outputs[0])
            return policy, torch.zeros_like(policy), torch.from_numpy(outputs[1]).reshape(-1)

    return Network()


def to_official(state):
    activate_utttai()
    from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe
    data = bytearray(93)
    data[:81] = bytes(2 if cell == -1 else cell for cell in state.cells)
    data[81:90] = bytes(2 if b == -1 else 3 if b == 2 else b for b in state.boards)
    data[90] = 1 if state.turn == 1 else 2
    data[91] = state.forced if state.forced >= 0 else 9
    data[92] = 0 if state.result is None else 3 if state.result == 0 else 1 if state.result == 1 else 2
    official = UltimateTicTacToe(data)
    official._verify_state()
    if set(official.get_legal_indexes()) != set(state.legal_actions()):
        raise ValueError('uttt.ai/local legal actions disagree')
    return official


class UTTTAIBot:
    def __init__(self, simulations=128, model=DEFAULT_MODEL, seed=0):
        if simulations < 2:
            raise ValueError('uttt.ai needs at least two simulations')
        self.simulations = simulations
        self.network = load_network(str(Path(model).resolve()))
        self.seed = seed
        self.reset()

    def reset(self):
        self.random_state = random.Random(self.seed).getstate()

    def choose(self, state, rng=None):
        from utttpy.selfplay.neural_monte_carlo_tree_search import NeuralMonteCarloTreeSearch
        if state.result is not None:
            raise ValueError('Cannot choose a move in a terminal state')
        # Fresh root matches the previous reference benchmark: the budget is
        # total simulations this move, independent of inherited tree visits.
        search = NeuralMonteCarloTreeSearch(to_official(state), self.simulations, 2., self.network)
        previous = random.getstate()
        try:
            random.setstate(self.random_state)
            search.run(progress_bar=False)
            action = search.select_action(search.get_evaluated_actions(), 'argmax').index
            self.random_state = random.getstate()
        finally:
            random.setstate(previous)
        if action not in state.legal_actions():
            raise ValueError('uttt.ai returned an illegal move')
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulations', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--protocol', choices=['state_json', 'action_index'], default='state_json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    bot = UTTTAIBot(args.simulations, args.model, args.seed)
    state = State()
    for line in sys.stdin:
        if args.protocol == 'state_json':
            data = json.loads(line)
            state = State(tuple(data['cells']), tuple(data['boards']),
                          data['turn'], data['forced'], data['result'])
        else:
            previous = int(line)
            if previous != -1:
                state = state.play(previous)
            count = int(sys.stdin.readline())
            legal = {int(sys.stdin.readline()) for _ in range(count)}
            if legal != set(state.legal_actions()):
                raise ValueError('Action protocol state/legal moves disagree')
        action = bot.choose(state)
        state = state.play(action)
        print(action, flush=True)


if __name__ == '__main__':
    main()
