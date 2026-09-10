"""Persistent online Bayesian mixture of behavioral policies."""
import json
from pathlib import Path
import numpy as np

NAMES = ('random', 'center', 'corners', 'local-win', 'global-win')

def policies(state):
    legal = state.legal_actions()
    p = np.zeros((len(NAMES),81))
    p[:,legal] = 1
    for a in legal:
        b,c = divmod(a,9)
        child = state.play(a)
        p[1,a] += 4*(c == 4)
        p[2,a] += 3*(c in (0,2,6,8))
        p[3,a] += 20*(child.boards[b] == state.turn)
        p[4,a] += 100*(child.result == state.turn) + 4*(child.boards[b] == state.turn)
    return p / p.sum(axis=1,keepdims=True)

class Opponent:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.log_weights = np.zeros(len(NAMES))
        self.observations = 0
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            self.log_weights = np.array(data['log_weights'], dtype=float)
            if self.log_weights.shape != (len(NAMES),) or not np.isfinite(self.log_weights).all():
                raise ValueError('Invalid opponent profile')
            self.observations = data['observations']

    def weights(self):
        p = np.exp(self.log_weights-self.log_weights.max())
        return p/p.sum()

    def predict(self, state, base):
        # Conservative cap: behavioral experts are imperfect models of humans.
        trust = min(0.5, self.observations/100)
        return (1-trust)*base + trust*(self.weights() @ policies(state))

    def observe(self, state, action):
        if action not in state.legal_actions():
            raise ValueError('Cannot observe illegal move')
        self.log_weights = .98*self.log_weights + np.log(policies(state)[:,action]+1e-12)
        self.log_weights -= self.log_weights.max()
        self.observations += 1
        if self.path:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps({'log_weights':self.log_weights.tolist(),
                                        'observations':self.observations}))
            temp.replace(self.path)
