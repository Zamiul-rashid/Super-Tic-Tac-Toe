"""PUCT with optional sampled human responses; values use player-to-move perspective."""
import math
import numpy as np

class Node:
    def __init__(self, state, prior=1.):
        self.state, self.prior = state, prior
        self.n, self.total = 0, 0.
        self.children = {}

def search(state, model, simulations, rng, opponent=None, noise=False):
    if simulations < 1 or state.result is not None:
        raise ValueError('Search requires a nonterminal state and positive simulations')
    root = Node(state)

    def expand(node):
        p,v = model.evaluate(node.state)
        if opponent is not None and node.state.turn != state.turn:
            p = opponent.predict(node.state,p)
        for a in node.state.legal_actions():
            node.children[a] = Node(node.state.play(a),p[a])
        return v

    expand(root)
    if noise:
        for child,eta in zip(root.children.values(),rng.dirichlet([.3]*len(root.children))):
            child.prior = .75*child.prior + .25*eta
    for _ in range(simulations):
        node, path = root, [root]
        while node.children:
            children = list(node.children.values())
            if opponent is not None and node.state.turn != state.turn:
                p = np.array([c.prior for c in children]); p /= p.sum()
                node = children[rng.choice(len(children),p=p)]
            else:
                node = max(children,key=lambda c: -c.total/max(1,c.n) +
                           1.5*c.prior*math.sqrt(node.n+1)/(1+c.n))
            path.append(node)
        value = (node.state.result*node.state.turn if node.state.result is not None
                 else expand(node))
        for visited in reversed(path):
            visited.n += 1
            visited.total += value
            value = -value
    counts = np.zeros(81,dtype=np.float32)
    for a,child in root.children.items():
        counts[a] = child.n
    return counts/counts.sum()
