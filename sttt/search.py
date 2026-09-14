"""Reusable, batched PUCT. Node values use the player-to-move perspective."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SearchConfig:
    soft_pruning: bool = True
    proofs: bool = True
    reuse: bool = True
    c_puct: float = 1.5
    soft_margin: float = .2
    soft_strength: float = 3.
    min_visits: int = 8
    revisit_interval: int = 64


class Node:
    def __init__(self, state, prior=1.):
        self.state, self.prior = state, prior
        self.n, self.total = 0, 0.
        self.children = {}
        self.pending = False
        self.in_flight = 0
        self.last_selected = self.selections = 0
        # Neural predictions must NEVER set this field.
        self.solved = state.result * state.turn if state.result is not None else None


def root_action_values(root):
    """Q(s, a) targets from a finished search: mean backed-up value of each
    visited root child from the root player's view, and which actions were
    visited. Works for both `Node` and the native `FastNode` (same `n`,
    `total`, `solved`, `children` attributes). Proven children report their
    exact proof value. Unvisited and illegal actions are 0 / False.
    """
    q = np.zeros(81, dtype=np.float32)
    visited = np.zeros(81, dtype=bool)
    for action, child in root.children.items():
        solved = getattr(child, 'solved', None)
        if solved is not None:
            q[action] = -float(solved)
            visited[action] = True
        elif child.n > 0:
            q[action] = -float(child.total) / child.n
            visited[action] = True
    return q, visited


class TreeSearch:
    def __init__(self, model, rng=None, config=None, opponent=None, agent_side=None):
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng()
        self.config = config or SearchConfig()
        self.opponent, self.agent_side = opponent, agent_side
        self.root = None
        self.stats, self.root_priors = {}, {}

    @property
    def use_proofs(self):
        # Minimax losses need not be losses against a fallible human policy.
        return self.config.proofs and self.opponent is None

    def reset(self):
        self.root = None

    def advance(self, action):
        """Retain the played subtree, release siblings. Advance on BOTH players' moves."""
        if self.root is not None:
            child = self.root.children.get(action)
            self.root = child if child is not None else Node(self.root.state.play(action))

    def _prove(self, node):
        if not self.use_proofs or not node.children or node.solved is not None:
            return
        values = [c.solved for c in node.children.values()]
        if -1 in values:
            node.solved = 1
        elif all(v is not None for v in values):
            node.solved = max(-v for v in values)

    def _expand(self, node, policy):
        legal = node.state.legal_actions()
        policy = np.asarray(policy, dtype=float)
        if policy.shape != (81,) or not np.isfinite(policy).all() or (policy < 0).any():
            raise ValueError('Evaluator returned an invalid policy')
        if self.opponent is not None and node.state.turn != self.agent_side:
            policy = self.opponent.predict(node.state, policy)
        probabilities = np.maximum(policy[legal], 1e-5)
        probabilities /= probabilities.sum()
        node.children = {a: Node(node.state.play(a), p) for a, p in zip(legal, probabilities)}
        self._prove(node)

    def _eligible(self, node):
        children = list(node.children.values())
        if self.use_proofs:
            if node.solved is not None:
                return [c for c in children if c.solved == -node.solved]
            alternatives = [c for c in children if c.solved != 1]
            if alternatives:
                self.stats['hard_pruned_choices'] += len(children) - len(alternatives)
                children = alternatives
        return children

    def _q(self, child):
        if self.use_proofs and child.solved is not None:
            return -child.solved
        # Virtual losses spread pending work across branches.
        return -(child.total + child.in_flight) / max(1, child.n + child.in_flight)

    def _select(self, node, path):
        if node.pending:
            return None
        if node.state.result is not None or (self.use_proofs and node.solved is not None) or not node.children:
            return path
        children = self._eligible(node)
        while children:
            if self.opponent is not None and node.state.turn != self.agent_side:
                weights = np.array([c.prior for c in children]); weights /= weights.sum()
                child = children[self.rng.choice(len(children), p=weights)]
            elif (self.config.soft_pruning and node.selections > 0
                  and node.selections % self.config.revisit_interval == 0):
                child = min(children, key=lambda c: (c.last_selected, c.n + c.in_flight))
                self.stats['soft_rechecks'] += 1
            else:
                reliable = [self._q(c) for c in children if c.n >= self.config.min_visits]
                best = max(reliable) if reliable else None

                def priority(c):
                    q, weight = self._q(c), 1.
                    if self.config.soft_pruning and best is not None and c.n >= self.config.min_visits:
                        gap = max(0., best - q - self.config.soft_margin)
                        weight = max(.2, 1. / (1. + self.config.soft_strength * gap))
                    prior = self.root_priors.get(id(c), c.prior) if node is self.root else c.prior
                    return q + self.config.c_puct * prior * weight * math.sqrt(node.n + node.in_flight + 1) / (1 + c.n + c.in_flight)

                child = max(children, key=priority)
            node.selections += 1
            child.last_selected = node.selections
            selected = self._select(child, path + [child])
            if selected is not None:
                return selected
            children.remove(child)
        return None

    def _backup(self, path, value):
        for node in reversed(path):
            self._prove(node)
            if self.use_proofs and node.solved is not None:
                value = node.solved
            node.n += 1
            node.total += value
            value = -value
        self.stats['completed_simulations'] += 1
        self.stats['max_depth'] = max(self.stats['max_depth'], len(path) - 1)

    @staticmethod
    def _release(path):
        path[-1].pending = False
        for node in path:
            node.in_flight -= 1

    def run(self, state, simulations, batch_size=1, noise=False):
        if simulations < 1 or batch_size < 1 or state.result is not None:
            raise ValueError('Search requires a nonterminal state and positive budgets')
        if self.root is None or self.root.state != state or not self.config.reuse:
            self.root = Node(state)
        if self.agent_side is None:
            self.agent_side = state.turn
        self.stats = dict(completed_simulations=0, neural_positions=0, inference_batches=0,
                          max_depth=0, hard_pruned_choices=0, soft_rechecks=0,
                          retained_visits=self.root.n)
        if not self.root.children:
            policy, _ = self.model.evaluate(state)
            self._expand(self.root, policy)
            self.stats['neural_positions'] += 1
            self.stats['inference_batches'] += 1
        self.root_priors = {}
        if noise:
            children = list(self.root.children.values())
            for child, eta in zip(children, self.rng.dirichlet([.3] * len(children))):
                self.root_priors[id(child)] = .75 * child.prior + .25 * eta
        while self.stats['completed_simulations'] < simulations:
            if self.use_proofs and self.root.solved is not None:
                break
            pending = []
            while len(pending) < batch_size and self.stats['completed_simulations'] + len(pending) < simulations:
                path = self._select(self.root, [self.root])
                if path is None:
                    break
                leaf = path[-1]
                if leaf.state.result is not None or (self.use_proofs and leaf.solved is not None):
                    self._backup(path, leaf.solved)
                    if self.use_proofs and self.root.solved is not None:
                        break
                else:
                    leaf.pending = True
                    for node in path:
                        node.in_flight += 1
                    pending.append(path)
            if pending:
                try:
                    states = [path[-1].state for path in pending]
                    outputs = (self.model.evaluate_many(states) if hasattr(self.model, 'evaluate_many')
                               else [self.model.evaluate(s) for s in states])
                    if len(outputs) != len(pending):
                        raise ValueError('Evaluator batch length mismatch')
                    self.stats['inference_batches'] += 1
                    self.stats['neural_positions'] += len(pending)
                    for path, (policy, value) in zip(pending, outputs):
                        if not np.isfinite(value) or not -1.00001 <= value <= 1.00001:
                            raise ValueError('Evaluator returned an invalid value')
                        self._expand(path[-1], policy)
                        self._backup(path, float(value))
                finally:
                    for path in pending:
                        self._release(path)
        self.stats['root_solved'] = self.root.solved if self.use_proofs else None
        eligible = self._eligible(self.root)
        probabilities = np.zeros(81, dtype=np.float64)
        allowed = {id(c) for c in eligible}
        for a, child in self.root.children.items():
            if id(child) in allowed:
                probabilities[a] = child.n
        if probabilities.sum() == 0:
            for a, child in self.root.children.items():
                if id(child) in allowed:
                    probabilities[a] = child.prior
        return (probabilities / probabilities.sum()).astype(np.float32)


def search(state, model, simulations, rng, opponent=None, noise=False):
    """Compatibility entry point. Use TreeSearch for reuse and batched evaluation."""
    return TreeSearch(model, rng, opponent=opponent).run(state, simulations, noise=noise)


try:
    from .cpp_env import CppTreeSearch
except ImportError:
    CppTreeSearch = None
