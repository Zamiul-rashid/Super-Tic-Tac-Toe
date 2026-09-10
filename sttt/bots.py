"""Independent, deterministic tactical and bounded alpha-beta baselines."""
import math
from .env import LINES


def _lines_value(cells, player, weights):
    value = 0.
    for line in LINES:
        values = [cells[i] for i in line]
        if 2 in values:
            continue  # A drawn global board cannot complete a winning line.
        own, other = values.count(player), values.count(-player)
        if not other:
            value += weights[own]
        if not own:
            value -= weights[other]
    return value


def value(state):
    """Zero-sum heuristic. Cutoff evaluations are estimates, never search proofs."""
    if state.result is not None:
        return state.result * state.turn
    score = _lines_value(state.boards, state.turn, (0, 3, 16, 100))
    score += 5 * (state.boards.count(state.turn) - state.boards.count(-state.turn))
    for b in range(9):
        if state.boards[b] == 0:
            score += _lines_value(state.cells[b*9:b*9+9], state.turn, (0, .15, 1., 4.))
    return .95 * math.tanh(score / 30.)


class BudgetExceeded(Exception):
    pass


class AlphaBetaBot:
    def __init__(self, depth=3, node_budget=3000):
        self.depth, self.node_budget = depth, node_budget
        self.nodes = 0

    def _negamax(self, state, depth, alpha, beta):
        self.nodes += 1
        if self.nodes > self.node_budget:
            raise BudgetExceeded()
        if state.result is not None or depth == 0:
            return value(state)
        best = -2.
        children = [state.play(a) for a in state.legal_actions()]
        children.sort(key=lambda s: -value(s), reverse=True)
        for child in children:
            score = -self._negamax(child, depth-1, -beta, -alpha)
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def choose(self, state, rng):
        legal = state.legal_actions()
        if not legal:
            raise ValueError('Cannot choose a move in a terminal state')
        self.nodes = 0
        children = [(a, state.play(a)) for a in legal]
        rng.shuffle(children)
        wins = [a for a, s in children if s.result == state.turn]
        if wins:
            return wins[0]
        # Exact one-reply safety screening, including the forced-board routing rule.
        safe = [(a, child) for a, child in children
                if not any(child.play(reply).result == -state.turn for reply in child.legal_actions())]
        candidates = safe or children
        candidates.sort(key=lambda pair: -value(pair[1]), reverse=True)
        chosen = candidates[0][0]
        for depth in range(1, self.depth+1):
            best, best_action, alpha = -2., chosen, -2.
            try:
                for a, child in candidates:
                    score = -self._negamax(child, depth-1, -2., -alpha)
                    if score > best:
                        best, best_action = score, a
                    alpha = max(alpha, score)
            except BudgetExceeded:
                break  # Only completed depths replace the previous decision.
            chosen = best_action
            candidates.sort(key=lambda pair: pair[0] != chosen)
            if best == 1:
                break
        return chosen


class TacticalBot(AlphaBetaBot):
    """Two-ply replies: immediate wins, global-loss avoidance and local threat blocking."""
    def __init__(self):
        super().__init__(depth=2, node_budget=10000)
