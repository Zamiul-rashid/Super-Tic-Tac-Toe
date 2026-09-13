"""Immutable Ultimate Tic-Tac-Toe; completed boards are closed."""
from dataclasses import dataclass

LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

def winner(cells):
    for a,b,c in LINES:
        if cells[a] in (-1, 1) and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return 0

def state_key(state):
    """Immutable canonical identity for a state from any backend.

    ``State`` is a frozen dataclass and is hashable, but ``FastState`` and
    ``CppState`` are mutable (``play_inplace``) and deliberately are not. Key
    dictionaries and sets on this tuple instead; it compares equal across all
    three backends, so keys built from one are found by another.
    """
    result = state.result
    return (tuple(state.cells), tuple(state.boards), int(state.turn),
            int(state.forced), None if result is None else int(result))


@dataclass(frozen=True)
class State:
    cells: tuple = (0,) * 81
    boards: tuple = (0,) * 9
    turn: int = 1
    forced: int = -1
    result: int | None = None

    def legal_actions(self):
        if self.result is not None:
            return []
        boards = range(9) if self.forced == -1 else (self.forced,)
        return [b*9+c for b in boards if self.boards[b] == 0
                for c in range(9) if self.cells[b*9+c] == 0]

    def play(self, action):
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")
        cells, boards = list(self.cells), list(self.boards)
        b, c = divmod(action, 9)
        cells[action] = self.turn
        local = cells[b*9:b*9+9]
        boards[b] = winner(local) or (2 if all(local) else 0)
        win = winner(boards)
        result = win if win else (0 if all(boards) else None)
        return State(tuple(cells), tuple(boards), -self.turn,
                     c if boards[c] == 0 else -1, result)

    def render(self):
        symbols = {0: '.', 1: 'X', -1: 'O', 2: '='}
        rows = []
        for br in range(3):
            for r in range(3):
                rows.append(' | '.join(' '.join(symbols[self.cells[(br*3+bc)*9+r*3+c]]
                                               for c in range(3)) for bc in range(3)))
            if br < 2:
                rows.append('------+-------+------')
        rows.append('Local boards: ' + ' '.join(symbols[b] for b in self.boards))
        rows.append('Next board: ' + ('any open board' if self.forced < 0 else str(self.forced+1)))
        return '\n'.join(rows)
