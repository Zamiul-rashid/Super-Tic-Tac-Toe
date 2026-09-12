"""Uniform Bot subsystem, baseline search bots, external engine adapter, and factory."""
import abc
import json
import math
import os
from pathlib import Path
import select
import shlex
import subprocess
import time
import numpy as np
import torch

from .env import LINES, State
from .opponent import policies, NAMES
from .search import TreeSearch, SearchConfig
from .learning import load_model


def action_to_coord(action: int) -> tuple[int, int]:
    """Convert global action index (0..80) to 2D grid coordinates (row, col) in 0..8."""
    b, c = divmod(action, 9)
    br, bc = divmod(b, 3)
    cr, cc = divmod(c, 3)
    return br * 3 + cr, bc * 3 + cc


def coord_to_action(row: int, col: int) -> int:
    """Convert 2D grid coordinates (row, col) in 0..8 to global action index (0..80)."""
    b = (row // 3) * 3 + (col // 3)
    c = (row % 3) * 3 + (col % 3)
    return b * 9 + c


def sttt_to_openspiel_action(action: int) -> int:
    """Legacy flat-grid coordinate mapping, not official OpenSpiel decisions.

    Official OpenSpiel uses board/cell actions; use OpenSpielBot or the state
    replay helpers for that engine. Kept only for old flat-grid consumers.
    """
    if not (0 <= action < 81):
        raise ValueError(f"Invalid sttt action {action}: must be in range 0..80")
    row, col = action_to_coord(action)
    return row * 9 + col


def openspiel_to_sttt_action(action: int) -> int:
    """Inverse legacy flat-grid mapping; not the official decision protocol."""
    if not (0 <= action < 81):
        raise ValueError(f"Invalid OpenSpiel action {action}: must be in range 0..80")
    row, col = divmod(action, 9)
    return coord_to_action(row, col)


def _verify_action_sequence(state: State, actions: list[int] | None) -> bool:
    """Check whether actions played sequentially from State() legally reach state."""
    if not actions:
        return state == State()
    try:
        s = State()
        for a in actions:
            if not (0 <= a < 81) or a not in s.legal_actions():
                return False
            s = s.play(a)
        return s == state
    except Exception:
        return False


def _find_action_sequence(
    start: State, target: State, max_depth: int = 10, node_budget: int = 2000
) -> list[int] | None:
    """Find a sequence of legal actions from start reaching target State with search bounds."""
    if start == target:
        return []
    start_cells = start.cells
    target_cells = target.cells
    for i in range(81):
        if start_cells[i] != 0 and target_cells[i] != start_cells[i]:
            return None
    diff_x = sum(1 for i, c in enumerate(target_cells) if c == 1 and start_cells[i] != 1)
    diff_o = sum(1 for i, c in enumerate(target_cells) if c == -1 and start_cells[i] != -1)
    moves_needed = diff_x + diff_o
    if moves_needed > max_depth or moves_needed <= 0:
        return None
    expected_turn = start.turn if (moves_needed % 2 == 0) else -start.turn
    if target.turn != expected_turn:
        return None
    if start.turn == 1 and not (diff_x == diff_o or diff_x == diff_o + 1):
        return None
    if start.turn == -1 and not (diff_o == diff_x or diff_o == diff_x + 1):
        return None

    visited = 0

    def dfs(curr: State, moves_left: int) -> list[int] | None:
        nonlocal visited
        visited += 1
        if visited > node_budget:
            return None
        if moves_left == 0:
            return [] if curr == target else None
        legal = curr.legal_actions()
        turn = curr.turn
        candidates = [a for a in legal if target_cells[a] == turn]
        for a in candidates:
            nxt = curr.play(a)
            b = a // 9
            if nxt.boards[b] != 0 and nxt.boards[b] != target.boards[b]:
                continue
            if nxt.result is not None and moves_left > 1:
                continue
            res = dfs(nxt, moves_left - 1)
            if res is not None:
                return [a] + res
        return None

    return dfs(start, moves_needed)


def _find_action_sequence_to_state(
    target: State, max_depth: int = 10, node_budget: int = 2000
) -> list[int] | None:
    """Find a sequence of legal actions from State() reaching target State."""
    return _find_action_sequence(State(), target, max_depth=max_depth, node_budget=node_budget)


def sttt_to_openspiel_state(state: State, game=None, actions=None):
    """Replay local moves into official OpenSpiel's board/cell decisions."""
    from .engine_runtime import activate_runtime
    activate_runtime()
    import pyspiel
    if game is None:
        game = pyspiel.load_game('ultimate_tic_tac_toe')
    if actions is None:
        actions = _find_action_sequence_to_state(state, max_depth=8, node_budget=4000)
    if actions is None or not _verify_action_sequence(state, actions):
        raise ValueError('Provide a complete legal action history for this position')
    result = game.new_initial_state()
    local = State()
    for action in actions:
        board, cell = divmod(action, 9)
        if local.forced < 0:
            result.apply_action(board)
        result.apply_action(cell)
        local = local.play(action)
    return result


def openspiel_to_sttt_state(os_state) -> State:
    """Convert official history, coalescing board selection and cell moves."""
    local = State()
    board = None
    for decision in os_state.history():
        if local.forced < 0 and board is None:
            board = decision
            continue
        action = (local.forced if local.forced >= 0 else board) * 9 + decision
        local = local.play(action)
        board = None
    if board is not None:
        raise ValueError('OpenSpiel is between board and cell selection')
    return local


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


class Bot(abc.ABC):
    """Abstract base class for all Super Tic-Tac-Toe bots."""

    @property
    def name(self) -> str:
        """Human-readable identifier for reporting and tournament matrices."""
        return getattr(self, "_name", None) or self.__class__.__name__

    @abc.abstractmethod
    def choose(self, state: State, rng: np.random.Generator) -> int:
        """Select a legal action (0..80) for the current state.

        Args:
            state: Current immutable State.
            rng: NumPy random Generator for reproducible stochastic choices.

        Returns:
            An integer action index in state.legal_actions().

        Raises:
            ValueError: If called on a terminal state (state.result is not None).
        """
        raise NotImplementedError

    def advance(self, action: int) -> None:
        """Optional hook to advance internal state or MCTS subtree."""
        pass

    def reset(self) -> None:
        """Optional hook to reset internal state between games."""
        pass

    def close(self) -> None:
        """Optional hook to release external resources, sub-processes, or threads."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AlphaBetaBot(Bot):
    def __init__(self, depth=3, node_budget=3000, name=None):
        self.depth, self.node_budget = depth, node_budget
        self.nodes = 0
        self._name = name or f"alphabeta-d{depth}"

    @property
    def name(self) -> str:
        return self._name

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
        if not legal or state.result is not None:
            raise ValueError("Cannot choose a move in a terminal state")
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
    def __init__(self, name=None):
        super().__init__(depth=2, node_budget=10000, name=name or "tactical")


class ThreatBlockBot(Bot):
    """Adversarial threat bot prioritizing wins and denying immediate replies.

    This is deliberately different from the value-driven AlphaBeta bot: it
    first screens every move for an opponent's immediate global win, then
    ranks the surviving moves by immediate counter-threats and the normal
    heuristic. Random ties make it useful as a family of blocking variants.
    """

    def __init__(self, node_budget=12000, name="threat-block"):
        self.node_budget = int(node_budget)
        self._name = name

    def _immediate_wins(self, state, player):
        return [action for action in state.legal_actions()
                if state.play(action).result == player]

    def choose(self, state, rng):
        legal = state.legal_actions()
        if not legal or state.result is not None:
            raise ValueError("Cannot choose a move in a terminal state")

        wins = self._immediate_wins(state, state.turn)
        if wins:
            return int(rng.choice(wins))

        children = [(action, state.play(action)) for action in legal]
        safe = []
        for action, child in children:
            if not self._immediate_wins(child, -state.turn):
                safe.append((action, child))
        candidates = safe or children

        scored = []
        for action, child in candidates:
            own_threats = sum(
                [child.boards[i] for i in line].count(state.turn) == 2
                and [child.boards[i] for i in line].count(0) == 1
                for line in LINES)
            opponent_threats = len(self._immediate_wins(child, -state.turn))
            local_win = child.boards[action // 9] == state.turn
            score = -value(child) + 0.5 * own_threats - 0.75 * opponent_threats
            score += 0.2 if local_win else 0.
            scored.append((score, action))
        best = max(score for score, _ in scored)
        ties = [action for score, action in scored if math.isclose(score, best, abs_tol=1e-12)]
        return int(rng.choice(ties))


class StyleBot(Bot):
    """Wraps behavioral policies ('center', 'corners', 'local-win', 'global-win', 'random')."""

    def __init__(self, style: str = "center", deterministic: bool = False, name: str | None = None):
        if style not in NAMES:
            raise ValueError(f"Unknown style '{style}'. Must be one of {NAMES}")
        self.style = style
        self.style_idx = NAMES.index(style)
        self.deterministic = bool(deterministic)
        self._name = name or f"style-{style}"

    @property
    def name(self) -> str:
        return self._name

    def choose(self, state: State, rng: np.random.Generator) -> int:
        legal = state.legal_actions()
        if not legal or state.result is not None:
            raise ValueError("Cannot choose a move in a terminal state")

        p = policies(state)[self.style_idx]
        if self.deterministic:
            return max(legal, key=lambda a: p[a])
        else:
            return int(rng.choice(81, p=p))


class ExternalProcessBot(Bot):
    """External process adapter supporting CodinGame and integer action protocols.

    Communicates via subprocess stdin/stdout pipes with strict timeout enforcement,
    process crash/EOF detection, invalid output handling, and tactical fallback moves.
    """

    def __init__(
        self,
        command: list[str] | str,
        timeout: float = 5.0,
        protocol: str = "codingame",
        fallback: str = "tactical",
        auto_restart: bool = True,
        restart_on_reset: bool = True,
        cwd: str | None = None,
        name: str | None = None,
    ):
        self.process: subprocess.Popen | None = None
        self.cmd = shlex.split(command) if isinstance(command, str) else list(command)
        if not self.cmd:
            raise ValueError("Command cannot be empty")
        self.timeout = max(0.001, min(float(timeout), 5.0))
        self.protocol = protocol
        self.fallback = fallback
        self.auto_restart = auto_restart
        self.restart_on_reset = restart_on_reset
        self.cwd = cwd
        self._name = name or Path(self.cmd[0]).name
        self.last_state: State | None = None
        self.last_action: int | None = None
        self._crashed_in_game: bool = False
        self.tactical_fallback = TacticalBot()
        self._start_process()

    @property
    def name(self) -> str:
        return self._name

    def _start_process(self) -> None:
        self.close()
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.cwd,
            )
            if self.process.stdout is not None:
                try:
                    os.set_blocking(self.process.stdout.fileno(), False)
                except Exception:
                    pass
            self._crashed_in_game = False
        except Exception as e:
            self.process = None
            self._crashed_in_game = True
            if self.fallback == "raise":
                raise e

    def close(self) -> None:
        if getattr(self, "process", None) is not None:
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=0.5)
            except Exception:
                pass
            finally:
                if getattr(self, "process", None) is not None:
                    for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                        if pipe is not None:
                            try:
                                pipe.close()
                            except Exception:
                                pass
                self.process = None

    def __del__(self):
        self.close()

    def reset(self) -> None:
        self.last_state = None
        self.last_action = None
        self._crashed_in_game = False
        if self.restart_on_reset or (self.process is not None and self.process.poll() is not None):
            self.close()

    def advance(self, action: int) -> None:
        self.last_action = action
        if self.last_state is not None:
            try:
                self.last_state = self.last_state.play(action)
            except Exception:
                self.last_state = None

    def _fallback_move(self, state: State, rng: np.random.Generator, reason: str) -> int:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("Cannot choose a move in a terminal state")
        if self.fallback == "raise":
            raise RuntimeError(f"External bot '{self.name}' failed ({reason})")
        if self.fallback == "tactical":
            try:
                return self.tactical_fallback.choose(state, rng)
            except Exception:
                return int(rng.choice(legal))
        return int(rng.choice(legal))

    def _determine_opp_action(self, state: State) -> int:
        if self.last_state is not None:
            diffs = [i for i in range(81) if self.last_state.cells[i] == 0 and state.cells[i] != 0]
            if len(diffs) == 1:
                return diffs[0]
            for a in self.last_state.legal_actions():
                if self.last_state.play(a) == state:
                    return a
        if self.last_action is not None:
            return self.last_action
        # First move of game: check if board already has exactly 1 mark
        non_zero = [i for i, c in enumerate(state.cells) if c != 0]
        if len(non_zero) == 1:
            return non_zero[0]
        return -1

    def choose(self, state: State, rng: np.random.Generator) -> int:
        if state.result is not None:
            raise ValueError("Cannot choose a move in a terminal state")
        legal = state.legal_actions()
        if not legal:
            raise ValueError("Cannot choose a move in a terminal state")

        # Automatically reset crashed status if playing from an initial clean board
        if all(c == 0 for c in state.cells):
            self._crashed_in_game = False

        if self.process is None or self.process.poll() is not None:
            if self.auto_restart and not self._crashed_in_game:
                self._start_process()
            if self.process is None or self.process.poll() is not None:
                return self._fallback_move(state, rng, "process not running")

        try:
            opp_action = self._determine_opp_action(state)

            if self.protocol == 'state_json':
                payload = json.dumps(dict(cells=state.cells, boards=state.boards,
                                          turn=state.turn, forced=state.forced,
                                          result=state.result)) + '\n'
            elif self.protocol == "codingame":
                opp_r, opp_c = (-1, -1) if opp_action == -1 else action_to_coord(opp_action)
                payload = f"{opp_r} {opp_c}\n{len(legal)}\n"
                for a in legal:
                    r, c = action_to_coord(a)
                    payload += f"{r} {c}\n"
            elif self.protocol in ("action", "action_index"):
                payload = f"{opp_action}\n{len(legal)}\n"
                for a in legal:
                    payload += f"{a}\n"
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")

            try:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self.close()
                self._crashed_in_game = True
                return self._fallback_move(state, rng, f"broken pipe: {e}")

            if self.process.stdout is None:
                self.close()
                self._crashed_in_game = True
                return self._fallback_move(state, rng, "stdout pipe unavailable")

            try:
                fd = self.process.stdout.fileno()
                os.set_blocking(fd, False)
            except Exception:
                self.close()
                self._crashed_in_game = True
                return self._fallback_move(state, rng, "stdout pipe unavailable")

            deadline = time.monotonic() + self.timeout
            buf = ""
            line = None
            timed_out = False

            while time.monotonic() < deadline:
                rem = max(0.0, deadline - time.monotonic())
                try:
                    rlist, _, _ = select.select([fd], [], [], rem)
                except (ValueError, OSError):
                    break
                if not rlist:
                    timed_out = True
                    break
                try:
                    raw = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                except OSError:
                    break
                if not raw:  # EOF / process died
                    break
                buf += raw.decode("utf-8", errors="replace")
                if "\n" in buf:
                    line = buf.split("\n", 1)[0]
                    break

            if line is None:
                self.close()
                self._crashed_in_game = True
                reason = "timeout" if (timed_out or time.monotonic() >= deadline) else "process crash / EOF"
                return self._fallback_move(state, rng, reason)

            parts = line.strip().split()
            if not parts:
                self.close()
                self._crashed_in_game = True
                return self._fallback_move(state, rng, "empty output")

            candidate_action = None
            if len(parts) == 1:
                try:
                    candidate_action = int(parts[0])
                except ValueError:
                    pass
            elif len(parts) >= 2:
                if self.protocol in ("action", "action_index"):
                    try:
                        cand = int(parts[0])
                        if cand in legal:
                            candidate_action = cand
                    except ValueError:
                        pass
                if candidate_action is None:
                    try:
                        row, col = int(parts[0]), int(parts[1])
                        if 0 <= row <= 8 and 0 <= col <= 8:
                            candidate_action = coord_to_action(row, col)
                    except ValueError:
                        pass
                if candidate_action is None:
                    try:
                        candidate_action = int(parts[0])
                    except ValueError:
                        pass

            if candidate_action is None or candidate_action not in legal:
                self.close()
                self._crashed_in_game = True
                return self._fallback_move(state, rng, f"invalid/illegal action: {line.strip()}")

            self.last_action = candidate_action
            self.last_state = state.play(candidate_action)
            return candidate_action

        except Exception as e:
            self.close()
            self._crashed_in_game = True
            return self._fallback_move(state, rng, f"exception: {e}")


class CheckpointBot(Bot):
    """MCTS bot powered by a saved neural network checkpoint.

    Loads model weights on CPU to avoid interfering with active GPU training runs.
    Maintains TreeSearch instance and supports subtree reuse across turns.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        simulations: int = 512,
        leaf_batch: int = 16,
        device: str = "cpu",
        config: SearchConfig | None = None,
        name: str | None = None,
        checkpoint_path: str | Path | None = None,
    ):
        p = path or checkpoint_path
        if p is None:
            raise ValueError("Must provide path to checkpoint")
        self.path = Path(p).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.path}")

        self.model, self.checkpoint_data = load_model(str(self.path))
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.simulations = simulations
        self.leaf_batch = leaf_batch
        self.config = config or SearchConfig()
        iteration = self.checkpoint_data.get("iteration")
        self._name = name or (f"ckpt-iter{iteration:04d}" if iteration is not None else self.path.stem)
        self.tree: TreeSearch | None = None

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self.tree = None

    def advance(self, action: int) -> None:
        if self.tree is not None:
            self.tree.advance(action)

    def choose(self, state: State, rng: np.random.Generator) -> int:
        if state.result is not None:
            raise ValueError("Cannot choose a move in a terminal state")
        legal = state.legal_actions()
        if not legal:
            raise ValueError("Cannot choose a move in a terminal state")

        if self.tree is None or self.tree.root is None or self.tree.root.state != state:
            matched = False
            if self.tree is not None and self.tree.root is not None:
                for child_action, child_node in self.tree.root.children.items():
                    if child_node.state == state:
                        self.tree.root = child_node
                        matched = True
                        break
            if not matched:
                self.tree = TreeSearch(self.model, rng=rng, config=self.config)
        else:
            self.tree.rng = rng

        pi = self.tree.run(state, self.simulations, batch_size=self.leaf_batch)
        action = int(pi.argmax())
        self.tree.advance(action)
        return action


class OpenSpielBot(Bot):
    """Official OpenSpiel MCTS; budgets are per board/cell decision."""

    def __init__(self, algorithm="mcts", simulations=100, uct_c=1.414,
                 name=None, seed=None, **kwargs):
        from .engine_runtime import activate_runtime
        activate_runtime()
        import pyspiel
        if Path(pyspiel.__file__).suffix == '.py':
            raise RuntimeError('Official OpenSpiel extension required')
        if algorithm not in ('mcts', 'random', 'uniform_random'):
            raise ValueError(f'Unsupported OpenSpiel algorithm: {algorithm}')
        self.algorithm, self.simulations = algorithm, max(2, int(simulations))
        self.seed, self.uct_c = seed, uct_c
        self._name = name or f'openspiel-{algorithm}-{self.simulations}'
        self.game = pyspiel.load_game('ultimate_tic_tac_toe')
        self.reset()

    def reset(self):
        from open_spiel.python.algorithms import mcts
        self.os_state = self.game.new_initial_state()
        self.local_state = State()
        self.history = []
        random_state = np.random.RandomState(self.seed)
        self._bot = mcts.MCTSBot(
            self.game, self.uct_c, self.simulations,
            mcts.RandomRolloutEvaluator(1, random_state=random_state),
            solve=True, random_state=random_state)

    def advance(self, action):
        action = int(action)
        child = self.local_state.play(action)
        board, cell = divmod(action, 9)
        if self.local_state.forced < 0:
            self.os_state.apply_action(board)
        self.os_state.apply_action(cell)
        self.history.append(action)
        self.local_state = child
        if self.os_state.is_terminal() != (child.result is not None):
            raise ValueError('OpenSpiel/local termination mismatch')
        if child.result is not None and self.os_state.returns()[0] != child.result:
            raise ValueError('OpenSpiel/local result mismatch')

    def _sync_state(self, state):
        if state == self.local_state:
            return
        sequence = _find_action_sequence(self.local_state, state, max_depth=8, node_budget=4000)
        if sequence is None:
            sequence = _find_action_sequence_to_state(state, max_depth=8, node_budget=4000)
            if sequence is None:
                raise ValueError('OpenSpiel needs advance() calls or a replayable opening')
            self.reset()
        for action in sequence:
            self.advance(action)

    def choose(self, state, rng=None):
        if state.result is not None:
            raise ValueError('Cannot choose from a terminal state')
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        self._sync_state(state)
        if self.seed is None:
            random_state = np.random.RandomState(int(rng.integers(2**31 - 1)))
            self._bot.random_state = random_state
            self._bot.evaluator.random_state = random_state
        work = self.os_state.clone()

        def decision():
            if self.algorithm != 'mcts':
                return int(rng.choice(work.legal_actions()))
            return int(self._bot.step(work))

        board = state.forced
        if board < 0:
            board = decision()
            work.apply_action(board)
        cell = decision()
        action = 9 * board + cell
        if action not in state.legal_actions():
            raise ValueError('OpenSpiel returned an illegal move')
        self.advance(action)
        return action

    def close(self):
        self._bot = None
        self.os_state = None


def create_bot(spec: str | Bot, **kwargs) -> Bot:
    """Factory creating a Bot instance from a specification string or existing Bot.

    Supported specifications:
      - Existing Bot instance: returned as-is.
      - 'tactical': TacticalBot(**kwargs)
      - 'threat' or 'threat-block': ThreatBlockBot(**kwargs)
      - 'alphabeta': AlphaBetaBot(depth=kwargs.get('depth', 3), node_budget=kwargs.get('node_budget', 3000), ...)
      - 'alphabeta:<depth>': AlphaBetaBot with specified depth
      - 'openspiel-mcts': OpenSpielBot(algorithm='mcts', simulations=kwargs.get('simulations', 100), ...)
      - 'openspiel-random': OpenSpielBot(algorithm='random', ...)
      - 'openspiel:<algorithm>:<simulations>': OpenSpielBot with specified algorithm and simulations
      - 'random', 'center', 'corners', 'local-win', 'global-win': StyleBot(style=spec, ...)
      - 'style:<style>': StyleBot(style=...)
      - 'checkpoint:<path>' or 'ckpt:<path>': CheckpointBot(path=..., **kwargs)
      - '<path>.pt' or existing file path: CheckpointBot(path=..., **kwargs)
      - 'cmd:<command>' or 'external:<command>': ExternalProcessBot(command=..., **kwargs)
    """
    if isinstance(spec, Bot):
        return spec

    if not isinstance(spec, str):
        if isinstance(spec, (list, tuple)):
            return ExternalProcessBot(command=spec, **kwargs)
        raise TypeError(f"Bot specification must be str or Bot, got {type(spec).__name__}")

    s = spec.strip()
    if s == "tactical":
        return TacticalBot(name=kwargs.get("name"))

    if s in ("threat", "threat-block"):
        return ThreatBlockBot(node_budget=kwargs.get("node_budget", 12000),
                              name=kwargs.get("name", "threat-block"))

    if s == "alphabeta":
        depth = kwargs.get("depth", 3)
        node_budget = kwargs.get("node_budget", 3000)
        return AlphaBetaBot(depth=depth, node_budget=node_budget, name=kwargs.get("name"))

    if s.startswith("alphabeta:"):
        depth_str = s.split(":", 1)[1]
        depth = int(depth_str) if depth_str else kwargs.get("depth", 3)
        node_budget = kwargs.get("node_budget", 3000)
        return AlphaBetaBot(depth=depth, node_budget=node_budget, name=kwargs.get("name"))

    if s == "openspiel-mcts" or s == "openspiel":
        kw = dict(kwargs)
        sims = kw.pop("simulations", 100)
        default_name = "openspiel-mcts" if "simulations" not in kwargs else f"openspiel-mcts-{sims}"
        name = kw.pop("name", default_name)
        return OpenSpielBot(algorithm="mcts", simulations=sims, name=name, **kw)

    if s == "openspiel-random":
        kw = dict(kwargs)
        name = kw.pop("name", "openspiel-random")
        return OpenSpielBot(algorithm="random", name=name, **kw)

    if s.startswith("openspiel-mcts:"):
        kw = dict(kwargs)
        sims_str = s.split(":", 1)[1]
        sims = int(sims_str) if sims_str else kw.pop("simulations", 100)
        name = kw.pop("name", f"openspiel-mcts-{sims}")
        return OpenSpielBot(algorithm="mcts", simulations=sims, name=name, **kw)

    if s.startswith("openspiel:"):
        kw = dict(kwargs)
        parts = s.split(":")[1:]
        algo = parts[0] if parts else "mcts"
        sims = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else kw.pop("simulations", 100)
        default_name = f"openspiel-{algo}" if sims == 100 else f"openspiel-{algo}-{sims}"
        name = kw.pop("name", default_name)
        return OpenSpielBot(algorithm=algo, simulations=sims, name=name, **kw)

    if s in NAMES:
        return StyleBot(style=s, deterministic=kwargs.get("deterministic", False), name=kwargs.get("name"))

    if s.startswith("style:"):
        style = s.split(":", 1)[1]
        return StyleBot(style=style, deterministic=kwargs.get("deterministic", False), name=kwargs.get("name"))

    if s.startswith("checkpoint:"):
        path = s.split(":", 1)[1]
        return CheckpointBot(path=path, **kwargs)

    if s.startswith("ckpt:"):
        path = s.split(":", 1)[1]
        return CheckpointBot(path=path, **kwargs)

    if s.startswith("cmd:"):
        cmd = s.split(":", 1)[1]
        return ExternalProcessBot(command=cmd, **kwargs)

    if s.startswith("external:"):
        cmd = s.split(":", 1)[1]
        return ExternalProcessBot(command=cmd, **kwargs)

    if s.endswith(".pt") or Path(s).exists():
        return CheckpointBot(path=s, **kwargs)

    raise ValueError(f"Unrecognized bot specification: '{spec}'")


# Backward compatibility alias
get_bot = create_bot
