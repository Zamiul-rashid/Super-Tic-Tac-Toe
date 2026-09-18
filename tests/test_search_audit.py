"""Search correctness audit: controlled evaluators with known outputs, run
against BOTH search backends (Python ``TreeSearch`` and native
``CppTreeSearch``). See docs/engineering/search-audit-iter4530.md.

Conventions pinned here:
* Evaluator values and node totals are from the player-to-move perspective;
  ``root_action_values`` reports Q(s, a) from the root player's perspective.
* ``simulations`` is the number of NEW backups requested per ``run`` call.
  Retained subtree visits (``stats['retained_visits']``) do not count toward it.
* ``completed_simulations`` counts backups: neural leaf evaluations plus
  terminal/proven-leaf backups (no evaluator call). It is below the budget only
  when the root is proven (early proof termination).
* ``neural_positions`` counts evaluator positions INCLUDING the root expansion,
  which is not a simulation (its value is not backed up).
"""
import unittest
import zlib

import numpy as np

from sttt.env import State
from sttt.search import TreeSearch, SearchConfig, root_action_values
from sttt.cpp_env import is_cpp_available, CppTreeSearch

BACKENDS = [TreeSearch] + ([CppTreeSearch] if is_cpp_available() and CppTreeSearch is not None else [])
NO_PROOFS = SearchConfig(proofs=False, soft_pruning=False, reuse=True)

DRAW = [1, -1, 1, 1, -1, -1, -1, 1, 1]       # full local board, no line
O_WON = [-1, -1, -1, 0, 0, 0, 0, 0, 0]
X_WON = [1, 1, 1, 0, 0, 0, 0, 0, 0]


def build(board_cells, boards, turn, forced):
    cells = [0] * 81
    for b, local in board_cells.items():
        cells[b * 9:b * 9 + 9] = local
    return State(tuple(cells), tuple(boards), turn, forced)


# X completes board 2 and the global top row with 20.
WIN = build({0: X_WON, 1: X_WON, 2: [1, 1, 0, 0, 0, 0, 0, 0, 0]}, (1, 1, 0, 0, 0, 0, 0, 0, 0), 1, 2)
# O threatens 20 (global top row); every X move except 20 hands O a free choice. Exact value: draw.
BLOCK = build({0: O_WON, 1: O_WON, 2: [-1, -1, 0, 0, 1, 0, 0, 0, 0], **{b: DRAW for b in range(3, 9)}},
              (-1, -1, 0, 2, 2, 2, 2, 2, 2), 1, 2)
# X on board 4: 38 sends O to board 2, 41 to a closed board (free choice); only 39
# (to board 3) is safe. Exact values: 38 and 41 lose, 39 draws.
AVOID = build({0: O_WON, 1: O_WON, 2: [-1, -1, 0, 0, 0, 0, 0, 0, 0], 3: [0, 0, 0, 1, -1, -1, -1, 1, 1],
               4: [1, -1, 0, 0, 1, 0, -1, 1, -1], **{b: DRAW for b in (5, 6, 7, 8)}},
              (-1, -1, 0, 0, 0, 2, 2, 2, 2), 1, 4)
# As AVOID but board 3 is closed too: every X move lets O win next move.
LOSE = build({0: O_WON, 1: O_WON, 2: [-1, -1, 0, 0, 0, 0, 0, 0, 0], 4: [1, -1, 0, -1, 1, 0, -1, 1, -1],
              **{b: DRAW for b in (3, 5, 6, 7, 8)}}, (-1, -1, 0, 2, 0, 2, 2, 2, 2), 1, 4)
# Only board 8 open, no global line possible: every continuation is a draw.
DRAWN = build({**{b: DRAW for b in range(8)}, 8: [1, -1, 1, 1, -1, -1, 0, 0, 1]}, (2,) * 8 + (0,), 1, 8)
# 78 wins board 8 and ends the game as a draw (no global line is possible);
# 79 and 80 continue the game.
TERMINAL_OPTION = build({**{b: DRAW for b in range(8)}, 8: [1, -1, 1, 1, -1, -1, 0, 0, 0]}, (2,) * 8 + (0,), 1, 8)
# Previous move was sent to closed board 0 (won by X): free choice over the open boards only.
CLOSED_TARGET = build({0: X_WON, 5: DRAW, 4: [0, 0, 0, 0, -1, 0, 0, 0, 0]}, (1, 0, 0, 0, 0, 2, 0, 0, 0), -1, -1)


class Stub:
    """Deterministic evaluator. The policy deliberately puts mass on ILLEGAL
    actions too; the search must ignore it. ``value(state)`` is mover-relative."""

    def __init__(self, value, policy=None):
        self.value, self.policy = value, policy
        self.calls = self.batch_calls = 0

    def evaluate(self, state):
        self.calls += 1
        if self.policy is not None:
            return self.policy(state), float(self.value(state))
        return np.full(81, 1 / 81, dtype=np.float32), float(self.value(state))

    def evaluate_many(self, states):
        self.batch_calls += 1
        return [self.evaluate(s) for s in states]


def mover_winning(v):
    return Stub(lambda s: v)


def x_ahead(v):
    """X is winning by v from every position: mover-relative v * turn."""
    return Stub(lambda s: v * s.turn)


def hashed():
    """Tie-free pseudo-random policy/value keyed on the state. Values are float32
    representable so both backends see bit-identical inputs."""
    def draw(s):
        key = zlib.crc32(bytes(c % 256 for c in s.cells) + bytes([s.turn % 256, s.forced % 256]))
        return np.random.default_rng(key)
    return Stub(lambda s: float(np.float32(draw(s).uniform(-.9, .9))),
                policy=lambda s: draw(s).random(82)[1:].astype(np.float32))


def nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


def search(backend, state, sims, model, batch=1, config=None):
    tree = backend(model, config=config or SearchConfig())
    pi = tree.run(state, sims, batch_size=batch)
    return tree, pi


class ValuePerspectiveTests(unittest.TestCase):
    def test_backups_alternate_sign_and_stay_mover_relative(self):
        for backend in BACKENDS:
            for state in (State(), State().play(40)):          # X and O to move
                with self.subTest(backend=backend.__name__, turn=state.turn):
                    tree, _ = search(backend, state, 256, x_ahead(.5), batch=8, config=NO_PROOFS)
                    visited = 0
                    for node in nodes(tree.root):
                        if node.n:
                            visited += 1
                            self.assertAlmostEqual(node.total / node.n, .5 * node.state.turn, places=6)
                    self.assertGreater(visited, 50)
                    q, mask = root_action_values(tree.root)
                    np.testing.assert_allclose(q[mask], .5 * state.turn, atol=1e-6)

    def test_mover_winning_stub_gives_negative_root_q(self):
        state = State().play(40)
        legal = state.legal_actions()
        for backend in BACKENDS:
            with self.subTest(backend=backend.__name__):
                tree, pi = search(backend, state, len(legal), mover_winning(.6), config=NO_PROOFS)
                q, mask = root_action_values(tree.root)
                self.assertEqual(np.flatnonzero(mask).tolist(), legal)   # one visit each
                np.testing.assert_allclose(q[legal], -.6, atol=1e-6)


class TerminalTests(unittest.TestCase):
    BUDGETS, BATCHES = (128, 512), (1, 16, 64)

    def cases(self):
        for backend in BACKENDS:
            for sims in self.BUDGETS:
                for batch in self.BATCHES:
                    yield backend, sims, batch

    def test_immediate_win_is_proven_at_root_expansion(self):
        for backend, sims, batch in self.cases():
            with self.subTest(backend=backend.__name__, sims=sims, batch=batch):
                tree, pi = search(backend, WIN, sims, mover_winning(-1.), batch)
                self.assertEqual(tree.root.solved, 1)
                self.assertEqual(np.flatnonzero(pi).tolist(), [20])
                self.assertEqual(tree.stats['completed_simulations'], 0)
                self.assertEqual(float(root_action_values(tree.root)[0][20]), 1.)

    def test_forced_block_is_the_only_move_with_mass(self):
        for backend, sims, batch in self.cases():
            with self.subTest(backend=backend.__name__, sims=sims, batch=batch):
                # Misleading net: says whoever is to move (O, after X's move) is lost.
                tree, pi = search(backend, BLOCK, sims, mover_winning(-.9), batch)
                self.assertEqual(np.flatnonzero(pi).tolist(), [20])
                self.assertIn(tree.root.solved, (None, 0))           # exact value is a draw
                q, mask = root_action_values(tree.root)
                self.assertTrue(all(q[a] == -1. for a in (21, 23, 24, 25, 26) if mask[a]))

    def test_never_allows_an_immediate_loss_when_a_safe_move_exists(self):
        for backend, sims, batch in self.cases():
            with self.subTest(backend=backend.__name__, sims=sims, batch=batch):
                tree, pi = search(backend, AVOID, sims, mover_winning(-.9), batch)
                self.assertEqual(float(pi[38]), 0.)
                self.assertEqual(float(pi[41]), 0.)
                self.assertEqual(int(pi.argmax()), 39)
                self.assertNotEqual(tree.root.solved, -1)

    def test_forced_loss_is_proven_and_stops_early(self):
        for backend, sims, batch in self.cases():
            with self.subTest(backend=backend.__name__, sims=sims, batch=batch):
                tree, pi = search(backend, LOSE, sims, mover_winning(.9), batch)
                self.assertEqual(tree.root.solved, -1)
                self.assertEqual(tree.stats['root_solved'], -1)
                self.assertLess(tree.stats['completed_simulations'], sims)
                q, mask = root_action_values(tree.root)
                self.assertEqual(np.flatnonzero(mask).tolist(), [38, 41])
                np.testing.assert_array_equal(q[mask], -1.)
                self.assertAlmostEqual(float(pi.sum()), 1., places=6)

    def test_draw_is_proven_exactly(self):
        for backend, sims, batch in self.cases():
            with self.subTest(backend=backend.__name__, sims=sims, batch=batch):
                tree, _ = search(backend, DRAWN, sims, mover_winning(.9), batch)
                self.assertEqual(tree.root.solved, 0)
                q, mask = root_action_values(tree.root)
                self.assertEqual(np.flatnonzero(mask).tolist(), [78, 79])
                np.testing.assert_array_equal(q, 0.)
                self.assertLess(tree.stats['completed_simulations'], sims)


class LegalActionTests(unittest.TestCase):
    def test_no_illegal_children_visits_or_probability(self):
        positions = {'forced': State().play(40), 'sent_to_closed': CLOSED_TARGET, 'free': State()}
        self.assertEqual(CLOSED_TARGET.forced, -1)
        for name, state in positions.items():
            legal = set(state.legal_actions())
            illegal = np.setdiff1d(np.arange(81), sorted(legal))
            for backend in BACKENDS:
                for batch in (1, 16):
                    with self.subTest(position=name, backend=backend.__name__, batch=batch):
                        tree, pi = search(backend, state, 256, x_ahead(.3), batch)
                        self.assertEqual(set(tree.root.children), legal)
                        self.assertTrue((pi[illegal] == 0).all())
                        self.assertAlmostEqual(float(pi.sum()), 1., places=5)
                        for node in nodes(tree.root):
                            if node.children:
                                self.assertEqual(set(node.children), set(node.state.legal_actions()))
                        self.assertFalse(root_action_values(tree.root)[1][illegal].any())


class SimulationAccountingTests(unittest.TestCase):
    def test_budget_counts_new_backups_and_every_evaluation(self):
        for backend in BACKENDS:
            for sims in (128, 512, 2000):
                for batch in (1, 16):
                    with self.subTest(backend=backend.__name__, sims=sims, batch=batch):
                        model = x_ahead(.2)
                        tree, _ = search(backend, State(), sims, model, batch, NO_PROOFS)
                        st = tree.stats
                        self.assertEqual(st['completed_simulations'], sims)
                        self.assertEqual(st['retained_visits'], 0)
                        self.assertEqual(tree.root.n, sims)
                        terminal = sum(n.n for n in nodes(tree.root) if n.state.result is not None)
                        # +1: the root expansion is evaluated but is not a simulation.
                        self.assertEqual(st['neural_positions'], model.calls)
                        self.assertEqual(st['neural_positions'], sims - terminal + 1)
                        self.assertEqual(st['inference_batches'], model.batch_calls + 1)
                        if batch == 1:
                            self.assertEqual(st['inference_batches'], st['neural_positions'])

                        # Reuse: retained visits are reported, not charged to the new budget.
                        action = max(tree.root.children, key=lambda a: tree.root.children[a].n)
                        retained = tree.root.children[action].n
                        tree.advance(action)
                        tree.run(State().play(action), 128, batch_size=batch)
                        self.assertEqual(tree.stats['retained_visits'], retained)
                        self.assertEqual(tree.stats['completed_simulations'], 128)
                        self.assertEqual(tree.root.n, retained + 128)

    def test_proven_retained_root_does_no_new_work(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend.__name__):
                model = mover_winning(.9)
                tree, _ = search(backend, LOSE, 64, model)
                calls = model.calls
                tree.run(LOSE, 64)
                self.assertEqual(tree.stats['completed_simulations'], 0)
                self.assertEqual(tree.stats['neural_positions'], 0)
                self.assertEqual(model.calls, calls)


class BatchConsistencyTests(unittest.TestCase):
    def test_no_stale_virtual_visits_and_consistent_totals(self):
        for backend in BACKENDS:
            for config in (NO_PROOFS, SearchConfig()):
                for batch in (1, 16, 64):
                    with self.subTest(backend=backend.__name__, proofs=config.proofs, batch=batch):
                        tree, pi = search(backend, State().play(40), 512, hashed(), batch, config)
                        self.assertEqual(tree.stats['completed_simulations'], 512)
                        root = tree.root
                        self.assertEqual(root.n, sum(c.n for c in root.children.values()))
                        for node in nodes(root):
                            self.assertEqual(node.in_flight, 0)
                            self.assertFalse(node.pending)
                            self.assertLessEqual(abs(node.total), node.n + 1e-9)
                            if node is root or not node.children:
                                continue
                            below = sum(c.n for c in node.children.values())
                            if config.proofs:
                                self.assertGreaterEqual(node.n, below + 1)
                            else:    # exactly one expansion visit, then one per child visit
                                self.assertEqual(node.n, below + 1)
                        visits = np.array([root.children[a].n if a in root.children else 0
                                           for a in range(81)], dtype=float)
                        np.testing.assert_allclose(pi, visits / visits.sum(), atol=1e-6)

    def test_pending_evaluations_do_not_starve_behind_a_terminal_child(self):
        # Regression: while the non-terminal children were pending, batch
        # collection re-selected the terminal child 78 until the whole budget
        # was spent (198/200 visits, 2 evaluations, root never proven).
        sims = 200
        for backend in BACKENDS:
            for batch in (1, 16, 64):
                with self.subTest(backend=backend.__name__, batch=batch):
                    tree, pi = search(backend, TERMINAL_OPTION, sims, mover_winning(-.9), batch)
                    self.assertEqual(tree.root.solved, 0)          # exact: every line draws
                    self.assertLess(tree.stats['completed_simulations'], 20)
                    tree, pi = search(backend, TERMINAL_OPTION, sims, mover_winning(-.9), batch, NO_PROOFS)
                    self.assertEqual(tree.stats['completed_simulations'], sims)
                    self.assertLess(tree.root.children[78].n, sims // 2)   # batch 1: 70/200


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(len(BACKENDS) == 2, 'C++ MCTS Engine not available')
class BackendAgreementTests(unittest.TestCase):
    """Identical states + a tie-free deterministic evaluator + no noise => the
    two backends build the same tree. Documented tie-breaking differences (only
    reachable with exact PUCT ties, e.g. dyadic stub priors): Python normalizes
    priors with numpy's pairwise sum and keeps values in float64, C++ sums
    sequentially and stores values as float32 (soft_margin is 0.2f); after a
    pending child is skipped in a batch, Python keeps the remaining order and
    C++ swaps the last candidate into its slot. Ties go to the first candidate."""

    def test_identical_visit_counts_and_q(self):
        rng = np.random.default_rng(0)
        states = []
        while len(states) < 8:
            state = State()
            for _ in range(int(rng.integers(0, 40))):
                if state.result is None:
                    state = state.play(int(rng.choice(state.legal_actions())))
            if state.result is None:
                states.append(state)
        configs = (SearchConfig(), NO_PROOFS, SearchConfig(proofs=False))
        for i, state in enumerate(states):
            for config in configs:
                for batch in (1, 16):
                    with self.subTest(position=i, proofs=config.proofs, soft=config.soft_pruning, batch=batch):
                        trees = [backend(hashed(), config=config) for backend in BACKENDS]
                        current = state
                        for step in range(2):          # second step exercises subtree reuse
                            pis = [t.run(current, 300, batch_size=batch) for t in trees]
                            py, cpp = trees
                            self.assertEqual({a: c.n for a, c in py.root.children.items()},
                                             {a: c.n for a, c in cpp.root.children.items()})
                            for key in ('completed_simulations', 'neural_positions',
                                        'inference_batches', 'retained_visits', 'root_solved'):
                                self.assertEqual(py.stats[key], cpp.stats[key], key)
                            np.testing.assert_allclose(pis[0], pis[1], atol=1e-6)
                            np.testing.assert_allclose(root_action_values(py.root)[0],
                                                       root_action_values(cpp.root)[0], atol=1e-5)
                            action = int(pis[0].argmax())
                            if py.root.solved is not None or current.play(action).result is not None:
                                break
                            for t in trees:
                                t.advance(action)
                            current = current.play(action)
