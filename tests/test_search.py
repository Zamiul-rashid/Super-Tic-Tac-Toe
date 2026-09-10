"""Proof correctness against exact game-tree answers; batching and reuse invariants."""
from dataclasses import dataclass
import unittest
import numpy as np
import torch
from sttt.env import State
from sttt.search import TreeSearch, SearchConfig, Node
from sttt.learning import Network, ResNet
from sttt.bots import TacticalBot, AlphaBetaBot


@dataclass(frozen=True)
class ToyState:
    """Small fully specified adversarial tree. Leaves hold X's actual result."""
    tree: object
    turn: int = 1

    @property
    def result(self):
        return self.tree if isinstance(self.tree, int) else None

    def legal_actions(self):
        return list(range(len(self.tree))) if self.result is None else []

    def play(self, action):
        return ToyState(self.tree[action], -self.turn)


def exact(state):
    if state.result is not None:
        return state.result * state.turn
    return max(-exact(state.play(a)) for a in state.legal_actions())


class FakeEvaluator:
    def __init__(self, estimate=0.):
        self.estimate, self.batches = estimate, []

    def evaluate(self, state):
        p = np.zeros(81)
        p[state.legal_actions()] = 1 / len(state.legal_actions())
        return p, self.estimate

    def evaluate_many(self, states):
        self.batches.append(len(states))
        return [self.evaluate(s) for s in states]


class SearchTests(unittest.TestCase):
    def test_exact_proofs_for_all_small_binary_games(self):
        # 3 outcomes at each of four leaves: every depth-2 binary game.
        import itertools
        for leaves in itertools.product((-1,0,1), repeat=4):
            state = ToyState((leaves[:2], leaves[2:]))
            tree = TreeSearch(FakeEvaluator(.99))
            p = tree.run(state, 100, batch_size=4)
            self.assertEqual(tree.root.solved, exact(state))
            self.assertEqual(-exact(state.play(int(p.argmax()))), exact(state))

    def test_sacrifice_and_misleading_network_are_not_proofs(self):
        state = ToyState((((1,1),(1,1)), ((-1,-1),(-1,-1))))
        tree = TreeSearch(FakeEvaluator(-1.))
        p = tree.run(state, 100, batch_size=4)
        self.assertEqual(tree.root.solved,1)
        self.assertEqual(int(p.argmax()),0)
        incomplete = TreeSearch(FakeEvaluator(1.))
        incomplete.run(State(),1)
        self.assertIsNone(incomplete.root.solved)

    def test_proven_loss_excluded_when_unknown_alternative_exists(self):
        tree = TreeSearch(FakeEvaluator())
        tree.root = Node(ToyState(((1,-1), ((1,1),(1,1)))))
        tree.stats = {'hard_pruned_choices':0}
        tree._expand(tree.root, tree.model.evaluate(tree.root.state)[0])
        bad = tree.root.children[0]
        tree._expand(bad,tree.model.evaluate(bad.state)[0])
        self.assertEqual(bad.solved,1)  # Opponent has a winning reply.
        self.assertEqual(tree._eligible(tree.root),[tree.root.children[1]])

    def test_reuse_and_noise_do_not_overwrite_base_priors(self):
        tree = TreeSearch(FakeEvaluator(), np.random.default_rng(4))
        tree.run(State(),64,batch_size=8,noise=True)
        self.assertTrue(all(abs(c.prior-1/81)<1e-7 for c in tree.root.children.values()))
        action = max(tree.root.children,key=lambda a: tree.root.children[a].n)
        child = tree.root.children[action]
        visits = child.n
        tree.advance(action)
        self.assertIs(tree.root,child)
        tree.run(child.state,8,batch_size=4)
        self.assertEqual(tree.stats['retained_visits'],visits)
        self.assertGreaterEqual(tree.root.n,visits)
        tree.config = SearchConfig(reuse=False)
        tree.run(child.state,4)
        self.assertEqual(tree.stats['retained_visits'],0)

    def test_soft_rechecks_preserve_bad_branches(self):
        tree = TreeSearch(FakeEvaluator(),config=SearchConfig(proofs=False, revisit_interval=4))
        tree.run(ToyState((1,-1)),100,batch_size=4)
        self.assertGreater(tree.root.children[0].n,tree.root.children[1].n)
        self.assertGreater(tree.root.children[1].n,1)
        self.assertGreater(tree.stats['soft_rechecks'],0)

    def test_batch_virtual_visits_released_even_on_failure(self):
        model = FakeEvaluator()
        tree = TreeSearch(model)
        tree.run(State(),32,batch_size=8)
        self.assertEqual(tree.stats['completed_simulations'],32)
        self.assertGreater(max(model.batches),1)
        self.assertEqual(tree.root.in_flight,0)
        self.assertTrue(all(c.in_flight==0 and not c.pending for c in tree.root.children.values()))
        def fail(states):
            raise RuntimeError('inference failed')
        model.evaluate_many = fail
        with self.assertRaises(RuntimeError):
            tree.run(tree.root.state,8,batch_size=8)
        pending = [tree.root]
        while pending:
            node = pending.pop()
            self.assertEqual(node.in_flight,0)
            self.assertFalse(node.pending)
            pending.extend(node.children.values())

    def test_batch_neural_evaluation_matches_single(self):
        torch.set_num_threads(1)
        states = [State(),State().play(4),State().play(4).play(40)]
        for model in (Network().eval(),ResNet().eval()):
            batch = model.evaluate_many(states)
            for state,(p,v) in zip(states,batch):
                single,sv = model.evaluate(state)
                np.testing.assert_allclose(p,single,atol=1e-6)
                self.assertAlmostEqual(v,sv,places=5)
                illegal = [a for a in range(81) if a not in state.legal_actions()]
                self.assertTrue((p[illegal]==0).all())

    def test_bots_take_global_win(self):
        cells = [0]*81; cells[18:20] = [1,1]
        s = State(tuple(cells),(1,1,0,0,0,0,0,0,0),forced=2)
        for bot in (TacticalBot(),AlphaBetaBot()):
            self.assertEqual(bot.choose(s,np.random.default_rng(0)),20)

    def test_bots_avoid_sending_opponent_to_global_win(self):
        cells = [0]*81; cells[18:20] = [-1,-1]
        # X is on board 4; cell 2 would send O to board 2 to complete the global line.
        s = State(tuple(cells),(-1,-1,0,0,0,0,0,0,0),forced=4)
        for bot in (TacticalBot(),AlphaBetaBot(depth=2,node_budget=500)):
            action = bot.choose(s,np.random.default_rng(0))
            child = s.play(action)
            self.assertFalse(any(child.play(a).result == -1 for a in child.legal_actions()))

    def test_bots_block_local_loss(self):
        cells = [0]*81; cells[36:38] = [-1,-1]
        # Only the threat board remains open; all moves route back to it.
        s = State(tuple(cells),(2,2,2,2,0,2,2,2,2),forced=4)
        # With no global line possible every result is a draw. Instead keep board 0 open.
        s = State(tuple(cells),(0,2,2,2,0,2,2,2,2),forced=4)
        for bot in (TacticalBot(),AlphaBetaBot(depth=2)):
            child = s.play(bot.choose(s,np.random.default_rng(1)))
            self.assertFalse(any(child.play(a).boards[4] == -1 for a in child.legal_actions()))

    def test_real_late_games_match_exhaustive_solver(self):
        rng = np.random.default_rng(37)
        checked = 0
        for _ in range(150):
            state = State()
            while state.result is None:
                remaining = sum(state.cells[b*9:b*9+9].count(0) for b in range(9) if state.boards[b] == 0)
                if remaining <= 5:
                    tree = TreeSearch(FakeEvaluator(.8))
                    p = tree.run(state,5000,batch_size=8)
                    answer = exact(state)
                    self.assertEqual(tree.root.solved,answer)
                    self.assertEqual(-exact(state.play(int(p.argmax()))),answer)
                    checked += 1
                    break
                state = state.play(int(rng.choice(state.legal_actions())))
            if checked == 10:
                break
        self.assertEqual(checked,10)

    def test_minimax_proof_pruning_disabled_with_human_model(self):
        class Human:
            def predict(self, state, base):
                return base
        tree = TreeSearch(FakeEvaluator(),opponent=Human(),agent_side=1)
        tree.run(ToyState(((1,-1),(0,0))),32,batch_size=4)
        self.assertFalse(tree.use_proofs)
        self.assertIsNone(tree.root.solved)


if __name__ == '__main__':
    unittest.main()
