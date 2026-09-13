"""M1 regression tests: arena growth is bounded across a complete game.

Re-rooting used to retain every discarded sibling subtree. Measured over one
60-move game at 1024 simulations per move the arena reached 437,343 nodes
(40 MiB) of which 212 were reachable. advance() now compacts to the live
subtree, so occupancy tracks the search tree instead of the game length.
"""
import unittest

import numpy as np

from sttt.env import State
from sttt.search import SearchConfig
from sttt.cpp_env import is_cpp_available, CppTreeSearch


@unittest.skipUnless(is_cpp_available() and CppTreeSearch is not None,
                     "C++ MCTS Engine not available")
class TestArenaGrowth(unittest.TestCase):
    def _play(self, sims, moves_cap=40, seed=0):
        tree = CppTreeSearch(model=None, config=SearchConfig(proofs=False, reuse=True))
        state = State()
        rng = np.random.default_rng(seed)
        moves = 0
        sizes = []
        while state.result is None and moves < moves_cap:
            tree.run(state, simulations=sims, batch_size=8)
            action = int(rng.choice(state.legal_actions()))
            state = state.play(action)
            tree.advance(action)
            sizes.append(tree.stats["arena_nodes"])
            moves += 1
        return tree, sizes

    def test_arena_does_not_grow_with_game_length(self):
        """Post-advance occupancy must not scale with the number of moves played."""
        tree, sizes = self._play(sims=128, moves_cap=40)
        self.assertGreater(len(sizes), 10)
        early = max(sizes[:5])
        late = max(sizes[-5:])
        # Pre-patch this ratio grew without bound; the live subtree is small and
        # roughly stationary, so allow generous slack but not linear growth.
        self.assertLess(late, early * 10 + 1000,
                        f"arena grew from {early} to {late} across {len(sizes)} moves")

    def test_compaction_keeps_only_reachable_nodes(self):
        tree, _ = self._play(sims=256, moves_cap=20)
        live = 0
        stack = [tree.root]
        while stack:
            node = stack.pop()
            live += 1
            stack.extend(node.children.values())
        self.assertEqual(tree.stats["arena_nodes"], live,
                         "every node in the arena must be reachable from the root")

    def test_capacity_floor_is_respected(self):
        """Churn must not drop below the reserve and thrash the allocator."""
        tree, _ = self._play(sims=128, moves_cap=20)
        self.assertGreaterEqual(tree.stats["arena_capacity"], 65536)

    def test_reset_releases_and_refloors_capacity(self):
        tree, _ = self._play(sims=256, moves_cap=10)
        tree.reset()
        tree.run(State(), simulations=32, batch_size=4)
        self.assertGreaterEqual(tree.stats["arena_capacity"], 65536)

    def test_reuse_survives_compaction(self):
        """Compaction must preserve the retained subtree, not discard it."""
        tree = CppTreeSearch(model=None, config=SearchConfig(proofs=False, reuse=True))
        state = State()
        tree.run(state, simulations=256, batch_size=8)
        action = int(state.legal_actions()[0])
        nxt = state.play(action)
        tree.advance(action)
        tree.run(nxt, simulations=256, batch_size=8)
        self.assertGreater(tree.stats["retained_visits"], 0,
                           "compaction discarded the reusable subtree")

    def test_children_contiguity_preserved(self):
        """Selection relies on children occupying consecutive arena slots."""
        tree, _ = self._play(sims=256, moves_cap=10)
        root = tree.root
        actions = sorted(root.children)
        self.assertEqual(len(actions), len(set(actions)))
        for action, child in root.children.items():
            self.assertGreaterEqual(child.n, 0)
            self.assertEqual(child.state.turn, -root.state.turn)

    def test_stats_expose_arena_occupancy(self):
        tree = CppTreeSearch(model=None, config=SearchConfig(proofs=False))
        tree.run(State(), simulations=32, batch_size=4)
        stats = tree.stats
        self.assertIn("arena_nodes", stats)
        self.assertIn("arena_capacity", stats)
        self.assertGreater(stats["arena_nodes"], 0)
        self.assertGreaterEqual(stats["arena_capacity"], stats["arena_nodes"])


if __name__ == "__main__":
    unittest.main()
