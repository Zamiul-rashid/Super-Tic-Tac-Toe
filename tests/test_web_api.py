"""HTTP contract for the self-hosted board.

Runs against a stub evaluator rather than the real network: these tests are
about the API's rules-correctness and its behaviour at the trust boundary, and
they should not need a 6 MB model file or a second of search to say so.
"""
import unittest

import numpy as np
from fastapi.testclient import TestClient

from sttt.env import State
from sttt.web.difficulty import Tiers
from sttt.web.server import create_app
from sttt.web.sessions import SessionStore


class StubEvaluator:
    """Uniform priors over legal actions, neutral value. Deterministic."""

    path = ""

    def evaluate_many(self, states):
        out = []
        for state in states:
            policy = np.zeros(81, dtype=np.float64)
            legal = state.legal_actions()
            policy[legal] = 1.0 / len(legal)
            out.append((policy, 0.0))
        return out

    def evaluate(self, state):
        return self.evaluate_many([state])[0]


def client(max_sessions=8):
    tiers = Tiers()
    store = SessionStore(StubEvaluator(), tiers, max_sessions=max_sessions, ttl=3600)
    return TestClient(create_app(evaluator=StubEvaluator(), tiers=tiers, store=store))


class NewGameTests(unittest.TestCase):
    def test_new_game_starts_empty_with_every_action_legal(self):
        response = client().post("/api/game", json={})
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(len(body["session"]), 16)
        state = body["state"]
        self.assertEqual(state["cells"], [0] * 81)
        self.assertEqual(state["turn"], 1)
        self.assertIsNone(state["result"])
        self.assertEqual(len(state["legal"]), 81)
        self.assertIsNone(body["engine_action"])

    def test_choosing_the_second_seat_makes_the_engine_open(self):
        body = client().post("/api/game", json={"human_side": -1}).json()
        self.assertIsNotNone(body["engine_action"])
        self.assertEqual(sum(1 for c in body["state"]["cells"] if c != 0), 1)
        self.assertEqual(body["state"]["turn"], -1)

    def test_unknown_difficulty_is_rejected(self):
        response = client().post("/api/game", json={"difficulty": "impossible"})
        self.assertEqual(response.status_code, 400)

    def test_bad_seat_is_rejected(self):
        self.assertEqual(client().post("/api/game", json={"human_side": 0}).status_code, 400)


class MoveTests(unittest.TestCase):
    def setUp(self):
        self.client = client()
        body = self.client.post("/api/game", json={"difficulty": "casual"}).json()
        self.session = body["session"]

    def test_a_legal_move_advances_and_the_engine_replies(self):
        response = self.client.post(f"/api/game/{self.session}/move", json={"action": 40})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["state"]["cells"][40], 1)
        self.assertIsNotNone(body["engine_action"])
        self.assertEqual(body["state"]["cells"][body["engine_action"]], -1)

    def test_an_illegal_move_is_rejected_and_changes_nothing(self):
        self.client.post(f"/api/game/{self.session}/move", json={"action": 40})
        before = self.client.get(f"/api/game/{self.session}").json()["state"]
        response = self.client.post(f"/api/game/{self.session}/move", json={"action": 40})
        self.assertEqual(response.status_code, 400)
        after = self.client.get(f"/api/game/{self.session}").json()["state"]
        self.assertEqual(before, after)

    def test_out_of_range_actions_never_reach_the_engine(self):
        for action in (-1, 81, 9999):
            with self.subTest(action=action):
                response = self.client.post(f"/api/game/{self.session}/move",
                                            json={"action": action})
                self.assertEqual(response.status_code, 422)

    def test_the_forced_board_constraint_is_reported(self):
        """After the engine plays cell c, the player is confined to board c."""
        body = self.client.post(f"/api/game/{self.session}/move", json={"action": 40}).json()
        state, engine = body["state"], body["engine_action"]
        if state["result"] is not None:
            self.skipTest("game ended")
        target = engine % 9
        if state["boards"][target] != 0:
            self.assertEqual(state["forced"], -1)
        else:
            self.assertEqual(state["forced"], target)
            self.assertTrue(all(a // 9 == target for a in state["legal"]))

    def test_unknown_session_is_404(self):
        response = self.client.post("/api/game/nope/move", json={"action": 0})
        self.assertEqual(response.status_code, 404)


class RulesAgreementTests(unittest.TestCase):
    def test_served_legal_list_equals_the_engine(self):
        """The client trusts `legal` completely, so it must be the engine's own."""
        c = client()
        session = c.post("/api/game", json={"difficulty": "casual"}).json()["session"]
        state = State()
        for _ in range(30):
            served = c.get(f"/api/game/{session}").json()["state"]
            state = State(tuple(served["cells"]), tuple(served["boards"]),
                          served["turn"], served["forced"], served["result"])
            self.assertEqual(sorted(served["legal"]), sorted(state.legal_actions()))
            if served["result"] is not None:
                break
            body = c.post(f"/api/game/{session}/move",
                          json={"action": served["legal"][0]}).json()
            if body["state"]["result"] is not None:
                self.assertEqual(body["state"]["legal"], [])
                break


class FullGameTests(unittest.TestCase):
    def test_a_scripted_game_reaches_a_terminal_result(self):
        c = client()
        session = c.post("/api/game", json={"difficulty": "casual"}).json()["session"]
        rng = np.random.default_rng(0)
        for _ in range(81):
            state = c.get(f"/api/game/{session}").json()["state"]
            if state["result"] is not None:
                self.assertIn(state["result"], (-1, 0, 1))
                return
            action = int(rng.choice(state["legal"]))
            c.post(f"/api/game/{session}/move", json={"action": action})
        self.fail("game did not terminate within 81 plies")

    def test_moving_in_a_finished_game_is_rejected(self):
        c = client()
        session = c.post("/api/game", json={"difficulty": "casual"}).json()["session"]
        rng = np.random.default_rng(1)
        for _ in range(81):
            state = c.get(f"/api/game/{session}").json()["state"]
            if state["result"] is not None:
                break
            c.post(f"/api/game/{session}/move", json={"action": int(rng.choice(state["legal"]))})
        response = c.post(f"/api/game/{session}/move", json={"action": 0})
        self.assertEqual(response.status_code, 409)


class CapacityTests(unittest.TestCase):
    def test_the_session_cap_is_enforced(self):
        c = client(max_sessions=2)
        self.assertEqual(c.post("/api/game", json={}).status_code, 201)
        self.assertEqual(c.post("/api/game", json={}).status_code, 201)
        response = c.post("/api/game", json={})
        self.assertEqual(response.status_code, 429)
        self.assertIn("STTT_WEB_MAX_SESSIONS", response.json()["detail"])

    def test_ending_a_game_frees_a_slot(self):
        c = client(max_sessions=1)
        session = c.post("/api/game", json={}).json()["session"]
        self.assertEqual(c.post("/api/game", json={}).status_code, 429)
        self.assertEqual(c.delete(f"/api/game/{session}").status_code, 204)
        self.assertEqual(c.post("/api/game", json={}).status_code, 201)


class HealthTests(unittest.TestCase):
    def test_health_reports_tiers_and_capacity(self):
        body = client().get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["sessions"]["max"], 8)
        keys = {tier["key"] for tier in body["tiers"]}
        self.assertEqual(keys, {"casual", "standard", "strong", "championship"})
        self.assertIn(body["default_difficulty"], keys)


if __name__ == "__main__":
    unittest.main()
