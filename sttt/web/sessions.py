"""In-memory game sessions, capped and expiring.

A session exists so the search tree survives between turns: the MCTS subtree
below the played move is reused rather than rebuilt, which is most of why the
engine answers as fast as it does. A stateless "post me the whole board" API
would throw that away every move.

The cap is not a tuning knob. Sessions hold trees and consume CPU, and anyone
who can reach the port can create them.
"""
import os
import secrets
import threading
import time

from ..backends import create_search
from ..env import State


class SessionLimit(RuntimeError):
    """Raised when the concurrent-game cap is reached."""


class GameSession:
    def __init__(self, evaluator, tier, human_side, seed, backend="auto"):
        self.search = create_search(evaluator, backend=backend, seed=seed)
        self.state = State()
        self.tier = tier
        self.human_side = human_side
        self.lock = threading.Lock()
        self.last_seen = time.monotonic()
        self.history = []
        # The tree has no root until the first search runs, so there is nothing
        # to advance through before then.
        self._primed = False

    @property
    def backend(self):
        return self.search.backend

    def _advance(self, action):
        if self._primed:
            self.search.advance(action)

    def play_human(self, action):
        """Apply the player's move. Raises ValueError if it is not legal."""
        self.state = self.state.play(action)   # validates; raises on illegal
        self._advance(action)
        self.history.append(action)
        self.last_seen = time.monotonic()

    def play_engine(self, simulations, leaf_batch):
        """Run the search and play its choice. Returns the action, or None if
        the game is already over."""
        if self.state.result is not None:
            return None
        policy = self.search.run(self.state, simulations, batch_size=leaf_batch)
        self._primed = True
        action = int(policy.argmax())
        self.search.advance(action)
        self.state = self.state.play(action)
        self.history.append(action)
        self.last_seen = time.monotonic()
        return action


class SessionStore:
    def __init__(self, evaluator, tiers, max_sessions=None, ttl=None, backend=None):
        self.evaluator = evaluator
        self.tiers = tiers
        self.max_sessions = int(max_sessions if max_sessions is not None
                                else os.environ.get("STTT_WEB_MAX_SESSIONS", 8))
        self.ttl = float(ttl if ttl is not None
                         else os.environ.get("STTT_WEB_SESSION_TTL", 1800))
        self.backend = backend or os.environ.get("STTT_WEB_BACKEND", "auto")
        self._sessions = {}
        self._lock = threading.Lock()
        self._seed = 0

    def _sweep(self):
        """Drop idle sessions. Lazy, on each request -- no background thread."""
        cutoff = time.monotonic() - self.ttl
        for key in [k for k, s in self._sessions.items() if s.last_seen < cutoff]:
            del self._sessions[key]

    def create(self, tier, human_side):
        with self._lock:
            self._sweep()
            if len(self._sessions) >= self.max_sessions:
                raise SessionLimit(
                    f"this server is already running {self.max_sessions} games "
                    f"(STTT_WEB_MAX_SESSIONS); finish one or try again later")
            self._seed += 1
            session = GameSession(self.evaluator, tier, human_side,
                                  seed=self._seed, backend=self.backend)
            key = secrets.token_urlsafe(12)
            self._sessions[key] = session
            return key, session

    def get(self, key):
        with self._lock:
            self._sweep()
            return self._sessions.get(key)

    def drop(self, key):
        with self._lock:
            self._sessions.pop(key, None)

    def __len__(self):
        return len(self._sessions)
