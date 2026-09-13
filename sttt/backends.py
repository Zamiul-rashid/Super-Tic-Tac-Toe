"""One explicit search-backend contract (plan M2).

Backend selection used to be three unrelated snippets: ``ai.py`` resolved a
bare ``use_cpp`` boolean that recorded nothing, ``bots.py`` re-probed for the
extension inside ``choose()`` on every move, and ``CppTreeSearch`` was a raw
alias to the native type. The worst consequence was silent: asking for
``backend="cpp"`` on a box without a built extension quietly ran the *Python*
search instead, and every artifact produced from that run claimed "cpp".

This module owns the decision instead. ``create_search`` resolves the backend
exactly once, up front, and returns an adapter whose ``backend_info`` records
what was requested, what is actually running, and why they differ. A strict
``cpp`` request raises rather than falling back.

``sttt.search.TreeSearch`` remains the Python reference implementation and is
deliberately *not* aliased to the native search anywhere.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .search import SearchConfig, TreeSearch

PYTHON = "python"
CPP = "cpp"
AUTO = "auto"
BACKENDS = (PYTHON, CPP, AUTO)

# Fixed, named RNG stream identifiers. A game seed derives one independent
# stream per role, so search noise, move sampling and the opponent model cannot
# consume each other's draws and a rerun of the same game reproduces all three.
# These numbers are part of the reproducibility contract: never renumber them.
STREAM_SEARCH = 1
STREAM_MOVE_SAMPLING = 2
STREAM_OPPONENT = 3
STREAM_NAMES = {
    STREAM_SEARCH: "search",
    STREAM_MOVE_SAMPLING: "move_sampling",
    STREAM_OPPONENT: "opponent",
}


class BackendUnavailable(RuntimeError):
    """A strictly requested backend cannot be provided on this machine."""


def derive_rng(game_seed: int, stream: int) -> np.random.Generator:
    """An independent, reproducible generator for one named role.

    Streams are derived from ``(game_seed, stream)`` rather than by reseeding a
    shared generator, so drawing from one role never shifts another's sequence.
    """
    if stream not in STREAM_NAMES:
        raise ValueError(f"unknown RNG stream id {stream!r}; "
                         f"expected one of {sorted(STREAM_NAMES)}")
    return np.random.default_rng([int(game_seed), int(stream)])


def cpp_available() -> bool:
    """True when the native search API -- not merely the module -- is importable."""
    try:
        from .cpp_env import CppTreeSearch, is_cpp_available
    except ImportError:
        return False
    return bool(is_cpp_available()) and CppTreeSearch is not None


def _cpp_version() -> str | None:
    try:
        from .cpp_env import get_cpp_provenance
    except ImportError:
        return None
    prov = get_cpp_provenance()
    return prov.get("build_id") or prov.get("version")


class SearchAdapter:
    """Uniform surface over the Python and native searches.

    Callers use this and nothing else: reaching into ``.root`` or ``.rng`` of the
    underlying tree is what made the two backends diverge in the first place.
    """

    __slots__ = ("_tree", "_info", "_seed")

    def __init__(self, tree, info: dict[str, Any], seed: int):
        self._tree = tree
        self._info = info
        self._seed = seed

    # -- search -------------------------------------------------------------
    def run(self, state, simulations: int, batch_size: int = 1, noise: bool = False):
        """Return the float32[81] visit-count policy for ``state``."""
        return self._tree.run(state, simulations, batch_size=batch_size, noise=noise)

    def advance(self, action: int) -> None:
        """Advance the tree by one *legal* move. Call once per played move."""
        self._tree.advance(action)

    def reset(self) -> None:
        """Forget the tree, keep the configured RNG stream."""
        self._tree.reset()

    # -- read-only views ----------------------------------------------------
    @property
    def root_state(self):
        root = self._tree.root
        return None if root is None else root.state

    @property
    def stats(self) -> dict:
        """A snapshot; fetch once per search and aggregate that copy."""
        stats = self._tree.stats
        return dict(stats) if stats else {}

    @property
    def backend_info(self) -> dict[str, Any]:
        """What was requested, what is running, and why they differ."""
        return dict(self._info)

    @property
    def backend(self) -> str:
        return self._info["actual"]

    @property
    def seed(self) -> int:
        return self._seed

    def __repr__(self) -> str:
        return (f"SearchAdapter(backend={self._info['actual']!r}, "
                f"requested={self._info['requested']!r}, seed={self._seed})")


def resolve_backend(backend: str, *, opponent=None) -> dict[str, Any]:
    """Decide the backend once, recording the reason. Raises on a strict miss.

    Called before any game starts so a missing native build fails immediately
    rather than part-way through a run whose earlier games claimed ``cpp``.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {list(BACKENDS)}")

    available = cpp_available()
    info: dict[str, Any] = {
        "requested": backend,
        "actual": None,
        "version": None,
        "fallback_reason": None,
        "cpp_available": available,
    }

    # Adaptive opponent modelling is Python-only: the native tree has no hook to
    # call opponent.predict() at non-agent nodes. A strict cpp request that also
    # asks for adaptation is a contradiction, not something to silently drop.
    if opponent is not None:
        if backend == CPP:
            raise NotImplementedError(
                "backend='cpp' cannot serve an adaptive opponent model: the "
                "native search has no opponent.predict() hook. Use "
                "backend='python' for adaptive play, or drop the opponent.")
        info["actual"] = PYTHON
        if backend == AUTO:
            info["fallback_reason"] = "adaptive opponent modelling is Python-only"
        return info

    if backend == PYTHON:
        info["actual"] = PYTHON
        return info

    if backend == CPP:
        if not available:
            raise BackendUnavailable(
                "backend='cpp' was requested but the native search is not "
                "available (build it with `make -C cpp`). Refusing to fall back "
                "to the Python search, which would misreport this run's backend.")
        info["actual"] = CPP
        info["version"] = _cpp_version()
        return info

    # auto
    if available:
        info["actual"] = CPP
        info["version"] = _cpp_version()
    else:
        info["actual"] = PYTHON
        info["fallback_reason"] = "native search unavailable (extension not built)"
    return info


def create_search(model, *, backend: str = AUTO, seed: int = 0,
                  config: SearchConfig | None = None, opponent=None,
                  agent_side=None) -> SearchAdapter:
    """Build a search for one role, on one explicitly resolved backend."""
    info = resolve_backend(backend, opponent=opponent)
    config = config or SearchConfig()
    rng = derive_rng(seed, STREAM_SEARCH)

    if info["actual"] == CPP:
        from .cpp_env import CppTreeSearch
        tree = CppTreeSearch(model, rng=rng, config=config)
    else:
        tree = TreeSearch(model, rng=rng, config=config,
                          opponent=opponent, agent_side=agent_side)
    return SearchAdapter(tree, info, seed)


def describe_backend(info: dict[str, Any]) -> str:
    """One log line stating the actual choice and why -- required for `auto`."""
    line = f"search backend: {info['actual']} (requested {info['requested']}"
    if info.get("version"):
        line += f", build {info['version']}"
    line += ")"
    if info.get("fallback_reason"):
        line += f" -- {info['fallback_reason']}"
    return line
