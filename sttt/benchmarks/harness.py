"""M8: shared measurement vocabulary for benchmarks.

Two rules this module exists to enforce.

**Every measurement declares what it measured.** Dummy-evaluator tree traversal,
native heuristic search and real neural inference were previously reported in
one table as though they were the same quantity, and parallel independent-tree
throughput sat beside single-move latency. They are different categories and
cannot be compared to each other.

**Every rate denominator is completed work.** A search may finish early on an
exact proof, so `requested` and `completed` differ; the rate uses `completed`
and `requested` stays visible.
"""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

#: What a measurement is allowed to claim it measured.
CATEGORIES = (
    'native-primitive',            # one C++ operation in a loop
    'python-primitive',            # the Python reference of the same operation
    'dummy-evaluator-traversal',   # tree work with a stub evaluator; no network
    'native-heuristic-search',     # C++ search with the built-in heuristic
    'neural-inference',            # real forward passes through a loaded model
    'selfplay-game',               # complete games
    'league-game',                 # complete games against the opponent mix
    'training-iteration',          # a full collect + optimize + save iteration
    'parallel-throughput',         # independent work across threads/processes
)


@dataclass
class Measurement:
    """One timed quantity, with the category that makes it interpretable."""
    name: str
    category: str
    seconds: float
    completed: int
    requested: int | None = None
    threads: int = 1
    checksum: int | None = None
    notes: str = ''

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f'Unknown benchmark category {self.category!r}; '
                             f'expected one of {CATEGORIES}')
        if self.seconds < 0:
            raise ValueError(f'Negative elapsed time for {self.name!r}')
        if self.completed < 0:
            raise ValueError(f'Negative completed work for {self.name!r}')

    @property
    def rate(self) -> float | None:
        """Completed work per second, or None when nothing completed."""
        if not self.completed or self.seconds <= 0:
            return None
        return self.completed / self.seconds

    def to_dict(self) -> dict[str, Any]:
        data = {'name': self.name, 'category': self.category,
                'seconds': self.seconds, 'completed': self.completed,
                'rate_per_sec': self.rate, 'threads': self.threads}
        if self.requested is not None:
            data['requested'] = self.requested
        if self.checksum is not None:
            data['checksum'] = self.checksum
        if self.notes:
            data['notes'] = self.notes
        return data


def summarize_repeats(samples) -> dict[str, float]:
    """Median and spread over repeated timed blocks.

    A single run of a timed block is one sample of a noisy process; reporting it
    as the result hides the variance entirely.
    """
    values = [float(s) for s in samples]
    if not values:
        raise ValueError('At least one sample is required')
    return {'median': statistics.median(values),
            'mean': statistics.fmean(values),
            'min': min(values),
            'max': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'repeats': len(values)}


@dataclass
class StageTimer:
    """Wall-clock time attributed to non-overlapping pipeline stages.

    Worker time is deliberately separate. Four workers busy for ten seconds
    contribute forty worker-seconds but only ten seconds of elapsed time, and
    summing them into the stage total produces a pipeline that appears to take
    longer than it did.
    """
    stages: dict = field(default_factory=dict)
    aggregates: dict = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        started = time.monotonic()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.monotonic() - started)

    def add(self, name: str, seconds: float):
        """Record a stage measured elsewhere (already non-overlapping)."""
        self.stages[name] = self.stages.get(name, 0.0) + float(seconds)

    def add_aggregate(self, name: str, seconds: float, workers: int = 1):
        """Record summed concurrent time, which is NOT elapsed time."""
        entry = self.aggregates.setdefault(name, {'seconds': 0.0, 'workers': workers})
        entry['seconds'] += float(seconds)
        entry['workers'] = workers

    def report(self, elapsed_seconds: float) -> dict[str, Any]:
        accounted = sum(self.stages.values())
        return {'elapsed_seconds': float(elapsed_seconds),
                'stages': dict(self.stages),
                'aggregates': {k: dict(v) for k, v in self.aggregates.items()},
                'accounted_seconds': accounted,
                'unaccounted_seconds': float(elapsed_seconds) - accounted}
