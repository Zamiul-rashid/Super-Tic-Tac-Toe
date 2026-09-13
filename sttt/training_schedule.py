"""Resumable learning-rate schedules in units of completed iterations (plan M5).

The unit matters more than the curve. A cosine horizon expressed in optimizer
*updates* and stepped once per update is a completely different schedule from
the same horizon expressed in training *iterations*: with 100 updates per
iteration, ``T_max=5000`` stepped per update reaches its floor after 50
iterations, not 5,000. This module counts completed iterations only.

Nothing here is a torch scheduler object. Schedules are serialised as plain
fields so they load under ``weights_only=True``; pickling a scheduler instance
or a callable would defeat that.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

CONSTANT = "constant"
COSINE = "cosine"
SCHEDULES = (CONSTANT, COSINE)

DEFAULT_LR = 1e-3


@dataclass
class LRSchedule:
    """Learning rate as a function of completed iterations in this phase.

    ``completed`` counts iterations finished *within the current phase*, so a
    resume continues the curve rather than restarting it, and an explicitly
    declared phase change resets it deliberately.
    """

    kind: str = CONSTANT
    lr_start: float = DEFAULT_LR
    lr_min: float = 0.0
    horizon: int = 0          # iterations; only meaningful for cosine
    completed: int = 0
    phase: int = 0            # bumped by an explicit --reset-lr-schedule

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.kind not in SCHEDULES:
            raise ValueError(f"unknown lr schedule {self.kind!r}; expected one of {list(SCHEDULES)}")
        if not math.isfinite(self.lr_start) or self.lr_start <= 0:
            raise ValueError(f"lr must be finite and positive, got {self.lr_start!r}")
        if not math.isfinite(self.lr_min) or self.lr_min < 0:
            raise ValueError(f"lr_min must be finite and non-negative, got {self.lr_min!r}")
        if self.lr_min > self.lr_start:
            raise ValueError(f"lr_min ({self.lr_min}) must not exceed lr ({self.lr_start})")
        if self.kind == COSINE:
            if int(self.horizon) != self.horizon or self.horizon < 1:
                raise ValueError(f"cosine needs a positive integer horizon, got {self.horizon!r}")
        if self.completed < 0:
            raise ValueError("completed iterations cannot be negative")
        return self

    # -- the curve ----------------------------------------------------------
    def lr_at(self, completed: int) -> float:
        """LR for the iteration that follows ``completed`` finished iterations."""
        if self.kind == CONSTANT:
            return self.lr_start
        # Clamped at both ends: start LR at k=0, midpoint at H/2, floor at and
        # after H. It never restarts and never climbs back up.
        k = min(max(int(completed), 0), int(self.horizon))
        cosine = 0.5 * (1.0 + math.cos(math.pi * k / self.horizon))
        return self.lr_min + (self.lr_start - self.lr_min) * cosine

    @property
    def current_lr(self) -> float:
        """LR to use for the next iteration."""
        return self.lr_at(self.completed)

    @property
    def next_lr(self) -> float:
        """LR the iteration after this one will use."""
        return self.lr_at(self.completed + 1)

    def advance(self) -> float:
        """Record one COMPLETED iteration. Call after its optimization block.

        Deliberately not called for an interrupted iteration: a partial
        iteration that never saved a checkpoint must not move the schedule, or
        a resume would skip a step of the curve.
        """
        self.completed += 1
        return self.current_lr

    # -- persistence --------------------------------------------------------
    def state_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state(cls, state: dict | None) -> "LRSchedule | None":
        if not state:
            return None
        known = {f: state[f] for f in cls.__dataclass_fields__ if f in state}
        return cls(**known)

    def describe(self) -> str:
        if self.kind == CONSTANT:
            return f"lr schedule: constant {self.lr_start:.3e} (phase {self.phase})"
        return (f"lr schedule: cosine {self.lr_start:.3e} -> {self.lr_min:.3e} over "
                f"{self.horizon} iterations, {self.completed} completed "
                f"(phase {self.phase})")


def build_schedule(args, saved_state=None, *, legacy_lr=None):
    """Resolve the schedule for a run, honouring resume semantics.

    A legacy resume with no saved schedule keeps running at its restored
    optimizer LR unless the caller explicitly asks for something else -- the
    default must never silently re-raise a decayed run back to 1e-3.
    """
    requested_kind = getattr(args, "lr_schedule", None)
    explicit_lr = getattr(args, "lr", None)
    reset = bool(getattr(args, "reset_lr_schedule", False))
    saved = LRSchedule.from_state(saved_state)

    if saved is not None and not reset:
        # Continue the existing phase. An explicit --lr / --lr-min / --lr-iterations
        # without --reset-lr-schedule is a mistake worth naming rather than
        # silently applying mid-curve.
        changes = []
        if explicit_lr is not None and not math.isclose(explicit_lr, saved.lr_start):
            changes.append(f"--lr {explicit_lr} (schedule holds {saved.lr_start})")
        if requested_kind is not None and requested_kind != saved.kind:
            changes.append(f"--lr-schedule {requested_kind} (schedule is {saved.kind})")
        if changes:
            raise ValueError(
                "refusing to change a running learning-rate schedule mid-phase: "
                + "; ".join(changes)
                + ". Pass --reset-lr-schedule (ideally with a new --output) to "
                  "start a new, recorded phase.")
        return saved

    kind = requested_kind or CONSTANT
    lr_start = explicit_lr if explicit_lr is not None else (
        legacy_lr if (legacy_lr is not None and requested_kind is None) else DEFAULT_LR)
    schedule = LRSchedule(
        kind=kind,
        lr_start=float(lr_start),
        lr_min=float(getattr(args, "lr_min", 0.0) or 0.0),
        horizon=int(getattr(args, "lr_iterations", 0) or 0),
        completed=0,
        phase=(saved.phase + 1) if (saved is not None and reset) else 0,
    )
    return schedule


def apply_lr(optimizer, lr: float) -> None:
    """Set the LR without disturbing optimizer moments."""
    for group in optimizer.param_groups:
        group["lr"] = lr
