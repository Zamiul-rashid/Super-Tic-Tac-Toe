"""Game-phase-stratified sampling from the replay buffer.

uttt.ai trains from per-depth files and round-robins across them, so every
batch is balanced over opening, midgame and endgame no matter how many
positions each phase produced. Our buffer is one FIFO; this module recovers
the same balance at sampling time. Depth is read off the encoded row itself
(81 minus the empty-cell plane), so legacy checkpoints need no migration.
"""
import numpy as np

EMPTY_PLANE = slice(0, 81)
PLY_BINS = 8
_EDGES = [(9 * i, 9 * i + 8) for i in range(PLY_BINS - 1)] + [(63, 80)]
BIN_LABELS = tuple(f'{lo}-{hi}' for lo, hi in _EDGES)


def row_ply(x):
    """Plies played so far in the encoded position `x` (shape (289,))."""
    return 81 - int(round(float(x[EMPTY_PLANE].sum())))


def ply_bin(ply):
    return min(int(ply) // 9, PLY_BINS - 1)


class StratifiedSampler:
    """Equal share per non-empty ply bin, without replacement.

    Built once per iteration from the buffer's bin ids; `sample` is then cheap
    enough to call once per optimizer step. A bin thinner than its share is
    exhausted and the leftover share is spread over the remaining bins, so the
    batch is always min(batch, len(bins)) rows.
    """

    def __init__(self, bins):
        self.bins = np.asarray(bins)
        self.members = [np.flatnonzero(self.bins == b) for b in range(PLY_BINS)]

    def __len__(self):
        return len(self.bins)

    def histogram(self):
        return {BIN_LABELS[b]: int(len(m)) for b, m in enumerate(self.members) if len(m)}

    def sample(self, batch, rng):
        batch = min(int(batch), len(self.bins))
        if batch <= 0:
            return np.empty(0, dtype=np.int64)
        quota = np.zeros(PLY_BINS, dtype=np.int64)
        open_bins = [b for b in range(PLY_BINS) if len(self.members[b])]
        remaining = batch
        while remaining > 0 and open_bins:
            share = max(1, remaining // len(open_bins))
            for b in list(open_bins):
                take = min(share, len(self.members[b]) - quota[b], remaining)
                quota[b] += take
                remaining -= take
                if quota[b] == len(self.members[b]):
                    open_bins.remove(b)
                if remaining == 0:
                    break
        picks = [rng.choice(self.members[b], size=int(quota[b]), replace=False)
                 for b in range(PLY_BINS) if quota[b]]
        return np.concatenate(picks).astype(np.int64)


def sample_bin_fractions(bins, indices):
    """Fraction of a sampled batch that came from each bin, by label."""
    indices = np.asarray(indices)
    if len(indices) == 0:
        return {}
    counts = np.bincount(np.asarray(bins)[indices], minlength=PLY_BINS)
    return {BIN_LABELS[b]: float(counts[b]) / len(indices) for b in range(PLY_BINS) if counts[b]}
