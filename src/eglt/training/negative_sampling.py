from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def build_unigram_table(
    items: Sequence[str],
    counts: Sequence[int],
    power: float = 0.75,
    table_size: int = 1_000_000,
) -> np.ndarray:
    """
    Build a word2vec-style unigram table for negative sampling.

    Each item i gets probability proportional to count_i ** power.
    We then create a large table of indices for O(1) sampling.

    Args:
      items: vocabulary items (only used for length checks).
      counts: raw counts aligned with items.
      power: smoothing power, default 0.75.
      table_size: size of sampling table (tradeoff memory vs approximation quality).

    Returns:
      np.ndarray[int64] of shape (table_size,) with indices into items.

    Examples (minimal):
    >>> import numpy as np
    >>> tbl = build_unigram_table(["a","b"], [1,3], power=1.0, table_size=1000)
    >>> int((tbl==1).sum()) > int((tbl==0).sum())
    True
    """
    if len(items) != len(counts):
        raise ValueError("items and counts must have same length")
    if len(items) == 0:
        raise ValueError("empty vocabulary")
    if table_size <= 0:
        raise ValueError("table_size must be > 0")
    if power <= 0:
        raise ValueError("power must be > 0")

    c = np.asarray(counts, dtype=np.float64)
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    if float(c.sum()) <= 0.0:
        raise ValueError("sum(counts) must be > 0")

    probs = np.power(c, power)
    probs_sum = float(probs.sum())
    probs = probs / probs_sum

    # Allocate per-item slots then fill; ensure exact table_size
    slots = np.floor(probs * table_size).astype(np.int64)
    # Fix rounding drift by distributing remaining slots to largest residuals
    residual = (probs * table_size) - slots
    missing = int(table_size - int(slots.sum()))
    if missing > 0:
        # indices of largest residuals
        add_idx = np.argsort(-residual)[:missing]
        slots[add_idx] += 1
    elif missing < 0:
        # remove from smallest residuals where slots>0
        rem = -missing
        order = np.argsort(residual)  # smallest first
        for i in order:
            if rem <= 0:
                break
            if slots[i] > 0:
                dec = min(int(slots[i]), rem)
                slots[i] -= dec
                rem -= dec

    # Build table
    table = np.empty((table_size,), dtype=np.int64)
    pos = 0
    for i, n in enumerate(slots.tolist()):
        if n <= 0:
            continue
        table[pos : pos + n] = i
        pos += n

    # Safety: fill any remaining due to edge drift
    if pos < table_size:
        table[pos:] = int(np.argmax(probs))
    elif pos > table_size:
        table = table[:table_size]

    return table


@dataclass
class UnigramSampler:
    """
    Fast negative sampler using a prebuilt unigram table.

    Use .draw(k) to get k indices (with replacement).
    """
    table: np.ndarray
    seed: int = 0

    def __post_init__(self) -> None:
        if self.table.dtype != np.int64:
            self.table = self.table.astype(np.int64)
        if self.table.ndim != 1 or self.table.size == 0:
            raise ValueError("table must be 1D and non-empty")
        self._rng = np.random.default_rng(self.seed)

    def draw(self, k: int) -> np.ndarray:
        if k <= 0:
            return np.empty((0,), dtype=np.int64)
        idx = self._rng.integers(0, self.table.size, size=k, dtype=np.int64)
        return self.table[idx]
