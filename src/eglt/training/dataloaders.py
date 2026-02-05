from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from eglt.dataset.schema import DeltaRecord
from eglt.utils.io import read_jsonl


def _stable_int_seed(seed: int, key: str) -> int:
    """
    Stable (cross-run) derived seed from (seed, key), independent of Python hash randomization.
    """
    h = hashlib.sha256(f"{seed}::{key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)  # 64-bit-ish


def iter_delta_records_jsonl(path: str | Path) -> Iterator[DeltaRecord]:
    for row in read_jsonl(path):
        yield DeltaRecord.from_dict(row)


@dataclass(frozen=True)
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @staticmethod
    def build(items: Iterable[str], min_count: int = 1) -> "Vocab":
        cnt = Counter(items)
        itos = [w for w, c in cnt.items() if c >= min_count]
        itos.sort()
        stoi = {w: i for i, w in enumerate(itos)}
        return Vocab(stoi=stoi, itos=itos)

    def __len__(self) -> int:
        return len(self.itos)


def reservoir_cap_by_tactic(
    records: Iterable[DeltaRecord],
    k_per_tactic: int = 5000,
    seed: int = 0,
) -> List[DeltaRecord]:
    """
    Cap examples per tactic_head via deterministic reservoir sampling.

    Strategy:
      - Maintain a reservoir list per tactic.
      - For the n-th seen record of a tactic:
          if reservoir size < K: append
          else replace an existing element with probability K/n.

    Determinism:
      - Use a per-tactic RNG seeded by (seed, tactic_head) with stable hashing.

    Returns:
      A *materialized* list of sampled records (order: concatenated by tactic, then insertion order).
      This is fine because total is bounded by (#tactics * K).

    Note: for very large #tactics, memory can still grow; but SPEC explicitly says cap K=5000 per tactic.
    """
    reservoirs: dict[str, list[DeltaRecord]] = defaultdict(list)
    seen_counts: dict[str, int] = defaultdict(int)

    rngs: dict[str, random.Random] = {}

    for r in records:
        t = r.tactic_head
        if t not in rngs:
            rngs[t] = random.Random(_stable_int_seed(seed, t))

        rng = rngs[t]
        seen_counts[t] += 1
        n = seen_counts[t]
        res = reservoirs[t]

        if len(res) < k_per_tactic:
            res.append(r)
        else:
            j = rng.randrange(n)  # 0..n-1
            if j < k_per_tactic:
                res[j] = r

    # Flatten deterministically: sort tactics alphabetically for stable ordering
    out: list[DeltaRecord] = []
    for t in sorted(reservoirs.keys()):
        out.extend(reservoirs[t])
    return out


def build_vocabs_from_records(
    records: Iterable[DeltaRecord],
    min_count_context: int = 1,
    min_count_tactic: int = 1,
) -> tuple[Vocab, Vocab, Counter[str], Counter[str]]:
    """
    Build vocabularies for:
      - context tokens (delta_context entries)
      - tactic heads

    Returns:
      (ctx_vocab, tactic_vocab, ctx_counts, tactic_counts)
    """
    ctx_counts: Counter[str] = Counter()
    tactic_counts: Counter[str] = Counter()

    # records may be list; keep one pass
    recs = list(records)
    for r in recs:
        tactic_counts[r.tactic_head] += 1
        ctx_counts.update(r.delta_context)

    ctx_vocab = Vocab.build(ctx_counts.elements(), min_count=min_count_context)
    tactic_vocab = Vocab.build(tactic_counts.elements(), min_count=min_count_tactic)
    return ctx_vocab, tactic_vocab, ctx_counts, tactic_counts


@dataclass
class SGNSBatch:
    tactic_ids: List[int]
    ctx_ids: List[int]


@dataclass
class CBOWBatch:
    ctx_ids: List[List[int]]  # list of bag-lists
    tactic_ids: List[int]


def iter_sgns_batches(
    records: Iterable[DeltaRecord],
    ctx_vocab: Vocab,
    tactic_vocab: Vocab,
    batch_size: int = 1024,
) -> Iterator[SGNSBatch]:
    """
    SGNS batches over positive pairs (tactic, context_token) for each DeltaRecord.

    - Each DeltaRecord contributes len(delta_context) positive pairs.
    - Unknown tokens/tactics are skipped.
    """
    t_ids: list[int] = []
    c_ids: list[int] = []

    for r in records:
        t = tactic_vocab.stoi.get(r.tactic_head, None)
        if t is None:
            continue
        for tok in r.delta_context:
            c = ctx_vocab.stoi.get(tok, None)
            if c is None:
                continue
            t_ids.append(t)
            c_ids.append(c)
            if len(t_ids) >= batch_size:
                yield SGNSBatch(tactic_ids=t_ids, ctx_ids=c_ids)
                t_ids, c_ids = [], []

    if t_ids:
        yield SGNSBatch(tactic_ids=t_ids, ctx_ids=c_ids)


def iter_cbow_batches(
    records: Iterable[DeltaRecord],
    ctx_vocab: Vocab,
    tactic_vocab: Vocab,
    batch_size: int = 256,
    min_ctx_len: int = 1,
) -> Iterator[CBOWBatch]:
    """
    CBOW-Δ batches:
      - Input: bag of context token ids (delta_context)
      - Target: tactic_id

    Unknown tokens are dropped; if context becomes empty, sample is skipped.
    """
    ctx_bags: list[list[int]] = []
    t_ids: list[int] = []

    for r in records:
        t = tactic_vocab.stoi.get(r.tactic_head, None)
        if t is None:
            continue
        bag = [ctx_vocab.stoi[tok] for tok in r.delta_context if tok in ctx_vocab.stoi]
        if len(bag) < min_ctx_len:
            continue
        ctx_bags.append(bag)
        t_ids.append(t)

        if len(t_ids) >= batch_size:
            yield CBOWBatch(ctx_ids=ctx_bags, tactic_ids=t_ids)
            ctx_bags, t_ids = [], []

    if t_ids:
        yield CBOWBatch(ctx_ids=ctx_bags, tactic_ids=t_ids)
