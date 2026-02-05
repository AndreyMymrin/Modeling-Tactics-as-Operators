from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from eglt.dataset.schema import StepRecord
from eglt.models.sgns import DeltaSGNS, SGNSBatchTensors
from eglt.training.negative_sampling import UnigramSampler, build_unigram_table
from eglt.utils.io import read_jsonl


def iter_step_records_jsonl(path: str | Path) -> Iterator[StepRecord]:
    for row in read_jsonl(path):
        yield StepRecord.from_dict(row)


def _safe_step_sort_key(step_id: str) -> Tuple[int, str]:
    """
    Try numeric sort if possible, else fallback to lexicographic.
    """
    try:
        return (0, f"{int(step_id):012d}")
    except Exception:
        return (1, step_id)


def group_tactic_sequences_by_proof(records: Iterable[StepRecord]) -> Dict[str, List[str]]:
    """
    Build tactic-head sequences per proof_id.

    Assumes StepRecord.tactic_head is already extracted/canonicalized.
    Sorts steps by step_id (numeric if possible).
    """
    by_proof: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in records:
        if not r.proof_id:
            continue
        if not r.tactic_head:
            continue
        by_proof[r.proof_id].append((r.step_id, r.tactic_head))

    out: Dict[str, List[str]] = {}
    for pid, items in by_proof.items():
        items.sort(key=lambda x: _safe_step_sort_key(x[0]))
        out[pid] = [t for _, t in items]
    return out


def generate_skipgram_pairs_from_sequences(
    sequences: Dict[str, List[str]],
    window: int = 2,
) -> Iterator[Tuple[str, str]]:
    """
    Sequence SGNS baseline: generate (center_tactic, context_tactic) pairs
    within a fixed window on tactic sequence.

    For each position i, for j in [i-window, i+window], j!=i:
        yield (seq[i], seq[j])

    Deterministic given sequences/order.
    """
    for _, seq in sequences.items():
        n = len(seq)
        for i in range(n):
            ci = seq[i]
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                yield (ci, seq[j])


@dataclass(frozen=True)
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @staticmethod
    def from_counts(cnt: Counter[str], min_count: int = 1) -> "Vocab":
        itos = [w for w, c in cnt.items() if c >= min_count]
        itos.sort()
        return Vocab(stoi={w: i for i, w in enumerate(itos)}, itos=itos)

    def __len__(self) -> int:
        return len(self.itos)


@dataclass(frozen=True)
class SeqBaselineCfg:
    step_records_path: str
    run_name: str = "seq_baseline"
    device: str = "cpu"
    seed: int = 0

    dim: int = 128
    epochs: int = 1
    lr: float = 2e-3

    window: int = 2
    batch_size: int = 8192

    neg_k: int = 10
    unigram_power: float = 0.75
    unigram_table_size: int = 1_000_000
    min_count_tactic: int = 1


def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def train_seq_baseline_sgns(cfg: dict, run_dir: str | Path) -> None:
    """
    SGNS baseline on tactic sequences (tactic -> neighboring tactics).

    Input: JSONL StepRecord (raw stepwise corpus) or any jsonl with proof_id/step_id/tactic_head.
    Output: run_dir/model.pt + vocab_tactic.json

    Notes:
      - Uses DeltaSGNS with n_tactics = n_context = |tactics| because both sides are tactics.
      - Negative sampling over tactics using unigram^0.75.
    """
    c = SeqBaselineCfg(
        step_records_path=str(cfg["step_records_path"]),
        run_name=str(cfg.get("run_name", "seq_baseline")),
        device=str(cfg.get("device", "cpu")),
        seed=int(cfg.get("seed", 0)),
        dim=int(cfg.get("dim", 128)),
        epochs=int(cfg.get("epochs", 1)),
        lr=float(cfg.get("lr", 2e-3)),
        window=int(cfg.get("window", 2)),
        batch_size=int(cfg.get("batch_size", 8192)),
        neg_k=int(cfg.get("neg_k", 10)),
        unigram_power=float(cfg.get("unigram_power", 0.75)),
        unigram_table_size=int(cfg.get("unigram_table_size", 1_000_000)),
        min_count_tactic=int(cfg.get("min_count_tactic", 1)),
    )

    run_dir = Path(run_dir)
    device = torch.device(c.device if (c.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Load and group sequences
    steps = list(iter_step_records_jsonl(c.step_records_path))
    seqs = group_tactic_sequences_by_proof(steps)

    # Build tactic vocab + counts
    tac_counts: Counter[str] = Counter()
    for seq in seqs.values():
        tac_counts.update(seq)
    vocab = Vocab.from_counts(tac_counts, min_count=c.min_count_tactic)

    # Negative sampler over tactics
    tac_items = vocab.itos
    tac_cnts = [int(tac_counts.get(w, 0)) for w in tac_items]
    table = build_unigram_table(tac_items, tac_cnts, power=c.unigram_power, table_size=c.unigram_table_size)
    neg_sampler = UnigramSampler(table=table, seed=c.seed)

    # Model: tactic->tactic, so n_tactics == n_context == |vocab|
    model = DeltaSGNS(n_tactics=len(vocab), n_context=len(vocab), dim=c.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr)

    _save_json(run_dir / "config.json", {"cfg": cfg})
    _save_json(run_dir / "vocab_tactic.json", {"itos": vocab.itos})

    # Iterate pairs and train
    def iter_batches() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        t_buf: list[int] = []
        c_buf: list[int] = []
        for t_str, c_str in generate_skipgram_pairs_from_sequences(seqs, window=c.window):
            t_id = vocab.stoi.get(t_str, None)
            c_id = vocab.stoi.get(c_str, None)
            if t_id is None or c_id is None:
                continue
            t_buf.append(t_id)
            c_buf.append(c_id)
            if len(t_buf) >= c.batch_size:
                yield (
                    torch.tensor(t_buf, dtype=torch.long, device=device),
                    torch.tensor(c_buf, dtype=torch.long, device=device),
                )
                t_buf, c_buf = [], []
        if t_buf:
            yield (
                torch.tensor(t_buf, dtype=torch.long, device=device),
                torch.tensor(c_buf, dtype=torch.long, device=device),
            )

    for epoch in range(1, c.epochs + 1):
        total = 0.0
        n = 0
        for t_pos, c_pos in tqdm(iter_batches(), desc=f"SEQ-SGNS epoch {epoch}/{c.epochs}", leave=False):
            B = t_pos.numel()
            neg_flat = neg_sampler.draw(B * c.neg_k)
            c_neg = torch.tensor(neg_flat, dtype=torch.long, device=device).view(B, c.neg_k)

            batch = SGNSBatchTensors(t_pos=t_pos, c_pos=c_pos, c_neg=c_neg)
            loss_val = model.train_step(batch, opt)
            total += loss_val
            n += 1

        _save_json(run_dir / f"metrics_epoch_{epoch}.json", {"epoch": epoch, "avg_loss": total / max(1, n)})

    torch.save(model.state_dict(), run_dir / "model.pt")
