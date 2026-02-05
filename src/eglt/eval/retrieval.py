from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from eglt.dataset.schema import DeltaRecord
from eglt.training.dataloaders import Vocab, iter_delta_records_jsonl


@dataclass(frozen=True)
class RetrievalMetrics:
    mrr: float
    recall_at: Dict[int, float]
    n: int


def _load_vocab_json(path: Path) -> Vocab:
    obj = json.loads(path.read_text(encoding="utf-8"))
    itos = obj["itos"]
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def _infer_model_kind(state: Dict[str, torch.Tensor]) -> str:
    """
    Infer which embedding names exist:
      - SGNSModel: 't_in.weight' and 'c_out.weight' (training/trainer.py)
      - DeltaSGNS: 'tables.tactic_emb.weight' and 'tables.context_emb.weight' (models/sgns.py)
      - CBOWDeltaModel: 'c_in.weight' and 't_out.weight' (training/trainer.py)
      - CBOWDelta: 'tables.context_emb.weight' and 'tables.tactic_emb.weight' (models/cbow_delta.py)

    Returns: one of {"sgns_trainer","sgns_models","cbow_trainer","cbow_models"}.
    """
    keys = set(state.keys())
    if "t_in.weight" in keys and "c_out.weight" in keys:
        return "sgns_trainer"
    if "tables.tactic_emb.weight" in keys and "tables.context_emb.weight" in keys:
        return "sgns_models"  # also used by CBOWDelta models variant; handled below
    if "c_in.weight" in keys and "t_out.weight" in keys:
        return "cbow_trainer"
    # models version could be either; disambiguate by usage later
    if "tables.context_emb.weight" in keys and "tables.tactic_emb.weight" in keys:
        return "cbow_models"
    raise ValueError(f"Unrecognized model state_dict keys (sample): {sorted(list(keys))[:10]}")


def _get_embedding_mats(
    state: Dict[str, torch.Tensor],
    kind: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (tactic_emb [T,D], ctx_emb [C,D]) to use for delta->tactic retrieval.

    For retrieval we want:
      - tactic vectors: tactic embedding table
      - ctx vectors: context embedding table
    """
    if kind == "sgns_trainer":
        return state["t_in.weight"], state["c_out.weight"]
    if kind == "cbow_trainer":
        return state["t_out.weight"], state["c_in.weight"]
    if kind == "sgns_models":
        # In models/sgns.py and models/cbow_delta.py, tables.tactic_emb/context_emb exist.
        return state["tables.tactic_emb.weight"], state["tables.context_emb.weight"]
    if kind == "cbow_models":
        return state["tables.tactic_emb.weight"], state["tables.context_emb.weight"]
    raise AssertionError(kind)


def delta_record_query_vec(
    rec: DeltaRecord,
    ctx_vocab: Vocab,
    ctx_emb: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Embed a delta_context multiset by mean-pooling context embeddings for tokens in vocab.
    Returns [D]. If no known tokens, returns zeros.
    """
    ids = [ctx_vocab.stoi[t] for t in rec.delta_context if t in ctx_vocab.stoi]
    if not ids:
        return torch.zeros((ctx_emb.shape[1],), device=device)
    t = ctx_emb[torch.tensor(ids, dtype=torch.long, device=device)]  # [N,D]
    return t.mean(dim=0)


def evaluate_retrieval(
    deltas_path: str | Path,
    run_dir: str | Path,
    ks: List[int] | None = None,
    max_examples: int | None = None,
    device: str = "cpu",
) -> RetrievalMetrics:
    """
    Compute MRR and Recall@k for predicting tactic from delta_context.
    Ranking is over ALL tactics in the tactic vocab (SPEC).

    Args:
      deltas_path: JSONL of DeltaRecord
      run_dir: trained run directory containing model.pt, vocab_context.json, vocab_tactic.json
      ks: list of k values (default [1,5,10])
      max_examples: optional cap for faster iteration
      device: "cpu" (default) or "cuda"

    Returns metrics over examples whose (tactic, at least one ctx token) are known.
    """
    ks = ks or [1, 5, 10]
    run_dir = Path(run_dir)
    deltas_path = Path(deltas_path)

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    ctx_vocab = _load_vocab_json(run_dir / "vocab_context.json")
    tac_vocab = _load_vocab_json(run_dir / "vocab_tactic.json")
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    kind = _infer_model_kind(state)
    tac_emb, ctx_emb = _get_embedding_mats(state, kind)
    tac_emb = tac_emb.to(dev)
    ctx_emb = ctx_emb.to(dev)

    # Pre-normalize tactic vectors for fast cosine via dot
    tac_emb_n = F.normalize(tac_emb, dim=-1)  # [T,D]

    rr_sum = 0.0
    recall_hits = {k: 0 for k in ks}
    n = 0

    for rec in iter_delta_records_jsonl(deltas_path):
        if max_examples is not None and n >= max_examples:
            break
        if rec.tactic_head not in tac_vocab.stoi:
            continue

        q = delta_record_query_vec(rec, ctx_vocab, ctx_emb, dev)
        if torch.all(q == 0):
            continue

        qn = F.normalize(q.view(1, -1), dim=-1)  # [1,D]
        sims = (qn @ tac_emb_n.T).view(-1)  # [T]

        true_id = tac_vocab.stoi[rec.tactic_head]
        # rank (1-based): count how many are strictly greater + 1
        true_sim = sims[true_id].item()
        rank = int((sims > true_sim).sum().item()) + 1

        rr_sum += 1.0 / rank
        for k in ks:
            if rank <= k:
                recall_hits[k] += 1
        n += 1

    if n == 0:
        return RetrievalMetrics(mrr=0.0, recall_at={k: 0.0 for k in ks}, n=0)

    return RetrievalMetrics(
        mrr=rr_sum / n,
        recall_at={k: recall_hits[k] / n for k in ks},
        n=n,
    )
