from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from eglt.dataset.schema import DeltaRecord
from eglt.eval.state_embed import mean_pool_tokens
from eglt.training.dataloaders import Vocab, iter_delta_records_jsonl
from eglt.eval.retrieval import _infer_model_kind, _get_embedding_mats, _load_vocab_json


@dataclass(frozen=True)
class AnalogyMetrics:
    avg_cosine: float
    median_rank: float
    recall_at: Dict[int, float]
    n: int


def evaluate_analogy(
    deltas_path: str | Path,
    run_dir: str | Path,
    ks: List[int] | None = None,
    max_examples: int | None = None,
    device: str = "cpu",
) -> AnalogyMetrics:
    """
    Analogy task (paper-style): v(before_state) + e(tactic) should be close to v(after_state).
    We compute:
      - cosine(pred, true_after) per example
      - rank of true_after among all after vectors in the evaluated set (SPEC: "rank по after-states")

    Implementation details:
      - v(state) = mean pooling of token embeddings using context embedding table
        (works if your state tokens live in same token space as delta TOK_* + edits;
         if not, you can later swap to a separate state vocab/model).
      - e(tactic) = tactic embedding table vector.

    Requires DeltaRecord with state_before_tokens and state_after_tokens present.
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

    # Collect examples with states
    recs: list[DeltaRecord] = []
    for r in iter_delta_records_jsonl(deltas_path):
        if max_examples is not None and len(recs) >= max_examples:
            break
        if r.tactic_head not in tac_vocab.stoi:
            continue
        if not r.state_before_tokens or not r.state_after_tokens:
            continue
        recs.append(r)

    if not recs:
        return AnalogyMetrics(avg_cosine=0.0, median_rank=0.0, recall_at={k: 0.0 for k in ks}, n=0)

    # Precompute after vectors for ranking set
    after_vecs = []
    for r in recs:
        va = mean_pool_tokens(r.state_after_tokens or [], ctx_vocab.stoi, ctx_emb, dev)
        after_vecs.append(va)
    after_mat = torch.stack(after_vecs, dim=0)  # [N,D]
    after_mat_n = F.normalize(after_mat, dim=-1)

    cos_sum = 0.0
    ranks: list[int] = []
    recall_hits = {k: 0 for k in ks}

    for i, r in enumerate(recs):
        vb = mean_pool_tokens(r.state_before_tokens or [], ctx_vocab.stoi, ctx_emb, dev)
        et = tac_emb[tac_vocab.stoi[r.tactic_head]]

        pred = vb + et
        pred_n = F.normalize(pred.view(1, -1), dim=-1)  # [1,D]

        true_after = after_mat_n[i].view(1, -1)
        cos = float((pred_n * true_after).sum().item())
        cos_sum += cos

        sims = (pred_n @ after_mat_n.T).view(-1)  # [N]
        true_sim = sims[i].item()
        rank = int((sims > true_sim).sum().item()) + 1
        ranks.append(rank)

        for k in ks:
            if rank <= k:
                recall_hits[k] += 1

    ranks_sorted = sorted(ranks)
    mid = len(ranks_sorted) // 2
    median_rank = float(ranks_sorted[mid]) if ranks_sorted else 0.0

    n = len(recs)
    return AnalogyMetrics(
        avg_cosine=cos_sum / n,
        median_rank=median_rank,
        recall_at={k: recall_hits[k] / n for k in ks},
        n=n,
    )
