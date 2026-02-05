from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eglt.models.embeddings import EmbeddingTables


@dataclass
class CBOWBatchTensors:
    """
    CBOW-∆ batch:
      - ctx_bags: list of 1D Long tensors with token ids (ragged)
      - t_pos: [B] tactic ids
      - t_neg: [B, K] negative tactic ids
    """
    ctx_bags: List[torch.Tensor]
    t_pos: torch.Tensor
    t_neg: torch.Tensor


class CBOWDelta(nn.Module):
    """
    CBOW-∆ model: predict tactic from a bag of delta-context tokens.

    ctx_vec = mean(context_emb[token] for token in bag)
    Score(ctx, t) = dot(ctx_vec, tactic_emb[t])
    """
    def __init__(self, n_tactics: int, n_context: int, dim: int) -> None:
        super().__init__()
        # Here, "tactic_emb" acts like output embeddings, "context_emb" like input embeddings.
        self.tables = EmbeddingTables(n_tactics=n_tactics, n_context=n_context, dim=dim)

    def ctx_mean(self, ctx_bags: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute [B, D] mean vectors from ragged bags.
        """
        vecs: list[torch.Tensor] = []
        for ids in ctx_bags:
            v = self.tables.context_emb(ids).mean(dim=0)
            vecs.append(v)
        return torch.stack(vecs, dim=0)

    def score(self, ctx_vecs: torch.Tensor, t_ids: torch.Tensor) -> torch.Tensor:
        t = self.tables.tactic_emb(t_ids)  # [*, D]
        return (ctx_vecs * t).sum(dim=-1)

    def forward(self, batch: CBOWBatchTensors) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (pos_scores [B], neg_scores [B, K]).
        """
        ctx_vecs = self.ctx_mean(batch.ctx_bags)  # [B, D]
        pos_scores = self.score(ctx_vecs, batch.t_pos)  # [B]

        B, K = batch.t_neg.shape
        ctx_exp = ctx_vecs.view(B, 1, -1).expand(B, K, -1).reshape(-1, ctx_vecs.size(-1))
        t_neg_flat = batch.t_neg.reshape(-1)
        neg_scores = self.score(ctx_exp, t_neg_flat).view(B, K)
        return pos_scores, neg_scores

    @staticmethod
    def loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return (-F.logsigmoid(pos_scores).mean()) + (-F.logsigmoid(-neg_scores).mean())

    def train_step(self, batch: CBOWBatchTensors, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        pos, neg = self.forward(batch)
        loss = self.loss(pos, neg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.item())
