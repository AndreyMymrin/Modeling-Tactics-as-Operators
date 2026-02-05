from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eglt.models.embeddings import EmbeddingTables


@dataclass
class SGNSBatchTensors:
    """
    Batch for SGNS:
      - t_pos: [B] tactic ids
      - c_pos: [B] context ids
      - c_neg: [B, K] negative context ids
    """
    t_pos: torch.Tensor
    c_pos: torch.Tensor
    c_neg: torch.Tensor


class DeltaSGNS(nn.Module):
    """
    ∆-SGNS model: predict context tokens from tactic.

    Score(t, c) = dot(tactic_emb[t], context_emb[c])
    """
    def __init__(self, n_tactics: int, n_context: int, dim: int) -> None:
        super().__init__()
        self.tables = EmbeddingTables(n_tactics=n_tactics, n_context=n_context, dim=dim)

    def score(self, t_ids: torch.Tensor, c_ids: torch.Tensor) -> torch.Tensor:
        t = self.tables.tactic_emb(t_ids)  # [*, D]
        c = self.tables.context_emb(c_ids)  # [*, D]
        return (t * c).sum(dim=-1)

    def forward(self, batch: SGNSBatchTensors) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (pos_scores [B], neg_scores [B, K]).
        """
        pos_scores = self.score(batch.t_pos, batch.c_pos)  # [B]

        B, K = batch.c_neg.shape
        t_exp = batch.t_pos.view(B, 1).expand(B, K).reshape(-1)
        c_neg_flat = batch.c_neg.reshape(-1)
        neg_scores = self.score(t_exp, c_neg_flat).view(B, K)  # [B, K]
        return pos_scores, neg_scores

    @staticmethod
    def loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Negative sampling loss:
          L = -E[log σ(pos)] - E[log σ(-neg)]
        """
        return (-F.logsigmoid(pos_scores).mean()) + (-F.logsigmoid(-neg_scores).mean())

    def train_step(self, batch: SGNSBatchTensors, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        pos, neg = self.forward(batch)
        loss = self.loss(pos, neg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.item())
