from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def init_embedding(emb: nn.Embedding, mode: str = "uniform") -> None:
    """
    Initialize embeddings in a word2vec-friendly way.

    mode:
      - "uniform": U(-0.5/d, 0.5/d)
      - "zeros": zeros
    """
    d = emb.embedding_dim
    if mode == "uniform":
        nn.init.uniform_(emb.weight, -0.5 / d, 0.5 / d)
    elif mode == "zeros":
        nn.init.zeros_(emb.weight)
    else:
        raise ValueError(f"Unknown init mode: {mode}")


@dataclass
class EmbeddingTables(nn.Module):
    """
    Shared container for (tactic_emb, context_emb).

    Used by:
      - ∆-SGNS: tactic_in + context_out
      - CBOW-∆: context_in + tactic_out
      - Seq baseline SGNS: tactic_in + tactic_out (context is tactic)
    """
    n_tactics: int
    n_context: int
    dim: int
    sparse: bool = False

    tactic_emb: nn.Embedding | None = None
    context_emb: nn.Embedding | None = None

    def __post_init__(self) -> None:
        super().__init__()
        self.tactic_emb = nn.Embedding(self.n_tactics, self.dim, sparse=self.sparse)
        self.context_emb = nn.Embedding(self.n_context, self.dim, sparse=self.sparse)

        # Typical SGNS: input initialized, output zeros
        init_embedding(self.tactic_emb, "uniform")
        init_embedding(self.context_emb, "zeros")

    def forward(self) -> None:  # pragma: no cover
        raise RuntimeError("EmbeddingTables is a container; use tactic_emb/context_emb directly.")
