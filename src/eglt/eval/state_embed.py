from __future__ import annotations

from typing import List, Optional

import torch


def mean_pool_tokens(
    tokens: List[str],
    stoi: dict[str, int],
    emb: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Mean pooling for arbitrary token lists given an embedding table.

    If none of the tokens exist in stoi, returns zeros.

    Example:
    >>> import torch
    >>> emb = torch.randn(3, 4)
    >>> v = mean_pool_tokens(["a","b"], {"a":0,"b":1}, emb, torch.device("cpu"))
    >>> list(v.shape)
    [4]
    """
    ids = [stoi[t] for t in tokens if t in stoi]
    if not ids:
        return torch.zeros((emb.shape[1],), device=device)
    x = emb[torch.tensor(ids, dtype=torch.long, device=device)]
    return x.mean(dim=0)


def state_vector_before_after(
    before_tokens: Optional[List[str]],
    after_tokens: Optional[List[str]],
    stoi: dict[str, int],
    emb: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience to embed (before_state_tokens, after_state_tokens) with same pooling.
    """
    vb = mean_pool_tokens(before_tokens or [], stoi, emb, device)
    va = mean_pool_tokens(after_tokens or [], stoi, emb, device)
    return vb, va
