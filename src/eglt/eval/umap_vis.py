from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from eglt.eval.retrieval import _infer_model_kind, _get_embedding_mats, _load_vocab_json


@dataclass(frozen=True)
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    top_n_labels: int = 50  # label only most frequent tactics (optional, if you later pass counts)


def save_umap_figure(
    run_dir: str | Path,
    out_path: str | Path,
    cfg: UMAPConfig | None = None,
    device: str = "cpu",
) -> Path:
    """
    Save "Figure 1"-style 2D visualization of tactic embeddings using UMAP.

    Writes a PNG to results/figures by SPEC.

    Note: requires `umap-learn` installed. If missing, raises RuntimeError with install hint.
    """
    cfg = cfg or UMAPConfig()
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import umap  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("UMAP requires `umap-learn`. Install: pip install umap-learn") from e

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    tac_vocab = _load_vocab_json(run_dir / "vocab_tactic.json")
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    kind = _infer_model_kind(state)
    tac_emb, _ctx_emb = _get_embedding_mats(state, kind)

    X = F.normalize(tac_emb.to(dev), dim=-1).detach().cpu().numpy()

    reducer = umap.UMAP(
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        metric=cfg.metric,
        random_state=42,
    )
    X2 = reducer.fit_transform(X)  # [T,2]

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], s=8)
    plt.title("UMAP of tactic embeddings")

    # Label a subset (first N in vocab order; you can later improve by frequency)
    nlab = min(cfg.top_n_labels, len(tac_vocab.itos))
    for i in range(nlab):
        plt.text(X2[i, 0], X2[i, 1], tac_vocab.itos[i], fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
