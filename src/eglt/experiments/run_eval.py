from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eglt.eval.analogy import evaluate_analogy
from eglt.eval.retrieval import evaluate_retrieval
from eglt.eval.umap_vis import UMAPConfig, save_umap_figure
from eglt.paths import paths
from eglt.utils.logging import setup_logging


def run_eval(cfg: dict, run_dir: str | Path) -> None:
    """
    Evaluate retrieval/analogy/umap.

    Expected cfg keys:
      - deltas_path: DeltaRecord JSONL for retrieval + analogy
      - ks: list of ints (optional)
      - device: "cpu" or "cuda"
      - umap: optional dict (n_neighbors, min_dist, metric, top_n_labels)
    """
    log = setup_logging()
    P = paths(Path.cwd())
    run_dir = Path(run_dir)

    deltas_path = Path(cfg["deltas_path"])
    if not deltas_path.is_absolute():
        deltas_path = (P.root / deltas_path).resolve()

    ks = cfg.get("ks", [1, 5, 10])
    device = str(cfg.get("device", "cpu"))

    # Retrieval
    ret = evaluate_retrieval(deltas_path, run_dir, ks=list(ks), device=device, max_examples=cfg.get("max_examples"))
    (run_dir / "eval_retrieval.yaml").write_text(
        yaml.safe_dump({"mrr": ret.mrr, "recall_at": ret.recall_at, "n": ret.n}, sort_keys=False),
        encoding="utf-8",
    )
    log.info(f"Retrieval: MRR={ret.mrr:.4f}, n={ret.n}")

    # Analogy (only uses examples with before/after states)
    an = evaluate_analogy(deltas_path, run_dir, ks=list(ks), device=device, max_examples=cfg.get("max_examples"))
    (run_dir / "eval_analogy.yaml").write_text(
        yaml.safe_dump(
            {"avg_cosine": an.avg_cosine, "median_rank": an.median_rank, "recall_at": an.recall_at, "n": an.n},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    log.info(f"Analogy: avg_cos={an.avg_cosine:.4f}, median_rank={an.median_rank:.1f}, n={an.n}")

    # UMAP Figure 1
    um = cfg.get("umap", {}) or {}
    um_cfg = UMAPConfig(
        n_neighbors=int(um.get("n_neighbors", 15)),
        min_dist=float(um.get("min_dist", 0.1)),
        metric=str(um.get("metric", "cosine")),
        top_n_labels=int(um.get("top_n_labels", 50)),
    )
    fig_path = P.results_figures / f"figure1_umap_{run_dir.name}.png"
    save_umap_figure(run_dir=run_dir, out_path=fig_path, cfg=um_cfg, device=device)
    log.info(f"Saved UMAP figure: {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run evaluations (retrieval/analogy/umap).")
    ap.add_argument("--config", type=str, default="configs/eval.yaml")
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    P = paths(Path.cwd())
    cfg = yaml.safe_load((P.root / args.config).read_text(encoding="utf-8"))
    run_eval(cfg, (P.root / args.run_dir).resolve())


if __name__ == "__main__":
    main()
