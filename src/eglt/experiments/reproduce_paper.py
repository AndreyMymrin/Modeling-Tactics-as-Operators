from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eglt.experiments.run_preprocess import run_preprocess
from eglt.experiments.run_train import run_train
from eglt.experiments.run_eval import run_eval
from eglt.paths import paths
from eglt.utils.logging import setup_logging


def reproduce(
    dataset_cfg_path: str,
    preprocess_cfg_path: str,
    train_cfg_path: str,
    eval_cfg_path: str,
) -> None:
    """
    End-to-end pipeline driven by config files (SPEC):
      1) preprocess raw -> deltas
      2) train (sgns/cbow/seq)
      3) eval (retrieval/analogy/umap)

    Conventions:
      - dataset.yaml contains raw_jsonl and out_deltas_jsonl
      - train_*.yaml contains method + run_name + dataset_path/step_records_path
      - eval.yaml references deltas_path and uses the produced run_dir
    """
    log = setup_logging()
    P = paths(Path.cwd())

    dataset_cfg = yaml.safe_load((P.root / dataset_cfg_path).read_text(encoding="utf-8"))
    preprocess_cfg = yaml.safe_load((P.root / preprocess_cfg_path).read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((P.root / train_cfg_path).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load((P.root / eval_cfg_path).read_text(encoding="utf-8"))

    # 1) Preprocess
    deltas_path = run_preprocess(dataset_cfg, preprocess_cfg)

    # 2) Train: ensure train_cfg points to produced delta file if needed
    method = str(train_cfg.get("method", "sgns")).lower()
    if method in {"sgns", "cbow"}:
        train_cfg["dataset_path"] = str(deltas_path)
    elif method == "seq":
        # expects step_records_path; if absent, try raw_jsonl
        train_cfg.setdefault("step_records_path", dataset_cfg.get("raw_jsonl"))

    run_name = str(train_cfg.get("run_name", "run"))
    run_dir = (P.results_runs / run_name).resolve()
    log.info(f"Run dir: {run_dir}")

    run_train(train_cfg, run_dir)

    # 3) Eval: ensure eval_cfg points to produced deltas file
    eval_cfg["deltas_path"] = str(deltas_path)
    run_eval(eval_cfg, run_dir)

    log.info("Reproduction pipeline finished.")


def main() -> None:
    ap = argparse.ArgumentParser(description="One-shot reproduction pipeline (configs-driven).")
    ap.add_argument("--dataset", type=str, default="configs/dataset.yaml")
    ap.add_argument("--preprocess", type=str, default="configs/preprocess.yaml")
    ap.add_argument("--train", type=str, default="configs/train_sgns.yaml")
    ap.add_argument("--eval", type=str, default="configs/eval.yaml")
    args = ap.parse_args()

    reproduce(
        dataset_cfg_path=args.dataset,
        preprocess_cfg_path=args.preprocess,
        train_cfg_path=args.train,
        eval_cfg_path=args.eval,
    )


if __name__ == "__main__":
    main()
