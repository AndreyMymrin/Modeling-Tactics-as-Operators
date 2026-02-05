from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eglt.paths import paths
from eglt.utils.logging import setup_logging


def run_train(train_cfg: dict, run_dir: Path) -> None:
    """
    Dispatch training based on train_cfg["method"] in {"sgns","cbow","seq"}.

    Expected keys:
      - method: "sgns" | "cbow" | "seq"
      - dataset_path: for sgns/cbow (DeltaRecord JSONL)
      - step_records_path: for seq baseline (StepRecord JSONL)
      - run_name: used by wrappers; run_dir is already chosen by caller
    """
    log = setup_logging()
    method = str(train_cfg.get("method", "sgns")).lower()
    run_dir.mkdir(parents=True, exist_ok=True)

    if method == "sgns":
        from eglt.training.trainer import train_sgns
        train_sgns(train_cfg, run_dir)
        log.info("Trained SGNS.")
    elif method == "cbow":
        from eglt.training.trainer import train_cbow_delta
        train_cbow_delta(train_cfg, run_dir)
        log.info("Trained CBOW-Δ.")
    elif method == "seq":
        from eglt.models.seq_baseline import train_seq_baseline_sgns
        train_seq_baseline_sgns(train_cfg, run_dir)
        log.info("Trained SEQ baseline.")
    else:
        raise ValueError(f"Unknown method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run training for SGNS / CBOW-Δ / sequence baseline.")
    ap.add_argument("--config", type=str, required=True, help="Path to train_*.yaml")
    ap.add_argument("--run_dir", type=str, default="", help="Override output run dir (optional)")
    args = ap.parse_args()

    P = paths(Path.cwd())
    train_cfg = yaml.safe_load((P.root / args.config).read_text(encoding="utf-8"))

    run_name = str(train_cfg.get("run_name", "run"))
    run_dir = (P.results_runs / run_name) if not args.run_dir else (P.root / args.run_dir)
    run_train(train_cfg, run_dir.resolve())


if __name__ == "__main__":
    main()
