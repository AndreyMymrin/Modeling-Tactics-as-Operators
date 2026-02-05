from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eglt.paths import paths
from eglt.utils.logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Run retrieval/analogy/umap evaluations.")
    ap.add_argument("--config", type=str, default="configs/eval.yaml")
    ap.add_argument("--run_dir", type=str, required=True, help="Path to a trained run directory")
    args = ap.parse_args()

    log = setup_logging()
    P = paths(Path.cwd())

    cfg = yaml.safe_load((P.root / args.config).read_text(encoding="utf-8"))
    run_dir = (P.root / args.run_dir).resolve()

    # Deferred import: you will implement this later
    from eglt.experiments.run_eval import run_eval  # type: ignore

    run_eval(cfg=cfg, run_dir=run_dir)
    log.info("Eval complete.")


if __name__ == "__main__":
    main()
