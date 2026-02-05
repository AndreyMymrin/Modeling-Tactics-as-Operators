from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eglt.paths import paths
from eglt.utils.logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Create train/val/test split by proof_id.")
    ap.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset.yaml")
    ap.add_argument("--out", type=str, default="data/interim/splits.jsonl", help="Output splits file")
    args = ap.parse_args()

    log = setup_logging()
    P = paths(Path.cwd())
    cfg_path = P.root / args.config
    out_path = P.root / args.out

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    log.info(f"Loaded config: {cfg_path}")

    # Deferred import: you will implement this later
    from eglt.dataset.split import make_splits  # type: ignore

    make_splits(cfg=cfg, out_path=out_path)
    log.info(f"Wrote splits: {out_path}")


if __name__ == "__main__":
    main()
