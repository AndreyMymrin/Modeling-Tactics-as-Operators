from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eglt.paths import paths
from eglt.utils.logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Build delta contexts (TOK_w + typed edits) from stepwise states.")
    ap.add_argument("--dataset", type=str, default="configs/dataset.yaml")
    ap.add_argument("--preprocess", type=str, default="configs/preprocess.yaml")
    ap.add_argument("--out", type=str, default="data/processed/deltas.jsonl")
    args = ap.parse_args()

    log = setup_logging()
    P = paths(Path.cwd())

    dataset_cfg = yaml.safe_load((P.root / args.dataset).read_text(encoding="utf-8"))
    prep_cfg = yaml.safe_load((P.root / args.preprocess).read_text(encoding="utf-8"))
    out_path = P.root / args.out

    log.info("Loaded configs.")
    # Deferred import: you will implement this later
    from eglt.delta.build_context import build_delta_dataset  # type: ignore

    build_delta_dataset(dataset_cfg=dataset_cfg, preprocess_cfg=prep_cfg, out_path=out_path)
    log.info(f"Wrote deltas: {out_path}")


if __name__ == "__main__":
    main()
