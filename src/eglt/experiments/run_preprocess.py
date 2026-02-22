from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from eglt.dataset.load_raw import load_step_records_jsonl
from eglt.delta.build_context import BuildDeltaConfig, build_delta_context
from eglt.paths import paths
from eglt.utils.io import write_jsonl
from eglt.utils.logging import setup_logging



def _build_delta_cfg_from_yaml(preprocess_cfg: dict) -> BuildDeltaConfig:
    canonicalize = preprocess_cfg.get("canonicalize", {})
    alpha = preprocess_cfg.get("alpha_rename", {})
    delta = preprocess_cfg.get("delta_context", {})

    return BuildDeltaConfig(
        normalize_numbers=bool(canonicalize.get("normalize_numbers", False)),
        alpha_rename_enabled=bool(alpha.get("enabled", True)),
        alpha_rename_prefix=str(alpha.get("prefix", "_x")),
        alpha_rename_shared_mapping=bool(alpha.get("shared_mapping", True)),
        include_tok_deltas=bool(delta.get("include_tok_deltas", True)),
        include_typed_edits=bool(delta.get("include_typed_edits", True)),
    )


def run_preprocess(dataset_cfg: dict, preprocess_cfg: dict) -> Path:
    """
    Build DeltaRecord JSONL from StepRecord JSONL.
    Thin orchestrator; heavy logic is in build_delta_context.

    Expected dataset_cfg keys:
      - raw_jsonl: path under data/raw or absolute
      - out_deltas_jsonl: path under data/processed

    preprocess_cfg controls normalization / alpha-renaming / feature toggles
    passed into build_delta_context.
    """
    log = setup_logging()
    P = paths(Path.cwd())

    raw_path = Path(dataset_cfg["raw_jsonl"])
    if not raw_path.is_absolute():
        raw_path = (P.root / raw_path).resolve()

    out_path = Path(dataset_cfg.get("out_deltas_jsonl", "data/processed/deltas.jsonl"))
    if not out_path.is_absolute():
        out_path = (P.root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading raw JSONL: {raw_path}")
    records = list(load_step_records_jsonl(raw_path))

    delta_cfg = _build_delta_cfg_from_yaml(preprocess_cfg)

    # Build deltas
    out_rows = []
    for r in tqdm(records, desc="build_delta_context"):
        d = build_delta_context(r, cfg=delta_cfg)
        out_rows.append(d.to_dict())

    write_jsonl(out_path, out_rows)
    log.info(f"Wrote DeltaRecord JSONL: {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess raw StepRecord JSONL into DeltaRecord JSONL.")
    ap.add_argument("--dataset", type=str, default="configs/dataset.yaml")
    ap.add_argument("--preprocess", type=str, default="configs/preprocess.yaml")
    args = ap.parse_args()

    P = paths(Path.cwd())
    dataset_cfg = yaml.safe_load((P.root / args.dataset).read_text(encoding="utf-8"))
    preprocess_cfg = yaml.safe_load((P.root / args.preprocess).read_text(encoding="utf-8"))

    run_preprocess(dataset_cfg, preprocess_cfg)


if __name__ == "__main__":
    main()
