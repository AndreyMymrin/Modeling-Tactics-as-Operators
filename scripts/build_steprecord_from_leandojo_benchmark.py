from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple
from urllib.request import urlretrieve


ZENODO_TARBALL_URL = "https://zenodo.org/records/15372180/files/leandojo_benchmark_4.tar.gz?download=1"


def extract_tactic_head(tactic: str) -> str:
    s = (tactic or "").strip()
    if s.startswith("by "):
        s = s[3:].lstrip()
    return s.split()[0] if s else ""


def ensure_extracted(tar_gz_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted.ok"
    if marker.exists():
        return out_dir

    with tarfile.open(tar_gz_path, "r:gz") as tf:
        tf.extractall(out_dir)

    marker.write_text("ok\n", encoding="utf-8")
    return out_dir


def iter_theorems(json_path: Path) -> Iterator[Dict[str, Any]]:
    """
    LeanDojo split files are JSON (not JSONL).
    Usually it's a list of theorem dicts.
    """
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
    elif isinstance(obj, dict):
        # just in case some version wraps list
        for k in ("theorems", "data", "items"):
            v = obj.get(k)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        yield x
                return
        raise ValueError(f"Unsupported JSON structure in {json_path}")
    else:
        raise ValueError(f"Unsupported JSON type in {json_path}: {type(obj)}")


def build_proof_id(thm: Dict[str, Any]) -> str:
    file_path = str(thm.get("file_path", ""))
    full_name = str(thm.get("full_name", ""))
    return f"{file_path}::{full_name}"


def convert_split_to_steprecords(
    split_json: Path,
    out_jsonl: Path,
    limit_theorems: int | None = None,
    limit_steps_per_theorem: int | None = None,
) -> Tuple[int, int]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_thm = 0
    n_steps = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for thm in iter_theorems(split_json):
            if limit_theorems is not None and n_thm >= limit_theorems:
                break

            proof_id = build_proof_id(thm)
            traced = thm.get("traced_tactics", [])
            if not isinstance(traced, list) or len(traced) == 0:
                continue

            step_id = 0
            for tt in traced:
                if limit_steps_per_theorem is not None and step_id >= limit_steps_per_theorem:
                    break
                if not isinstance(tt, dict):
                    continue

                tactic = str(tt.get("tactic", "")).strip()
                state_before = str(tt.get("state_before", ""))
                state_after = str(tt.get("state_after", ""))

                # Minimal StepRecord-compatible row
                row = {
                    "proof_id": proof_id,
                    "step_id": str(step_id),
                    "tactic_head": extract_tactic_head(tactic),
                    # Keep raw states in extra fields that your pipeline reads in typed_edits
                    "state_before": state_before,
                    "state_after": state_after,
                    # Optional: keep original tactic text
                    "tactic": tactic,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

                step_id += 1
                n_steps += 1

            n_thm += 1

    return n_thm, n_steps


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert LeanDojo Benchmark split JSON to StepRecord JSONL for EGLT."
    )
    ap.add_argument("--root", type=str, default="data/raw/leandojo_benchmark_4",
                    help="Directory containing extracted LeanDojo Benchmark 4 files.")
    ap.add_argument("--download", action="store_true",
                    help="Download leandojo_benchmark_4.tar.gz from Zenodo into --tar and extract into --root.")
    ap.add_argument("--tar", type=str, default="data/raw/leandojo_benchmark_4.tar.gz",
                    help="Where to store the downloaded tar.gz (used if --download).")
    ap.add_argument("--split_kind", type=str, default="random", choices=["random", "novel_premises"],
                    help="Which split strategy to use.")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                    help="Which split file to convert.")
    ap.add_argument("--out", type=str, default="data/raw/steps.jsonl",
                    help="Output StepRecord JSONL path.")
    ap.add_argument("--limit_theorems", type=int, default=0,
                    help="Optional limit theorems (0 = no limit).")
    ap.add_argument("--limit_steps", type=int, default=0,
                    help="Optional limit steps per theorem (0 = no limit).")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    if args.download:
        tar_path = Path(args.tar)
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading: {ZENODO_TARBALL_URL}")
        urlretrieve(ZENODO_TARBALL_URL, tar_path)
        print(f"Saved tar.gz: {tar_path.resolve()}")
        ensure_extracted(tar_path, root)
        print(f"Extracted to: {root.resolve()}")

    # Find split json
    split_json = root / args.split_kind / f"{args.split}.json"
    if not split_json.exists():
        # some tarballs contain an extra top-level folder; try to locate it
        cands = list(root.rglob(f"{args.split_kind}/{args.split}.json"))
        if len(cands) == 1:
            split_json = cands[0]
        else:
            raise FileNotFoundError(f"Could not find split file: {root}/{args.split_kind}/{args.split}.json")

    lt = None if args.limit_theorems <= 0 else int(args.limit_theorems)
    ls = None if args.limit_steps <= 0 else int(args.limit_steps)

    n_thm, n_steps = convert_split_to_steprecords(
        split_json=split_json,
        out_jsonl=out,
        limit_theorems=lt,
        limit_steps_per_theorem=ls,
    )
    print(f"DONE. theorems={n_thm}, steps={n_steps}")
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
