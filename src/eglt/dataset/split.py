from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from eglt.paths import ProjectPaths, paths as get_paths
from eglt.utils.io import write_jsonl


DEFAULT_SEED = 0


@dataclass(frozen=True)
class SplitLists:
    train: list[str]
    val: list[str]
    test: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def split_proof_ids(
    proof_ids: list[str],
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> SplitLists:
    """
    Deterministic split by proof_id with fixed seed.

    Note: ratios must sum to 1.0 (within float tolerance).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    uniq = sorted(set(proof_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # ensure all assigned
    n_test = n - n_train - n_val

    train = uniq[:n_train]
    val = uniq[n_train : n_train + n_val]
    test = uniq[n_train + n_val : n_train + n_val + n_test]
    return SplitLists(train=train, val=val, test=test)


def save_split_lists_txt(P: ProjectPaths, splits: SplitLists) -> Path:
    """
    Save proof_id lists to:
      data/processed/splits/train.txt
      data/processed/splits/val.txt
      data/processed/splits/test.txt
    """
    out_dir = P.data_processed / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "train.txt").write_text("\n".join(splits.train) + ("\n" if splits.train else ""), encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(splits.val) + ("\n" if splits.val else ""), encoding="utf-8")
    (out_dir / "test.txt").write_text("\n".join(splits.test) + ("\n" if splits.test else ""), encoding="utf-8")
    return out_dir


def read_split_list_txt(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    return [line.strip() for line in txt.splitlines() if line.strip()]


def make_splits(cfg: dict, out_path: str | Path | None = None) -> SplitLists:
    """
    Create splits from config.

    Expected cfg keys (minimal):
      - proof_ids: optional explicit list of proof_id
      - proof_ids_file: optional path to a txt/jsonl containing proof ids
      - seed: optional seed (default 0)

    Saves lists to data/processed/splits/*.txt by spec.

    Optionally writes a jsonl summary to out_path (thin compatibility with scripts/make_splits.py).
    """
    P = get_paths(Path.cwd())
    seed = int(cfg.get("seed", DEFAULT_SEED))

    proof_ids: list[str] = []
    if "proof_ids" in cfg and cfg["proof_ids"] is not None:
        proof_ids = [str(x) for x in cfg["proof_ids"]]
    elif "proof_ids_file" in cfg and cfg["proof_ids_file"]:
        p = (P.root / str(cfg["proof_ids_file"])).resolve()
        # accept simple txt list
        proof_ids = read_split_list_txt(p)
    else:
        raise ValueError("Config must provide either 'proof_ids' or 'proof_ids_file'.")

    splits = split_proof_ids(proof_ids, seed=seed)
    save_split_lists_txt(P, splits)

    if out_path is not None:
        out_path = Path(out_path)
        # write one-line jsonl with lists for convenience/debugging
        write_jsonl(out_path, [{"seed": seed, **splits.to_dict()}])

    return splits
