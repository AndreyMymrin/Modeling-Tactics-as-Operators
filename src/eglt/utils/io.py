from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def read_parquet(path: str | Path) -> pd.DataFrame:
    """
    Optional parquet support if pyarrow is installed.
    """
    if not has_pyarrow():
        raise RuntimeError("pyarrow is not installed. Install with: pip install .[parquet]")
    return pd.read_parquet(Path(path))


def write_parquet(path: str | Path, df: pd.DataFrame) -> None:
    """
    Optional parquet support if pyarrow is installed.
    """
    if not has_pyarrow():
        raise RuntimeError("pyarrow is not installed. Install with: pip install .[parquet]")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
