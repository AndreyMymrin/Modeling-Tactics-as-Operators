from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_json_dumps(obj: Any) -> str:
    """
    Stable, reproducible JSON serialization for config hashing.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_config(cfg: dict[str, Any]) -> str:
    return sha256_bytes(stable_json_dumps(cfg).encode("utf-8"))
