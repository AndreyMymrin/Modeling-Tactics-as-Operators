from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from eglt.dataset.schema import StepRecord
from eglt.utils.io import read_jsonl


def load_step_records_jsonl(path: str | Path) -> Iterator[StepRecord]:
    """
    Read JSONL where each line is a dict for StepRecord.
    """
    for row in read_jsonl(path):
        yield StepRecord.from_dict(row)


def load_many_jsonl(paths: Iterable[str | Path]) -> Iterator[StepRecord]:
    """
    Convenience: stream multiple JSONL files.
    """
    for p in paths:
        yield from load_step_records_jsonl(p)
