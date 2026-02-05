from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """
    Single source of truth for repository-relative paths.

    Creates required directories:
      - data/raw
      - data/interim
      - data/processed
      - results/runs
    """
    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def data_raw(self) -> Path:
        return self.data / "raw"

    @property
    def data_interim(self) -> Path:
        return self.data / "interim"

    @property
    def data_processed(self) -> Path:
        return self.data / "processed"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def results_tables(self) -> Path:
        return self.results / "tables"

    @property
    def results_figures(self) -> Path:
        return self.results / "figures"

    @property
    def results_runs(self) -> Path:
        return self.results / "runs"

    def ensure(self) -> "ProjectPaths":
        # required by spec
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.data_interim.mkdir(parents=True, exist_ok=True)
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.results_runs.mkdir(parents=True, exist_ok=True)

        # convenient extras (won't hurt)
        self.results_tables.mkdir(parents=True, exist_ok=True)
        self.results_figures.mkdir(parents=True, exist_ok=True)
        return self


def get_repo_root(start: Path | None = None) -> Path:
    """
    Best-effort repo root discovery.
    - If called from scripts/, start is typically that script's location.
    - Walk up until pyproject.toml found; else fallback to cwd.
    """
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd().resolve()


def paths(start: Path | None = None) -> ProjectPaths:
    return ProjectPaths(root=get_repo_root(start)).ensure()
