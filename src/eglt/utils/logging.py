from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from rich.logging import RichHandler


@dataclass(frozen=True)
class LogConfig:
    level: str = "INFO"
    rich: bool = True


def setup_logging(cfg: LogConfig | None = None) -> logging.Logger:
    """
    Setup a consistent logger for both library + scripts.
    """
    cfg = cfg or LogConfig()
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    handlers: list[logging.Handler]
    if cfg.rich:
        handlers = [RichHandler(rich_tracebacks=True, show_path=False)]
    else:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        h.setFormatter(fmt)
        handlers = [h]

    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(message)s" if cfg.rich else None,
        datefmt="[%X]",
        force=True,
    )
    return logging.getLogger("eglt")
