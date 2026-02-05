from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int) -> None:
    """
    Make runs deterministic-ish.
    Note: true determinism on GPU can still vary by ops / drivers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Reasonable defaults for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
