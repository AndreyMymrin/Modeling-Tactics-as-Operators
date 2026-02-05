from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Heuristic for identifiers we want to rename:
# - starts with a letter/_ and continues with letters/digits/_/' (Lean often uses ').
# - excludes obvious keywords/symbols handled elsewhere.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")


def alpha_rename(tokens: List[str], prefix: str = "_x") -> Tuple[List[str], Dict[str, str]]:
    """
    Deterministically alpha-rename identifier-like tokens by order of first appearance.

    Returns (new_tokens, mapping).

    Only renames tokens matching IDENT_RE (heuristic).
    Mapping is stable within a call and depends only on token order.

    >>> alpha_rename(["x", ":", "Nat", "⊢", "x", "=", "y"])[0]
    ['_x1', ':', 'Nat', '⊢', '_x1', '=', '_x2']
    >>> alpha_rename(["h1", "h1", "h2"])[1]
    {'h1': '_x1', 'h2': '_x2'}
    """
    mapping: Dict[str, str] = {}
    out: List[str] = []
    k = 0

    for t in tokens:
        if _IDENT_RE.match(t):
            if t not in mapping:
                k += 1
                mapping[t] = f"{prefix}{k}"
            out.append(mapping[t])
        else:
            out.append(t)
    return out, mapping

