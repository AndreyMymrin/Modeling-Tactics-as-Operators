from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

# Optional normalization of numerals:
# - keep exact numbers (default) OR map all to a placeholder.
_NUM_RE = re.compile(r"^(?:[0-9]+|0x[0-9a-fA-F]+)$")


def canonicalize_text(text: str) -> str:
    """
    Basic canonicalization for Lean pretty-print text:
    - unicode normalization (NFKC)
    - whitespace collapse

    >>> canonicalize_text("A\\u00a0\\tB")
    'A B'
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = _WS_RE.sub(" ", s.strip())
    return s


def canonicalize_tokens(tokens: List[str], normalize_numbers: bool = False) -> List[str]:
    """
    Token-level canonicalization:
    - NFKC each token
    - optionally map numeric literals to "<NUM>"

    >>> canonicalize_tokens(["x", "≤", "10"], normalize_numbers=True)
    ['x', '≤', '<NUM>']
    >>> canonicalize_tokens(["0xFF"], normalize_numbers=True)
    ['<NUM>']
    """
    out: List[str] = []
    for t in tokens:
        t2 = unicodedata.normalize("NFKC", t)
        if normalize_numbers and _NUM_RE.match(t2):
            out.append("<NUM>")
        else:
            out.append(t2)
    return out
