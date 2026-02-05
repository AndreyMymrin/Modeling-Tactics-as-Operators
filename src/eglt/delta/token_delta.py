from __future__ import annotations

from collections import Counter
from typing import List


def positive_token_deltas(before_ctx_tokens: list[str], after_ctx_tokens: list[str]) -> List[str]:
    """
    Positive-only multiset delta over context tokens.

    Returns list of tokens already prefixed as "TOK_<token>" with multiplicities.
    If a token appears k more times in after than before, we emit it k times.

    Notes:
    - This is order-invariant (multiset).
    - Negative changes (removals) are ignored by design.

    Examples (doctest):
    >>> positive_token_deltas(["h", ":", "P"], ["h", ":", "P", "x", "x"])
    ['TOK_x', 'TOK_x']
    >>> positive_token_deltas(["a", "a"], ["a"])
    []
    """
    b = Counter(before_ctx_tokens)
    a = Counter(after_ctx_tokens)

    out: list[str] = []
    for tok, a_cnt in a.items():
        diff = a_cnt - b.get(tok, 0)
        if diff > 0:
            out.extend([f"TOK_{tok}"] * diff)
    return out
