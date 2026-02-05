from __future__ import annotations

import re

# Strip comments and normalize whitespace; extract "head" tactic:
# - remove leading `by` / `simp` wrappers
# - take first token-like identifier (including `simp`, `rw`, `intro`, etc.)
# - handle `tactic; tactic` by taking the first head
# - handle `tactic <;> tactic` similarly
#
# This is a pragmatic parser, not full Lean parsing.

_WS_RE = re.compile(r"\s+")
# Remove line comments `-- ...` and block comments `/- ... -/` (non-nested heuristic)
_LINE_COMMENT = re.compile(r"--.*?$", flags=re.MULTILINE)
_BLOCK_COMMENT = re.compile(r"/-.*?-/", flags=re.DOTALL)

# Split on combinators ; and <;>
_SPLIT_COMBINATORS = re.compile(r"(<;>|;)")


def extract_tactic_head(tactic_str: str) -> str:
    """
    Extract a canonical "head" tactic label from a tactic string.

    Examples (doctest):
    >>> extract_tactic_head("intro x")
    'intro'
    >>> extract_tactic_head("  simp [h]  ")
    'simp'
    >>> extract_tactic_head("rw [h]; simp")
    'rw'
    >>> extract_tactic_head("by exact h")
    'exact'
    >>> extract_tactic_head("")
    ''
    """
    if not tactic_str:
        return ""

    s = tactic_str.strip()
    s = _BLOCK_COMMENT.sub(" ", s)
    s = _LINE_COMMENT.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""

    # drop leading "by"
    if s.startswith("by "):
        s = s[3:].strip()

    # take first segment before combinators
    parts = _SPLIT_COMBINATORS.split(s)
    if parts:
        s0 = parts[0].strip()
    else:
        s0 = s

    if not s0:
        return ""

    # token head: first identifier-ish token (Lean allows `_` and `'`)
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_']*)", s0)
    if m:
        return m.group(1)

    # fallback: sometimes tactic starts with `(` or `[` etc.
    # try to find first identifier anywhere
    m2 = re.search(r"([A-Za-z_][A-Za-z0-9_']*)", s0)
    return m2.group(1) if m2 else ""
