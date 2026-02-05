from __future__ import annotations

import re
from typing import List

# Lean-ish unicode operators and symbols frequently appearing in pretty-printed goals.
# We treat them as standalone tokens.
_UNICODE_OPS = [
    "⊢", "→", "←", "↔", "∀", "∃", "λ", "Π", "Σ",
    "¬", "∧", "∨", "⊑", "≤", "≥", "≠", "≈",
    "∈", "∉", "⊆", "⊂", "⊇", "⊃", "∪", "∩",
    "⊕", "⊗", "⋆", "⋅", "∘",
    "⟂", "⊥", "⊤",
    "≃", "≅", "≡",
    "⟦", "⟧", "⟨", "⟩",
    "“", "”", "‘", "’",
    "·", "•", "⟮", "⟯",
    "↦", "⇒", "⇐", "⇑", "⇓", "⇔",
]

# Also split on common ASCII operator-ish chars.
_ASCII_OPS = list("()[]{}.,;:+-*/=<>|&!^~?:")

# Build a regex that captures operators/punct as separate tokens.
# Order matters: longer tokens first.
_ops_sorted = sorted(set(_UNICODE_OPS + _ASCII_OPS), key=len, reverse=True)
_OPS_RE = re.compile("(" + "|".join(re.escape(x) for x in _ops_sorted) + ")")

# Whitespace collapse
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def tokenize_state(text: str) -> List[str]:
    """
    Tokenize Lean pretty-printed state / goal text into a list of tokens.

    - Collapses whitespace.
    - Splits unicode and ascii operators/punctuation into separate tokens.
    - Keeps identifiers and numerals as tokens.

    This is intentionally simple and deterministic; downstream canonicalize/alpha_rename
    handle normalization.

    Examples (doctest):
    >>> tokenize_state("h : α → β\\n⊢ ∀ x, P x")
    ['h', ':', 'α', '→', 'β', '⊢', '∀', 'x', ',', 'P', 'x']
    >>> tokenize_state("⊢ x≤y ∧ y≠z")
    ['⊢', 'x', '≤', 'y', '∧', 'y', '≠', 'z']
    >>> tokenize_state("f (g x)=y")
    ['f', '(', 'g', 'x', ')', '=', 'y']
    """
    if not text:
        return []

    s = _WS_RE.sub(" ", text.strip())
    # Surround operators with spaces, then split.
    s = _OPS_RE.sub(r" \1 ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return []
    return s.split(" ")

