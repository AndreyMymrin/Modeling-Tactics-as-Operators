from __future__ import annotations

import re
from typing import List


def _looks_solved(goal_tokens: list[str], goal_raw: str) -> bool:
    """
    Heuristic for "goal solved / no goals".

    We intentionally keep it conservative and easy to adjust.
    """
    if not goal_tokens:
        return True

    s = (goal_raw or "").strip().lower()
    if "no goals" in s:
        return True

    # common patterns in pretty-print states
    if len(goal_tokens) >= 3 and goal_tokens[0] == "⊢" and goal_tokens[1].lower() == "no" and goal_tokens[2].lower().startswith("goals"):
        return True

    return False


def extract_typed_edits(
    before_raw: str | None,
    after_raw: str | None,
    before_ctx_tokens: list[str],
    after_ctx_tokens: list[str],
    before_goal_tokens: list[str],
    after_goal_tokens: list[str],
) -> List[str]:
    """
    Extract typed edit features between before/after proof states.

    SPEC intent (as used in the paper-style pipeline):
      - Add coarse, *typed* edit markers capturing structural changes that
        token-level deltas alone might miss or dilute.
      - Output is a list of string features like:
          ∆ADD_HYP, ∆DEL_HYP, ∆GOAL_SOLVED, ∆GOALS_INC, ∆GOALS_DEC,
          ∆CTX_GROW, ∆CTX_SHRINK, ∆GOAL_CHANGED, ∆CTX_CHANGED, ∆NO_CHANGE
        plus a small set of "flip/op" markers (see below).

    Implemented heuristics (documented and deterministic):
      1) Hypothesis/context changes:
         - Compare context token lengths and token equality to produce:
           ∆CTX_GROW / ∆CTX_SHRINK / ∆CTX_SAME
         - If after_ctx has *more* tokens than before_ctx -> ∆ADD_HYP
           (coarse proxy: new locals/hyps introduced).
         - If after_ctx has *fewer* tokens -> ∆DEL_HYP (proxy: hyps removed/consumed).

      2) Goal solved:
         - If after goal looks solved (no goals / empty goal segment) -> ∆GOAL_SOLVED

      3) Multi-goal dynamics:
         - Estimate num goals by counting '⊢' markers (if absent, assume 1 when tokens exist).
         - If count increases -> ∆GOALS_INC
         - If decreases -> ∆GOALS_DEC

      4) Goal/content change:
         - If goal tokens differ -> ∆GOAL_CHANGED
         - If context tokens differ -> ∆CTX_CHANGED
         - If both unchanged -> ∆NO_CHANGE (useful sanity signal)

      5) "Flips / ops" markers (very lightweight):
         We add operator-presence markers when *new* operators appear in AFTER goal tokens.
         This is a proxy for structural transformations:
           - ∆OP_INTRODUCED_<op>
         Operators tracked: {→, ∧, ∨, ¬, ∀, ∃, =, ≠, ≤, ≥, ∈}
         This is intentionally simple: paper-SPEC may define more specific typed edits
         (e.g., GOAL_SOLVED, ADD_HYP, FLIP_IMP, etc.). Replace/extend here once you have
         the exact definitions.

    Returns:
      List[str] of typed edit markers (order deterministic).

    Examples (doctest):
    >>> extract_typed_edits("h : P\\n⊢ Q", "h : P\\nx : Nat\\n⊢ Q", ["h",":","P"], ["h",":","P","x",":","Nat"], ["⊢","Q"], ["⊢","Q"])
    ['∆CTX_GROW', '∆ADD_HYP', '∆CTX_CHANGED']
    >>> extract_typed_edits("⊢ P", "⊢ no goals", [], [], ["⊢","P"], ["⊢","no","goals"])
    ['∆GOAL_SOLVED', '∆GOAL_CHANGED']
    """
    before_raw = before_raw or ""
    after_raw = after_raw or ""

    edits: list[str] = []

    # --- Context growth/shrink signals ---
    if len(after_ctx_tokens) > len(before_ctx_tokens):
        edits.append("∆CTX_GROW")
        edits.append("∆ADD_HYP")
    elif len(after_ctx_tokens) < len(before_ctx_tokens):
        edits.append("∆CTX_SHRINK")
        edits.append("∆DEL_HYP")
    else:
        edits.append("∆CTX_SAME")

    # --- Goal solved ---
    # Try to isolate raw goal-ish substrings for the solved heuristic:
    # If raw contains '⊢', take substring from first ⊢ onward.
    def goal_raw(s: str) -> str:
        i = s.find("⊢")
        return s[i:] if i >= 0 else s

    after_goal_raw = goal_raw(after_raw)
    if _looks_solved(after_goal_tokens, after_goal_raw):
        edits.append("∆GOAL_SOLVED")

    # --- Multi-goal dynamics (count ⊢ markers) ---
    def count_goals(tokens: list[str]) -> int:
        if not tokens:
            return 0
        c = sum(1 for t in tokens if t == "⊢")
        return c if c > 0 else 1

    g_before = count_goals(before_goal_tokens)
    g_after = count_goals(after_goal_tokens)
    if g_after > g_before:
        edits.append("∆GOALS_INC")
    elif g_after < g_before:
        edits.append("∆GOALS_DEC")

    # --- Content changes ---
    ctx_changed = before_ctx_tokens != after_ctx_tokens
    goal_changed = before_goal_tokens != after_goal_tokens

    if goal_changed:
        edits.append("∆GOAL_CHANGED")
    if ctx_changed:
        edits.append("∆CTX_CHANGED")
    if (not goal_changed) and (not ctx_changed):
        edits.append("∆NO_CHANGE")

    # --- Operator "op introduced" markers (proxy for flips/ops) ---
    tracked_ops = {"→", "∧", "∨", "¬", "∀", "∃", "=", "≠", "≤", "≥", "∈"}
    before_ops = {t for t in before_goal_tokens if t in tracked_ops}
    after_ops = {t for t in after_goal_tokens if t in tracked_ops}
    introduced = sorted(after_ops - before_ops)
    for op in introduced:
        # keep ASCII-safe marker
        tag = op
        tag = tag.replace("→", "IMP").replace("∧", "AND").replace("∨", "OR").replace("¬", "NOT")
        tag = tag.replace("∀", "FORALL").replace("∃", "EXISTS").replace("≠", "NEQ")
        tag = tag.replace("≤", "LE").replace("≥", "GE").replace("∈", "IN")
        edits.append(f"∆OP_INTRODUCED_{tag}")

    # Deterministic order already ensured by append + sorted(introduced)
    # Remove ∆CTX_SAME if context actually changed (can happen if equal lengths but different tokens)
    if "∆CTX_SAME" in edits and ctx_changed:
        edits = [e for e in edits if e != "∆CTX_SAME"]

    # Also, if we added ∆CTX_GROW/SHRINK we don't need ∆CTX_SAME anyway
    if ("∆CTX_GROW" in edits) or ("∆CTX_SHRINK" in edits):
        edits = [e for e in edits if e != "∆CTX_SAME"]

    # De-duplicate while preserving order (safety)
    seen = set()
    out: list[str] = []
    for e in edits:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out
