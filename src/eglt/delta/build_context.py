from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

from eglt.dataset.schema import DeltaRecord, StepRecord
from eglt.delta.token_delta import positive_token_deltas
from eglt.delta.typed_edits import extract_typed_edits
from eglt.preprocess.alpha_rename import alpha_rename
from eglt.preprocess.canonicalize import canonicalize_tokens, canonicalize_text
from eglt.preprocess.goal_context import split_context_goal
from eglt.preprocess.tactic_head import extract_tactic_head
from eglt.preprocess.tokenize import tokenize_state


def _get_raw_states(step: StepRecord) -> tuple[str | None, str | None]:
    """
    Try to retrieve raw before/after texts from StepRecord.extra.
    Common keys (you can adapt to your dataset):
      - "state_before", "state_after"
      - "before_state", "after_state"
      - "state_before_raw", "state_after_raw"
    """
    ex = step.extra or {}
    before_raw = ex.get("state_before") or ex.get("before_state") or ex.get("state_before_raw")
    after_raw = ex.get("state_after") or ex.get("after_state") or ex.get("state_after_raw")
    return (str(before_raw) if before_raw is not None else None,
            str(after_raw) if after_raw is not None else None)


def _ensure_tokens(raw: str | None, fallback_tokens: list[str] | None) -> list[str]:
    """
    Prefer provided token lists from StepRecord; otherwise tokenize raw if available.
    """
    if fallback_tokens is not None:
        return list(fallback_tokens)
    if raw is None:
        return []
    raw = canonicalize_text(raw)
    return tokenize_state(raw)


def build_delta_context(step: StepRecord) -> DeltaRecord:
    """
    Build DeltaRecord for a single StepRecord.

    Pipeline (deterministic):
      1) Obtain before/after tokens:
         - use step.state_before_tokens/state_after_tokens if present,
           else tokenize raw strings (if available in step.extra)
      2) Canonicalize tokens (NFKC etc.)
      3) Split into context vs goal via '⊢'
      4) Alpha-rename identifiers consistently within each state
         (NOTE: by default we rename *within each list* independently.
          If you want a shared mapping across before/after, you can change this.)
      5) Compute:
         - positive TOK deltas over CONTEXT tokens only (multiset)
         - typed edits (coarse structural markers)
      6) Combine into delta_context multiset C (list with multiplicities)

    Output:
      DeltaRecord(proof_id, step_id, tactic_head, delta_context, optional state tokens)

    Minimal examples (doctest):
    >>> s = StepRecord(proof_id="p1", step_id="0", tactic_head="intro", state_before_tokens=["⊢","P","x"], state_after_tokens=["h",":","P","x","⊢","Q"])
    >>> d = build_delta_context(s)
    >>> d.tactic_head
    'intro'
    >>> "∆ADD_HYP" in d.delta_context
    True
    """
    before_raw, after_raw = _get_raw_states(step)

    before_tokens = _ensure_tokens(before_raw, step.state_before_tokens)
    after_tokens = _ensure_tokens(after_raw, step.state_after_tokens)

    before_tokens = canonicalize_tokens(before_tokens, normalize_numbers=False)
    after_tokens = canonicalize_tokens(after_tokens, normalize_numbers=False)

    b_ctx, b_goal, _, _ = split_context_goal(before_tokens)
    a_ctx, a_goal, _, _ = split_context_goal(after_tokens)

    # alpha-rename within each state independently (deterministic)
    b_ctx_r, _ = alpha_rename(b_ctx)
    a_ctx_r, _ = alpha_rename(a_ctx)
    b_goal_r, _ = alpha_rename(b_goal)
    a_goal_r, _ = alpha_rename(a_goal)

    tok_deltas = positive_token_deltas(b_ctx_r, a_ctx_r)
    typed = extract_typed_edits(
        before_raw=before_raw,
        after_raw=after_raw,
        before_ctx_tokens=b_ctx_r,
        after_ctx_tokens=a_ctx_r,
        before_goal_tokens=b_goal_r,
        after_goal_tokens=a_goal_r,
    )

    # Combine into multiset C: typed edits + TOK deltas (positive only)
    delta_context: list[str] = []
    delta_context.extend(typed)
    delta_context.extend(tok_deltas)

    # Tactic head: prefer already pre-extracted label, but normalize if needed
    th = step.tactic_head or ""
    th2 = extract_tactic_head(th) if th else ""
    tactic_head = th2 if th2 else th

    return DeltaRecord(
        proof_id=step.proof_id,
        step_id=step.step_id,
        tactic_head=tactic_head,
        delta_context=delta_context,
        # Keep original (pre-rename) tokens if present; useful for analogy/state embeddings later
        state_before_tokens=before_tokens if before_tokens else None,
        state_after_tokens=after_tokens if after_tokens else None,
    )
