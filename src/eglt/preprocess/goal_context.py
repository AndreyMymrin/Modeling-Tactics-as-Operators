from __future__ import annotations

from typing import List, Tuple

# In Lean pretty-printed states, goals are usually separated by '⊢' markers.
# We'll treat everything before the FIRST '⊢' as context (hypotheses/locals),
# and the segment after as goal tokens (potentially multiple goals).
#
# If there is no '⊢', we assume the whole thing is "goal" (context empty).
#
# solved_flag: heuristic: if goal part is empty OR contains common "no goals" phrases.

_SOLVED_PHRASES = {
    "no", "goals", "No", "Goals", "goals.", "goals!", "⊢",
}


def split_context_goal(tokens: List[str]) -> Tuple[List[str], List[str], int, bool]:
    """
    Split tokens into (context_tokens, goal_tokens, num_goals, solved_flag).

    Heuristics:
      - num_goals = count of '⊢' markers (0 -> 1 implicit goal if tokens non-empty)
      - solved_flag = True if goal_tokens is empty OR contains a "no goals" phrase.

    Examples (doctest):
    >>> split_context_goal(["h", ":", "P", "x", "⊢", "Q", "x"])[:3]
    (['h', ':', 'P', 'x'], ['⊢', 'Q', 'x'], 1)
    >>> split_context_goal(["⊢", "no", "goals"])[3]
    True
    >>> split_context_goal(["Q", "x"])[0]
    []
    """
    if not tokens:
        return [], [], 0, True

    idx = None
    for i, t in enumerate(tokens):
        if t == "⊢":
            idx = i
            break

    if idx is None:
        # no marker: treat as goal-only
        context_tokens = []
        goal_tokens = tokens
        num_goals = 1 if tokens else 0
    else:
        context_tokens = tokens[:idx]
        goal_tokens = tokens[idx:]
        num_goals = sum(1 for t in tokens if t == "⊢")
        if num_goals == 0:
            num_goals = 1

    # solved heuristic
    solved_flag = False
    if not goal_tokens:
        solved_flag = True
    else:
        # "⊢ no goals" etc.
        # Check small windows for phrase presence
        gt = goal_tokens
        if len(gt) >= 3 and gt[0] == "⊢" and gt[1] == "no" and gt[2].startswith("goals"):
            solved_flag = True
        elif any(t in _SOLVED_PHRASES for t in gt):
            # overly permissive, but safe as a sanity signal
            # (downstream can ignore / recompute)
            solved_flag = ("no" in gt and any(x.startswith("goals") for x in gt))

    return context_tokens, goal_tokens, num_goals, solved_flag
