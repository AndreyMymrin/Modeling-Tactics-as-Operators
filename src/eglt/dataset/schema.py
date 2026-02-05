from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class StepRecord:
    """
    Raw / stepwise record (JSONL-friendly).

    Required:
      - proof_id: groups steps belonging to a single proof
      - step_id: step index within proof (can be int or str; keep as str for safety)
      - tactic_head: canonicalized "head" tactic label (as in paper)
    Optional:
      - state_before_tokens/state_after_tokens: tokenized goal states (if available)
      - extra: passthrough bag for any other fields in raw dataset
    """
    proof_id: str
    step_id: str
    tactic_head: str

    state_before_tokens: Optional[list[str]] = None
    state_after_tokens: Optional[list[str]] = None

    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StepRecord":
        # tolerant parsing: keep unknown fields in extra
        proof_id = str(d.get("proof_id", ""))
        step_id = str(d.get("step_id", ""))
        tactic_head = str(d.get("tactic_head", ""))

        state_before_tokens = d.get("state_before_tokens", None)
        state_after_tokens = d.get("state_after_tokens", None)

        # normalize empty -> None
        if state_before_tokens == []:
            state_before_tokens = None
        if state_after_tokens == []:
            state_after_tokens = None

        known = {"proof_id", "step_id", "tactic_head", "state_before_tokens", "state_after_tokens"}
        extra = {k: v for k, v in d.items() if k not in known}

        return StepRecord(
            proof_id=proof_id,
            step_id=step_id,
            tactic_head=tactic_head,
            state_before_tokens=state_before_tokens,
            state_after_tokens=state_after_tokens,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "proof_id": self.proof_id,
            "step_id": self.step_id,
            "tactic_head": self.tactic_head,
        }
        if self.state_before_tokens is not None:
            d["state_before_tokens"] = self.state_before_tokens
        if self.state_after_tokens is not None:
            d["state_after_tokens"] = self.state_after_tokens
        # include extra fields
        d.update(self.extra)
        return d


@dataclass(frozen=True)
class DeltaRecord:
    """
    Processed record for learning/eval.

    Required:
      - proof_id, step_id, tactic_head
      - delta_context: List[str] (tokens representing Δ / typed edits / TOK_w, etc.)
    Optional:
      - state_before_tokens/state_after_tokens: for analogy/state embedding experiments
    """
    proof_id: str
    step_id: str
    tactic_head: str
    delta_context: list[str]

    state_before_tokens: Optional[list[str]] = None
    state_after_tokens: Optional[list[str]] = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DeltaRecord":
        proof_id = str(d.get("proof_id", ""))
        step_id = str(d.get("step_id", ""))
        tactic_head = str(d.get("tactic_head", ""))
        delta_context = d.get("delta_context", [])
        if not isinstance(delta_context, list):
            raise TypeError("delta_context must be a list of strings")

        state_before_tokens = d.get("state_before_tokens", None)
        state_after_tokens = d.get("state_after_tokens", None)
        if state_before_tokens == []:
            state_before_tokens = None
        if state_after_tokens == []:
            state_after_tokens = None

        return DeltaRecord(
            proof_id=proof_id,
            step_id=step_id,
            tactic_head=tactic_head,
            delta_context=[str(x) for x in delta_context],
            state_before_tokens=state_before_tokens,
            state_after_tokens=state_after_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "proof_id": self.proof_id,
            "step_id": self.step_id,
            "tactic_head": self.tactic_head,
            "delta_context": self.delta_context,
        }
        if self.state_before_tokens is not None:
            d["state_before_tokens"] = self.state_before_tokens
        if self.state_after_tokens is not None:
            d["state_after_tokens"] = self.state_after_tokens
        return d
