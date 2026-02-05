from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from eglt.dataset.schema import StepRecord


@dataclass(frozen=True)
class SanityReport:
    n_records: int
    n_proofs: int

    n_missing_proof_id: int
    n_missing_step_id: int
    n_missing_tactic_head: int

    n_empty_before: int
    n_empty_after: int

    tactic_counts: Counter[str]

    def top_tactics(self, k: int = 20) -> list[tuple[str, int]]:
        return self.tactic_counts.most_common(k)


def compute_stats(records: Iterable[StepRecord]) -> SanityReport:
    """
    Compute tactic frequencies and basic sanity checks.

    Checks:
      - missing/empty proof_id, step_id, tactic_head
      - empty/None state_before_tokens / state_after_tokens counts
      - unique proof count
    """
    tactic_counts: Counter[str] = Counter()
    proofs = set()

    n_records = 0
    n_missing_proof_id = 0
    n_missing_step_id = 0
    n_missing_tactic_head = 0
    n_empty_before = 0
    n_empty_after = 0

    for r in records:
        n_records += 1

        if not r.proof_id:
            n_missing_proof_id += 1
        else:
            proofs.add(r.proof_id)

        if not r.step_id:
            n_missing_step_id += 1

        if not r.tactic_head:
            n_missing_tactic_head += 1
        else:
            tactic_counts[r.tactic_head] += 1

        if not r.state_before_tokens:
            n_empty_before += 1
        if not r.state_after_tokens:
            n_empty_after += 1

    return SanityReport(
        n_records=n_records,
        n_proofs=len(proofs),
        n_missing_proof_id=n_missing_proof_id,
        n_missing_step_id=n_missing_step_id,
        n_missing_tactic_head=n_missing_tactic_head,
        n_empty_before=n_empty_before,
        n_empty_after=n_empty_after,
        tactic_counts=tactic_counts,
    )


def format_report(rep: SanityReport, top_k: int = 20) -> str:
    lines: list[str] = []
    lines.append(f"Records: {rep.n_records}")
    lines.append(f"Unique proofs: {rep.n_proofs}")
    lines.append("")
    lines.append("Missing fields:")
    lines.append(f"  proof_id: {rep.n_missing_proof_id}")
    lines.append(f"  step_id: {rep.n_missing_step_id}")
    lines.append(f"  tactic_head: {rep.n_missing_tactic_head}")
    lines.append("")
    lines.append("Empty/None token fields:")
    lines.append(f"  state_before_tokens: {rep.n_empty_before}")
    lines.append(f"  state_after_tokens: {rep.n_empty_after}")
    lines.append("")
    lines.append(f"Top-{top_k} tactics:")
    for t, c in rep.top_tactics(top_k):
        lines.append(f"  {t}\t{c}")
    return "\n".join(lines)
