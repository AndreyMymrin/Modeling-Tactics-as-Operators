from __future__ import annotations

import argparse
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from eglt.dataset.schema import StepRecord
from eglt.delta.build_context import build_delta_context
from eglt.eval.retrieval import evaluate_retrieval
from eglt.paths import paths
from eglt.utils.io import write_jsonl
from eglt.utils.logging import setup_logging
from eglt.training.trainer import train_sgns


def make_synthetic_steps() -> list[StepRecord]:
    """
    5 synthetic StepRecord with minimal before/after states encoded via tokens.
    We intentionally design changes to trigger:
      - ∆ADD_HYP, ∆CTX_GROW, TOK_* deltas
      - ∆GOAL_CHANGED / ∆GOAL_SOLVED
    """
    steps: list[StepRecord] = []

    # proof p1: intro adds a hypothesis-like context; goal changes
    steps.append(
        StepRecord(
            proof_id="p1",
            step_id="0",
            tactic_head="intro x",
            state_before_tokens=["⊢", "P", "x"],
            state_after_tokens=["x", ":", "Nat", "⊢", "P", "x"],
            extra={"state_before": "⊢ P x", "state_after": "x : Nat\n⊢ P x"},
        )
    )

    # proof p1: simp changes goal
    steps.append(
        StepRecord(
            proof_id="p1",
            step_id="1",
            tactic_head="simp",
            state_before_tokens=["x", ":", "Nat", "⊢", "P", "x"],
            state_after_tokens=["x", ":", "Nat", "⊢", "Q", "x"],
            extra={"state_before": "x : Nat\n⊢ P x", "state_after": "x : Nat\n⊢ Q x"},
        )
    )

    # proof p2: have adds hypothesis token(s) to context
    steps.append(
        StepRecord(
            proof_id="p2",
            step_id="0",
            tactic_head="have h : R x",
            state_before_tokens=["⊢", "R", "x", "→", "S", "x"],
            state_after_tokens=["h", ":", "R", "x", "⊢", "S", "x"],
            extra={"state_before": "⊢ R x → S x", "state_after": "h : R x\n⊢ S x"},
        )
    )

    # proof p2: exact solves goal
    steps.append(
        StepRecord(
            proof_id="p2",
            step_id="1",
            tactic_head="exact h",
            state_before_tokens=["h", ":", "R", "x", "⊢", "S", "x"],
            state_after_tokens=["⊢", "no", "goals"],
            extra={"state_before": "h : R x\n⊢ S x", "state_after": "⊢ no goals"},
        )
    )

    # proof p3: rw introduces '=' in goal (op marker)
    steps.append(
        StepRecord(
            proof_id="p3",
            step_id="0",
            tactic_head="rw [h]",
            state_before_tokens=["⊢", "f", "x", "=", "y"],
            state_after_tokens=["⊢", "g", "x", "=", "y"],
            extra={"state_before": "⊢ f x = y", "state_after": "⊢ g x = y"},
        )
    )

    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test: preprocess->train(1 epoch)->eval retrieval on synthetic data.")
    ap.add_argument("--run_name", type=str, default="smoke_test", help="Run dir name under results/runs/")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda (cuda used only if available)")
    args = ap.parse_args()

    log = setup_logging()
    P = paths(Path.cwd())

    # --- 1) Create synthetic raw JSONL (StepRecord) ---
    steps = make_synthetic_steps()
    raw_path = (P.data_interim / "smoke_raw_steps.jsonl").resolve()
    write_jsonl(raw_path, [s.to_dict() for s in steps])
    log.info(f"Wrote synthetic StepRecord JSONL: {raw_path}")

    # --- 2) Build delta contexts (DeltaRecord) ---
    deltas = [build_delta_context(s) for s in steps]
    assert all(isinstance(d.delta_context, list) and len(d.delta_context) > 0 for d in deltas), \
        "Delta contexts were not built or empty."

    deltas_path = (P.data_processed / "smoke_deltas.jsonl").resolve()
    write_jsonl(deltas_path, [d.to_dict() for d in deltas])
    log.info(f"Wrote synthetic DeltaRecord JSONL: {deltas_path}")

    # --- 3) Train SGNS for 1 epoch ---
    run_dir = (P.results_runs / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = {
        "method": "sgns",
        "run_name": args.run_name,
        "dataset_path": str(deltas_path),
        "device": args.device,
        "seed": 0,
        "dim": 32,
        "epochs": 1,
        "lr": 5e-3,
        "batch_size": 64,
        "cap_per_tactic": 5000,
        "min_count_context": 1,
        "min_count_tactic": 1,
        "neg_k": 5,
        "unigram_power": 0.75,
        # small table for speed
        "unigram_table_size": 50_000,
    }

    train_sgns(train_cfg, run_dir)
    model_path = run_dir / "model.pt"
    assert model_path.exists(), "Training did not produce model.pt"
    assert (run_dir / "vocab_context.json").exists(), "Missing vocab_context.json"
    assert (run_dir / "vocab_tactic.json").exists(), "Missing vocab_tactic.json"
    log.info(f"Training OK. Model saved: {model_path}")

    # --- 4) Eval retrieval (MRR) ---
    ret = evaluate_retrieval(
        deltas_path=deltas_path,
        run_dir=run_dir,
        ks=[1, 5, 10],
        device=args.device,
    )
    assert ret.n > 0, "Retrieval evaluated on 0 examples (likely vocab mismatch)."
    # We don't require a specific MRR threshold; just ensure it runs and returns a number.
    assert 0.0 <= ret.mrr <= 1.0, f"Invalid MRR: {ret.mrr}"
    log.info(f"Retrieval OK. n={ret.n}, MRR={ret.mrr:.4f}, Recall@1={ret.recall_at.get(1, 0.0):.4f}")

    log.info("SMOKE TEST PASSED ✅")


if __name__ == "__main__":
    main()
