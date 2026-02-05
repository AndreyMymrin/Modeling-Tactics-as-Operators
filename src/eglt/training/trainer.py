from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from eglt.dataset.schema import DeltaRecord
from eglt.training.dataloaders import (
    build_vocabs_from_records,
    iter_cbow_batches,
    iter_delta_records_jsonl,
    iter_sgns_batches,
    reservoir_cap_by_tactic,
)
from eglt.training.negative_sampling import UnigramSampler, build_unigram_table
from eglt.utils.hashing import sha256_config
from eglt.utils.logging import setup_logging


# -------------------------
# Models (kept here for backward-compat with earlier trainer)
# -------------------------

class SGNSModel(nn.Module):
    """
    SGNS: tactic_in embeddings + context_out embeddings.
    Score(t, c) = dot(t_in[t], c_out[c]).
    """
    def __init__(self, n_tactics: int, n_ctx: int, dim: int) -> None:
        super().__init__()
        self.t_in = nn.Embedding(n_tactics, dim)
        self.c_out = nn.Embedding(n_ctx, dim)
        self._init()

    def _init(self) -> None:
        d = self.t_in.embedding_dim
        nn.init.uniform_(self.t_in.weight, -0.5 / d, 0.5 / d)
        nn.init.zeros_(self.c_out.weight)

    def score(self, t_ids: torch.Tensor, c_ids: torch.Tensor) -> torch.Tensor:
        t = self.t_in(t_ids)
        c = self.c_out(c_ids)
        return (t * c).sum(dim=-1)


class CBOWDeltaModel(nn.Module):
    """
    CBOW-Δ: context_in embeddings averaged -> tactic_out embeddings.
    Score(ctx_bag, t) = dot(mean(ctx_in[bag]), t_out[t]).
    """
    def __init__(self, n_tactics: int, n_ctx: int, dim: int) -> None:
        super().__init__()
        self.c_in = nn.Embedding(n_ctx, dim)
        self.t_out = nn.Embedding(n_tactics, dim)
        self._init()

    def _init(self) -> None:
        d = self.c_in.embedding_dim
        nn.init.uniform_(self.c_in.weight, -0.5 / d, 0.5 / d)
        nn.init.zeros_(self.t_out.weight)

    def ctx_mean(self, bags: List[List[int]], device: torch.device) -> torch.Tensor:
        vecs = []
        for bag in bags:
            ids = torch.tensor(bag, dtype=torch.long, device=device)
            v = self.c_in(ids).mean(dim=0)
            vecs.append(v)
        return torch.stack(vecs, dim=0)

    def score(self, ctx_vecs: torch.Tensor, t_ids: torch.Tensor) -> torch.Tensor:
        t = self.t_out(t_ids)
        return (ctx_vecs * t).sum(dim=-1)


# -------------------------
# Losses
# -------------------------

def neg_sampling_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Standard negative sampling loss:
      L = -E[log σ(pos)] - E[log σ(-neg)]
    """
    loss_pos = -F.logsigmoid(pos_scores).mean()
    loss_neg = -F.logsigmoid(-neg_scores).mean()
    return loss_pos + loss_neg


# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class TrainCommonCfg:
    dataset_path: str
    run_name: str = "run"
    seed: int = 0
    device: str = "cpu"

    dim: int = 128
    epochs: int = 1
    lr: float = 2e-3

    # cap per tactic
    cap_per_tactic: int = 5000

    # vocab thresholds
    min_count_context: int = 1
    min_count_tactic: int = 1

    # negative sampling
    neg_k: int = 10
    unigram_power: float = 0.75
    unigram_table_size: int = 1_000_000

    # batching
    batch_size_sgns: int = 4096
    batch_size_cbow: int = 256
    min_ctx_len: int = 1

    # logging / checkpoints
    log_every_steps: int = 100
    save_checkpoints: bool = True

    # grad clip
    grad_clip_norm: float | None = None


def _parse_common_cfg(cfg: dict, mode: str) -> TrainCommonCfg:
    """
    mode in {"sgns","cbow"} just selects relevant batch size defaults.
    """
    batch_size_sgns = int(cfg.get("batch_size", cfg.get("batch_size_sgns", 4096)))
    batch_size_cbow = int(cfg.get("batch_size", cfg.get("batch_size_cbow", 256))) if mode == "cbow" else int(
        cfg.get("batch_size_cbow", 256)
    )

    gcn = cfg.get("grad_clip_norm", None)
    if gcn is not None:
        gcn = float(gcn)

    return TrainCommonCfg(
        dataset_path=str(cfg["dataset_path"]),
        run_name=str(cfg.get("run_name", "run")),
        seed=int(cfg.get("seed", 0)),
        device=str(cfg.get("device", "cpu")),
        dim=int(cfg.get("dim", 128)),
        epochs=int(cfg.get("epochs", 1)),
        lr=float(cfg.get("lr", 2e-3)),
        cap_per_tactic=int(cfg.get("cap_per_tactic", 5000)),
        min_count_context=int(cfg.get("min_count_context", 1)),
        min_count_tactic=int(cfg.get("min_count_tactic", 1)),
        neg_k=int(cfg.get("neg_k", 10)),
        unigram_power=float(cfg.get("unigram_power", 0.75)),
        unigram_table_size=int(cfg.get("unigram_table_size", 1_000_000)),
        batch_size_sgns=batch_size_sgns,
        batch_size_cbow=batch_size_cbow,
        min_ctx_len=int(cfg.get("min_ctx_len", 1)),
        log_every_steps=int(cfg.get("log_every_steps", 100)),
        save_checkpoints=bool(cfg.get("save_checkpoints", True)),
        grad_clip_norm=gcn,
    )


# -------------------------
# I/O helpers
# -------------------------

def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _device_from_string(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_records_and_vocabs(c: TrainCommonCfg):
    # Load and cap per tactic
    raw_iter = iter_delta_records_jsonl(c.dataset_path)
    capped = reservoir_cap_by_tactic(raw_iter, k_per_tactic=c.cap_per_tactic, seed=c.seed)

    ctx_vocab, tactic_vocab, ctx_counts, tactic_counts = build_vocabs_from_records(
        capped,
        min_count_context=c.min_count_context,
        min_count_tactic=c.min_count_tactic,
    )
    return capped, ctx_vocab, tactic_vocab, ctx_counts, tactic_counts


def _maybe_clip_grad(model: nn.Module, grad_clip_norm: float | None) -> float | None:
    if grad_clip_norm is None:
        return None
    if grad_clip_norm <= 0:
        return None
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    try:
        return float(total_norm.item())
    except Exception:
        return float(total_norm)


# -------------------------
# Train: SGNS
# -------------------------

def train_sgns(cfg: dict, run_dir: str | Path) -> None:
    """
    Train ∆-SGNS on DeltaRecord JSONL.
    Saves:
      - model.pt
      - vocab_context.json, vocab_tactic.json
      - config.json
      - metrics.jsonl (step logs)
      - metrics_epoch_*.json
      - checkpoint_epoch_*.pt (optional)
    """
    log = setup_logging()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    c = _parse_common_cfg(cfg, mode="sgns")
    device = _device_from_string(c.device)

    cfg_hash = sha256_config(cfg)
    _save_json(run_dir / "config.json", {"cfg": cfg, "cfg_hash": cfg_hash})

    records, ctx_vocab, tactic_vocab, ctx_counts, _tactic_counts = _prepare_records_and_vocabs(c)

    # negative sampler over context tokens
    ctx_items = ctx_vocab.itos
    ctx_cnts = [int(ctx_counts.get(w, 0)) for w in ctx_items]
    table = build_unigram_table(
        ctx_items,
        ctx_cnts,
        power=c.unigram_power,
        table_size=c.unigram_table_size,
    )
    neg_sampler = UnigramSampler(table=table, seed=c.seed)

    model = SGNSModel(n_tactics=len(tactic_vocab), n_ctx=len(ctx_vocab), dim=c.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr)

    # Save vocabs early
    _save_json(run_dir / "vocab_context.json", {"itos": ctx_vocab.itos})
    _save_json(run_dir / "vocab_tactic.json", {"itos": tactic_vocab.itos})

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()  # fresh run logs

    global_step = 0
    for epoch in range(1, c.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_pairs = 0

        t0 = time.time()
        pbar = tqdm(
            iter_sgns_batches(records, ctx_vocab, tactic_vocab, batch_size=c.batch_size_sgns),
            desc=f"SGNS epoch {epoch}/{c.epochs}",
            leave=False,
        )

        for batch in pbar:
            global_step += 1
            B = len(batch.ctx_ids)
            epoch_pairs += B

            t_ids = torch.tensor(batch.tactic_ids, dtype=torch.long, device=device)
            c_pos = torch.tensor(batch.ctx_ids, dtype=torch.long, device=device)

            neg_flat = neg_sampler.draw(B * c.neg_k)
            c_neg = torch.tensor(neg_flat, dtype=torch.long, device=device).view(B, c.neg_k)

            pos_scores = model.score(t_ids, c_pos)  # [B]
            t_exp = t_ids.view(-1, 1).expand(-1, c.neg_k).reshape(-1)
            c_neg_flat = c_neg.reshape(-1)
            neg_scores = model.score(t_exp, c_neg_flat).view(B, c.neg_k)  # [B,K]

            loss = neg_sampling_loss(pos_scores, neg_scores)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = _maybe_clip_grad(model, c.grad_clip_norm)
            opt.step()

            loss_val = float(loss.item())
            epoch_loss_sum += loss_val
            epoch_batches += 1

            if c.log_every_steps > 0 and (global_step % c.log_every_steps == 0):
                elapsed = max(1e-9, time.time() - t0)
                pairs_per_s = epoch_pairs / elapsed
                row = {
                    "mode": "sgns",
                    "epoch": epoch,
                    "step": global_step,
                    "batch_loss": loss_val,
                    "avg_loss_epoch_so_far": epoch_loss_sum / max(1, epoch_batches),
                    "pairs_seen_epoch": epoch_pairs,
                    "pairs_per_sec_epoch": pairs_per_s,
                    "lr": opt.param_groups[0]["lr"],
                }
                if grad_norm is not None:
                    row["grad_norm"] = grad_norm
                _append_jsonl(metrics_path, row)
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "avg": f"{row['avg_loss_epoch_so_far']:.4f}"})

        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        elapsed = max(1e-9, time.time() - t0)
        pairs_per_s = epoch_pairs / elapsed

        _save_json(
            run_dir / f"metrics_epoch_{epoch}.json",
            {
                "mode": "sgns",
                "epoch": epoch,
                "avg_loss": avg_loss,
                "batches": epoch_batches,
                "pairs": epoch_pairs,
                "seconds": elapsed,
                "pairs_per_sec": pairs_per_s,
                "lr": opt.param_groups[0]["lr"],
                "grad_clip_norm": c.grad_clip_norm,
            },
        )
        log.info(f"[SGNS] epoch={epoch} avg_loss={avg_loss:.6f} pairs/s={pairs_per_s:.1f}")

        if c.save_checkpoints:
            torch.save(model.state_dict(), run_dir / f"checkpoint_epoch_{epoch}.pt")

    torch.save(model.state_dict(), run_dir / "model.pt")


# -------------------------
# Train: CBOW-Δ
# -------------------------

def train_cbow_delta(cfg: dict, run_dir: str | Path) -> None:
    """
    Train CBOW-Δ: predict tactic from delta_context bag.
    Uses negative sampling over tactics.
    """
    log = setup_logging()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    c = _parse_common_cfg(cfg, mode="cbow")
    device = _device_from_string(c.device)

    cfg_hash = sha256_config(cfg)
    _save_json(run_dir / "config.json", {"cfg": cfg, "cfg_hash": cfg_hash})

    records, ctx_vocab, tactic_vocab, ctx_counts, tactic_counts = _prepare_records_and_vocabs(c)

    # negative sampler over tactics
    tac_items = tactic_vocab.itos
    tac_cnts = [int(tactic_counts.get(w, 0)) for w in tac_items]
    table = build_unigram_table(
        tac_items,
        tac_cnts,
        power=c.unigram_power,
        table_size=c.unigram_table_size,
    )
    neg_sampler = UnigramSampler(table=table, seed=c.seed)

    model = CBOWDeltaModel(n_tactics=len(tactic_vocab), n_ctx=len(ctx_vocab), dim=c.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr)

    _save_json(run_dir / "vocab_context.json", {"itos": ctx_vocab.itos})
    _save_json(run_dir / "vocab_tactic.json", {"itos": tactic_vocab.itos})

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    global_step = 0
    for epoch in range(1, c.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_examples = 0

        t0 = time.time()
        pbar = tqdm(
            iter_cbow_batches(
                records,
                ctx_vocab,
                tactic_vocab,
                batch_size=c.batch_size_cbow,
                min_ctx_len=c.min_ctx_len,
            ),
            desc=f"CBOW-Δ epoch {epoch}/{c.epochs}",
            leave=False,
        )

        for batch in pbar:
            global_step += 1
            B = len(batch.tactic_ids)
            epoch_examples += B

            ctx_vecs = model.ctx_mean(batch.ctx_ids, device=device)  # [B,D]
            t_pos = torch.tensor(batch.tactic_ids, dtype=torch.long, device=device)  # [B]

            neg_flat = neg_sampler.draw(B * c.neg_k)
            t_neg = torch.tensor(neg_flat, dtype=torch.long, device=device).view(B, c.neg_k)

            pos_scores = model.score(ctx_vecs, t_pos)  # [B]

            ctx_exp = ctx_vecs.unsqueeze(1).expand(-1, c.neg_k, -1).reshape(-1, ctx_vecs.size(-1))
            t_neg_flat = t_neg.reshape(-1)
            neg_scores = model.score(ctx_exp, t_neg_flat).view(B, c.neg_k)  # [B,K]

            loss = neg_sampling_loss(pos_scores, neg_scores)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = _maybe_clip_grad(model, c.grad_clip_norm)
            opt.step()

            loss_val = float(loss.item())
            epoch_loss_sum += loss_val
            epoch_batches += 1

            if c.log_every_steps > 0 and (global_step % c.log_every_steps == 0):
                elapsed = max(1e-9, time.time() - t0)
                ex_per_s = epoch_examples / elapsed
                row = {
                    "mode": "cbow_delta",
                    "epoch": epoch,
                    "step": global_step,
                    "batch_loss": loss_val,
                    "avg_loss_epoch_so_far": epoch_loss_sum / max(1, epoch_batches),
                    "examples_seen_epoch": epoch_examples,
                    "examples_per_sec_epoch": ex_per_s,
                    "lr": opt.param_groups[0]["lr"],
                }
                if grad_norm is not None:
                    row["grad_norm"] = grad_norm
                _append_jsonl(metrics_path, row)
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "avg": f"{row['avg_loss_epoch_so_far']:.4f}"})

        avg_loss = epoch_loss_sum / max(1, epoch_batches)
        elapsed = max(1e-9, time.time() - t0)
        ex_per_s = epoch_examples / elapsed

        _save_json(
            run_dir / f"metrics_epoch_{epoch}.json",
            {
                "mode": "cbow_delta",
                "epoch": epoch,
                "avg_loss": avg_loss,
                "batches": epoch_batches,
                "examples": epoch_examples,
                "seconds": elapsed,
                "examples_per_sec": ex_per_s,
                "lr": opt.param_groups[0]["lr"],
                "grad_clip_norm": c.grad_clip_norm,
            },
        )
        log.info(f"[CBOW-Δ] epoch={epoch} avg_loss={avg_loss:.6f} ex/s={ex_per_s:.1f}")

        if c.save_checkpoints:
            torch.save(model.state_dict(), run_dir / f"checkpoint_epoch_{epoch}.pt")

    torch.save(model.state_dict(), run_dir / "model.pt")


def train_seq_baseline(cfg: dict, run_dir: str | Path) -> None:
    """
    Keep stub: sequence baseline training is implemented in eglt.models.seq_baseline
    and dispatched via experiments/run_train.py.

    This stub remains for backward compatibility with scripts/train_seq.py wrapper.
    """
    raise NotImplementedError("Use eglt.models.seq_baseline.train_seq_baseline_sgns via method='seq'.")
