#!/usr/bin/env python
"""
Training Script for ABC2Vec.

Trains the ABC2Vec model using multi-objective pre-training (MMM + TI + SCL).

All loss values are logged at step level and written to training_history.json
so that generate_training_curves.py can produce figures from real data.

Usage:
    python run_training.py --data_dir ./data/processed --output_dir ./checkpoints
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.data import ABC2VecDataset, SectionPairDataset, load_processed_data
from core.model import ABC2VecConfig, ABC2VecModel, ABC2VecLoss
from core.tokenizer import ABCVocabulary, BarPatchifier


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup_steps: int, total_steps: int):
    """Return a lambda suitable for torch.optim.lr_scheduler.LambdaLR."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, loss_fn, device, amp_ctx=None):
    """Run one pass over val_loader; return per-component loss dict."""
    model.eval()
    accum = {"mmm": 0.0, "ti": 0.0, "total": 0.0}
    n = 0

    for batch in val_loader:
        bar_indices = batch["bar_indices"].to(device)
        char_mask   = batch["char_mask"].to(device)
        bar_mask    = batch["bar_mask"].to(device)

        ctx = amp_ctx() if amp_ctx is not None else torch.autocast("cpu", enabled=False)
        with ctx:
            mmm_out = model.forward_mmm(bar_indices, char_mask, bar_mask)

            ti_orig = ti_trans = None
            if "trans_bar_indices" in batch:
                ti_orig, ti_trans = model.forward_contrastive(
                    bar_indices, char_mask, bar_mask,
                    batch["trans_bar_indices"].to(device),
                    batch["trans_char_mask"].to(device),
                    batch["trans_bar_mask"].to(device),
                )

            loss, losses = loss_fn(
                mmm_logits=mmm_out["mmm_logits"],
                mmm_targets=mmm_out["mmm_targets"],
                mmm_mask=mmm_out["mmm_mask"],
                ti_emb_orig=ti_orig,
                ti_emb_trans=ti_trans,
            )

        accum["total"] += loss.item()
        accum["mmm"]   += losses.get("mmm", 0.0)
        accum["ti"]    += losses.get("ti",  0.0)
        n += 1

    return {k: v / max(n, 1) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------

def save_history(history: dict, path: Path):
    """Atomically write training_history.json."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ABC2Vec Training — multi-objective pre-training"
    )
    parser.add_argument("--data_dir",     type=str,   default="./data/processed",
                        help="Directory with processed data")
    parser.add_argument("--output_dir",   type=str,   default="./checkpoints",
                        help="Directory to save checkpoints and history")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to a .pt checkpoint to resume from")
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--n_layers",     type=int,   default=6)
    parser.add_argument("--n_heads",      type=int,   default=8)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--grad_accum",   type=int,   default=4,
                        help="Gradient accumulation steps "
                             "(effective batch = batch_size x grad_accum)")
    parser.add_argument("--epochs",       type=int,   default=40)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int,   default=1000)
    parser.add_argument("--lambda_mmm",   type=float, default=1.0)
    parser.add_argument("--lambda_ti",    type=float, default=0.5)
    parser.add_argument("--lambda_scl",   type=float, default=0.5)
    parser.add_argument("--log_interval",  type=int, default=100,
                        help="Log train loss every N optimizer steps")
    parser.add_argument("--eval_interval", type=int, default=2000,
                        help="Run validation every N optimizer steps")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Save step checkpoint every N optimizer steps")
    parser.add_argument("--num_workers",  type=int, default=8,
                        help="DataLoader worker processes (default: 8)")
    parser.add_argument("--compile",      action="store_true",
                        help="torch.compile the model (faster after warm-up)")
    parser.add_argument("--amp",          action="store_true",
                        help="Mixed-precision autocast with bfloat16")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else
                                "mps"  if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of optimizer steps (overrides epochs if set)")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ABC2Vec Training")
    print("=" * 80)

    # ── Data ────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_df = load_processed_data(args.data_dir, split="train")
    val_df   = load_processed_data(args.data_dir, split="val")
    print(f"  Train: {len(train_df):,} tunes")
    print(f"  Val:   {len(val_df):,} tunes")

    vocab_path = Path(args.data_dir) / "vocab.json"
    vocab      = ABCVocabulary.load(vocab_path)
    print(f"  Vocabulary: {vocab.size} tokens")

    patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)

    train_dataset = ABC2VecDataset(train_df, patchifier, augment_transpose=True)
    val_dataset   = ABC2VecDataset(val_df,   patchifier, augment_transpose=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # SCL: section pairs loader cycling indefinitely to keep pace with main loop
    scl_iter = None
    scl_path = Path(args.data_dir) / "section_pairs.json"
    if scl_path.exists():
        scl_dataset = SectionPairDataset(str(scl_path), patchifier)
        scl_loader  = DataLoader(
            scl_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True,
            drop_last=True, persistent_workers=True, prefetch_factor=2,
        )
        scl_iter = itertools.cycle(scl_loader)
        print(f"  Section pairs: {len(scl_dataset):,} (SCL enabled)")
    else:
        print("  section_pairs.json not found -- SCL disabled")

    # ── Model ───────────────────────────────────────────────────────────────
    print("\nInitializing model...")
    config = ABC2VecConfig(
        vocab_size=vocab.size,
        max_bar_length=64, max_bars=64,
        pad_idx=vocab.pad_idx, mask_idx=vocab.mask_idx,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_model * 4, d_embed=128,
    )
    model = ABC2VecModel(config).to(args.device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # AMP context: bfloat16 autocast on CUDA only; MPS doesn't support it
    _amp_device = args.device.split(":")[0]
    _amp_enabled = args.amp and _amp_device == "cuda"
    if args.amp and not _amp_enabled:
        print("  Note: --amp ignored on MPS (only supported on CUDA)")
    def amp_context():
        if _amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    loss_fn = ABC2VecLoss(
        config,
        lambda_mmm=args.lambda_mmm,
        lambda_ti=args.lambda_ti,
        lambda_scl=args.lambda_scl if scl_iter is not None else 0.0,
        lambda_vac=0.0,
    )

    # LR schedule: linear warmup then cosine decay
    batches_per_epoch = len(train_loader)
    steps_per_epoch   = batches_per_epoch // args.grad_accum
    total_steps       = steps_per_epoch * args.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(args.warmup_steps, total_steps)
    )
    config.save(output_path / "model_config.json")

    print(f"  Batches/epoch: {batches_per_epoch}  "
          f"optimizer steps/epoch: {steps_per_epoch}  "
          f"total steps: {total_steps:,}")

    # ── History structure (matches generate_training_curves.py exactly) ────
    history = {
        "train_losses": [],   # {step, mmm, scl, ti, total}
        "val_losses":   [],   # {step, mmm, ti, total}
        "config": {
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_epochs": args.epochs,
            "warmup_steps": args.warmup_steps,
            "lambda_mmm": args.lambda_mmm,
            "lambda_ti": args.lambda_ti,
            "lambda_scl": args.lambda_scl,
            "log_interval": args.log_interval,
            "eval_interval": args.eval_interval,
            "save_interval": args.save_interval,
        },
        "best_val_loss": float("inf"),
        "total_steps": 0,
    }
    history_path = output_path / "training_history.json"

    # ── Resume ──────────────────────────────────────────────────────────────
    global_step   = 0
    start_epoch   = 1
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        print(f"\nResuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=args.device)
        raw_sd = ckpt["model_state_dict"]
        # Strip "_orig_mod." prefix if checkpoint was saved from a compiled model
        if any(k.startswith("_orig_mod.") for k in raw_sd):
            raw_sd = {k.removeprefix("_orig_mod."): v for k, v in raw_sd.items()}
        model.load_state_dict(raw_sd)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step   = ckpt.get("step", 0)
        start_epoch   = ckpt.get("epoch", global_step // max(steps_per_epoch, 1)) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        for _ in range(global_step):
            scheduler.step()
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            history["best_val_loss"] = best_val_loss
        print(f"  Resumed at step {global_step:,}, starting from epoch {start_epoch}")

    # Compile after loading weights so checkpoint keys always match
    if args.compile:
        print("  Compiling model with torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")
        print("  Compilation scheduled (JIT happens on first forward pass)")

    # ── Training loop ────────────────────────────────────────────────────────
    # log_accum: averaged across log_interval optimizer steps before writing
    log_accum   = {"total": 0.0, "mmm": 0.0, "ti": 0.0, "scl": 0.0}
    log_accum_n = 0
    # micro_accum: averaged across grad_accum batches per optimizer step
    micro_accum = {"total": 0.0, "mmm": 0.0, "ti": 0.0, "scl": 0.0}
    micro_n     = 0

    optimizer.zero_grad()

    print(f"\nTraining for {args.epochs} epochs on {args.device}")
    print(f"  batch_size={args.batch_size}  grad_accum={args.grad_accum}  "
          f"effective_batch={args.batch_size * args.grad_accum}")
    print(f"  lr={args.lr}  warmup={args.warmup_steps} steps  "
          f"total={total_steps:,} steps")
    print(f"  num_workers={args.num_workers}  compile={args.compile}  amp={args.amp}")
    print(f"  log_interval={args.log_interval}  "
          f"eval_interval={args.eval_interval}  "
          f"save_interval={args.save_interval}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            bar_indices = batch["bar_indices"].to(args.device)
            char_mask   = batch["char_mask"].to(args.device)
            bar_mask    = batch["bar_mask"].to(args.device)

            # ── Forward passes (all inside AMP context) ───────────────────
            with amp_context():
                mmm_out = model.forward_mmm(bar_indices, char_mask, bar_mask)

                # ── TI forward ───────────────────────────────────────────
                ti_orig = ti_trans = None
                if "trans_bar_indices" in batch:
                    ti_orig, ti_trans = model.forward_contrastive(
                        bar_indices, char_mask, bar_mask,
                        batch["trans_bar_indices"].to(args.device),
                        batch["trans_char_mask"].to(args.device),
                        batch["trans_bar_mask"].to(args.device),
                    )

                # ── SCL forward ──────────────────────────────────────────
                scl_emb_a = scl_emb_b = None
                if scl_iter is not None:
                    scl_batch = next(scl_iter)
                    scl_emb_a = model.get_embedding(
                        scl_batch["a_bar_indices"].to(args.device),
                        scl_batch["a_char_mask"].to(args.device),
                        scl_batch["a_bar_mask"].to(args.device),
                    )
                    scl_emb_b = model.get_embedding(
                        scl_batch["b_bar_indices"].to(args.device),
                        scl_batch["b_char_mask"].to(args.device),
                        scl_batch["b_bar_mask"].to(args.device),
                    )

                # ── Combined loss ─────────────────────────────────────────
                loss, losses = loss_fn(
                    mmm_logits=mmm_out["mmm_logits"],
                    mmm_targets=mmm_out["mmm_targets"],
                    mmm_mask=mmm_out["mmm_mask"],
                    ti_emb_orig=ti_orig,
                    ti_emb_trans=ti_trans,
                    scl_emb_a=scl_emb_a,
                    scl_emb_b=scl_emb_b,
                )

            # Scale loss for gradient accumulation before backprop
            (loss / args.grad_accum).backward()

            micro_accum["total"] += loss.item()
            micro_accum["mmm"]   += losses.get("mmm", 0.0)
            micro_accum["ti"]    += losses.get("ti",  0.0)
            micro_accum["scl"]   += losses.get("scl", 0.0)
            micro_n              += 1

            # ── Optimizer step (every grad_accum micro-batches) ───────────
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                log_accum["total"] += micro_accum["total"] / micro_n
                log_accum["mmm"]   += micro_accum["mmm"]   / micro_n
                log_accum["ti"]    += micro_accum["ti"]    / micro_n
                log_accum["scl"]   += micro_accum["scl"]   / micro_n
                log_accum_n        += 1
                micro_accum = {"total": 0.0, "mmm": 0.0, "ti": 0.0, "scl": 0.0}
                micro_n     = 0

                cur_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "scl":  f"{losses.get('scl', 0.0):.4f}",
                    "lr":   f"{cur_lr:.2e}",
                    "step": global_step,
                })

                # ── Step-level train log ──────────────────────────────────
                if global_step % args.log_interval == 0:
                    entry = {
                        "step":  global_step,
                        "total": log_accum["total"] / log_accum_n,
                        "mmm":   log_accum["mmm"]   / log_accum_n,
                        "ti":    log_accum["ti"]    / log_accum_n,
                        "scl":   log_accum["scl"]   / log_accum_n,
                    }
                    history["train_losses"].append(entry)
                    log_accum   = {"total": 0.0, "mmm": 0.0, "ti": 0.0, "scl": 0.0}
                    log_accum_n = 0

                # ── Validation ────────────────────────────────────────────
                if global_step % args.eval_interval == 0:
                    val_metrics = validate(model, val_loader, loss_fn, args.device, amp_context)
                    val_entry = {
                        "step":  global_step,
                        "total": val_metrics["total"],
                        "mmm":   val_metrics["mmm"],
                        "ti":    val_metrics["ti"],
                    }
                    history["val_losses"].append(val_entry)

                    if val_metrics["total"] < best_val_loss:
                        best_val_loss = val_metrics["total"]
                        history["best_val_loss"] = best_val_loss
                        best_path = output_path / "best_model.pt"
                        torch.save({
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": best_val_loss,
                        }, best_path)
                        tqdm.write(f"  > New best model at step {global_step:,} "
                                   f"(val_loss={best_val_loss:.4f})")

                    history["total_steps"] = global_step
                    save_history(history, history_path)
                    tqdm.write(
                        f"  [step {global_step:,}] "
                        f"val={val_metrics['total']:.4f}  "
                        f"mmm={val_metrics['mmm']:.4f}  "
                        f"ti={val_metrics['ti']:.4f}  "
                        f"lr={cur_lr:.2e}"
                    )
                    model.train()

                # ── Step checkpoint ───────────────────────────────────────
                if global_step % args.save_interval == 0:
                    ckpt_path = output_path / f"checkpoint_step{global_step}.pt"
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, ckpt_path)
                    tqdm.write(f"  > Saved checkpoint: {ckpt_path.name}")

                # ── Check max_steps limit ─────────────────────────────────
                if args.max_steps is not None and global_step >= args.max_steps:
                    tqdm.write(f"\n  Reached max_steps={args.max_steps}, stopping training.")
                    break

        # Break outer epoch loop if max_steps reached
        if args.max_steps is not None and global_step >= args.max_steps:
            break

        # ── Epoch checkpoint ─────────────────────────────────────────────
        ckpt_path = output_path / f"checkpoint_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch, "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

    # ── Final save ───────────────────────────────────────────────────────────
    final_path = output_path / "final_model.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

    history["total_steps"] = global_step
    save_history(history, history_path)

    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print(f"  Total steps      : {global_step:,}")
    print(f"  Best val loss    : {best_val_loss:.6f}")
    print(f"  Final model      : {final_path}")
    print(f"  Training history : {history_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
