#!/usr/bin/env python
"""
Training Script for ABC2Vec.

Trains the ABC2Vec model using multi-objective pre-training.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from abc2vec.data import ABC2VecDataset, SectionPairDataset, load_processed_data
from abc2vec.model import ABC2VecConfig, ABC2VecModel, ABC2VecLoss
from abc2vec.tokenizer import ABCVocabulary, BarPatchifier


def train_epoch(
    model: ABC2VecModel,
    loader: DataLoader,
    loss_fn: ABC2VecLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {"mmm": 0, "ti": 0}
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        bar_indices = batch["bar_indices"].to(device)
        char_mask = batch["char_mask"].to(device)
        bar_mask = batch["bar_mask"].to(device)

        # Forward pass for MMM
        mmm_out = model.forward_mmm(bar_indices, char_mask, bar_mask)

        # Forward pass for TI
        if "trans_bar_indices" in batch:
            trans_bar_indices = batch["trans_bar_indices"].to(device)
            trans_char_mask = batch["trans_char_mask"].to(device)
            trans_bar_mask = batch["trans_bar_mask"].to(device)

            emb_orig, emb_trans = model.forward_contrastive(
                bar_indices, char_mask, bar_mask,
                trans_bar_indices, trans_char_mask, trans_bar_mask,
            )
        else:
            emb_orig, emb_trans = None, None

        # Compute loss
        loss, losses = loss_fn(
            mmm_logits=mmm_out["mmm_logits"],
            mmm_targets=mmm_out["mmm_targets"],
            mmm_mask=mmm_out["mmm_mask"],
            ti_emb_orig=emb_orig,
            ti_emb_trans=emb_trans,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        for k in ["mmm", "ti"]:
            if k in losses:
                loss_components[k] += losses[k]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Average metrics
    metrics = {"total_loss": total_loss / num_batches}
    for k, v in loss_components.items():
        if num_batches > 0:
            metrics[f"{k}_loss"] = v / num_batches

    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Training - Multi-objective pre-training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Model hidden dimension",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=6,
        help="Number of Transformer layers",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Training")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    train_df = load_processed_data(args.data_dir, split="train")
    val_df = load_processed_data(args.data_dir, split="val")

    print(f"  Train: {len(train_df):,} tunes")
    print(f"  Val:   {len(val_df):,} tunes")

    # Load vocabulary
    vocab_path = Path(args.data_dir) / "vocab.json"
    vocab = ABCVocabulary.load(vocab_path)
    print(f"  Vocabulary: {vocab.size} tokens")

    # Create patchifier
    patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)

    # Create datasets
    train_dataset = ABC2VecDataset(
        train_df, patchifier, augment_transpose=True
    )
    val_dataset = ABC2VecDataset(
        val_df, patchifier, augment_transpose=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    print("\nInitializing model...")
    config = ABC2VecConfig(
        vocab_size=vocab.size,
        max_bar_length=64,
        max_bars=64,
        pad_idx=vocab.pad_idx,
        mask_idx=vocab.mask_idx,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        d_embed=128,
    )

    model = ABC2VecModel(config).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Loss and optimizer
    loss_fn = ABC2VecLoss(
        config,
        lambda_mmm=1.0,
        lambda_scl=0.0,  # Not used in simple training
        lambda_ti=0.5,
        lambda_vac=0.0,  # Not used in simple training
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_path / "model_config.json")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Device: {args.device}")
    print()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, args.device, epoch
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_metrics['total_loss']:.4f}")
        for k, v in train_metrics.items():
            if k != "total_loss":
                print(f"    {k}: {v:.4f}")

        # Save checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:
            checkpoint_path = output_path / f"checkpoint_epoch_{epoch}.pt"
            model.save_pretrained(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = output_path / "final_model.pt"
    model.save_pretrained(final_path)

    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print(f"  Final model saved: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
