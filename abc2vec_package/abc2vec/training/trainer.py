"""Training utilities and trainer class."""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    """
    Trainer for ABC2Vec model.

    Handles:
        - Training loop with progress tracking
        - Validation
        - Checkpoint saving
        - Learning rate scheduling
        - Logging

    Args:
        model: ABC2VecModel instance
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary with average metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        loss_components = {}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            bar_indices = batch["bar_indices"].to(self.device)
            char_mask = batch["char_mask"].to(self.device)
            bar_mask = batch["bar_mask"].to(self.device)

            # Forward pass (MMM + optional TI)
            mmm_out = self.model.forward_mmm(bar_indices, char_mask, bar_mask)

            # Transposition invariance if available
            ti_orig, ti_trans = None, None
            if "trans_bar_indices" in batch:
                trans_bi = batch["trans_bar_indices"].to(self.device)
                trans_cm = batch["trans_char_mask"].to(self.device)
                trans_bm = batch["trans_bar_mask"].to(self.device)

                ti_orig, ti_trans = self.model.forward_contrastive(
                    bar_indices, char_mask, bar_mask,
                    trans_bi, trans_cm, trans_bm,
                )

            # Compute loss
            loss, losses = self.loss_fn(
                mmm_logits=mmm_out["mmm_logits"],
                mmm_targets=mmm_out["mmm_targets"],
                mmm_mask=mmm_out["mmm_mask"],
                ti_emb_orig=ti_orig,
                ti_emb_trans=ti_trans,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0
                loss_components[k] += v

            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average metrics
        metrics = {"total_loss": total_loss / num_batches}
        for k, v in loss_components.items():
            metrics[k] = v / num_batches

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                bar_indices = batch["bar_indices"].to(self.device)
                char_mask = batch["char_mask"].to(self.device)
                bar_mask = batch["bar_mask"].to(self.device)

                # Forward pass
                mmm_out = self.model.forward_mmm(
                    bar_indices, char_mask, bar_mask
                )

                # Compute loss
                loss, _ = self.loss_fn(
                    mmm_logits=mmm_out["mmm_logits"],
                    mmm_targets=mmm_out["mmm_targets"],
                    mmm_mask=mmm_out["mmm_mask"],
                )

                total_loss += loss.item()
                num_batches += 1

        return {"val_loss": total_loss / num_batches}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        save_every: int = 5,
    ) -> None:
        """
        Complete training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print()

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            epoch_time = time.time() - start_time

            # Log
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train loss: {train_metrics['total_loss']:.4f}")

            for k, v in train_metrics.items():
                if k != "total_loss":
                    print(f"    {k}: {v:.4f}")

            if val_metrics:
                print(f"  Val loss:   {val_metrics['val_loss']:.4f}")

            # Save history
            self.history.append(
                {"epoch": epoch, **train_metrics, **val_metrics}
            )

            # Save checkpoint
            if epoch % save_every == 0 or epoch == num_epochs:
                checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_path)
                print(f"  ✓ Saved: {checkpoint_path}")

            # Save best model
            current_val_loss = val_metrics.get("val_loss", train_metrics["total_loss"])
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                best_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_path)
                print(f"  ✓ New best model saved!")

        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n✓ Training complete! History saved to {history_path}")

    def save_checkpoint(self, path: str) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
