"""
Complete workflow example: from data to trained model to retrieval.

This script demonstrates the entire ABC2Vec pipeline in one place.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import ABC2Vec components
from core import ABC2VecModel, ABCVocabulary, BarPatchifier
from core.data import ABC2VecDataset, download_and_process_dataset
from core.model import ABC2VecConfig, ABC2VecLoss


def run_complete_workflow(
    data_dir: str = "./example_data",
    checkpoint_dir: str = "./example_checkpoints",
    quick_test: bool = True,
):
    """
    Run the complete ABC2Vec workflow.

    Args:
        data_dir: Directory for processed data
        checkpoint_dir: Directory for model checkpoints
        quick_test: If True, use reduced settings for fast testing
    """
    print("=" * 80)
    print("ABC2Vec Complete Workflow Example")
    print("=" * 80)

    # ============================================================================
    # STEP 1: Data Processing
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Data Processing")
    print("=" * 80 + "\n")

    data_path = Path(data_dir)

    # Check if data already exists
    if (data_path / "train.parquet").exists():
        print("✓ Processed data already exists, skipping download...")
    else:
        print("Downloading and processing IrishMAN dataset...")
        download_and_process_dataset(
            output_dir=data_dir,
            test_size=0.05,
            random_seed=42,
            verbose=True,
        )

    # Build vocabulary
    vocab_path = data_path / "vocab.json"
    if vocab_path.exists():
        print("\n✓ Loading existing vocabulary...")
        vocab = ABCVocabulary.load(vocab_path)
    else:
        print("\n✗ Vocabulary not found. Run data pipeline first.")
        return

    print(f"  Vocabulary size: {vocab.size}")

    # ============================================================================
    # STEP 2: Model Setup
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Model Setup")
    print("=" * 80 + "\n")

    # Configuration
    if quick_test:
        config = ABC2VecConfig(
            vocab_size=vocab.size,
            d_model=128,        # Smaller for quick test
            n_layers=3,         # Fewer layers
            n_heads=4,
            d_embed=64,
        )
        epochs = 2
        batch_size = 32
    else:
        config = ABC2VecConfig(
            vocab_size=vocab.size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_embed=128,
        )
        epochs = 10
        batch_size = 64

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ABC2VecModel(config).to(device)

    print(f"Model configuration:")
    print(f"  d_model:    {config.d_model}")
    print(f"  n_layers:   {config.n_layers}")
    print(f"  n_heads:    {config.n_heads}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device:     {device}")

    # ============================================================================
    # STEP 3: Training
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Training")
    print("=" * 80 + "\n")

    # Load data
    import pandas as pd

    train_df = pd.read_parquet(data_path / "train.parquet")

    # Use subset for quick test
    if quick_test:
        train_df = train_df.head(10000)
        print(f"Quick test mode: using {len(train_df):,} tunes")

    # Create dataset and loader
    patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)
    train_dataset = ABC2VecDataset(train_df, patchifier, augment_transpose=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    # Loss and optimizer
    loss_fn = ABC2VecLoss(
        config,
        lambda_mmm=1.0,
        lambda_ti=0.5,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    print(f"Training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            # Move to device
            bar_indices = batch["bar_indices"].to(device)
            char_mask = batch["char_mask"].to(device)
            bar_mask = batch["bar_mask"].to(device)

            # Forward
            mmm_out = model.forward_mmm(bar_indices, char_mask, bar_mask)

            # Transposition if available
            ti_orig, ti_trans = None, None
            if "trans_bar_indices" in batch:
                trans_bi = batch["trans_bar_indices"].to(device)
                trans_cm = batch["trans_char_mask"].to(device)
                trans_bm = batch["trans_bar_mask"].to(device)

                ti_orig, ti_trans = model.forward_contrastive(
                    bar_indices, char_mask, bar_mask,
                    trans_bi, trans_cm, trans_bm,
                )

            # Loss
            loss, _ = loss_fn(
                mmm_logits=mmm_out["mmm_logits"],
                mmm_targets=mmm_out["mmm_targets"],
                mmm_mask=mmm_out["mmm_mask"],
                ti_emb_orig=ti_orig,
                ti_emb_trans=ti_trans,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Early stopping for quick test
            if quick_test and num_batches >= 50:
                break

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    # Save model
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_path / "example_model.pt"
    model.save_pretrained(model_path)
    print(f"\n✓ Model saved to {model_path}")

    # ============================================================================
    # STEP 4: Inference & Similarity
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Inference & Similarity")
    print("=" * 80 + "\n")

    model.eval()

    # Example tunes
    example_tunes = {
        "Irish Reel in D": "D2 EF G2 AB | c2 dc BAGF | D2 EF G2 AB | c2 dc BAGF |",
        "Similar Reel in D": "D2 FE G2 BA | c2 cd BGAF | D2 FE G2 BA | c2 cd BGAF |",
        "Different Jig in G": "G2 B d2 d | e2 c BAG | G2 B d2 d | e2 c BAG |",
    }

    embeddings = {}

    print("Encoding example tunes...\n")
    for name, abc_body in example_tunes.items():
        patches = patchifier.patchify(abc_body)

        with torch.no_grad():
            embedding = model.get_embedding(
                patches["bar_indices"].unsqueeze(0).to(device),
                patches["char_mask"].unsqueeze(0).to(device),
                patches["bar_mask"].unsqueeze(0).to(device),
            )

        embeddings[name] = embedding.cpu()
        print(f"  ✓ {name}")

    # Compute similarities
    print("\nPairwise similarities:")
    print()

    names = list(embeddings.keys())
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                sim = F.cosine_similarity(
                    embeddings[name_i], embeddings[name_j]
                ).item()
                print(f"  {name_i} <-> {name_j}")
                print(f"    Similarity: {sim:.4f}")
                print()

    # ============================================================================
    # Summary
    # ============================================================================
    print("=" * 80)
    print("✓ Workflow Complete!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  Data:       {data_dir}")
    print(f"  Model:      {model_path}")
    print(f"  Vocabulary: {vocab_path}")
    print(f"\nNext steps:")
    print(f"  - Train longer: set quick_test=False")
    print(f"  - Evaluate: python scripts/run_evaluation.py")
    print(f"  - Fine-tune hyperparameters in config.yaml")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Complete workflow example for ABC2Vec"
    )
    parser.add_argument(
        "--data_dir",
        default="./example_data",
        help="Data directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./example_checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        default=True,
        help="Run quick test with reduced settings",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full training (overrides quick_test)",
    )

    args = parser.parse_args()

    run_complete_workflow(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        quick_test=not args.full,
    )
