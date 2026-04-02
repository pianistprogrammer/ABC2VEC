#!/usr/bin/env python
"""
Data Pipeline Script for ABC2Vec.

Downloads IrishMAN dataset, processes ABC notation, and creates train/val/test splits.
"""

import argparse
from pathlib import Path

from core.data import download_and_process_dataset
from core.tokenizer import ABCVocabulary


def main():
    """Run the complete data processing pipeline."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Data Pipeline - Download and process ABC notation data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        help="Fraction of data for validation split",
    )
    parser.add_argument(
        "--min_char_freq",
        type=int,
        default=5,
        help="Minimum character frequency for vocabulary",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Data Pipeline")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Test size:        {args.test_size}")
    print(f"  Min char freq:    {args.min_char_freq}")
    print(f"  Random seed:      {args.seed}")
    print()

    # Run pipeline
    data_splits = download_and_process_dataset(
        output_dir=args.output_dir,
        test_size=args.test_size,
        min_char_freq=args.min_char_freq,
        random_seed=args.seed,
        verbose=True,
    )

    # Build vocabulary
    print("\nBuilding character vocabulary...")
    vocab = ABCVocabulary()
    train_bodies = data_splits["train"]["abc_body"].tolist()
    vocab.build_from_corpus(train_bodies, min_freq=args.min_char_freq)

    vocab_path = Path(args.output_dir) / "vocab.json"
    vocab.save(vocab_path)
    print(f"  Vocabulary saved: {vocab_path}")
    print(f"  Vocabulary size:  {vocab.size}")

    # Save tokenizer config
    tokenizer_config = {
        "vocab_size": vocab.size,
        "max_bar_length": 64,
        "max_bars": 64,
        "d_char": 64,
        "d_model": 256,
        "pad_idx": vocab.pad_idx,
        "mask_idx": vocab.mask_idx,
        "cls_idx": vocab.cls_idx,
        "sep_idx": vocab.sep_idx,
    }

    config_path = Path(args.output_dir) / "tokenizer_config.json"
    with open(config_path, "w") as f:
        import json
        json.dump(tokenizer_config, f, indent=2)

    print(f"  Tokenizer config saved: {config_path}")

    print("\n" + "=" * 80)
    print("✓ Pipeline complete! Data ready for training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
