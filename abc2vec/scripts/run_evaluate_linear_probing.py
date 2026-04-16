#!/usr/bin/env python
"""
Linear Probing Analysis for ABC2Vec.

Evaluates linear separability of musical properties using linear classifiers.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data import load_processed_data
from core.model import ABC2VecModel
from core.model.encoder import ABC2VecConfig
from core.tokenizer import ABCVocabulary, BarPatchifier


def compute_embeddings(
    model: ABC2VecModel,
    df: pd.DataFrame,
    patchifier: BarPatchifier,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Compute embeddings for all tunes in dataset."""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Computing embeddings"):
            batch_df = df.iloc[i : i + batch_size]

            batch_bar_indices = []
            batch_char_mask = []
            batch_bar_mask = []

            for _, row in batch_df.iterrows():
                patches = patchifier.patchify(row["abc_body"])
                batch_bar_indices.append(patches["bar_indices"])
                batch_char_mask.append(patches["char_mask"])
                batch_bar_mask.append(patches["bar_mask"])

            bar_indices = torch.stack(batch_bar_indices).to(device)
            char_mask = torch.stack(batch_char_mask).to(device)
            bar_mask = torch.stack(batch_bar_mask).to(device)

            embeddings = model.get_embedding(bar_indices, char_mask, bar_mask)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def probe_property(
    embeddings: np.ndarray,
    labels: np.ndarray,
    property_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Probe a single property with linear classifier."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Train Ridge Classifier with L2 regularization
        clf = RidgeClassifier(alpha=0.01, random_state=random_state)
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_val)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    # Chance baseline
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    chance_baseline = 1.0 / n_classes

    return {
        "property": property_name,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "fold_accuracies": fold_accuracies,
        "n_classes": n_classes,
        "n_samples": len(labels),
        "chance_baseline": chance_baseline,
        "above_chance": mean_accuracy - chance_baseline,
        "class_distribution": {
            str(cls): int(np.sum(labels == cls))
            for cls in unique_classes
        }
    }


def bin_tune_length(df: pd.DataFrame) -> pd.Series:
    """Bin tune lengths into 3 categories."""
    # Count bars per tune (assuming bar_count column exists, or we can compute it)
    if 'bar_count' in df.columns:
        bar_counts = df['bar_count']
    else:
        # Estimate from abc_body by counting '|' characters
        bar_counts = df['abc_body'].str.count('\|')

    # Use tertiles for binning
    low_thresh = bar_counts.quantile(0.33)
    high_thresh = bar_counts.quantile(0.67)

    def categorize(count):
        if count <= low_thresh:
            return 'short'
        elif count <= high_thresh:
            return 'medium'
        else:
            return 'long'

    return bar_counts.apply(categorize)


def main():
    parser = argparse.ArgumentParser(description="Linear probing analysis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./checkpoints/linear_probing_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Linear Probing Analysis")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    config_path = Path(args.checkpoint).parent / "model_config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    config = ABC2VecConfig(**config_dict)
    model = ABC2VecModel(config)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    step = checkpoint.get("step", "unknown")
    val_loss = checkpoint.get("val_loss", "unknown")
    print(f"✓ Model loaded (step {step}, val_loss={val_loss})")

    # Load data
    data_dir = Path(args.data_dir)
    df = load_processed_data(data_dir, args.split)
    print(f"✓ Dataset: {len(df)} tunes")

    # Create patchifier
    vocab = ABCVocabulary.load(data_dir / "vocab.json")
    patchifier = BarPatchifier(
        vocab=vocab,
        max_bars=config.max_bars,
        max_bar_length=config.max_bar_length,
    )
    print(f"✓ Vocabulary size: {vocab.size}")

    # Compute embeddings once
    print("\nComputing embeddings...")
    embeddings = compute_embeddings(model, df, patchifier, args.device)
    print(f"✓ Embeddings shape: {embeddings.shape}")

    # Probe each property
    results = {}

    # 1. Tune Type
    print("\n" + "=" * 80)
    print("Probing: Tune Type")
    print("=" * 80)
    df_filtered = df[df['tune_type'].notna() & (df['tune_type'] != '')]
    tune_type_embeddings = embeddings[df_filtered.index]
    tune_type_labels = df_filtered['tune_type'].values

    result = probe_property(tune_type_embeddings, tune_type_labels, "tune_type", args.n_splits)
    results['tune_type'] = result
    print(f"Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    print(f"Classes: {result['n_classes']}, Chance: {result['chance_baseline']:.4f}")

    # 2. Mode
    print("\n" + "=" * 80)
    print("Probing: Mode")
    print("=" * 80)
    main_modes = ['major', 'minor', 'dorian', 'mixolydian']
    df_filtered = df[df['mode'].isin(main_modes)]
    mode_embeddings = embeddings[df_filtered.index]
    mode_labels = df_filtered['mode'].values

    result = probe_property(mode_embeddings, mode_labels, "mode", args.n_splits)
    results['mode'] = result
    print(f"Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    print(f"Classes: {result['n_classes']}, Chance: {result['chance_baseline']:.4f}")

    # 3. Key Root
    print("\n" + "=" * 80)
    print("Probing: Key Root")
    print("=" * 80)
    # Get 8 most common keys
    top_keys = df['key'].value_counts().head(8).index.tolist()
    df_filtered = df[df['key'].isin(top_keys)]
    key_embeddings = embeddings[df_filtered.index]
    key_labels = df_filtered['key'].values

    result = probe_property(key_embeddings, key_labels, "key_root", args.n_splits)
    results['key_root'] = result
    print(f"Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    print(f"Classes: {result['n_classes']}, Chance: {result['chance_baseline']:.4f}")

    # 4. Tune Length
    print("\n" + "=" * 80)
    print("Probing: Tune Length")
    print("=" * 80)
    tune_length_labels = bin_tune_length(df).values

    result = probe_property(embeddings, tune_length_labels, "tune_length", args.n_splits)
    results['tune_length'] = result
    print(f"Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    print(f"Classes: {result['n_classes']}, Chance: {result['chance_baseline']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for prop_name, result in results.items():
        print(f"{prop_name:15s}: {result['mean_accuracy']*100:5.1f}% ± {result['std_accuracy']*100:4.1f}% "
              f"(chance: {result['chance_baseline']*100:5.1f}%, +{result['above_chance']*100:5.1f}%)")

    print(f"\n✓ Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
