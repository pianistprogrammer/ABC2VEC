#!/usr/bin/env python
"""
Tune Type Classification Evaluation.

Evaluates tune type classification using logistic regression on frozen embeddings,
as described in the paper.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
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


def evaluate_classification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Evaluate classification with stratified k-fold cross-validation.

    Args:
        embeddings: Embedding matrix (num_tunes, d_embed)
        labels: Tune type labels
        n_splits: Number of folds for cross-validation
        random_state: Random seed

    Returns:
        Dictionary with metrics
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accuracies = []
    fold_f1_scores = []
    all_predictions = []
    all_true_labels = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Train logistic regression with L2 regularization
        clf = LogisticRegression(
            C=1.0 / 0.01,  # C = 1/lambda, lambda=0.01
            max_iter=1000,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_val)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')

        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)

        all_predictions.extend(y_pred)
        all_true_labels.extend(y_val)

        print(f"  Fold {fold_idx + 1}/{n_splits}: Accuracy={acc:.4f}, F1={f1:.4f}")

    # Compute overall metrics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)

    # Get class distribution for chance baseline
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    chance_baseline = 1.0 / n_classes

    # Compute confusion matrix on aggregated predictions
    cm = confusion_matrix(all_true_labels, all_predictions, labels=unique_classes)

    # Per-class metrics
    report = classification_report(
        all_true_labels, all_predictions,
        labels=unique_classes,
        target_names=unique_classes,
        output_dict=True
    )

    return {
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "fold_accuracies": fold_accuracies,
        "fold_f1_scores": fold_f1_scores,
        "chance_baseline": chance_baseline,
        "above_chance": mean_accuracy - chance_baseline,
        "n_classes": n_classes,
        "n_samples": len(labels),
        "confusion_matrix": cm.tolist(),
        "class_labels": unique_classes.tolist(),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tune type classification")
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
        default="./checkpoints/tune_type_classification_results.json",
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
    print("ABC2Vec Tune Type Classification Evaluation")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Split:       {args.split}")
    print(f"  Device:      {args.device}")
    print(f"  N-folds:     {args.n_splits}")
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

    # Filter out tunes with missing tune_type
    df = df[df['tune_type'].notna() & (df['tune_type'] != '')]

    print(f"✓ Dataset: {len(df)} tunes")

    # Check tune type distribution
    tune_type_counts = df['tune_type'].value_counts()
    print(f"\nTune type distribution:")
    for tt, count in tune_type_counts.items():
        pct = 100 * count / len(df)
        print(f"  {tt}: {count} ({pct:.1f}%)")

    # Create patchifier
    vocab = ABCVocabulary.load(data_dir / "vocab.json")
    patchifier = BarPatchifier(
        vocab=vocab,
        max_bars=config.max_bars,
        max_bar_length=config.max_bar_length,
    )
    print(f"✓ Vocabulary size: {vocab.size}")

    # Compute embeddings
    print("\nComputing embeddings...")
    embeddings = compute_embeddings(model, df, patchifier, args.device)
    print(f"✓ Embeddings shape: {embeddings.shape}")

    # Get labels
    labels = df['tune_type'].values

    # Evaluate classification
    print(f"\nEvaluating with {args.n_splits}-fold cross-validation...")
    results = evaluate_classification(embeddings, labels, n_splits=args.n_splits)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Mean F1 Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"Chance Baseline: {results['chance_baseline']:.4f}")
    print(f"Above Chance: {results['above_chance']:.4f} ({results['above_chance']*100:.1f}%)")
    print(f"Number of Classes: {results['n_classes']}")
    print(f"Number of Samples: {results['n_samples']}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
