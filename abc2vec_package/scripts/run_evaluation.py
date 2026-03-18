#!/usr/bin/env python
"""
Evaluation Script for ABC2Vec.

Evaluates trained model on retrieval benchmarks.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.auto import tqdm

from abc2vec.data import load_processed_data
from abc2vec.model import ABC2VecModel
from abc2vec.tokenizer import ABCVocabulary, BarPatchifier


def compute_embeddings(
    model: ABC2VecModel,
    df: pd.DataFrame,
    patchifier: BarPatchifier,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Compute embeddings for all tunes in dataset.

    Args:
        model: Trained ABC2VecModel
        df: DataFrame with tune data
        patchifier: BarPatchifier instance
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        Embeddings array of shape (num_tunes, d_embed)
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Computing embeddings"):
            batch_df = df.iloc[i : i + batch_size]

            # Prepare batch
            batch_bar_indices = []
            batch_char_mask = []
            batch_bar_mask = []

            for _, row in batch_df.iterrows():
                patches = patchifier.patchify(row["abc_body"])
                batch_bar_indices.append(patches["bar_indices"])
                batch_char_mask.append(patches["char_mask"])
                batch_bar_mask.append(patches["bar_mask"])

            # Stack into tensors
            bar_indices = torch.stack(batch_bar_indices).to(device)
            char_mask = torch.stack(batch_char_mask).to(device)
            bar_mask = torch.stack(batch_bar_mask).to(device)

            # Get embeddings
            embeddings = model.get_embedding(bar_indices, char_mask, bar_mask)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def evaluate_retrieval(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    k_values: list = [1, 5, 10, 20],
) -> tuple:
    """
    Evaluate retrieval performance.

    For each query tune, find nearest neighbors and measure
    how many share the same tune type or key.

    Args:
        embeddings: Embedding matrix (num_tunes, d_embed)
        metadata: DataFrame with tune_type and key columns
        k_values: List of k values for Recall@k

    Returns:
        Tuple of (metrics dict, predictions array, similarities matrix)
    """
    num_tunes = embeddings.shape[0]

    # Compute pairwise cosine similarities
    embeddings_norm = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    similarities = embeddings_norm @ embeddings_norm.T

    metrics = {}
    predictions = []  # Store predictions for confusion matrix

    # Evaluate tune type retrieval
    tune_types = metadata["tune_type"].values

    for k in k_values:
        correct = 0
        for i in range(num_tunes):
            # Get top-k neighbors (excluding self)
            sims = similarities[i].copy()
            sims[i] = -np.inf  # Exclude self
            top_k_indices = np.argsort(sims)[-k:]

            # Check if any neighbor has same tune type
            query_type = tune_types[i]
            neighbor_types = tune_types[top_k_indices]

            if query_type in neighbor_types:
                correct += 1

            # For k=1, save predictions for confusion matrix
            if k == 1:
                predicted_type = neighbor_types[-1]  # Top 1 result
                predictions.append({
                    'true': query_type,
                    'predicted': predicted_type,
                    'correct': query_type == predicted_type
                })

        recall = correct / num_tunes
        metrics[f"tune_type_recall@{k}"] = recall

    return metrics, predictions, similarities


def plot_confusion_matrix(predictions: list, output_path: Path):
    """
    Generate confusion matrix for tune type retrieval.

    Args:
        predictions: List of dicts with 'true' and 'predicted' tune types
        output_path: Path to save plot
    """
    from sklearn.metrics import confusion_matrix

    pred_df = pd.DataFrame(predictions)

    # Filter out NaN values
    pred_df = pred_df[pred_df['true'].notna() & pred_df['predicted'].notna()]

    if len(pred_df) == 0:
        print("  ⚠ No valid predictions for confusion matrix")
        return

    true_labels = pred_df['true'].values
    pred_labels = pred_df['predicted'].values

    # Get unique tune types
    unique_types = sorted(set(true_labels) | set(pred_labels))

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_types)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=unique_types, yticklabels=unique_types,
        ax=ax, cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel('Predicted Tune Type', fontsize=12)
    ax.set_ylabel('True Tune Type', fontsize=12)
    ax.set_title('Confusion Matrix: Tune Type Retrieval (Top-1)', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved to: {output_path}")


def plot_recall_curves(metrics: dict, output_path: Path):
    """
    Plot Recall@K curves.

    Args:
        metrics: Dictionary with recall metrics
        output_path: Path to save plot
    """
    # Extract recall values
    k_values = [1, 5, 10, 20]
    recall_values = [metrics[f"tune_type_recall@{k}"] for k in k_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, recall_values, 'o-', linewidth=2, markersize=8, color='#e6445a')
    ax.set_xlabel('K (Number of Retrieved Results)', fontsize=12)
    ax.set_ylabel('Recall@K', fontsize=12)
    ax.set_title('ABC2Vec Retrieval Performance', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add value labels
    for k, r in zip(k_values, recall_values):
        ax.text(k, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Recall curve saved to: {output_path}")


def plot_similarity_distribution(similarities: np.ndarray, output_path: Path):
    """
    Plot distribution of similarity scores.

    Args:
        similarities: Similarity matrix
        output_path: Path to save plot
    """
    # Get all similarities (excluding diagonal)
    mask = ~np.eye(similarities.shape[0], dtype=bool)
    all_sims = similarities[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(all_sims, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0].set_xlabel('Cosine Similarity', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of Pairwise Similarities', fontsize=13)
    axes[0].axvline(x=all_sims.mean(), color='red', linestyle='--',
                    label=f'Mean={all_sims.mean():.3f}')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(all_sims, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[1].set_ylabel('Cosine Similarity', fontsize=12)
    axes[1].set_title('Similarity Score Distribution', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Similarity distribution saved to: {output_path}")


def plot_type_performance(predictions: list, output_path: Path):
    """
    Plot per-type retrieval accuracy.

    Args:
        predictions: List of dicts with 'true', 'predicted', 'correct'
        output_path: Path to save plot
    """
    pred_df = pd.DataFrame(predictions)

    # Compute accuracy per type
    type_accuracy = pred_df.groupby('true')['correct'].agg(['mean', 'count']).reset_index()
    type_accuracy.columns = ['Tune Type', 'Accuracy', 'Count']
    type_accuracy = type_accuracy.sort_values('Accuracy', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(type_accuracy) * 0.4)))
    bars = ax.barh(type_accuracy['Tune Type'], type_accuracy['Accuracy'],
                   color='steelblue', alpha=0.8)

    # Add count labels
    for i, (idx, row) in enumerate(type_accuracy.iterrows()):
        ax.text(row['Accuracy'] + 0.01, i, f"n={row['Count']}",
                va='center', fontsize=9, color='gray')

    ax.set_xlabel('Recall@1 Accuracy', fontsize=12)
    ax.set_ylabel('Tune Type', fontsize=12)
    ax.set_title('Retrieval Accuracy by Tune Type', fontsize=14, pad=20)
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Per-type performance saved to: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Evaluation - Retrieval benchmarks"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Evaluation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Split:       {args.split}")
    print(f"  Device:      {args.device}")
    print()

    # Load model
    print("Loading model...")
    model = ABC2VecModel.load_pretrained(args.checkpoint, device=args.device)
    print(f"  ✓ Model loaded from {args.checkpoint}")

    # Load vocabulary and data
    print("\nLoading data...")
    vocab_path = Path(args.data_dir) / "vocab.json"
    vocab = ABCVocabulary.load(vocab_path)

    df = load_processed_data(args.data_dir, split=args.split)
    metadata_df = pd.read_csv(Path(args.data_dir) / "metadata.csv")

    # Filter metadata to current split
    split_metadata = metadata_df[metadata_df["tune_id"].isin(df["tune_id"])]

    print(f"  {len(df):,} tunes in {args.split} set")

    # Create patchifier
    patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)

    # Compute embeddings
    print("\nComputing embeddings...")
    embeddings = compute_embeddings(
        model, df, patchifier, args.device, args.batch_size
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # Evaluate retrieval
    print("\nEvaluating retrieval...")
    metrics, predictions, similarities = evaluate_retrieval(
        embeddings, split_metadata, k_values=[1, 5, 10, 20]
    )

    print("\nRetrieval Results (Tune Type):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    checkpoint_dir = Path(args.checkpoint).parent
    results_path = checkpoint_dir / f"eval_{args.split}.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # Generate visualizations
    print("\nGenerating plots...")
    plot_dir = checkpoint_dir / "evaluation_plots"
    plot_dir.mkdir(exist_ok=True)

    plot_confusion_matrix(predictions, plot_dir / f"confusion_matrix_{args.split}.png")
    plot_recall_curves(metrics, plot_dir / f"recall_curves_{args.split}.png")
    plot_similarity_distribution(similarities, plot_dir / f"similarity_dist_{args.split}.png")
    plot_type_performance(predictions, plot_dir / f"type_performance_{args.split}.png")

    print(f"\n✓ All plots saved to: {plot_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
