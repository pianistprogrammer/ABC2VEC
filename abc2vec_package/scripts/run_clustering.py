#!/usr/bin/env python
"""
Clustering Analysis Script for ABC2Vec.

Evaluates whether ABC2Vec embeddings capture musically meaningful structure:
- Tune type clustering (jig, reel, hornpipe, etc.)
- Mode clustering (major, minor, dorian, mixolydian)
- Key root clustering
- UMAP/t-SNE visualization
- Linear probe classification
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

try:
    import umap
except ImportError:
    print("Installing umap-learn...")
    import subprocess
    subprocess.check_call(["pip", "install", "umap-learn"])
    import umap

from abc2vec.data import load_processed_data
from abc2vec.model import ABC2VecModel
from abc2vec.tokenizer import ABCVocabulary, BarPatchifier


def compute_embeddings(
    model: ABC2VecModel,
    df: pd.DataFrame,
    patchifier: BarPatchifier,
    device: str,
    batch_size: int = 64,
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


def linear_probe_classification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_name: str = "label",
) -> dict:
    """
    Train linear probe for classification.

    Args:
        embeddings: Embedding matrix
        labels: Label array
        label_name: Name of the label type

    Returns:
        Dictionary with classification metrics
    """
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv_scores = cross_val_score(clf, embeddings, labels, cv=5, scoring="accuracy")

    # Full fit for confusion matrix
    clf.fit(embeddings, labels)
    y_pred = clf.predict(embeddings)

    return {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores.tolist(),
        "predictions": y_pred.tolist(),
    }


def clustering_metrics(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int,
) -> dict:
    """
    Compute clustering quality metrics.

    Args:
        embeddings: Embedding matrix
        true_labels: True labels
        n_clusters: Number of clusters for KMeans

    Returns:
        Dictionary with clustering metrics
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    sil = silhouette_score(embeddings, true_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)

    return {
        "silhouette_score": sil,
        "nmi": nmi,
        "adjusted_rand_index": ari,
    }


def plot_confusion_matrix_clustering(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list,
    output_path: Path,
    title: str = "Confusion Matrix",
):
    """Plot confusion matrix for classification results."""
    cm = confusion_matrix(true_labels, pred_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_umap_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    output_path: Path,
    title: str = "UMAP Visualization",
):
    """Generate UMAP visualization colored by labels."""
    print("  Computing UMAP projection...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42
    )
    umap_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names[i] if i < len(label_names) else str(label)
        ax.scatter(
            umap_2d[mask, 0],
            umap_2d[mask, 1],
            c=[colors[i]],
            label=label_name,
            s=10,
            alpha=0.6,
        )

    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(markerscale=3, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tsne_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    output_path: Path,
    title: str = "t-SNE Visualization",
    max_samples: int = 5000,
):
    """Generate t-SNE visualization colored by labels."""
    print("  Computing t-SNE projection...")

    # Subsample if too large
    if len(embeddings) > max_samples:
        idx = np.random.RandomState(42).choice(
            len(embeddings), max_samples, replace=False
        )
        embeddings = embeddings[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = sorted(set(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names[i] if i < len(label_names) else str(label)
        ax.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            c=[colors[i]],
            label=label_name,
            s=10,
            alpha=0.6,
        )

    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(markerscale=3, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Main clustering analysis function."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Clustering Analysis - UMAP, t-SNE, Linear Probes"
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
        "--output_dir",
        type=str,
        default="./results/clustering",
        help="Directory to save results",
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum samples for t-SNE (for performance)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Clustering Analysis")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Device:      {args.device}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = ABC2VecModel.load_pretrained(args.checkpoint, device=args.device)
    print(f"  ✓ Model loaded")

    # Load vocabulary and data
    print("\nLoading data...")
    vocab_path = Path(args.data_dir) / "vocab.json"
    vocab = ABCVocabulary.load(vocab_path)

    test_df = load_processed_data(args.data_dir, split="test")
    metadata_df = pd.read_csv(Path(args.data_dir) / "metadata.csv")

    # Filter metadata to test split
    test_metadata = metadata_df[metadata_df["tune_id"].isin(test_df["tune_id"])]

    print(f"  {len(test_df):,} tunes in test set")

    # Create patchifier
    patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)

    # Compute embeddings
    print("\nComputing embeddings...")
    embeddings = compute_embeddings(
        model, test_df, patchifier, args.device, args.batch_size
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save embeddings for future use
    np.save(output_dir / "test_embeddings.npy", embeddings)
    print(f"  ✓ Embeddings saved")

    all_metrics = {}

    # ========================================================================
    # 1. Tune Type Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. TUNE TYPE CLASSIFICATION")
    print("=" * 80)

    if "tune_type" in test_metadata.columns:
        tune_types = test_metadata["tune_type"].values

        # Filter out NaN/empty
        valid_mask = pd.notna(tune_types) & (tune_types != "")
        if valid_mask.sum() > 0:
            valid_embeddings = embeddings[valid_mask]
            valid_types = tune_types[valid_mask]

            # Encode labels
            le_type = LabelEncoder()
            y_type = le_type.fit_transform(valid_types)

            print(f"  Classes: {list(le_type.classes_)}")
            print(f"  Samples: {len(y_type)}")

            # Linear probe
            probe_results = linear_probe_classification(
                valid_embeddings, y_type, "tune_type"
            )
            print(f"\n  Linear Probe (5-fold CV):")
            print(f"    Accuracy: {probe_results['cv_mean']:.4f} ± {probe_results['cv_std']:.4f}")

            all_metrics["tune_type_probe"] = {
                "accuracy_mean": probe_results["cv_mean"],
                "accuracy_std": probe_results["cv_std"],
            }

            # Clustering metrics
            cluster_metrics = clustering_metrics(
                valid_embeddings, y_type, len(le_type.classes_)
            )
            print(f"\n  Clustering Metrics:")
            print(f"    Silhouette: {cluster_metrics['silhouette_score']:.4f}")
            print(f"    NMI:        {cluster_metrics['nmi']:.4f}")
            print(f"    ARI:        {cluster_metrics['adjusted_rand_index']:.4f}")

            all_metrics["tune_type_clustering"] = cluster_metrics

            # Plot confusion matrix
            print("\n  Generating confusion matrix...")
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clf.fit(valid_embeddings, y_type)
            y_pred = clf.predict(valid_embeddings)

            plot_confusion_matrix_clustering(
                y_type,
                y_pred,
                le_type.classes_.tolist(),
                output_dir / "tune_type_confusion.png",
                title="Tune Type Classification - Confusion Matrix",
            )
            print(f"    ✓ Saved to {output_dir / 'tune_type_confusion.png'}")

            # UMAP visualization
            print("\n  Generating UMAP visualization...")
            plot_umap_visualization(
                valid_embeddings,
                y_type,
                le_type.classes_.tolist(),
                output_dir / "umap_tune_type.png",
                title="UMAP - Colored by Tune Type",
            )
            print(f"    ✓ Saved to {output_dir / 'umap_tune_type.png'}")

            # t-SNE visualization
            print("\n  Generating t-SNE visualization...")
            plot_tsne_visualization(
                valid_embeddings,
                y_type,
                le_type.classes_.tolist(),
                output_dir / "tsne_tune_type.png",
                title="t-SNE - Colored by Tune Type",
                max_samples=args.max_samples,
            )
            print(f"    ✓ Saved to {output_dir / 'tsne_tune_type.png'}")

    else:
        print("  ⚠ No tune_type column in metadata")

    # ========================================================================
    # 2. Mode Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. MODE CLUSTERING")
    print("=" * 80)

    if "mode" in test_metadata.columns:
        modes = test_metadata["mode"].values

        # Filter out NaN/empty
        valid_mask = pd.notna(modes) & (modes != "")
        if valid_mask.sum() > 0:
            valid_embeddings = embeddings[valid_mask]
            valid_modes = modes[valid_mask]

            # Encode labels
            le_mode = LabelEncoder()
            y_mode = le_mode.fit_transform(valid_modes)

            print(f"  Modes: {list(le_mode.classes_)}")
            print(f"  Samples: {len(y_mode)}")

            # Linear probe
            probe_results = linear_probe_classification(valid_embeddings, y_mode, "mode")
            print(f"\n  Linear Probe (5-fold CV):")
            print(f"    Accuracy: {probe_results['cv_mean']:.4f} ± {probe_results['cv_std']:.4f}")

            all_metrics["mode_probe"] = {
                "accuracy_mean": probe_results["cv_mean"],
                "accuracy_std": probe_results["cv_std"],
            }

            # Clustering metrics
            cluster_metrics = clustering_metrics(
                valid_embeddings, y_mode, len(le_mode.classes_)
            )
            print(f"\n  Clustering Metrics:")
            print(f"    Silhouette: {cluster_metrics['silhouette_score']:.4f}")
            print(f"    NMI:        {cluster_metrics['nmi']:.4f}")
            print(f"    ARI:        {cluster_metrics['adjusted_rand_index']:.4f}")

            all_metrics["mode_clustering"] = cluster_metrics

            # UMAP visualization
            print("\n  Generating UMAP visualization...")
            plot_umap_visualization(
                valid_embeddings,
                y_mode,
                le_mode.classes_.tolist(),
                output_dir / "umap_mode.png",
                title="UMAP - Colored by Mode",
            )
            print(f"    ✓ Saved to {output_dir / 'umap_mode.png'}")

    else:
        print("  ⚠ No mode column in metadata")

    # ========================================================================
    # 3. Key Root Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. KEY ROOT CLUSTERING")
    print("=" * 80)

    if "key" in test_metadata.columns:
        # Extract key root (first letter)
        test_metadata_copy = test_metadata.copy()
        test_metadata_copy["key_root"] = (
            test_metadata_copy["key"].astype(str).str[0].str.upper()
        )
        key_roots = test_metadata_copy["key_root"].values

        # Filter to keys with enough samples (>30)
        key_counts = pd.Series(key_roots).value_counts()
        valid_keys = key_counts[key_counts >= 30].index.tolist()
        valid_mask = pd.Series(key_roots).isin(valid_keys).values & (
            pd.notna(key_roots)
        )

        if valid_mask.sum() > 0:
            valid_embeddings = embeddings[valid_mask]
            valid_keys_arr = key_roots[valid_mask]

            # Encode labels
            le_key = LabelEncoder()
            y_key = le_key.fit_transform(valid_keys_arr)

            print(f"  Keys: {list(le_key.classes_)}")
            print(f"  Samples: {len(y_key)}")

            # Linear probe
            probe_results = linear_probe_classification(valid_embeddings, y_key, "key")
            print(f"\n  Linear Probe (5-fold CV):")
            print(f"    Accuracy: {probe_results['cv_mean']:.4f} ± {probe_results['cv_std']:.4f}")

            all_metrics["key_probe"] = {
                "accuracy_mean": probe_results["cv_mean"],
                "accuracy_std": probe_results["cv_std"],
            }

            # Clustering metrics
            cluster_metrics = clustering_metrics(
                valid_embeddings, y_key, len(le_key.classes_)
            )
            print(f"\n  Clustering Metrics:")
            print(f"    Silhouette: {cluster_metrics['silhouette_score']:.4f}")
            print(f"    NMI:        {cluster_metrics['nmi']:.4f}")
            print(f"    ARI:        {cluster_metrics['adjusted_rand_index']:.4f}")

            all_metrics["key_clustering"] = cluster_metrics

    else:
        print("  ⚠ No key column in metadata")

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_path = output_dir / "clustering_metrics.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✓ Metrics saved to: {results_path}")
    print(f"✓ All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
