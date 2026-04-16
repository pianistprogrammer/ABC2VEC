#!/usr/bin/env python
"""
Clustering Quality Analysis for ABC2Vec.

Evaluates unsupervised clustering quality using K-means.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
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


def evaluate_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    property_name: str,
    random_state: int = 42,
) -> dict:
    """Evaluate clustering quality for a property."""

    n_clusters = len(np.unique(labels))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute metrics
    silhouette = silhouette_score(embeddings, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)

    return {
        "property": property_name,
        "n_clusters": n_clusters,
        "n_samples": len(labels),
        "silhouette_score": silhouette,
        "nmi": nmi,
        "ari": ari,
    }


def main():
    parser = argparse.ArgumentParser(description="Clustering quality analysis")
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
        default="./checkpoints/clustering_results.json",
        help="Output path for results",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Clustering Quality Analysis")
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

    # Evaluate clustering for each property
    results = {}

    # 1. Tune Type
    print("\n" + "=" * 80)
    print("Clustering: Tune Type")
    print("=" * 80)
    df_filtered = df[df['tune_type'].notna() & (df['tune_type'] != '')]
    tune_type_embeddings = embeddings[df_filtered.index]
    tune_type_labels = df_filtered['tune_type'].values

    result = evaluate_clustering(tune_type_embeddings, tune_type_labels, "tune_type")
    results['tune_type'] = result
    print(f"Silhouette: {result['silhouette_score']:.4f}")
    print(f"NMI: {result['nmi']:.4f}")
    print(f"ARI: {result['ari']:.4f}")

    # 2. Mode
    print("\n" + "=" * 80)
    print("Clustering: Mode")
    print("=" * 80)
    main_modes = ['major', 'minor', 'dorian', 'mixolydian']
    df_filtered = df[df['mode'].isin(main_modes)]
    mode_embeddings = embeddings[df_filtered.index]
    mode_labels = df_filtered['mode'].values

    result = evaluate_clustering(mode_embeddings, mode_labels, "mode")
    results['mode'] = result
    print(f"Silhouette: {result['silhouette_score']:.4f}")
    print(f"NMI: {result['nmi']:.4f}")
    print(f"ARI: {result['ari']:.4f}")

    # 3. Key Root
    print("\n" + "=" * 80)
    print("Clustering: Key Root")
    print("=" * 80)
    # Get 8 most common keys
    top_keys = df['key'].value_counts().head(8).index.tolist()
    df_filtered = df[df['key'].isin(top_keys)]
    key_embeddings = embeddings[df_filtered.index]
    key_labels = df_filtered['key'].values

    result = evaluate_clustering(key_embeddings, key_labels, "key_root")
    results['key_root'] = result
    print(f"Silhouette: {result['silhouette_score']:.4f}")
    print(f"NMI: {result['nmi']:.4f}")
    print(f"ARI: {result['ari']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Property':15s} {'Silhouette':>12s} {'NMI':>8s} {'ARI':>8s}")
    print("-" * 50)
    for prop_name, result in results.items():
        print(f"{prop_name:15s} {result['silhouette_score']:>12.4f} "
              f"{result['nmi']:>8.4f} {result['ari']:>8.4f}")

    print(f"\n✓ Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
