#!/usr/bin/env python
"""
Generate UMAP and t-SNE visualizations from real ABC2Vec embeddings.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.manifold import TSNE
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


def plot_umap_combined(embeddings, df, output_path):
    """Generate combined UMAP figure with tune type and mode subplots."""

    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'figure.dpi': 300,
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
    })

    # Compute UMAP projection
    print("Computing UMAP projection...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Color schemes
    tune_type_colors = {
        'reel': '#1f77b4',
        'jig': '#ff7f0e',
        'polka': '#2ca02c',
        'waltz': '#d62728',
        'slip jig': '#9467bd',
        'slide': '#8c564b',
    }

    mode_colors = {
        'major': '#e41a1c',
        'minor': '#377eb8',
        'dorian': '#4daf4a',
        'mixolydian': '#ff7f00',
    }

    # Plot (a) - Tune Type
    tune_type_order = ['reel', 'jig', 'polka', 'waltz', 'slip jig', 'slide']
    for ttype in tune_type_order:
        mask = df['tune_type'] == ttype
        if mask.sum() > 0:
            ax1.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=tune_type_colors[ttype],
                label=ttype.title(),
                s=30,
                alpha=0.6,
                edgecolors='none'
            )

    ax1.set_xlabel('UMAP Dimension 1', fontweight='bold')
    ax1.set_ylabel('UMAP Dimension 2', fontweight='bold')
    ax1.set_title('(a) Colored by Tune Type', fontweight='bold', pad=10)
    ax1.legend(loc='best', frameon=True, fontsize=10, markerscale=1.5)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot (b) - Mode
    mode_order = ['major', 'minor', 'dorian', 'mixolydian']
    for mode in mode_order:
        mask = df['mode'] == mode
        if mask.sum() > 0:
            ax2.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=mode_colors[mode],
                label=mode.title(),
                s=30,
                alpha=0.6,
                edgecolors='none'
            )

    ax2.set_xlabel('UMAP Dimension 1', fontweight='bold')
    ax2.set_ylabel('UMAP Dimension 2', fontweight='bold')
    ax2.set_title('(b) Colored by Mode', fontweight='bold', pad=10)
    ax2.legend(loc='best', frameon=True, fontsize=10, markerscale=1.5)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")

    # Also save PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {png_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate UMAP visualizations")
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
        help="Dataset split",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./figures",
        help="Output directory",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec UMAP Visualization Generation")
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
    print(f"✓ Model loaded (step {checkpoint.get('step', 'unknown')})")

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

    # Compute embeddings
    embeddings = compute_embeddings(model, df, patchifier, args.device)
    print(f"✓ Embeddings shape: {embeddings.shape}")

    # Generate visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_umap_combined(embeddings, df, output_dir / "umap_combined.pdf")

    print("\n✓ Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
