#!/usr/bin/env python
"""
Folk Tune Evolution Visualization Script for ABC2Vec.

Analyzes temporal evolution of folk melodies using dated tunes:
- UMAP colored by decade/year
- Embedding drift over time
- Centroid trajectory visualization
- Intra-decade diversity analysis
- Interactive timeline visualization
"""

import argparse
import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

try:
    import umap
except ImportError:
    print("Installing umap-learn...")
    import subprocess
    subprocess.check_call(["pip", "install", "umap-learn"])
    import umap

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    print("Plotly not available - interactive plots will be skipped")
    HAS_PLOTLY = False

from abc2vec.model import ABC2VecModel
from abc2vec.tokenizer import ABCVocabulary, BarPatchifier


def compute_embeddings(
    model: ABC2VecModel,
    df: pd.DataFrame,
    patchifier: BarPatchifier,
    device: str,
    batch_size: int = 64,
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


def plot_umap_by_year(
    embeddings: np.ndarray, years: np.ndarray, output_path: Path
):
    """Generate UMAP visualization colored by year."""
    print("  Computing UMAP projection...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42
    )
    umap_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))
    sc = ax.scatter(
        umap_2d[:, 0], umap_2d[:, 1], c=years, cmap="viridis", s=10, alpha=0.6
    )
    plt.colorbar(sc, label="Year", ax=ax)
    ax.set_title("Folk Tune Evolution - UMAP Colored by Year", fontsize=14, pad=20)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return umap_2d


def plot_centroid_drift(
    embeddings: np.ndarray,
    umap_2d: np.ndarray,
    decades: np.ndarray,
    output_path: Path,
):
    """Visualize how tune centroids drift over decades."""
    print("  Computing decade centroids...")

    unique_decades = sorted(set(decades))
    decade_counts = pd.Series(decades).value_counts()
    valid_decades = [d for d in unique_decades if decade_counts.get(d, 0) >= 5]

    centroids_umap = []
    centroids_emb = []
    centroid_decades = []
    centroid_sizes = []

    for decade in valid_decades:
        mask = decades == decade
        n = mask.sum()

        centroid_emb = embeddings[mask].mean(axis=0)
        centroid_umap = umap_2d[mask].mean(axis=0)

        centroids_emb.append(centroid_emb)
        centroids_umap.append(centroid_umap)
        centroid_decades.append(decade)
        centroid_sizes.append(n)

    centroids_umap = np.array(centroids_umap)
    centroids_emb = np.array(centroids_emb)

    # Plot centroid trajectory
    fig, ax = plt.subplots(figsize=(14, 10))

    # Background: all points
    ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c="lightgray", s=5, alpha=0.3)

    # Centroid trajectory
    colors = cm.plasma(np.linspace(0, 1, len(centroid_decades)))
    for i in range(len(centroids_umap)):
        ax.scatter(
            centroids_umap[i, 0],
            centroids_umap[i, 1],
            c=[colors[i]],
            s=centroid_sizes[i] * 3 + 50,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            f"{centroid_decades[i]}s",
            (centroids_umap[i, 0], centroids_umap[i, 1]),
            fontsize=7,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Connect centroids with arrows
    for i in range(len(centroids_umap) - 1):
        ax.annotate(
            "",
            xy=centroids_umap[i + 1],
            xytext=centroids_umap[i],
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )

    ax.set_title("Folk Tune Centroid Drift Over Decades", fontsize=14, pad=20)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return centroids_emb, centroid_decades


def plot_decade_drift_metrics(
    centroids_emb: np.ndarray, centroid_decades: list, output_path: Path
):
    """Plot decade-to-decade drift metrics."""
    # Pairwise cosine similarity between decade centroids
    centroid_sim = cosine_similarity(centroids_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap of decade-decade similarity
    sns.heatmap(
        centroid_sim,
        xticklabels=[f"{d}s" for d in centroid_decades],
        yticklabels=[f"{d}s" for d in centroid_decades],
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        ax=axes[0],
    )
    axes[0].set_title("Decade Centroid Cosine Similarity", fontsize=13)

    # Consecutive decade drift
    consec_dist = []
    transitions = []
    for i in range(len(centroids_emb) - 1):
        cos_sim = cosine_similarity(
            centroids_emb[i : i + 1], centroids_emb[i + 1 : i + 2]
        )[0, 0]
        consec_dist.append(1 - cos_sim)
        transitions.append(f"{centroid_decades[i]}→{centroid_decades[i+1]}")

    axes[1].plot(range(len(consec_dist)), consec_dist, "o-", color="steelblue", linewidth=2)
    axes[1].set_xticks(range(len(transitions)))
    axes[1].set_xticklabels(transitions, rotation=45, ha="right")
    axes[1].set_xlabel("Decade Transition", fontsize=12)
    axes[1].set_ylabel("Cosine Distance", fontsize=12)
    axes[1].set_title("Consecutive Decade Centroid Drift", fontsize=13)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_intra_decade_diversity(
    embeddings: np.ndarray, decades: np.ndarray, output_path: Path
):
    """Analyze and plot diversity within each decade."""
    print("  Computing intra-decade diversity...")

    unique_decades = sorted(set(decades))
    decade_diversity = []

    for decade in unique_decades:
        mask = decades == decade
        emb_subset = embeddings[mask]

        if len(emb_subset) < 2:
            continue

        # Sample if too large
        if len(emb_subset) > 200:
            idx = np.random.choice(len(emb_subset), 200, replace=False)
            emb_subset = emb_subset[idx]

        sim = cosine_similarity(emb_subset)
        triu = np.triu_indices(len(emb_subset), k=1)
        mean_sim = sim[triu].mean()
        std_sim = sim[triu].std()

        decade_diversity.append(
            {
                "decade": decade,
                "n_tunes": mask.sum(),
                "mean_similarity": mean_sim,
                "std_similarity": std_sim,
                "diversity": 1 - mean_sim,
            }
        )

    diversity_df = pd.DataFrame(decade_diversity)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        diversity_df["decade"].astype(str),
        diversity_df["diversity"],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Decade", fontsize=12)
    ax.set_ylabel("Intra-Decade Diversity (1 - mean cosine sim)", fontsize=12)
    ax.set_title("Melodic Diversity Within Each Decade", fontsize=14, pad=20)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return diversity_df


def plot_type_distribution_over_time(
    df: pd.DataFrame, output_path: Path
):
    """Plot how tune types change over time."""
    if "tune_type" not in df.columns or "decade" not in df.columns:
        print("  ⚠ Missing tune_type or decade columns")
        return

    # Group by decade and tune type
    type_decade = df.groupby(["decade", "tune_type"]).size().unstack(fill_value=0)

    # Normalize per decade
    type_decade_pct = type_decade.div(type_decade.sum(axis=1), axis=0)

    # Keep only major types
    major_types = type_decade.sum().nlargest(6).index.tolist()
    type_decade_pct = type_decade_pct[major_types]

    fig, ax = plt.subplots(figsize=(14, 6))
    type_decade_pct.plot(kind="area", stacked=True, ax=ax, alpha=0.8)
    ax.set_xlabel("Decade", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Tune Type Distribution Over Time", fontsize=14, pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_interactive_timeline(
    umap_2d: np.ndarray, df: pd.DataFrame, output_path: Path
):
    """Create interactive Plotly timeline visualization."""
    if not HAS_PLOTLY:
        print("  ⚠ Plotly not available - skipping interactive plot")
        return

    print("  Creating interactive timeline...")

    plot_df = df.copy()
    plot_df["umap_x"] = umap_2d[:, 0]
    plot_df["umap_y"] = umap_2d[:, 1]
    plot_df["title_str"] = plot_df.get("title", plot_df.get("tune_name", "Unknown"))
    plot_df["type_str"] = plot_df.get("tune_type", "unknown")

    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="year",
        hover_data=["title_str", "year", "type_str"],
        title="Folk Tune Evolution - Interactive UMAP",
        color_continuous_scale="Viridis",
        opacity=0.6,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title=""),
        width=1000,
        height=800,
    )

    fig.write_html(output_path)
    print(f"    ✓ Interactive plot saved to {output_path}")


def main():
    """Main evolution analysis function."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Folk Tune Evolution Analysis"
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
        "--data_file",
        type=str,
        default=None,
        help="Specific data file with dated tunes (JSON or Parquet)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/evolution",
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
        "--min_year",
        type=int,
        default=1600,
        help="Minimum year to include",
    )
    parser.add_argument(
        "--max_year",
        type=int,
        default=2020,
        help="Maximum year to include",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Folk Tune Evolution Analysis")
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

    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab_path = Path(args.data_dir) / "vocab.json"
    vocab = ABCVocabulary.load(vocab_path)
    patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)

    # Load dated tunes
    print("\nLoading dated tunes...")

    if args.data_file:
        data_path = Path(args.data_file)
    else:
        # Try BFDB (British Folk Database)
        data_path = Path(args.data_dir) / "bfdb.parquet"
        if not data_path.exists():
            data_path = Path(args.data_dir) / "bfdb.json"

    if not data_path.exists():
        print(f"\n⚠ No dated tune data found at {data_path}")
        print("  Expected: bfdb.parquet, bfdb.json, or specify --data_file")
        print("  Exiting...")
        return

    if data_path.suffix == ".parquet":
        dated_df = pd.read_parquet(data_path)
    else:
        with open(data_path) as f:
            dated_df = pd.DataFrame(json.load(f))

    print(f"  Loaded {len(dated_df):,} tunes")

    # Filter to tunes with year metadata
    if "year" not in dated_df.columns:
        print("\n⚠ No 'year' column in data - cannot analyze temporal evolution")
        return

    dated_df = dated_df[dated_df["year"].notna()].copy()
    dated_df["year"] = dated_df["year"].astype(int)

    # Filter to reasonable year range
    dated_df = dated_df[
        (dated_df["year"] >= args.min_year) & (dated_df["year"] <= args.max_year)
    ]
    dated_df["decade"] = (dated_df["year"] // 10) * 10

    print(f"  Dated tunes: {len(dated_df):,}")
    print(f"  Year range: {dated_df['year'].min()} – {dated_df['year'].max()}")
    print(f"  Decades: {sorted(dated_df['decade'].unique())}")

    if len(dated_df) < 100:
        print("\n⚠ Too few dated tunes (<100) for meaningful analysis")
        return

    # Compute embeddings
    print("\nComputing embeddings...")
    embeddings = compute_embeddings(
        model, dated_df, patchifier, args.device, args.batch_size
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save embeddings
    np.save(output_dir / "evolution_embeddings.npy", embeddings)
    dated_df.to_parquet(output_dir / "evolution_metadata.parquet")
    print(f"  ✓ Embeddings and metadata saved")

    # ========================================================================
    # 1. UMAP by Year
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. UMAP COLORED BY YEAR")
    print("=" * 80)

    umap_2d = plot_umap_by_year(
        embeddings, dated_df["year"].values, output_dir / "umap_by_year.png"
    )
    print(f"  ✓ Saved to {output_dir / 'umap_by_year.png'}")

    # ========================================================================
    # 2. Centroid Drift
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. CENTROID DRIFT OVER DECADES")
    print("=" * 80)

    centroids_emb, centroid_decades = plot_centroid_drift(
        embeddings, umap_2d, dated_df["decade"].values, output_dir / "centroid_drift.png"
    )
    print(f"  ✓ Saved to {output_dir / 'centroid_drift.png'}")

    # ========================================================================
    # 3. Decade-to-Decade Drift Metrics
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. DECADE DRIFT METRICS")
    print("=" * 80)

    plot_decade_drift_metrics(
        centroids_emb, centroid_decades, output_dir / "decade_drift_metrics.png"
    )
    print(f"  ✓ Saved to {output_dir / 'decade_drift_metrics.png'}")

    # ========================================================================
    # 4. Intra-Decade Diversity
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. INTRA-DECADE DIVERSITY")
    print("=" * 80)

    diversity_df = plot_intra_decade_diversity(
        embeddings, dated_df["decade"].values, output_dir / "decade_diversity.png"
    )
    print(f"  ✓ Saved to {output_dir / 'decade_diversity.png'}")

    # Save diversity stats
    diversity_df.to_csv(output_dir / "decade_diversity.csv", index=False)
    print(f"  ✓ Diversity stats saved to {output_dir / 'decade_diversity.csv'}")

    print("\n  Diversity Summary:")
    print(diversity_df.to_string(index=False))

    # ========================================================================
    # 5. Tune Type Distribution Over Time
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. TUNE TYPE DISTRIBUTION OVER TIME")
    print("=" * 80)

    plot_type_distribution_over_time(
        dated_df, output_dir / "type_distribution_over_time.png"
    )
    print(f"  ✓ Saved to {output_dir / 'type_distribution_over_time.png'}")

    # ========================================================================
    # 6. Interactive Timeline (Optional)
    # ========================================================================
    if HAS_PLOTLY:
        print("\n" + "=" * 80)
        print("6. INTERACTIVE TIMELINE")
        print("=" * 80)

        dated_df["umap_x"] = umap_2d[:, 0]
        dated_df["umap_y"] = umap_2d[:, 1]

        create_interactive_timeline(
            umap_2d, dated_df, output_dir / "evolution_interactive.html"
        )

    # ========================================================================
    # Save Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    summary = {
        "n_tunes": int(len(dated_df)),
        "year_range": [int(dated_df["year"].min()), int(dated_df["year"].max())],
        "n_decades": int(dated_df["decade"].nunique()),
        "diversity_by_decade": diversity_df.to_dict(orient="records"),
    }

    results_path = output_dir / "evolution_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {results_path}")
    print(f"✓ All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
