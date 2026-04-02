#!/usr/bin/env python
"""
Cross-Tradition Transfer Script for ABC2Vec.

Tests whether ABC2Vec (trained on Irish folk) generalizes to other traditions:
- Zero-shot retrieval on British folk and Nottingham datasets
- Cross-tradition similarity analysis
- Tradition separability (how tradition-specific are embeddings?)
- Visualization of cross-tradition embedding space
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
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

try:
    import faiss
except ImportError:
    print("Installing faiss-cpu...")
    import subprocess
    subprocess.check_call(["pip", "install", "faiss-cpu"])
    import faiss

from core.data import load_processed_data
from core.model import ABC2VecModel
from core.tokenizer import ABCVocabulary, BarPatchifier


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


def plot_umap_by_tradition(
    embeddings: np.ndarray,
    traditions: np.ndarray,
    output_path: Path,
):
    """Generate UMAP visualization colored by tradition."""
    print("  Computing UMAP projection...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42
    )
    umap_2d = reducer.fit_transform(embeddings)

    tradition_colors = {
        "irish": "#2ecc71",
        "british": "#e74c3c",
        "nottingham": "#3498db",
    }

    fig, ax = plt.subplots(figsize=(12, 10))

    for tradition in sorted(set(traditions)):
        mask = traditions == tradition
        color = tradition_colors.get(tradition, "#999999")
        ax.scatter(
            umap_2d[mask, 0],
            umap_2d[mask, 1],
            c=color,
            label=f"{tradition.capitalize()} (n={mask.sum()})",
            s=8,
            alpha=0.5,
        )

    ax.set_title("UMAP - ABC2Vec Embeddings by Tradition", fontsize=14, pad=20)
    ax.legend(markerscale=3, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_similarity_distributions(
    irish_emb: np.ndarray,
    cross_emb: np.ndarray,
    output_path: Path,
):
    """Plot within and across tradition similarity distributions."""
    # Sample for efficiency
    n_sample = min(500, len(irish_emb), len(cross_emb))
    np.random.seed(42)
    irish_sample = irish_emb[
        np.random.choice(len(irish_emb), n_sample, replace=False)
    ]
    cross_sample = cross_emb[
        np.random.choice(len(cross_emb), n_sample, replace=False)
    ]

    # Within-Irish similarities
    sim_irish = cosine_similarity(irish_sample)
    triu_idx = np.triu_indices(n_sample, k=1)
    irish_sims = sim_irish[triu_idx]

    # Within-cross similarities
    sim_cross = cosine_similarity(cross_sample)
    cross_sims = sim_cross[triu_idx]

    # Cross-tradition similarities
    sim_across = cosine_similarity(irish_sample, cross_sample)
    across_sims = sim_across.flatten()

    # Subsample across for plotting
    if len(across_sims) > len(irish_sims):
        across_sims = np.random.choice(
            across_sims, len(irish_sims), replace=False
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        irish_sims,
        bins=80,
        alpha=0.5,
        label="Within Irish",
        color="green",
        density=True,
    )
    ax.hist(
        cross_sims,
        bins=80,
        alpha=0.5,
        label="Within British/Nottingham",
        color="red",
        density=True,
    )
    ax.hist(
        across_sims,
        bins=80,
        alpha=0.5,
        label="Irish ↔ British",
        color="blue",
        density=True,
    )
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pairwise Similarity Distributions by Tradition", fontsize=14, pad=20)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "within_irish_mean": float(irish_sims.mean()),
        "within_irish_std": float(irish_sims.std()),
        "within_cross_mean": float(cross_sims.mean()),
        "within_cross_std": float(cross_sims.std()),
        "across_mean": float(across_sims.mean()),
        "across_std": float(across_sims.std()),
    }


def main():
    """Main cross-tradition analysis function."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Cross-Tradition Transfer Analysis"
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
        default="./results/cross_tradition",
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
        "--irish_samples",
        type=int,
        default=2000,
        help="Number of Irish test samples to use",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Cross-Tradition Transfer Analysis")
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

    # Load Irish test data
    print("\nLoading Irish test data...")
    irish_test = load_processed_data(args.data_dir, split="test")
    if len(irish_test) > args.irish_samples:
        irish_test = irish_test.sample(n=args.irish_samples, random_state=42)
    print(f"  Irish test: {len(irish_test):,} tunes")

    # Try to load cross-tradition data
    print("\nLooking for cross-tradition data...")
    cross_dfs = []

    # Check for British folk
    bfdb_path = Path(args.data_dir) / "bfdb.parquet"
    if not bfdb_path.exists():
        bfdb_path = Path(args.data_dir) / "bfdb.json"

    if bfdb_path.exists():
        if bfdb_path.suffix == ".parquet":
            bfdb_df = pd.read_parquet(bfdb_path)
        else:
            with open(bfdb_path) as f:
                bfdb_df = pd.DataFrame(json.load(f))
        bfdb_df["tradition"] = "british"
        cross_dfs.append(bfdb_df)
        print(f"  ✓ British folk: {len(bfdb_df):,} tunes")

    # Check for Nottingham
    notts_path = Path(args.data_dir) / "nottingham.parquet"
    if not notts_path.exists():
        notts_path = Path(args.data_dir) / "nottingham.json"

    if notts_path.exists():
        if notts_path.suffix == ".parquet":
            notts_df = pd.read_parquet(notts_path)
        else:
            with open(notts_path) as f:
                notts_df = pd.DataFrame(json.load(f))
        notts_df["tradition"] = "nottingham"
        cross_dfs.append(notts_df)
        print(f"  ✓ Nottingham: {len(notts_df):,} tunes")

    if not cross_dfs:
        print("\n⚠ No cross-tradition data found.")
        print("  Expected files: bfdb.parquet/json or nottingham.parquet/json")
        print("  Exiting...")
        return

    cross_df = pd.concat(cross_dfs, ignore_index=True)
    print(f"\n  Total cross-tradition: {len(cross_df):,} tunes")

    # Encode Irish tunes
    print("\nEncoding Irish tunes...")
    irish_emb = compute_embeddings(
        model, irish_test, patchifier, args.device, args.batch_size
    )
    print(f"  Irish embeddings: {irish_emb.shape}")

    # Encode cross-tradition tunes
    print("\nEncoding cross-tradition tunes...")
    cross_emb = compute_embeddings(
        model, cross_df, patchifier, args.device, args.batch_size
    )
    print(f"  Cross embeddings: {cross_emb.shape}")

    # Combine for analysis
    all_emb = np.vstack([irish_emb, cross_emb])
    irish_test["tradition"] = "irish"
    all_traditions = np.concatenate(
        [irish_test["tradition"].values, cross_df["tradition"].values]
    )

    print(f"\n  Total embeddings: {len(all_emb):,}")

    all_metrics = {}

    # ========================================================================
    # 1. Tradition Separability
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. TRADITION SEPARABILITY")
    print("=" * 80)

    le_trad = LabelEncoder()
    y_trad = le_trad.fit_transform(all_traditions)

    clf_trad = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv_trad = cross_val_score(clf_trad, all_emb, y_trad, cv=5, scoring="accuracy")

    print(f"\n  Tradition Classification (linear probe):")
    print(f"    Accuracy: {cv_trad.mean():.4f} ± {cv_trad.std():.4f}")
    print(f"    Classes: {list(le_trad.classes_)}")

    print(f"\n  Interpretation:")
    if cv_trad.mean() > 0.7:
        print(
            f"    High accuracy ({cv_trad.mean():.2%}) → embeddings are tradition-specific"
        )
    else:
        print(
            f"    Low accuracy ({cv_trad.mean():.2%}) → embeddings are more universal"
        )

    all_metrics["tradition_separability"] = {
        "accuracy_mean": float(cv_trad.mean()),
        "accuracy_std": float(cv_trad.std()),
        "classes": list(le_trad.classes_),
    }

    # ========================================================================
    # 2. Similarity Distributions
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. SIMILARITY DISTRIBUTIONS")
    print("=" * 80)

    sim_stats = plot_similarity_distributions(
        irish_emb, cross_emb, output_dir / "similarity_distributions.png"
    )

    print(f"\n  Within Irish:    mean={sim_stats['within_irish_mean']:.4f}, std={sim_stats['within_irish_std']:.4f}")
    print(f"  Within Cross:    mean={sim_stats['within_cross_mean']:.4f}, std={sim_stats['within_cross_std']:.4f}")
    print(f"  Across:          mean={sim_stats['across_mean']:.4f}, std={sim_stats['across_std']:.4f}")

    all_metrics["similarity_distributions"] = sim_stats

    print(f"\n  ✓ Saved to {output_dir / 'similarity_distributions.png'}")

    # ========================================================================
    # 3. UMAP Visualization
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. UMAP VISUALIZATION")
    print("=" * 80)

    plot_umap_by_tradition(
        all_emb, all_traditions, output_dir / "umap_traditions.png"
    )
    print(f"  ✓ Saved to {output_dir / 'umap_traditions.png'}")

    # ========================================================================
    # 4. Cross-Tradition Nearest Neighbors
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. CROSS-TRADITION NEAREST NEIGHBORS")
    print("=" * 80)

    print("\n  Building FAISS index on Irish tunes...")
    irish_norm = irish_emb / np.linalg.norm(irish_emb, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(irish_emb.shape[1])
    index.add(irish_norm.astype(np.float32))

    print("  Querying with cross-tradition tunes...")
    cross_norm = cross_emb / np.linalg.norm(cross_emb, axis=1, keepdims=True)
    distances, nn_indices = index.search(cross_norm.astype(np.float32), 5)

    # Show examples
    print("\n  Cross-Tradition Nearest Neighbor Examples:")
    print("  " + "=" * 76)

    np.random.seed(42)
    sample_idx = np.random.choice(len(cross_df), min(10, len(cross_df)), replace=False)

    for idx in sample_idx:
        cross_tune = cross_df.iloc[idx]
        tradition = cross_tune.get("tradition", "unknown").upper()
        title = cross_tune.get("title", cross_tune.get("tune_name", "N/A"))
        abc_body = str(cross_tune.get("abc_body", ""))[:80]

        print(f"\n  {tradition} tune: {title}")
        print(f"    ABC: {abc_body}...")
        print(f"    Nearest Irish tunes:")

        for rank in range(3):
            i_idx = nn_indices[idx, rank]
            irish_tune = irish_test.iloc[i_idx]
            irish_title = irish_tune.get("title", irish_tune.get("tune_name", "N/A"))
            sim = distances[idx, rank]
            print(f"      {rank+1}. {irish_title} (sim={sim:.4f})")

    # Save nearest neighbors
    nn_results = []
    for idx in range(min(100, len(cross_df))):
        cross_tune = cross_df.iloc[idx]
        nn_results.append(
            {
                "cross_tradition": cross_tune.get("tradition", "unknown"),
                "cross_title": cross_tune.get(
                    "title", cross_tune.get("tune_name", "N/A")
                ),
                "nearest_irish": [
                    {
                        "title": irish_test.iloc[nn_indices[idx, i]].get(
                            "title", irish_test.iloc[nn_indices[idx, i]].get("tune_name", "N/A")
                        ),
                        "similarity": float(distances[idx, i]),
                    }
                    for i in range(5)
                ],
            }
        )

    nn_path = output_dir / "nearest_neighbors.json"
    with open(nn_path, "w") as f:
        json.dump(nn_results, f, indent=2)

    print(f"\n  ✓ Nearest neighbors saved to {nn_path}")

    # ========================================================================
    # 5. Cross-Tradition Tune Type Transfer (if available)
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. CROSS-TRADITION TUNE TYPE TRANSFER")
    print("=" * 80)

    irish_metadata = pd.read_csv(Path(args.data_dir) / "metadata.csv")
    irish_meta_test = irish_metadata[
        irish_metadata["tune_id"].isin(irish_test["tune_id"])
    ]

    if "tune_type" in irish_meta_test.columns and "tune_type" in cross_df.columns:
        VALID_TYPES = ["reel", "jig", "hornpipe", "polka", "waltz", "march"]

        # Train on Irish
        irish_mask = irish_meta_test["tune_type"].isin(VALID_TYPES)
        if irish_mask.sum() > 0:
            X_train = irish_emb[irish_mask.values]
            y_train = irish_meta_test.loc[irish_mask, "tune_type"].values

            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            clf.fit(X_train, y_train)

            print(f"  Trained on Irish: {len(X_train)} samples")

            # Test on cross-tradition
            cross_mask = cross_df["tune_type"].isin(VALID_TYPES)
            if cross_mask.sum() > 0:
                X_test = cross_emb[cross_mask.values]
                y_test = cross_df.loc[cross_mask, "tune_type"].values

                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                print(f"  Test on cross-tradition: {len(X_test)} samples")
                print(f"\n  Zero-shot Transfer Accuracy: {acc:.4f}")

                all_metrics["zero_shot_transfer"] = {
                    "accuracy": float(acc),
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                }

                print(f"\n  Classification Report:")
                print(classification_report(y_test, y_pred, zero_division=0))
            else:
                print("  ⚠ No cross-tradition samples with valid tune types")
        else:
            print("  ⚠ No Irish samples with valid tune types")
    else:
        print("  ⚠ tune_type column not available")

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_path = output_dir / "cross_tradition_metrics.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✓ Metrics saved to: {results_path}")
    print(f"✓ All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
