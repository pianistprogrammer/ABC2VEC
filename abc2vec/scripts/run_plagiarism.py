#!/usr/bin/env python
"""
Melodic Similarity / Plagiarism Detection Script for ABC2Vec.

Applies ABC2Vec embeddings to detect melodic similarity and potential
melodic borrowing between folk tunes:
- Known melodically related tune pair testing
- Nearest-neighbor search for similar tunes
- Threshold calibration (what similarity indicates "same tune"?)
- Cross-corpus duplicate detection
- Similarity distributions (same vs different families)
- Precision-Recall curves for similarity detection
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

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


def test_known_pairs(
    model: ABC2VecModel, patchifier: BarPatchifier, device: str
) -> list:
    """Test known melodically related tune pairs."""
    KNOWN_PAIRS = [
        {
            "name": "The Kesh Jig (variant pair)",
            "tune_a": "G2G GAB|ded BAG|G2G GAB|dBA AGF|G2G GAB|ded BAG|aba aga|BGG G3:|",
            "tune_b": "G2G GAB|d2d BAG|G2G GAB|d2A AGF|G2G GAB|d2d BAG|aba aga|BGG G3:|",
            "expected": "similar",
        },
        {
            "name": "Unrelated: Jig vs. Reel",
            "tune_a": "G2G GAB|ded BAG|G2G GAB|dBA AGF|",
            "tune_b": "ABAF ABdf|afdf aBBA|ABAF ABdf|afdf aBBA|",
            "expected": "different",
        },
        {
            "name": "Transposition: Same tune in D vs G",
            "tune_a": "D2D DEF|ABA FED|D2D DEF|AFE EDC|",
            "tune_b": "G2G GAB|ded BAG|G2G GAB|dBA AGF|",
            "expected": "similar (transposed)",
        },
    ]

    results = []

    print("\n  Known Melodic Pair Similarities:")
    print("  " + "=" * 76)

    for pair in KNOWN_PAIRS:
        # Create temporary dataframes
        df_a = pd.DataFrame([{"abc_body": pair["tune_a"]}])
        df_b = pd.DataFrame([{"abc_body": pair["tune_b"]}])

        emb_a = compute_embeddings(model, df_a, patchifier, device, batch_size=1)
        emb_b = compute_embeddings(model, df_b, patchifier, device, batch_size=1)

        sim = cosine_similarity(emb_a, emb_b)[0, 0]

        print(f"\n  {pair['name']}:")
        print(f"    Expected: {pair['expected']}")
        print(f"    Cosine similarity: {sim:.4f}")

        results.append(
            {
                "name": pair["name"],
                "expected": pair["expected"],
                "similarity": float(sim),
            }
        )

    return results


def calibrate_threshold(
    embeddings: np.ndarray, family_ids: np.ndarray, n_pairs: int = 50000
) -> dict:
    """
    Calibrate similarity threshold using same/different family pairs.

    Returns optimal threshold and metrics.
    """
    print("\n  Sampling pairwise similarities...")

    # Normalize embeddings
    en = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)

    n = len(en)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    np.random.seed(42)

    pos_sims = []  # Same family
    neg_sims = []  # Different family

    # Sample pairs
    sampled = 0
    while sampled < n_pairs:
        i, j = np.random.choice(n, 2, replace=False)
        sim = float(np.dot(en[i], en[j]))

        if family_ids[i] == family_ids[j]:
            pos_sims.append(sim)
        else:
            neg_sims.append(sim)

        sampled += 1

    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)

    print(f"    Same-family pairs: {len(pos_sims)}")
    print(f"      Mean sim: {pos_sims.mean():.4f}, Std: {pos_sims.std():.4f}")
    print(f"    Different-family pairs: {len(neg_sims)}")
    print(f"      Mean sim: {neg_sims.mean():.4f}, Std: {neg_sims.std():.4f}")

    # Find optimal threshold
    all_sims = np.concatenate([pos_sims, neg_sims])
    all_labels = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])

    precision, recall, thresholds = precision_recall_curve(all_labels, all_sims)
    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])

    ap = average_precision_score(all_labels, all_sims)

    print(f"\n    Optimal 'same-tune' threshold: {best_threshold:.4f}")
    print(f"      Precision: {precision[best_idx]:.4f}")
    print(f"      Recall: {recall[best_idx]:.4f}")
    print(f"      F1: {f1_scores[best_idx]:.4f}")
    print(f"      AP: {ap:.4f}")

    return {
        "pos_sims": pos_sims,
        "neg_sims": neg_sims,
        "best_threshold": best_threshold,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1_scores[best_idx]),
        "ap": float(ap),
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def plot_similarity_distributions(
    pos_sims: np.ndarray,
    neg_sims: np.ndarray,
    threshold: float,
    output_path: Path,
):
    """Plot similarity distributions for same vs different families."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        neg_sims, bins=100, alpha=0.6, label="Different tunes", color="blue", density=True
    )
    ax.hist(
        pos_sims, bins=100, alpha=0.6, label="Same tune family", color="red", density=True
    )
    ax.axvline(
        x=threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Optimal threshold: {threshold:.3f}",
    )
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Similarity Distribution: Same vs Different Tune Families", fontsize=14, pad=20)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curve(
    precision: list, recall: list, ap: float, best_idx: int, threshold: float, output_path: Path
):
    """Plot precision-recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="steelblue", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.2, color="steelblue")
    ax.scatter(
        [recall[best_idx]],
        [precision[best_idx]],
        color="red",
        s=100,
        zorder=5,
        label=f"Best F1 (t={threshold:.3f})",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Melodic Similarity Detection - PR Curve (AP={ap:.4f})", fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def find_near_duplicates(
    embeddings: np.ndarray, df: pd.DataFrame, threshold: float, output_path: Path
) -> list:
    """Find near-duplicates within corpus."""
    print("\n  Finding intra-corpus near-duplicates...")

    # Normalize
    en = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)

    # Build index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(en.astype(np.float32))

    # Search for top-2 (first will be self)
    distances, indices = index.search(en.astype(np.float32), 2)

    # Nearest neighbor (excluding self)
    nn_sims = distances[:, 1]
    nn_idx = indices[:, 1]

    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(nn_sims, bins=100, color="steelblue", edgecolor="white")
    ax.axvline(
        x=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {threshold:.3f}",
    )
    ax.set_xlabel("Cosine Similarity to Nearest Neighbor", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Nearest-Neighbor Similarity Distribution", fontsize=14, pad=20)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # List suspicious near-duplicates
    suspicious_mask = nn_sims >= threshold
    n_suspicious = suspicious_mask.sum()

    print(f"    Near-duplicates above threshold: {n_suspicious} / {len(df)}")

    # Get top 20 most similar pairs
    top_pairs = []
    top_indices = np.argsort(nn_sims)[::-1][:20]

    for idx in top_indices:
        nn = nn_idx[idx]
        sim = nn_sims[idx]

        title_a = df.iloc[idx].get("title", df.iloc[idx].get("tune_name", "N/A"))
        title_b = df.iloc[nn].get("title", df.iloc[nn].get("tune_name", "N/A"))

        top_pairs.append(
            {
                "idx_a": int(idx),
                "idx_b": int(nn),
                "similarity": float(sim),
                "title_a": str(title_a),
                "title_b": str(title_b),
            }
        )

    print(f"\n    Top 10 most similar pairs:")
    for pair in top_pairs[:10]:
        print(f"      sim={pair['similarity']:.4f}: '{pair['title_a']}' ↔ '{pair['title_b']}'")

    return top_pairs


def main():
    """Main plagiarism detection function."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Melodic Similarity / Plagiarism Detection"
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
        "--benchmark_dir",
        type=str,
        default="./data/benchmark",
        help="Directory with benchmark data (for threshold calibration)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/plagiarism",
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
        "--n_pairs",
        type=int,
        default=50000,
        help="Number of pairs to sample for threshold calibration",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Melodic Similarity / Plagiarism Detection")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Benchmark dir:  {args.benchmark_dir}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Device:         {args.device}")
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

    # ========================================================================
    # 1. Test Known Melodic Pairs
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. KNOWN MELODIC PAIR TESTING")
    print("=" * 80)

    known_pair_results = test_known_pairs(model, patchifier, args.device)

    # Save known pairs
    with open(output_dir / "known_pairs.json", "w") as f:
        json.dump(known_pair_results, f, indent=2)

    # ========================================================================
    # 2. Load Benchmark for Threshold Calibration
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. THRESHOLD CALIBRATION")
    print("=" * 80)

    benchmark_dir = Path(args.benchmark_dir)
    full_benchmark_path = benchmark_dir / "retrieval_full.parquet"

    if full_benchmark_path.exists():
        print(f"\n  Loading benchmark from {full_benchmark_path}")
        benchmark_df = pd.read_parquet(full_benchmark_path)
        family_col = "family_id"
    else:
        print(f"\n  Benchmark not found, using test set...")
        benchmark_df = load_processed_data(args.data_dir, split="test")
        metadata_df = pd.read_csv(Path(args.data_dir) / "metadata.csv")
        test_metadata = metadata_df[metadata_df["tune_id"].isin(benchmark_df["tune_id"])]
        benchmark_df = benchmark_df.merge(
            test_metadata[["tune_id", "tune_type"]], on="tune_id", how="left"
        )
        family_col = "tune_type"

    print(f"  Benchmark size: {len(benchmark_df):,} tunes")

    # Compute embeddings
    print("\n  Computing embeddings...")
    benchmark_emb = compute_embeddings(
        model, benchmark_df, patchifier, args.device, args.batch_size
    )

    # Calibrate threshold
    print("\n  Calibrating threshold...")
    calibration = calibrate_threshold(
        benchmark_emb, benchmark_df[family_col].values, args.n_pairs
    )

    threshold = calibration["best_threshold"]

    # ========================================================================
    # 3. Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Similarity distributions
    print("\n  Plotting similarity distributions...")
    plot_similarity_distributions(
        calibration["pos_sims"],
        calibration["neg_sims"],
        threshold,
        output_dir / "similarity_distributions.png",
    )
    print(f"    ✓ Saved to {output_dir / 'similarity_distributions.png'}")

    # PR curve
    print("\n  Plotting precision-recall curve...")
    best_idx = np.argmax(
        2
        * np.array(calibration["precision_curve"])
        * np.array(calibration["recall_curve"])
        / np.maximum(
            np.array(calibration["precision_curve"])
            + np.array(calibration["recall_curve"]),
            1e-8,
        )
    )
    plot_pr_curve(
        calibration["precision_curve"],
        calibration["recall_curve"],
        calibration["ap"],
        best_idx,
        threshold,
        output_dir / "pr_curve.png",
    )
    print(f"    ✓ Saved to {output_dir / 'pr_curve.png'}")

    # ========================================================================
    # 4. Intra-Corpus Near-Duplicate Detection
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. INTRA-CORPUS NEAR-DUPLICATE DETECTION")
    print("=" * 80)

    near_duplicates = find_near_duplicates(
        benchmark_emb, benchmark_df, threshold, output_dir / "nn_distribution.png"
    )
    print(f"    ✓ Saved to {output_dir / 'nn_distribution.png'}")

    # Save near-duplicates
    with open(output_dir / "near_duplicates.json", "w") as f:
        json.dump(near_duplicates, f, indent=2)

    # ========================================================================
    # Save All Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    summary = {
        "known_pairs": known_pair_results,
        "threshold_calibration": {
            "best_threshold": threshold,
            "precision": calibration["precision"],
            "recall": calibration["recall"],
            "f1": calibration["f1"],
            "ap": calibration["ap"],
            "same_family_mean": float(calibration["pos_sims"].mean()),
            "same_family_std": float(calibration["pos_sims"].std()),
            "diff_family_mean": float(calibration["neg_sims"].mean()),
            "diff_family_std": float(calibration["neg_sims"].std()),
        },
        "near_duplicates": {
            "n_found": len([p for p in near_duplicates if p["similarity"] >= threshold]),
            "top_pairs": near_duplicates[:10],
        },
    }

    results_path = output_dir / "plagiarism_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {results_path}")
    print(f"✓ Near-duplicates saved to: {output_dir / 'near_duplicates.json'}")
    print(f"✓ All plots saved to: {output_dir}")

    print("\n" + "=" * 80)
    print("APPLICATIONS")
    print("=" * 80)
    print("  1. Detect tune variants across collections")
    print("  2. Find cross-tradition borrowing")
    print("  3. Dataset deduplication")
    print("  4. Musicological analysis of tune relationships")
    print("=" * 80)


if __name__ == "__main__":
    main()
