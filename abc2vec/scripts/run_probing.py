#!/usr/bin/env python
"""
Probing Experiments Script for ABC2Vec.

Tests what musical properties are linearly decodable from embeddings:
- Mode (Dorian, Mixolydian, Major, Minor)
- Tune type/rhythm (Jig, Reel, Hornpipe, Polka)
- Key root (D, G, A, C, etc.)
- Time signature (4/4, 6/8, 3/4, etc.)
- Tune length (short, medium, long)
- Dimensionality analysis (which dimensions encode what)
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

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


def run_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    property_name: str,
    min_class_size: int = 30,
    n_folds: int = 5,
) -> dict:
    """
    Run a linear probing experiment.

    Returns dict with accuracy, per-class metrics, and feature importances.
    """
    # Filter NaN and empty
    valid_mask = pd.notna(labels) & (labels != "")
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]

    if len(labels) == 0:
        return None

    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_

    # Filter rare classes
    class_counts = np.bincount(y)
    valid_classes = np.where(class_counts >= min_class_size)[0]
    mask = np.isin(y, valid_classes)
    X = embeddings[mask]
    y = y[mask]

    if len(X) == 0:
        return None

    # Re-encode after filtering
    le2 = LabelEncoder()
    y = le2.fit_transform(y)
    classes = [classes[c] for c in valid_classes]

    print(f"\n  Probe: {property_name}")
    print(f"    Samples: {len(X)}, Classes: {len(classes)}")
    print(f"    Classes: {classes}")

    # Cross-validated logistic regression
    clf = LogisticRegression(
        max_iter=3000, C=1.0, solver="lbfgs", class_weight="balanced"
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")

    # Chance level
    chance = 1.0 / len(classes)

    print(f"    Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"    F1 (weighted): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"    Chance level: {chance:.4f}")
    print(f"    Above chance: +{scores.mean() - chance:.4f}")

    # Fit full model for feature importance
    clf.fit(X, y)

    # Feature importance: weight magnitudes across classes
    weights = np.abs(clf.coef_)  # (n_classes, d_embed)
    dim_importance = weights.mean(axis=0)  # Average across classes

    return {
        "property": property_name,
        "n_samples": int(len(X)),
        "n_classes": int(len(classes)),
        "classes": [str(c) for c in classes],
        "accuracy": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "f1": float(f1_scores.mean()),
        "f1_std": float(f1_scores.std()),
        "chance": float(chance),
        "dim_importance": dim_importance.tolist(),
    }


def extract_time_signature(abc_text: str) -> str:
    """Extract time signature from ABC notation."""
    m = re.search(r"M:\s*(\d+/\d+)", abc_text)
    if m:
        return m.group(1)
    return None


def plot_probe_summary(probes: list, output_path: Path):
    """Plot summary bar chart of all probing results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    props = [p["property"] for p in probes]
    accs = [p["accuracy"] for p in probes]
    chances = [p["chance"] for p in probes]

    x = np.arange(len(props))
    width = 0.35
    ax.bar(x - width / 2, accs, width, label="Probe Accuracy", color="steelblue")
    ax.bar(x + width / 2, chances, width, label="Chance Level", color="lightgray")
    ax.set_xticks(x)
    ax.set_xticklabels(props, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear Probing Accuracy vs. Chance")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dimension_importance(probes: list, output_path: Path):
    """Plot dimension importance for each property."""
    n_probes = len(probes)
    fig, axes = plt.subplots(n_probes, 1, figsize=(16, 4 * n_probes))
    if n_probes == 1:
        axes = [axes]

    for i, probe in enumerate(probes):
        importance = np.array(probe["dim_importance"])
        top_dims = np.argsort(importance)[::-1][:20]

        axes[i].bar(range(len(importance)), importance, color="steelblue", alpha=0.7)
        axes[i].set_xlabel("Embedding Dimension")
        axes[i].set_ylabel("Importance")
        axes[i].set_title(f'{probe["property"]} - Dimension Importance')

        # Highlight top 5 dimensions
        for d in top_dims[:5]:
            axes[i].bar(d, importance[d], color="red", alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pca_analysis(embeddings: np.ndarray, output_path: Path):
    """Analyze PCA of embedding space."""
    print("  Computing PCA...")
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    pca.fit(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Explained variance
    axes[0].plot(
        np.cumsum(pca.explained_variance_ratio_), "o-", color="steelblue"
    )
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("Cumulative Explained Variance")
    axes[0].set_title("PCA - Explained Variance")
    axes[0].axhline(y=0.9, color="red", linestyle="--", label="90%")
    axes[0].axhline(y=0.95, color="orange", linestyle="--", label="95%")
    axes[0].legend()

    # Individual component variance
    axes[1].bar(
        range(len(pca.explained_variance_ratio_)),
        pca.explained_variance_ratio_,
        color="steelblue",
    )
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained Variance Ratio")
    axes[1].set_title("PCA - Per-Component Variance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Find components for 90% / 95%
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.argmax(cumvar >= 0.9) + 1)
    n_95 = int(np.argmax(cumvar >= 0.95) + 1)

    return {"n_components_90": n_90, "n_components_95": n_95}


def main():
    """Main probing experiments function."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Probing Experiments - What do embeddings encode?"
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
        default="./results/probing",
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

    args = parser.parse_args()

    print("=" * 80)
    print("ABC2Vec Probing Experiments")
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
    test_metadata = metadata_df[metadata_df["tune_id"].isin(test_df["tune_id"])].copy()

    print(f"  {len(test_df):,} tunes in test set")

    # Create patchifier
    patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)

    # Compute embeddings
    print("\nComputing embeddings...")
    embeddings = compute_embeddings(
        model, test_df, patchifier, args.device, args.batch_size
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save embeddings
    np.save(output_dir / "test_embeddings.npy", embeddings)

    # ========================================================================
    # Run All Probes
    # ========================================================================
    print("\n" + "=" * 80)
    print("RUNNING PROBING EXPERIMENTS")
    print("=" * 80)

    all_probes = []

    # Probe 1: Mode
    print("\n[1/5] Mode Probe")
    if "mode" in test_metadata.columns:
        probe = run_probe(embeddings, test_metadata["mode"].values, "Mode")
        if probe:
            all_probes.append(probe)
    else:
        print("  ⚠ No mode column")

    # Probe 2: Tune Type
    print("\n[2/5] Tune Type Probe")
    if "tune_type" in test_metadata.columns:
        probe = run_probe(embeddings, test_metadata["tune_type"].values, "Tune Type")
        if probe:
            all_probes.append(probe)
    else:
        print("  ⚠ No tune_type column")

    # Probe 3: Key Root
    print("\n[3/5] Key Root Probe")
    if "key" in test_metadata.columns:
        test_metadata["key_root"] = (
            test_metadata["key"].astype(str).str[0].str.upper()
        )
        probe = run_probe(embeddings, test_metadata["key_root"].values, "Key Root")
        if probe:
            all_probes.append(probe)
    else:
        print("  ⚠ No key column")

    # Probe 4: Time Signature
    print("\n[4/5] Time Signature Probe")
    test_df["time_sig"] = test_df["abc_body"].apply(extract_time_signature)
    probe = run_probe(embeddings, test_df["time_sig"].values, "Time Signature")
    if probe:
        all_probes.append(probe)

    # Probe 5: Tune Length
    print("\n[5/5] Tune Length Probe")
    lengths = test_df["abc_body"].str.len()
    q33, q66 = lengths.quantile(0.33), lengths.quantile(0.66)

    def length_bin(l):
        if l < q33:
            return "short"
        elif l < q66:
            return "medium"
        else:
            return "long"

    length_labels = lengths.apply(length_bin).values
    probe = run_probe(embeddings, length_labels, "Tune Length")
    if probe:
        all_probes.append(probe)

    # ========================================================================
    # Summary Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("PROBING SUMMARY")
    print("=" * 80)

    summary_data = []
    for p in all_probes:
        summary_data.append(
            {
                "Property": p["property"],
                "N Classes": p["n_classes"],
                "Accuracy": f"{p['accuracy']:.4f} ± {p['accuracy_std']:.4f}",
                "F1": f"{p['f1']:.4f} ± {p['f1_std']:.4f}",
                "Chance": f"{p['chance']:.4f}",
                "Delta": f"+{p['accuracy'] - p['chance']:.4f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(output_dir / "probing_summary.csv", index=False)

    # Save detailed results
    probes_clean = []
    for p in all_probes:
        p_copy = p.copy()
        p_copy["dim_importance"] = None  # Don't save large arrays to JSON
        probes_clean.append(p_copy)

    with open(output_dir / "probing_metrics.json", "w") as f:
        json.dump(probes_clean, f, indent=2)

    # ========================================================================
    # Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Plot summary
    print("\n  Creating probe summary chart...")
    plot_probe_summary(all_probes, output_dir / "probing_summary.png")
    print(f"    ✓ Saved to {output_dir / 'probing_summary.png'}")

    # Plot dimension importance
    print("\n  Creating dimension importance plots...")
    plot_dimension_importance(all_probes, output_dir / "dimension_importance.png")
    print(f"    ✓ Saved to {output_dir / 'dimension_importance.png'}")

    # Dimension overlap analysis
    if len(all_probes) >= 2:
        print("\n  Analyzing dimension overlap...")
        top_10s = {}
        for probe in all_probes:
            importance = np.array(probe["dim_importance"])
            top_10 = set(np.argsort(importance)[::-1][:10].tolist())
            top_10s[probe["property"]] = top_10

        print("\n  Top-10 dimensions per property:")
        for prop, dims in top_10s.items():
            print(f"    {prop}: {sorted(dims)}")

        print("\n  Overlap between properties:")
        props = list(top_10s.keys())
        overlaps = []
        for i in range(len(props)):
            for j in range(i + 1, len(props)):
                overlap = top_10s[props[i]] & top_10s[props[j]]
                overlaps.append(
                    {
                        "property_a": props[i],
                        "property_b": props[j],
                        "overlap_size": len(overlap),
                        "overlap_dims": sorted(overlap),
                    }
                )
                print(
                    f"    {props[i]} ∩ {props[j]}: {len(overlap)} dims {sorted(overlap)}"
                )

        with open(output_dir / "dimension_overlap.json", "w") as f:
            json.dump(overlaps, f, indent=2)

    # PCA analysis
    print("\n  Running PCA analysis...")
    pca_results = plot_pca_analysis(embeddings, output_dir / "pca_analysis.png")
    print(f"    ✓ Saved to {output_dir / 'pca_analysis.png'}")
    print(f"    Components for 90% variance: {pca_results['n_components_90']}")
    print(f"    Components for 95% variance: {pca_results['n_components_95']}")

    # ========================================================================
    # Save All Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    final_results = {
        "probing_experiments": probes_clean,
        "pca_analysis": pca_results,
    }

    results_path = output_dir / "probing_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✓ Metrics saved to: {results_path}")
    print(f"✓ Summary saved to: {output_dir / 'probing_summary.csv'}")
    print(f"✓ All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
