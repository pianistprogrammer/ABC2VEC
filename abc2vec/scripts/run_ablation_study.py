#!/usr/bin/env python
"""
Ablation Study Evaluation Runner.

Evaluates all ablation model variants on multiple downstream tasks:
- Tune type classification
- Mode classification
- Linear probing (multiple properties)
- Clustering quality

Generates comparison tables, visualizations, and LaTeX output for the paper.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, silhouette_score,
    normalized_mutual_info_score, adjusted_rand_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data import load_processed_data
from core.model import ABC2VecModel
from core.model.encoder import ABC2VecConfig
from core.tokenizer import ABCVocabulary, BarPatchifier


# ============================================================================
# Model Loading and Embedding Computation
# ============================================================================

def load_ablation_model(checkpoint_dir: Path, device: str):
    """Load ablation model from checkpoint directory."""
    config_path = checkpoint_dir / "model_config.json"
    model_path = checkpoint_dir / "best_model.pt"

    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing config or model in {checkpoint_dir}")

    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = ABC2VecConfig(**config_dict)

    # Create model
    model = ABC2VecModel(config)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


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
        for i in tqdm(range(0, len(df), batch_size), desc="Computing embeddings", leave=False):
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


# ============================================================================
# Evaluation Tasks
# ============================================================================

def evaluate_classification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    classifier_type: str = "logistic",
) -> dict:
    """
    Evaluate classification with stratified k-fold cross-validation.

    Args:
        embeddings: Embedding matrix (num_tunes, d_embed)
        labels: Classification labels
        n_splits: Number of folds for cross-validation
        random_state: Random seed
        classifier_type: "logistic" or "ridge"

    Returns:
        Dictionary with metrics
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accuracies = []
    fold_f1_scores = []

    for train_idx, val_idx in skf.split(embeddings, labels):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Train classifier
        if classifier_type == "logistic":
            clf = LogisticRegression(
                C=1.0 / 0.01,  # C = 1/lambda, lambda=0.01
                max_iter=1000,
                random_state=random_state,
            )
        else:  # ridge
            clf = RidgeClassifier(alpha=0.01, random_state=random_state)

        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_val)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')

        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)

    # Compute overall metrics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)

    # Get class distribution for chance baseline
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    chance_baseline = 1.0 / n_classes

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
    }


def evaluate_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int = None,
    random_state: int = 42,
) -> dict:
    """
    Evaluate clustering quality using K-means.

    Args:
        embeddings: Embedding matrix (num_tunes, d_embed)
        labels: Ground truth labels for comparison
        n_clusters: Number of clusters (if None, use number of unique labels)
        random_state: Random seed

    Returns:
        Dictionary with clustering metrics
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels))

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_assignments = kmeans.fit_predict(embeddings)

    # Compute metrics
    silhouette = silhouette_score(embeddings, cluster_assignments)
    nmi = normalized_mutual_info_score(labels_encoded, cluster_assignments)
    ari = adjusted_rand_score(labels_encoded, cluster_assignments)

    return {
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "nmi": nmi,
        "ari": ari,
        "n_samples": len(labels),
    }


def evaluate_tune_type(model, patchifier, df, device, n_splits=5):
    """Evaluate tune type classification."""
    print("  Evaluating tune type classification...")

    # Filter tunes with valid tune_type
    df_filtered = df[df['tune_type'].str.strip() != ''].copy()

    if len(df_filtered) == 0:
        return {"error": "No valid tune types found"}

    embeddings = compute_embeddings(model, df_filtered, patchifier, device)
    labels = df_filtered['tune_type'].values

    return evaluate_classification(embeddings, labels, n_splits=n_splits)


def evaluate_mode(model, patchifier, df, device, n_splits=5):
    """Evaluate mode classification."""
    print("  Evaluating mode classification...")

    # Filter tunes with valid mode
    df_filtered = df[df['mode'].str.strip() != ''].copy()

    if len(df_filtered) == 0:
        return {"error": "No valid modes found"}

    embeddings = compute_embeddings(model, df_filtered, patchifier, device)
    labels = df_filtered['mode'].values

    return evaluate_classification(embeddings, labels, n_splits=n_splits)


def evaluate_linear_probing(model, patchifier, df, device, n_splits=5):
    """Evaluate linear probing on multiple properties."""
    print("  Evaluating linear probing on multiple properties...")

    results = {}

    # Compute embeddings once
    embeddings = compute_embeddings(model, df, patchifier, device)

    # 1. Tune type
    df_tune_type = df[df['tune_type'].str.strip() != ''].copy()
    if len(df_tune_type) > 0:
        tune_type_embeddings = embeddings[df['tune_type'].str.strip() != '']
        tune_type_labels = df_tune_type['tune_type'].values
        results['tune_type'] = evaluate_classification(
            tune_type_embeddings, tune_type_labels, n_splits=n_splits, classifier_type="ridge"
        )

    # 2. Mode
    df_mode = df[df['mode'].str.strip() != ''].copy()
    if len(df_mode) > 0:
        mode_embeddings = embeddings[df['mode'].str.strip() != '']
        mode_labels = df_mode['mode'].values
        results['mode'] = evaluate_classification(
            mode_embeddings, mode_labels, n_splits=n_splits, classifier_type="ridge"
        )

    # 3. Key root (if available)
    if 'root_note' in df.columns:
        df_root = df[df['root_note'].str.strip() != ''].copy()
        if len(df_root) > 0:
            root_embeddings = embeddings[df['root_note'].str.strip() != '']
            root_labels = df_root['root_note'].values
            results['key_root'] = evaluate_classification(
                root_embeddings, root_labels, n_splits=n_splits, classifier_type="ridge"
            )

    # 4. Tune length (binned)
    if 'num_bars' in df.columns:
        # Bin into short/medium/long
        df['tune_length_bin'] = pd.cut(
            df['num_bars'],
            bins=[0, 16, 32, 128],
            labels=['short', 'medium', 'long']
        )
        df_length = df[df['tune_length_bin'].notna()].copy()
        if len(df_length) > 0:
            length_embeddings = embeddings[df['tune_length_bin'].notna()]
            length_labels = df_length['tune_length_bin'].astype(str).values
            results['tune_length'] = evaluate_classification(
                length_embeddings, length_labels, n_splits=n_splits, classifier_type="ridge"
            )

    return results


def evaluate_clustering_quality(model, patchifier, df, device):
    """Evaluate clustering quality on multiple properties."""
    print("  Evaluating clustering quality...")

    results = {}

    # Compute embeddings once
    embeddings = compute_embeddings(model, df, patchifier, device)

    # 1. Tune type
    df_tune_type = df[df['tune_type'].str.strip() != ''].copy()
    if len(df_tune_type) > 0:
        tune_type_embeddings = embeddings[df['tune_type'].str.strip() != '']
        tune_type_labels = df_tune_type['tune_type'].values
        results['tune_type'] = evaluate_clustering(tune_type_embeddings, tune_type_labels)

    # 2. Mode
    df_mode = df[df['mode'].str.strip() != ''].copy()
    if len(df_mode) > 0:
        mode_embeddings = embeddings[df['mode'].str.strip() != '']
        mode_labels = df_mode['mode'].values
        results['mode'] = evaluate_clustering(mode_embeddings, mode_labels)

    # 3. Key root (if available)
    if 'root_note' in df.columns:
        df_root = df[df['root_note'].str.strip() != ''].copy()
        # Only use top 8 most common keys
        top_keys = df_root['root_note'].value_counts().head(8).index
        df_root = df_root[df_root['root_note'].isin(top_keys)].copy()
        if len(df_root) > 0:
            root_mask = (df['root_note'].str.strip() != '') & (df['root_note'].isin(top_keys))
            root_embeddings = embeddings[root_mask]
            root_labels = df.loc[root_mask, 'root_note'].values
            results['key_root'] = evaluate_clustering(root_embeddings, root_labels)

    return results


# ============================================================================
# Aggregation and Export
# ============================================================================

def aggregate_results(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """Aggregate results into comparison DataFrame."""
    rows = []

    for model_name, model_results in all_results.items():
        row = {'model': model_name}

        # Tune type metrics
        if 'tune_type' in model_results and 'mean_accuracy' in model_results['tune_type']:
            row['tune_type_acc'] = model_results['tune_type']['mean_accuracy']
            row['tune_type_std'] = model_results['tune_type']['std_accuracy']
            row['tune_type_f1'] = model_results['tune_type']['mean_f1']

        # Mode metrics
        if 'mode' in model_results and 'mean_accuracy' in model_results['mode']:
            row['mode_acc'] = model_results['mode']['mean_accuracy']
            row['mode_std'] = model_results['mode']['std_accuracy']
            row['mode_f1'] = model_results['mode']['mean_f1']

        # Linear probing metrics
        if 'linear_probing' in model_results:
            lp = model_results['linear_probing']
            if 'tune_type' in lp:
                row['lp_tune_type_acc'] = lp['tune_type']['mean_accuracy']
            if 'mode' in lp:
                row['lp_mode_acc'] = lp['mode']['mean_accuracy']
            if 'key_root' in lp:
                row['lp_key_root_acc'] = lp['key_root']['mean_accuracy']
            if 'tune_length' in lp:
                row['lp_tune_length_acc'] = lp['tune_length']['mean_accuracy']

        # Clustering metrics
        if 'clustering' in model_results:
            clust = model_results['clustering']
            if 'tune_type' in clust:
                row['clust_tune_type_sil'] = clust['tune_type']['silhouette_score']
                row['clust_tune_type_nmi'] = clust['tune_type']['nmi']
            if 'mode' in clust:
                row['clust_mode_sil'] = clust['mode']['silhouette_score']
                row['clust_mode_nmi'] = clust['mode']['nmi']

        rows.append(row)

    return pd.DataFrame(rows)


def export_latex_table(df: pd.DataFrame, output_path: Path, caption: str, label: str):
    """Export DataFrame as LaTeX table."""
    # Format numeric columns to 3 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "---")

    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format='l' + 'c' * (len(df.columns) - 1),
    )

    with open(output_path, 'w') as f:
        f.write(latex)


def generate_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12

    # 1. Tune type accuracy comparison
    if 'tune_type_acc' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(df))
        ax.bar(x, df['tune_type_acc'], yerr=df.get('tune_type_std', None),
               capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Tune Type Classification - Ablation Comparison')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'tune_type_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Mode accuracy comparison
    if 'mode_acc' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(df))
        ax.bar(x, df['mode_acc'], yerr=df.get('mode_std', None),
               capsize=5, alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Mode Classification - Ablation Comparison')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'mode_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Multi-task comparison (linear probing)
    lp_cols = [c for c in df.columns if c.startswith('lp_')]
    if lp_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df))
        width = 0.2

        for i, col in enumerate(lp_cols):
            offset = (i - len(lp_cols)/2 + 0.5) * width
            ax.bar(x + offset, df[col], width, label=col.replace('lp_', '').replace('_', ' '), alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Linear Probing - Multi-Task Comparison')
        ax.set_ylim(0, 1.0)
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'linear_probing_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Clustering quality comparison
    clust_cols = [c for c in df.columns if 'clust' in c and 'sil' in c]
    if clust_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(df))
        ax.bar(x, df[clust_cols[0]], alpha=0.7, color='mediumseagreen')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Clustering Quality - Ablation Comparison')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'clustering_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation Study Evaluation Runner")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/ablation",
                       help="Directory containing all ablation checkpoints")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, default="./results/ablation_study",
                       help="Output directory for results")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Which split to evaluate on")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--n_splits", type=int, default=5,
                       help="Number of CV folds")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for embedding computation")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover ablation checkpoints
    checkpoints_dir = Path(args.checkpoints_dir)
    model_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()])

    if not model_dirs:
        raise ValueError(f"No model directories found in {checkpoints_dir}")

    print(f"\n{'='*80}")
    print("ABLATION STUDY EVALUATION")
    print(f"{'='*80}")
    print(f"\nFound {len(model_dirs)} ablation models:")
    for d in model_dirs:
        print(f"  - {d.name}")
    print(f"\nEvaluating on: {args.split} split")
    print(f"Device: {args.device}")
    print(f"CV folds: {args.n_splits}")

    # Load data once
    print(f"\nLoading data from {args.data_dir}...")
    df = load_processed_data(args.data_dir, args.split)
    print(f"  Loaded {len(df):,} tunes")

    # Create vocabulary and patchifier (same for all models)
    vocab = ABCVocabulary()
    patchifier = BarPatchifier(vocab, max_bars=64, max_bar_length=64)

    # Evaluate each model
    all_results = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")

        try:
            # Load model
            print("Loading model...")
            model, config = load_ablation_model(model_dir, args.device)

            # Run evaluations
            model_results = {}

            # 1. Tune type classification
            model_results['tune_type'] = evaluate_tune_type(
                model, patchifier, df, args.device, n_splits=args.n_splits
            )

            # 2. Mode classification
            model_results['mode'] = evaluate_mode(
                model, patchifier, df, args.device, n_splits=args.n_splits
            )

            # 3. Linear probing
            model_results['linear_probing'] = evaluate_linear_probing(
                model, patchifier, df, args.device, n_splits=args.n_splits
            )

            # 4. Clustering quality
            model_results['clustering'] = evaluate_clustering_quality(
                model, patchifier, df, args.device
            )

            # Save individual results
            results_path = model_dir / "ablation_results.json"
            with open(results_path, 'w') as f:
                json.dump(model_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            print(f"  Saved results to {results_path}")

            all_results[model_name] = model_results

        except Exception as e:
            print(f"  ERROR: Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}")

    results_df = aggregate_results(all_results)

    # Save aggregated results
    json_path = output_dir / "ablation_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved JSON: {json_path}")

    csv_path = output_dir / "ablation_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    export_latex_table(
        results_df[['model', 'tune_type_acc', 'tune_type_std', 'tune_type_f1']],
        output_dir / "ablation_table_tune_type.tex",
        caption="Ablation study results for tune type classification.",
        label="tab:ablation_tune_type"
    )

    export_latex_table(
        results_df[['model', 'mode_acc', 'mode_std', 'mode_f1']],
        output_dir / "ablation_table_mode.tex",
        caption="Ablation study results for mode classification.",
        label="tab:ablation_mode"
    )

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    generate_comparison_plots(results_df, output_dir)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nTune Type Classification:")
    if 'tune_type_acc' in results_df.columns:
        for _, row in results_df.iterrows():
            acc = row.get('tune_type_acc', 0)
            std = row.get('tune_type_std', 0)
            print(f"  {row['model']:20s}: {acc:.4f} ± {std:.4f}")

    print("\nMode Classification:")
    if 'mode_acc' in results_df.columns:
        for _, row in results_df.iterrows():
            acc = row.get('mode_acc', 0)
            std = row.get('mode_std', 0)
            print(f"  {row['model']:20s}: {acc:.4f} ± {std:.4f}")

    print(f"\n{'='*80}")
    print("✓ Ablation study evaluation complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - ablation_comparison.json")
    print(f"  - ablation_comparison.csv")
    print(f"  - ablation_table_*.tex")
    print(f"  - *_comparison.png")
    print()


if __name__ == "__main__":
    main()
