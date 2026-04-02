#!/usr/bin/env python
"""
Ablation Study Script for ABC2Vec.

Measures the contribution of each pre-training objective by comparing models
trained with different objective combinations.

Ablation configurations:
- Full (MMM+TI) - All objectives
- MMM only - Only masked modeling
- TI only - Only transposition invariance
- Random - Untrained baseline

Each is evaluated on retrieval (MRR, Recall@5) and classification (accuracy).
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

try:
    import faiss
except ImportError:
    print("Installing faiss-cpu...")
    import subprocess
    subprocess.check_call(["pip", "install", "faiss-cpu"])
    import faiss

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
        for i in tqdm(range(0, len(df), batch_size), desc="Encoding", leave=False):
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


def evaluate_retrieval(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    query_labels: np.ndarray,
    corpus_labels: np.ndarray,
    k: int = 5,
) -> dict:
    """Quick retrieval evaluation: MRR and Recall@K."""
    d = query_emb.shape[1]

    # Normalize
    qn = query_emb / np.maximum(np.linalg.norm(query_emb, axis=1, keepdims=True), 1e-8)
    cn = corpus_emb / np.maximum(np.linalg.norm(corpus_emb, axis=1, keepdims=True), 1e-8)

    index = faiss.IndexFlatIP(d)
    index.add(cn.astype(np.float32))
    _, indices = index.search(qn.astype(np.float32), k)

    mrr, recall_at_k = 0.0, 0.0
    n_valid = 0

    for i in range(len(query_labels)):
        retrieved = corpus_labels[indices[i]]
        is_rel = retrieved == query_labels[i]
        n_relevant = np.sum(corpus_labels == query_labels[i])

        if n_relevant == 0:
            continue

        n_valid += 1
        positions = np.where(is_rel)[0]

        if len(positions) > 0:
            mrr += 1.0 / (positions[0] + 1)

        recall_at_k += np.sum(is_rel) / min(n_relevant, k)

    return {
        "MRR": mrr / max(n_valid, 1),
        f"Recall@{k}": recall_at_k / max(n_valid, 1),
    }


def evaluate_classification(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Quick classification with linear probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    # Filter valid labels
    valid_mask = pd.notna(labels) & (labels != "")
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]

    if len(labels) < 10:
        return {"Accuracy": 0.0}

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    scores = cross_val_score(clf, embeddings, y, cv=5, scoring="accuracy")

    return {"Accuracy": scores.mean()}


def load_ablation_checkpoints(checkpoint_dir: Path) -> dict:
    """Load all ablation checkpoints if they exist."""
    models = {}

    checkpoint_patterns = {
        "Full (MMM+TI)": "best_model.pt",
        "MMM only": "ablation_mmm_only.pt",
        "TI only": "ablation_ti_only.pt",
    }

    for name, pattern in checkpoint_patterns.items():
        ckpt_path = checkpoint_dir / pattern
        if ckpt_path.exists():
            models[name] = ckpt_path
            print(f"  ✓ Found {name}: {ckpt_path.name}")

    return models


def plot_ablation_results(results_df: pd.DataFrame, output_path: Path):
    """Plot ablation study results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    configs = results_df["Config"].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))

    # MRR
    axes[0].barh(configs, results_df["MRR"], color=colors)
    axes[0].set_xlabel("MRR")
    axes[0].set_title("Retrieval: MRR")
    axes[0].set_xlim(0, 1)

    # Recall@5
    axes[1].barh(configs, results_df["Recall@5"], color=colors)
    axes[1].set_xlabel("Recall@5")
    axes[1].set_title("Retrieval: Recall@5")
    axes[1].set_xlim(0, 1)

    # Classification accuracy
    axes[2].barh(configs, results_df["Accuracy"], color=colors)
    axes[2].set_xlabel("Accuracy")
    axes[2].set_title("Tune Type Classification")
    axes[2].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Main ablation study function."""
    parser = argparse.ArgumentParser(
        description="ABC2Vec Ablation Study - Measure objective contributions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to FULL model checkpoint (trained with all objectives)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing ablation checkpoints (optional)",
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
        help="Directory with benchmark data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/ablation",
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
    print("ABC2Vec Ablation Study")
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

    # Load vocabulary
    print("Loading vocabulary...")
    vocab_path = Path(args.data_dir) / "vocab.json"
    vocab = ABCVocabulary.load(vocab_path)
    patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)

    # Load benchmark data
    print("\nLoading benchmark data...")
    benchmark_dir = Path(args.benchmark_dir)

    queries_path = benchmark_dir / "retrieval_queries.parquet"
    corpus_path = benchmark_dir / "retrieval_corpus.parquet"

    if not queries_path.exists() or not corpus_path.exists():
        print("\n⚠ Benchmark data not found. Expected:")
        print(f"  {queries_path}")
        print(f"  {corpus_path}")
        print("\nUsing test set instead...")

        from core.data import load_processed_data

        test_df = load_processed_data(args.data_dir, split="test")
        queries_df = test_df.sample(min(500, len(test_df)), random_state=42)
        corpus_df = test_df

        metadata_df = pd.read_csv(Path(args.data_dir) / "metadata.csv")
        test_metadata = metadata_df[metadata_df["tune_id"].isin(test_df["tune_id"])]

        queries_df = queries_df.merge(
            test_metadata[["tune_id", "tune_type"]], on="tune_id", how="left"
        )
        corpus_df = corpus_df.merge(
            test_metadata[["tune_id", "tune_type"]], on="tune_id", how="left"
        )

        # Use tune_type as family_id
        queries_df["family_id"] = queries_df["tune_type"]
        corpus_df["family_id"] = corpus_df["tune_type"]

    else:
        queries_df = pd.read_parquet(queries_path)
        corpus_df = pd.read_parquet(corpus_path)

    print(f"  Queries: {len(queries_df):,}")
    print(f"  Corpus: {len(corpus_df):,}")

    # Load models
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    models_to_eval = {}

    # Full model
    print("\nLoading full model...")
    full_model = ABC2VecModel.load_pretrained(args.checkpoint, device=args.device)
    models_to_eval["Full (MMM+TI)"] = full_model
    print(f"  ✓ Full model loaded")

    # Check for ablation checkpoints
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        ablation_models = load_ablation_checkpoints(checkpoint_dir)
        for name, ckpt_path in ablation_models.items():
            if name not in models_to_eval:
                model = ABC2VecModel.load_pretrained(str(ckpt_path), device=args.device)
                models_to_eval[name] = model

    # Random baseline
    print("\n  Creating random baseline...")
    full_config = full_model.config
    random_model = ABC2VecModel(full_config).to(args.device)
    random_model.eval()
    models_to_eval["Random (untrained)"] = random_model
    print(f"  ✓ Random baseline created")

    print(f"\nTotal models to evaluate: {len(models_to_eval)}")

    # ========================================================================
    # Evaluate All Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING ALL MODELS")
    print("=" * 80)

    ablation_results = []

    for config_name, model in models_to_eval.items():
        print(f"\n  Evaluating: {config_name}")

        # Encode queries and corpus
        q_emb = compute_embeddings(
            model, queries_df, patchifier, args.device, args.batch_size
        )
        c_emb = compute_embeddings(
            model, corpus_df, patchifier, args.device, args.batch_size
        )

        # Retrieval evaluation
        ret_metrics = evaluate_retrieval(
            q_emb,
            c_emb,
            queries_df["family_id"].values,
            corpus_df["family_id"].values,
            k=5,
        )

        # Classification evaluation
        cls_metrics = evaluate_classification(
            c_emb, corpus_df.get("tune_type", corpus_df.get("family_id")).values
        )

        row = {"Config": config_name}
        row.update(ret_metrics)
        row.update(cls_metrics)
        ablation_results.append(row)

        print(
            f"    MRR={ret_metrics['MRR']:.4f}, "
            f"R@5={ret_metrics['Recall@5']:.4f}, "
            f"Acc={cls_metrics['Accuracy']:.4f}"
        )

    # ========================================================================
    # Results Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df = ablation_df.sort_values("MRR", ascending=False)

    print("\n" + ablation_df.to_string(index=False, float_format="%.4f"))

    # Save results
    ablation_df.to_csv(output_dir / "ablation_results.csv", index=False)

    # Contribution analysis
    full_row = ablation_df[ablation_df["Config"].str.contains("Full")].iloc[0]

    print("\n" + "=" * 80)
    print("MARGINAL CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print("(Measured as performance drop when objective is removed)")

    contribution_data = []
    for metric in ["MRR", "Recall@5", "Accuracy"]:
        print(f"\n{metric}:")
        full_val = full_row[metric]

        for _, row in ablation_df.iterrows():
            if row["Config"] == full_row["Config"]:
                continue

            delta = full_val - row[metric]
            direction = "↓" if delta > 0 else "↑"
            print(
                f"  {row['Config']:25s}: {row[metric]:.4f}  "
                f"({direction}{abs(delta):.4f})"
            )

            contribution_data.append(
                {
                    "Config": row["Config"],
                    "Metric": metric,
                    "Value": row[metric],
                    "Delta": delta,
                }
            )

    # Save contribution analysis
    contrib_df = pd.DataFrame(contribution_data)
    contrib_df.to_csv(output_dir / "contribution_analysis.csv", index=False)

    # ========================================================================
    # Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    print("\n  Creating ablation results chart...")
    plot_ablation_results(ablation_df, output_dir / "ablation_results.png")
    print(f"    ✓ Saved to {output_dir / 'ablation_results.png'}")

    # Contribution heatmap
    print("\n  Creating contribution heatmap...")
    pivot_contrib = contrib_df.pivot(
        index="Config", columns="Metric", values="Delta"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot_contrib,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Performance Drop"},
    )
    ax.set_title("Marginal Contribution of Each Objective", fontsize=14, pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_dir / "contribution_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved to {output_dir / 'contribution_heatmap.png'}")

    # ========================================================================
    # Save All Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    summary = {
        "models_evaluated": list(models_to_eval.keys()),
        "results": ablation_df.to_dict(orient="records"),
        "contributions": contrib_df.to_dict(orient="records"),
    }

    results_path = output_dir / "ablation_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {results_path}")
    print(f"✓ Results table saved to: {output_dir / 'ablation_results.csv'}")
    print(f"✓ All plots saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("KEY TAKEAWAY:")
    print("The objective with the largest ↓ when removed contributes the most.")
    print("=" * 80)


if __name__ == "__main__":
    main()
