#!/usr/bin/env python3
"""
Unified Figure Generation Script for ABC2Vec Paper.

Generates all figures needed for the paper:
- Dataset statistics and distribution figures
- Model architecture diagram
- Training curves
- Confusion matrices
- UMAP/t-SNE embeddings
- Ablation study figures (loss curves, heatmaps, comparisons)

Usage:
    # Generate all figures
    python scripts/generate_figures.py --all

    # Generate specific figure types
    python scripts/generate_figures.py --dataset --architecture
    python scripts/generate_figures.py --training --confusion
    python scripts/generate_figures.py --ablation --embeddings

    # With custom paths
    python scripts/generate_figures.py --all \
        --checkpoints_dir ../checkpoints \
        --results_dir ../results \
        --output_dir ../Paper/figures
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name, args_list):
    """Run a generation script with arguments."""
    cmd = [sys.executable, f"scripts/{script_name}"] + args_list
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures for ABC2Vec paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate all figures:
    python scripts/generate_figures.py --all

  Generate specific figure types:
    python scripts/generate_figures.py --dataset --architecture
    python scripts/generate_figures.py --training --ablation
    python scripts/generate_figures.py --embeddings --confusion

  With custom paths:
    python scripts/generate_figures.py --all \\
        --data_dir data/processed \\
        --checkpoints_dir checkpoints \\
        --results_dir results \\
        --output_dir ../Paper/figures
        """
    )

    # Figure selection flags
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all figures (overrides individual flags)"
    )
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Generate dataset statistics and distribution figures"
    )
    parser.add_argument(
        "--architecture",
        action="store_true",
        help="Generate model architecture diagram"
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Generate training curves"
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Generate confusion matrices"
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Generate UMAP/t-SNE embedding visualizations"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Generate ablation study figures (loss curves, heatmaps, comparisons)"
    )

    # Path arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory with processed data"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="../checkpoints",
        help="Directory with model checkpoints"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Directory with evaluation results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../Paper/figures",
        help="Output directory for generated figures"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device for model inference (mps, cuda, cpu)"
    )

    args = parser.parse_args()

    # Convert paths
    data_dir = Path(args.data_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine what to generate
    generate_all = args.all
    if not any([args.dataset, args.architecture, args.training, args.confusion,
                args.embeddings, args.ablation, generate_all]):
        print("Error: No figure types specified. Use --all or specific flags.")
        print("Run with --help for usage information.")
        sys.exit(1)

    print("=" * 80)
    print("ABC2Vec Figure Generation")
    print("=" * 80)
    print()
    print(f"Data directory:        {data_dir}")
    print(f"Checkpoints directory: {checkpoints_dir}")
    print(f"Results directory:     {results_dir}")
    print(f"Output directory:      {output_dir}")
    print()

    figures_to_generate = []
    if generate_all or args.dataset:
        figures_to_generate.append("dataset")
    if generate_all or args.architecture:
        figures_to_generate.append("architecture")
    if generate_all or args.training:
        figures_to_generate.append("training")
    if generate_all or args.confusion:
        figures_to_generate.append("confusion")
    if generate_all or args.embeddings:
        figures_to_generate.append("embeddings")
    if generate_all or args.ablation:
        figures_to_generate.append("ablation")

    print(f"Generating: {', '.join(figures_to_generate)}")
    print("=" * 80)
    print()

    success_count = 0
    total_count = len(figures_to_generate)

    # 1. Dataset figures
    if "dataset" in figures_to_generate:
        print("=" * 80)
        print("1. Generating Dataset Figures")
        print("=" * 80)
        success = run_script("generate_dataset_figures.py", [
            "--data_dir", str(data_dir),
            "--output_dir", str(output_dir)
        ])
        if success:
            print("✓ Dataset figures generated")
            success_count += 1
        print()

    # 2. Architecture diagram
    if "architecture" in figures_to_generate:
        print("=" * 80)
        print("2. Generating Architecture Diagram")
        print("=" * 80)
        success = run_script("generate_architecture_diagram.py", [
            "--output_dir", str(output_dir)
        ])
        if success:
            print("✓ Architecture diagram generated")
            success_count += 1
        print()

    # 3. Training curves
    if "training" in figures_to_generate:
        print("=" * 80)
        print("3. Generating Training Curves")
        print("=" * 80)
        success = run_script("generate_training_curves.py", [
            "--checkpoint_dir", str(checkpoints_dir),
            "--output_dir", str(output_dir)
        ])
        if success:
            print("✓ Training curves generated")
            success_count += 1
        print()

    # 4. Confusion matrices
    if "confusion" in figures_to_generate:
        print("=" * 80)
        print("4. Generating Confusion Matrices")
        print("=" * 80)
        success = run_script("generate_confusion_matrix.py", [
            "--checkpoint", str(checkpoints_dir / "best_model.pt"),
            "--data_dir", str(data_dir),
            "--output_dir", str(output_dir),
            "--device", args.device
        ])
        if success:
            print("✓ Confusion matrices generated")
            success_count += 1
        print()

    # 5. Embedding visualizations (UMAP/t-SNE)
    if "embeddings" in figures_to_generate:
        print("=" * 80)
        print("5. Generating Embedding Visualizations")
        print("=" * 80)
        success = run_script("generate_umap.py", [
            "--checkpoint", str(checkpoints_dir / "best_model.pt"),
            "--data_dir", str(data_dir),
            "--output_dir", str(output_dir),
            "--device", args.device
        ])
        if success:
            print("✓ Embedding visualizations generated")
            success_count += 1
        print()

    # 6. Ablation study figures
    if "ablation" in figures_to_generate:
        print("=" * 80)
        print("6. Generating Ablation Study Figures")
        print("=" * 80)
        ablation_dir = checkpoints_dir / "ablation"
        ablation_results_dir = results_dir / "ablation_study"

        success = run_script("generate_ablation_figures.py", [
            "--checkpoints_dir", str(ablation_dir),
            "--results_dir", str(ablation_results_dir),
            "--output_dir", str(output_dir)
        ])
        if success:
            print("✓ Ablation study figures generated")
            success_count += 1
        print()

    # Summary
    print("=" * 80)
    print("FIGURE GENERATION SUMMARY")
    print("=" * 80)
    print()
    print(f"Generated: {success_count}/{total_count} figure types")
    print()

    if success_count < total_count:
        print("⚠ Some figures failed to generate. Check errors above.")
        sys.exit(1)
    else:
        print("✓ All figures generated successfully!")
        print()
        print(f"Figures saved to: {output_dir}")
        print()


if __name__ == "__main__":
    main()
