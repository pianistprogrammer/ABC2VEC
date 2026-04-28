#!/usr/bin/env python3
"""
Train all ablation study models.

Trains 7 model variants with different objective combinations to measure
the contribution of each pre-training objective (MMM, TI, SCL).

Usage:
    python scripts/run_train_all_ablations.py --data_dir data/processed --device mps

All models use identical hyperparameters for fair comparison.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ABLATION_CONFIGS = [
    {
        "name": "mmm_only",
        "description": "MMM-only: Masked Music Modeling alone",
        "lambda_mmm": 1.0,
        "lambda_ti": 0.0,
        "lambda_scl": 0.0,
    },
    {
        "name": "ti_only",
        "description": "TI-only: Transposition Invariance alone",
        "lambda_mmm": 0.0,
        "lambda_ti": 1.0,
        "lambda_scl": 0.0,
    },
    {
        "name": "scl_only",
        "description": "SCL-only: Section Contrastive Learning alone",
        "lambda_mmm": 0.0,
        "lambda_ti": 0.0,
        "lambda_scl": 1.0,
    },
    {
        "name": "mmm_ti",
        "description": "MMM+TI: Reconstruction + Transposition Invariance",
        "lambda_mmm": 1.0,
        "lambda_ti": 0.5,
        "lambda_scl": 0.0,
    },
    {
        "name": "mmm_scl",
        "description": "MMM+SCL: Reconstruction + Section Structure (BEST: 80.8%)",
        "lambda_mmm": 1.0,
        "lambda_ti": 0.0,
        "lambda_scl": 0.5,
    },
    {
        "name": "ti_scl",
        "description": "TI+SCL: Transposition Invariance + Section Structure",
        "lambda_mmm": 0.0,
        "lambda_ti": 1.0,
        "lambda_scl": 0.5,
    },
    {
        "name": "mmm_ti_scl",
        "description": "MMM+TI+SCL: Full model with all three objectives",
        "lambda_mmm": 1.0,
        "lambda_ti": 0.5,
        "lambda_scl": 0.5,
    },
]

# Standard hyperparameters (same for all ablations)
STANDARD_PARAMS = {
    "epochs": 40,
    "batch_size": 32,
    "grad_accum": 4,
    "lr": 1e-4,
    "d_model": 256,
    "n_layers": 6,
    "n_heads": 8,
    "max_steps": 40000,
    "num_workers": 8,
}


def train_ablation(config, args):
    """Train a single ablation model."""
    output_dir = Path(args.output_dir) / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Training: {config['name']}")
    print(f"Description: {config['description']}")
    print("=" * 80)
    print()

    # Build command
    cmd = [
        sys.executable,
        "scripts/run_training.py",
        "--data_dir", args.data_dir,
        "--output_dir", str(output_dir),
        "--d_model", str(STANDARD_PARAMS["d_model"]),
        "--n_layers", str(STANDARD_PARAMS["n_layers"]),
        "--n_heads", str(STANDARD_PARAMS["n_heads"]),
        "--batch_size", str(STANDARD_PARAMS["batch_size"]),
        "--grad_accum", str(STANDARD_PARAMS["grad_accum"]),
        "--epochs", str(STANDARD_PARAMS["epochs"]),
        "--lr", str(STANDARD_PARAMS["lr"]),
        "--lambda_mmm", str(config["lambda_mmm"]),
        "--lambda_ti", str(config["lambda_ti"]),
        "--lambda_scl", str(config["lambda_scl"]),
        "--num_workers", str(STANDARD_PARAMS["num_workers"]),
        "--device", args.device,
        "--max_steps", str(STANDARD_PARAMS["max_steps"]),
        "--amp",
    ]

    # Check for existing checkpoint
    best_model = output_dir / "best_model.pt"
    if best_model.exists() and not args.force:
        print(f"⚠ Model already exists: {best_model}")
        print(f"  Use --force to retrain, or --skip-existing to skip")
        if args.skip_existing:
            print("  Skipping...")
            print()
            return True
        response = input("  Retrain? (y/n): ")
        if response.lower() != 'y':
            print("  Skipping...")
            print()
            return True

    # Run training
    try:
        subprocess.run(cmd, check=True)
        print()
        print(f"✓ {config['name']} training complete!")
        print(f"  Model saved to: {output_dir}/best_model.pt")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"✗ {config['name']} training failed!")
        print(f"  Error: {e}")
        print()
        return False
    except KeyboardInterrupt:
        print()
        print(f"⚠ {config['name']} training interrupted by user")
        print()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train all ablation study models with standardized hyperparameters"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory with processed data (train/val/test parquet files)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/ablation",
        help="Base directory for ablation checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (mps, cuda, cpu)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to train (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if checkpoints exist",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have checkpoints",
    )

    args = parser.parse_args()

    # Filter models if specified
    configs = ABLATION_CONFIGS
    if args.models:
        model_names = set(args.models)
        configs = [c for c in configs if c["name"] in model_names]
        if not configs:
            print(f"Error: No valid models specified")
            print(f"Available models: {[c['name'] for c in ABLATION_CONFIGS]}")
            sys.exit(1)

    print("=" * 80)
    print("ABLATION STUDY TRAINING")
    print("=" * 80)
    print()
    print(f"Training {len(configs)} ablation variants:")
    for c in configs:
        print(f"  • {c['name']}: {c['description']}")
    print()
    print("Standard hyperparameters:")
    for key, value in STANDARD_PARAMS.items():
        print(f"  {key}: {value}")
    print()
    print("=" * 80)
    print()

    # Train all models
    results = {}
    for config in configs:
        success = train_ablation(config, args)
        results[config["name"]] = success

    # Summary
    print()
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print()

    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name}: {status}")

    print()
    successful = sum(results.values())
    total = len(results)
    print(f"Completed: {successful}/{total} models")
    print()

    if successful < total:
        print("⚠ Some models failed to train. Check logs above for details.")
        sys.exit(1)
    else:
        print("=" * 80)
        print("✓ All ablation models trained successfully!")
        print("=" * 80)
        print()
        print("Next steps:")
        print(f"  1. Evaluate all models:")
        print(f"     python scripts/run_ablation_study.py --data_dir {args.data_dir}")
        print()
        print(f"  2. Generate comparison figures:")
        print(f"     python scripts/run_generate_ablation_figures.py")
        print()


if __name__ == "__main__":
    main()
