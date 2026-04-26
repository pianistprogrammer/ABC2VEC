#!/usr/bin/env python
"""
Generate publication-quality figures for ablation study.

Creates:
1. Training/validation loss curves comparison
2. Performance comparison bar charts
3. Multi-dimensional radar plot
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

sns.set_style("whitegrid")


def load_training_history(checkpoint_dir: Path) -> Dict:
    """Load training history from checkpoint directory."""
    history_path = checkpoint_dir / "training_history.json"

    if not history_path.exists():
        return None

    with open(history_path, 'r') as f:
        return json.load(f)


def plot_loss_curves(model_histories: Dict[str, Dict], output_path: Path):
    """Plot training and validation loss curves for all models."""
    # Create figure with more vertical spacing between rows
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Define colors for each model
    colors = {
        'mmm_only': '#e74c3c',
        'ti_only': '#3498db',
        'scl_only': '#2ecc71',
        'mmm_scl': '#f39c12',
        'ti_scl': '#9b59b6',
        'mmm_ti_scl': '#e67e22',
    }

    # 1. Total training loss
    ax = fig.add_subplot(gs[0, 0])
    for model_name, history in model_histories.items():
        if history is None:
            continue

        train_losses = history.get('train_losses', [])
        if not train_losses:
            continue

        steps = [entry['step'] for entry in train_losses]
        total_losses = [entry['total'] for entry in train_losses]

        ax.plot(steps, total_losses, label=model_name,
               color=colors.get(model_name, 'gray'), alpha=0.8, linewidth=2.0)

    ax.set_xlabel('Training Step', fontsize=18)
    ax.set_ylabel('Total Loss', fontsize=18)
    ax.set_title('(a) Training Loss Curves', fontweight='bold', fontsize=20, pad=15)
    ax.legend(loc='upper right', fontsize=14)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)

    # 2. Validation total loss
    ax = fig.add_subplot(gs[0, 1])
    for model_name, history in model_histories.items():
        if history is None:
            continue

        val_losses = history.get('val_losses', [])
        if not val_losses:
            continue

        steps = [entry['step'] for entry in val_losses]
        total_losses = [entry['total'] for entry in val_losses]

        ax.plot(steps, total_losses, label=model_name,
               color=colors.get(model_name, 'gray'), alpha=0.8, linewidth=2.0, marker='o', markersize=5)

    ax.set_xlabel('Training Step', fontsize=18)
    ax.set_ylabel('Validation Loss', fontsize=18)
    ax.set_title('(b) Validation Loss Curves', fontweight='bold', fontsize=20, pad=15)
    ax.legend(loc='upper right', fontsize=14)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)

    # 3. Final validation loss comparison (bar chart)
    ax = fig.add_subplot(gs[1, 0])
    final_val_losses = {}
    for model_name, history in model_histories.items():
        if history is None:
            continue

        val_losses = history.get('val_losses', [])
        if val_losses:
            final_val_losses[model_name] = val_losses[-1]['total']

    if final_val_losses:
        models = list(final_val_losses.keys())
        losses = list(final_val_losses.values())
        x_pos = np.arange(len(models))

        bars = ax.bar(x_pos, losses, color=[colors.get(m, 'gray') for m in models], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=15)
        ax.set_ylabel('Final Validation Loss', fontsize=18)
        ax.set_title('(c) Final Validation Loss Comparison', fontweight='bold', fontsize=20, pad=15)
        ax.tick_params(labelsize=16)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 4. Loss convergence rate (steps to reach threshold)
    ax = fig.add_subplot(gs[1, 1])
    convergence_steps = {}
    threshold = 2.5  # Define a reasonable threshold

    for model_name, history in model_histories.items():
        if history is None:
            continue

        val_losses = history.get('val_losses', [])
        if not val_losses:
            continue

        # Find first step where loss goes below threshold
        for entry in val_losses:
            if entry['total'] < threshold:
                convergence_steps[model_name] = entry['step']
                break

    if convergence_steps:
        models = list(convergence_steps.keys())
        steps = list(convergence_steps.values())
        x_pos = np.arange(len(models))

        bars = ax.bar(x_pos, steps, color=[colors.get(m, 'gray') for m in models], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=15)
        ax.set_ylabel(f'Steps to Reach Loss < {threshold}', fontsize=18)
        ax.set_title('(d) Convergence Speed Comparison', fontweight='bold', fontsize=20, pad=15)
        ax.tick_params(labelsize=16)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.suptitle('Ablation Study: Training Convergence Analysis', fontsize=24, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(str(output_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"✓ Saved loss curves: {output_path}")


def plot_performance_radar(results_df: pd.DataFrame, output_path: Path):
    """Create radar/spider plot for multi-dimensional performance comparison."""
    # Select metrics for radar plot
    metrics = ['tune_type_acc', 'mode_acc', 'lp_tune_type_acc', 'lp_mode_acc', 'lp_key_root_acc']
    metric_labels = ['Tune Type', 'Mode', 'LP: Tune Type', 'LP: Mode', 'LP: Key Root']

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    if not available_metrics:
        print("  Warning: No metrics available for radar plot")
        return

    available_labels = [metric_labels[i] for i, m in enumerate(metrics) if m in available_metrics]

    # Number of variables
    num_vars = len(available_metrics)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    # Define colors for each model
    colors = {
        'mmm_only': '#e74c3c',
        'ti_only': '#3498db',
        'scl_only': '#2ecc71',
        'mmm_scl': '#f39c12',
        'ti_scl': '#9b59b6',
        'mmm_ti_scl': '#e67e22',
    }

    # Plot each model
    for _, row in results_df.iterrows():
        model_name = row['model']
        values = [row.get(m, 0) for m in available_metrics]
        values += values[:1]  # Complete the circle

        color = colors.get(model_name, 'gray')
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color, alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_labels, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.title('Ablation Study: Multi-Task Performance Comparison',
             fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(str(output_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"✓ Saved radar plot: {output_path}")


def plot_objective_contribution_heatmap(results_df: pd.DataFrame, output_path: Path):
    """Create heatmap showing which objectives contribute to which tasks."""
    # Create a matrix: rows = models, columns = tasks
    tasks = ['tune_type_acc', 'mode_acc', 'lp_tune_type_acc', 'lp_mode_acc', 'lp_key_root_acc']
    task_labels = ['Tune Type\nClassification', 'Mode\nClassification',
                   'LP: Tune Type', 'LP: Mode', 'LP: Key Root']

    # Filter to available tasks
    available_tasks = [t for t in tasks if t in results_df.columns]
    if not available_tasks:
        print("  Warning: No tasks available for heatmap")
        return

    available_labels = [task_labels[i] for i, t in enumerate(tasks) if t in available_tasks]

    # Create matrix
    matrix = results_df[['model'] + available_tasks].set_index('model')

    # Sort models in a logical order
    model_order = ['mmm_only', 'ti_only', 'scl_only', 'mmm_scl', 'ti_scl', 'mmm_ti_scl']
    matrix = matrix.reindex([m for m in model_order if m in matrix.index])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(matrix.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1.0)

    # Set ticks
    ax.set_xticks(np.arange(len(available_labels)))
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_xticklabels(available_labels, fontsize=12, rotation=45, ha='right')
    ax.set_yticklabels(matrix.index, fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', fontsize=14, rotation=270, labelpad=25)

    # Add text annotations
    for i in range(len(matrix)):
        for j in range(len(available_labels)):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                text = ax.text(j, i, f'{value:.3f}',
                             ha="center", va="center", color="black", fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(len(available_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(matrix)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

    plt.title('Ablation Study: Performance Heatmap',
             fontsize=18, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(str(output_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"✓ Saved heatmap: {output_path}")


def plot_objective_importance(results_df: pd.DataFrame, output_path: Path):
    """Plot showing importance of each objective for different tasks."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Define model groups
    model_groups = {
        'Single': ['mmm_only', 'ti_only', 'scl_only'],
        'Pairs': ['mmm_scl', 'ti_scl'],
        'Full': ['mmm_ti_scl']
    }

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22']

    tasks = [
        ('tune_type_acc', 'Tune Type Classification'),
        ('mode_acc', 'Mode Classification'),
        ('lp_tune_type_acc', 'Linear Probing: Tune Type'),
        ('lp_mode_acc', 'Linear Probing: Mode')
    ]

    for idx, (task_col, task_name) in enumerate(tasks):
        if task_col not in results_df.columns:
            continue

        ax = axes[idx]

        # Prepare data
        task_data = results_df[['model', task_col]].copy()
        task_data = task_data[task_data['model'].isin([
            'mmm_only', 'ti_only', 'scl_only', 'mmm_scl', 'ti_scl', 'mmm_ti_scl'
        ])]

        # Sort by model order
        model_order = ['mmm_only', 'ti_only', 'scl_only', 'mmm_scl', 'ti_scl', 'mmm_ti_scl']
        task_data['model'] = pd.Categorical(task_data['model'], categories=model_order, ordered=True)
        task_data = task_data.sort_values('model')

        # Plot
        x_pos = np.arange(len(task_data))
        bars = ax.bar(x_pos, task_data[task_col], color=colors[:len(task_data)],
                     alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(task_data['model'], rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=13)
        ax.set_title(f'({chr(97+idx)}) {task_name}', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # Add horizontal line for best performance
        best_val = task_data[task_col].max()
        if pd.notna(best_val):
            ax.axhline(y=best_val, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Best')
            ax.legend(loc='upper left', fontsize=9)

    plt.suptitle('Ablation Study: Objective Importance Analysis',
                fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(str(output_path).replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"✓ Saved objective importance plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study figures")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/ablation",
                       help="Directory containing all ablation checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results/ablation_study",
                       help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="./figures/ablation",
                       help="Output directory for figures")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("ABLATION STUDY FIGURE GENERATION")
    print(f"{'='*80}\n")

    # 1. Load training histories
    print("Loading training histories...")
    checkpoints_dir = Path(args.checkpoints_dir)
    model_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()])

    model_histories = {}
    for model_dir in model_dirs:
        model_name = model_dir.name
        history = load_training_history(model_dir)
        if history:
            model_histories[model_name] = history
            print(f"  ✓ Loaded: {model_name}")
        else:
            print(f"  ✗ Missing: {model_name}")

    # 2. Plot loss curves
    print("\nGenerating loss curve comparison...")
    plot_loss_curves(model_histories, output_dir / "ablation_loss_curves.pdf")

    # 3. Load evaluation results
    print("\nLoading evaluation results...")
    results_path = Path(args.results_dir) / "ablation_comparison.csv"

    if results_path.exists():
        results_df = pd.read_csv(results_path)
        print(f"  ✓ Loaded results for {len(results_df)} models")

        # 4. Generate performance comparison plots
        print("\nGenerating performance comparison figures...")

        plot_performance_radar(results_df, output_dir / "ablation_radar_plot.pdf")
        plot_objective_contribution_heatmap(results_df, output_dir / "ablation_heatmap.pdf")
        plot_objective_importance(results_df, output_dir / "ablation_objective_importance.pdf")

    else:
        print(f"  ✗ Results not found at {results_path}")
        print("     Run run_ablation_study.py first to generate evaluation results")

    print(f"\n{'='*80}")
    print("✓ Figure generation complete!")
    print(f"{'='*80}")
    print(f"\nFigures saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - ablation_loss_curves.pdf (and .png)")
    print("  - ablation_radar_plot.pdf (and .png)")
    print("  - ablation_heatmap.pdf (and .png)")
    print("  - ablation_objective_importance.pdf (and .png)")
    print()


if __name__ == "__main__":
    main()
