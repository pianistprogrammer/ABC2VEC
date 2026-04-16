#!/usr/bin/env python
"""
Generate confusion matrix figure from tune type classification results.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def generate_confusion_matrix(results_path: Path, output_path: Path):
    """Generate confusion matrix visualization."""

    # Load results
    with open(results_path) as f:
        results = json.load(f)

    cm = np.array(results['confusion_matrix'])
    labels = results['class_labels']

    # Compute percentages for annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap (white to blue)
    cmap = LinearSegmentedColormap.from_list('custom_blue', ['#f7fbff', '#08519c'])

    # Plot heatmap
    sns.heatmap(
        cm_percent,
        annot=False,
        fmt='.1f',
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor='white'
    )

    # Add custom annotations with counts only
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            pct = cm_percent[i, j]

            # Use white text for dark cells, black for light cells
            text_color = 'white' if pct > 50 else 'black'

            # Format annotation - just the count
            text = str(count)

            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   color=text_color, fontsize=14,
                   weight='bold' if i == j else 'normal')

    # Labels and title
    ax.set_xlabel('Predicted Tune Type', fontsize=13, weight='bold')
    ax.set_ylabel('True Tune Type', fontsize=13, weight='bold')
    ax.set_title('Confusion Matrix: Tune Type Classification\n(5-fold Cross-Validation, n=2,078)',
                 fontsize=14, weight='bold', pad=15)

    # Rotate labels
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    # Add support counts to y-axis labels
    report = results['classification_report']
    new_ylabels = [f'{label}\n(n={int(report[label]["support"])})'
                   for label in labels]
    ax.set_yticklabels(new_ylabels, rotation=0, fontsize=11)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Confusion matrix saved to: {output_path}")

    # Also save PNG version
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PNG version saved to: {png_path}")

    plt.close()

    # Print summary
    print(f"\nConfusion Matrix Summary:")
    print(f"  Overall Accuracy: {results['mean_accuracy']*100:.1f}%")
    print(f"  Macro F1: {results['mean_f1']*100:.1f}%")
    print(f"  Weighted F1: {report['weighted avg']['f1-score']*100:.1f}%")
    print(f"\nDiagonal (correct predictions): {np.diag(cm).sum()} / {cm.sum()} = {np.diag(cm).sum()/cm.sum()*100:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate confusion matrix figure")
    parser.add_argument(
        "--results",
        type=str,
        default="./checkpoints/tune_type_classification_results.json",
        help="Path to classification results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./Paper/tune_type_confusion.pdf",
        help="Output path for figure",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_confusion_matrix(results_path, output_path)
