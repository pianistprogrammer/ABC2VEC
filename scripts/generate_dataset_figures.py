#!/usr/bin/env python
"""
Generate high-definition figures for dataset statistics.

This script creates publication-quality visualizations of the ABC2Vec dataset
including preprocessing statistics, distributions, and data characteristics.
"""

import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

sns.set_style("whitegrid")
sns.set_context("paper")


def load_data(data_dir: str):
    """Load train, val, and test datasets."""
    train_df = pd.read_parquet(Path(data_dir) / "train.parquet")
    val_df = pd.read_parquet(Path(data_dir) / "val.parquet")
    test_df = pd.read_parquet(Path(data_dir) / "test.parquet")

    # Add split labels
    train_df['split'] = 'Train'
    val_df['split'] = 'Val'
    test_df['split'] = 'Test'

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return train_df, val_df, test_df, all_df


def plot_dataset_statistics(all_df, train_df, val_df, test_df, output_path):
    """Create comprehensive dataset statistics figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.4, left=0.08, right=0.95, top=0.92, bottom=0.06)

    # 1. Dataset split sizes
    ax1 = fig.add_subplot(gs[0, 0])
    split_sizes = [len(train_df), len(val_df), len(test_df)]
    split_labels = ['Train', 'Val', 'Test']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax1.bar(split_labels, split_sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Number of Tunes')
    ax1.set_title('Dataset Split Sizes', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8)

    # 2. Bar length distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(all_df['num_bars'].clip(0, 64), bins=40, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Number of Bars')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Bar Length Distribution', fontweight='bold')
    ax2.axvline(all_df['num_bars'].median(), color='red', linestyle='--', linewidth=1.5, label=f"Median: {all_df['num_bars'].median():.0f}")
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Character length distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(all_df['char_length'].clip(0, 800), bins=40, color='#e67e22', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Character Length')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Tune Length (Characters)', fontweight='bold')
    ax3.axvline(all_df['char_length'].median(), color='red', linestyle='--', linewidth=1.5, label=f"Median: {all_df['char_length'].median():.0f}")
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Tune type distribution
    ax4 = fig.add_subplot(gs[1, :2])
    tune_type_counts = all_df['tune_type'].value_counts().head(8)
    colors_tune = sns.color_palette("husl", len(tune_type_counts))
    bars = ax4.barh(range(len(tune_type_counts)), tune_type_counts.values, color=colors_tune, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_yticks(range(len(tune_type_counts)))
    ax4.set_yticklabels(tune_type_counts.index, fontsize=9)
    ax4.set_xlabel('Number of Tunes', fontsize=10)
    ax4.set_title('Tune Type Distribution (Top 8)', fontweight='bold', pad=10)
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_xlim(0, max(tune_type_counts.values) * 1.15)

    # Add value labels with better positioning
    for i, (bar, val) in enumerate(zip(bars, tune_type_counts.values)):
        ax4.text(val + max(tune_type_counts.values)*0.01, i, f'{int(val):,}', va='center', fontsize=8, ha='left')

    # 5. Mode distribution
    ax5 = fig.add_subplot(gs[1, 2])
    mode_counts = all_df['mode'].value_counts().head(4)  # Top 4 to reduce clutter
    colors_mode = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(mode_counts)]

    # Create pie chart with better label positioning
    wedges, texts, autotexts = ax5.pie(mode_counts.values,
                                        autopct='%1.1f%%',
                                        colors=colors_mode,
                                        startangle=90,
                                        pctdistance=0.80,
                                        explode=[0.05, 0.05, 0.05, 0.05])  # Slightly separate slices

    # Improve text properties
    for text in texts:
        text.set_fontsize(9)
        text.set_fontweight('bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    # Add legend instead of labels on pie
    ax5.legend(wedges, mode_counts.index, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    ax5.set_title('Mode Distribution (Top 4)', fontweight='bold', pad=10)

    # 6. Key root distribution
    ax6 = fig.add_subplot(gs[2, :2])
    root_counts = all_df['root_note'].value_counts().head(10)  # Top 10 to reduce crowding
    colors_root = sns.color_palette("Spectral", len(root_counts))
    bars = ax6.bar(range(len(root_counts)), root_counts.values, color=colors_root, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax6.set_xticks(range(len(root_counts)))
    ax6.set_xticklabels(root_counts.index, fontsize=9)
    ax6.set_ylabel('Number of Tunes', fontsize=10)
    ax6.set_xlabel('Root Note', fontsize=10)
    ax6.set_title('Key Root Distribution (Top 10)', fontweight='bold', pad=10)
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_ylim(0, max(root_counts.values) * 1.15)

    # Add value labels with better spacing
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(root_counts.values)*0.01,
                f'{int(height/1000):.0f}k' if height >= 1000 else f'{int(height)}',
                ha='center', va='bottom', fontsize=8, rotation=0)

    # 7. Meter distribution
    ax7 = fig.add_subplot(gs[2, 2])
    meter_counts = all_df['meter'].value_counts().head(6)
    colors_meter = sns.color_palette("muted", len(meter_counts))
    bars = ax7.bar(range(len(meter_counts)), meter_counts.values, color=colors_meter, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax7.set_xticks(range(len(meter_counts)))
    ax7.set_xticklabels(meter_counts.index, rotation=45, ha='right', fontsize=9)
    ax7.set_ylabel('Number of Tunes', fontsize=10)
    ax7.set_xlabel('Meter Signature', fontsize=10)
    ax7.set_title('Meter Distribution (Top 6)', fontweight='bold', pad=10)
    ax7.grid(axis='y', alpha=0.3)
    ax7.set_ylim(0, max(meter_counts.values) * 1.15)

    # Add value labels with better spacing
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + max(meter_counts.values)*0.01,
                f'{int(height/1000):.0f}k' if height >= 1000 else f'{int(height)}',
                ha='center', va='bottom', fontsize=8)

    plt.suptitle('ABC2Vec Dataset Statistics', fontsize=15, fontweight='bold', y=0.985)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_key_mode_distribution(all_df, output_path):
    """Create detailed key and mode distribution heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Key root by mode heatmap
    ax1 = axes[0]
    key_mode_cross = pd.crosstab(all_df['root_note'], all_df['mode'])
    # Sort by total frequency
    key_mode_cross = key_mode_cross.loc[key_mode_cross.sum(axis=1).sort_values(ascending=False).index]

    sns.heatmap(key_mode_cross, annot=True, fmt='d', cmap='YlOrRd', ax=ax1,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    ax1.set_xlabel('Mode')
    ax1.set_ylabel('Root Note')
    ax1.set_title('Key Root × Mode Distribution', fontweight='bold')

    # 2. Tune type by mode
    ax2 = axes[1]
    type_mode_cross = pd.crosstab(all_df['tune_type'], all_df['mode'])
    # Keep top 10 tune types
    top_types = all_df['tune_type'].value_counts().head(10).index
    type_mode_cross = type_mode_cross.loc[type_mode_cross.index.isin(top_types)]
    type_mode_cross = type_mode_cross.loc[type_mode_cross.sum(axis=1).sort_values(ascending=False).index]

    sns.heatmap(type_mode_cross, annot=True, fmt='d', cmap='Blues', ax=ax2,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    ax2.set_xlabel('Mode')
    ax2.set_ylabel('Tune Type')
    ax2.set_title('Tune Type × Mode Distribution', fontweight='bold')

    plt.suptitle('Key and Mode Relationships', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf', pad_inches=0.2)
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_bar_patching_stats(all_df, output_path):
    """Create bar patching and preprocessing statistics."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35, left=0.08, right=0.95, top=0.90, bottom=0.08)

    # 1. Bars per tune (clipped for visualization)
    ax1 = fig.add_subplot(gs[0, 0])
    bars_clipped = all_df['num_bars'].clip(0, 64)
    ax1.hist(bars_clipped, bins=32, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(all_df['num_bars'].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {all_df["num_bars"].mean():.1f}')
    ax1.axvline(all_df['num_bars'].median(), color='orange', linestyle='--', linewidth=1.5, label=f'Median: {all_df["num_bars"].median():.0f}')
    ax1.set_xlabel('Number of Bars per Tune', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Bars per Tune Distribution', fontweight='bold', pad=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Sections per tune
    ax2 = fig.add_subplot(gs[0, 1])
    sections_clipped = all_df['num_sections'].clip(0, 10)
    ax2.hist(sections_clipped, bins=10, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Number of Sections', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Sections per Tune Distribution', fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Character length vs bar count scatter
    ax3 = fig.add_subplot(gs[0, 2])
    sample = all_df.sample(min(5000, len(all_df)), random_state=42)
    ax3.scatter(sample['num_bars'], sample['char_length'], alpha=0.3, s=10, color='#9b59b6')
    ax3.set_xlabel('Number of Bars', fontsize=10)
    ax3.set_ylabel('Character Length', fontsize=10)
    ax3.set_title('Bars vs Character Length', fontweight='bold', pad=10)
    ax3.grid(alpha=0.3)

    # Add correlation
    corr = all_df['num_bars'].corr(all_df['char_length'])
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes,
             verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Bar length categories
    ax4 = fig.add_subplot(gs[1, 0])
    bar_categories = pd.cut(all_df['num_bars'], bins=[0, 16, 32, 48, 64, 128],
                            labels=['0-16', '17-32', '33-48', '49-64', '64+'])
    cat_counts = bar_categories.value_counts().sort_index()
    colors = sns.color_palette("rocket", len(cat_counts))
    bars = ax4.bar(range(len(cat_counts)), cat_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_xticks(range(len(cat_counts)))
    ax4.set_xticklabels(cat_counts.index, fontsize=9)
    ax4.set_ylabel('Number of Tunes', fontsize=10)
    ax4.set_xlabel('Bar Count Category', fontsize=10)
    ax4.set_title('Tune Length Categories', fontweight='bold', pad=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, max(cat_counts.values) * 1.15)

    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(cat_counts.values)*0.01,
                f'{int(height/1000):.0f}k' if height >= 1000 else f'{int(height)}',
                ha='center', va='bottom', fontsize=8)

    # 5. Statistics table
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    # Add title at the top with more space from subplot above
    ax5.text(0.5, 1.0, 'Dataset Preprocessing Statistics Summary',
             ha='center', va='top', fontweight='bold', fontsize=12,
             transform=ax5.transAxes)

    stats_data = [
        ['Metric', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        ['Bars/Tune', f"{all_df['num_bars'].mean():.1f}", f"{all_df['num_bars'].median():.0f}",
         f"{all_df['num_bars'].std():.1f}", f"{all_df['num_bars'].min():.0f}", f"{all_df['num_bars'].max():.0f}"],
        ['Sections/Tune', f"{all_df['num_sections'].mean():.1f}", f"{all_df['num_sections'].median():.0f}",
         f"{all_df['num_sections'].std():.1f}", f"{all_df['num_sections'].min():.0f}", f"{all_df['num_sections'].max():.0f}"],
        ['Char Length', f"{all_df['char_length'].mean():.0f}", f"{all_df['char_length'].median():.0f}",
         f"{all_df['char_length'].std():.0f}", f"{all_df['char_length'].min():.0f}", f"{all_df['char_length'].max():.0f}"],
    ]

    table = ax5.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15],
                     bbox=[0.0, 0.0, 1.0, 0.85])  # Position table lower
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Increased from 9
    table.scale(1, 2.2)  # Increased row height for readability

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)

    # Alternate row colors
    for i in range(1, 4):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.suptitle('Bar Patchification and Preprocessing Statistics', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf', pad_inches=0.2)
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_dataset_summary(train_df, val_df, test_df, all_df):
    """Print comprehensive dataset summary statistics."""
    print("\n" + "="*80)
    print("ABC2VEC DATASET SUMMARY")
    print("="*80)

    print(f"\n📊 Dataset Splits:")
    print(f"  Training:   {len(train_df):>8,} tunes ({len(train_df)/len(all_df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):>8,} tunes ({len(val_df)/len(all_df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):>8,} tunes ({len(test_df)/len(all_df)*100:.1f}%)")
    print(f"  Total:      {len(all_df):>8,} tunes")

    print(f"\n📏 Structural Statistics:")
    print(f"  Bars per tune:        {all_df['num_bars'].mean():.1f} ± {all_df['num_bars'].std():.1f} (median: {all_df['num_bars'].median():.0f})")
    print(f"  Sections per tune:    {all_df['num_sections'].mean():.1f} ± {all_df['num_sections'].std():.1f}")
    print(f"  Character length:     {all_df['char_length'].mean():.0f} ± {all_df['char_length'].std():.0f}")

    print(f"\n🎵 Tune Type Distribution (Top 8):")
    for tune_type, count in all_df['tune_type'].value_counts().head(8).items():
        print(f"  {tune_type:>15s}: {count:>7,} ({count/len(all_df)*100:>5.1f}%)")

    print(f"\n🎼 Mode Distribution:")
    for mode, count in all_df['mode'].value_counts().items():
        print(f"  {mode:>12s}: {count:>7,} ({count/len(all_df)*100:>5.1f}%)")

    print(f"\n🎹 Key Root Distribution (Top 12):")
    for root, count in all_df['root_note'].value_counts().head(12).items():
        print(f"  {root:>3s}: {count:>7,} ({count/len(all_df)*100:>5.1f}%)")

    print(f"\n⏱️  Meter Distribution (Top 6):")
    for meter, count in all_df['meter'].value_counts().head(6).items():
        print(f"  {meter:>6s}: {count:>7,} ({count/len(all_df)*100:>5.1f}%)")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate HD dataset figures for ABC2Vec paper")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, default="./figures",
                       help="Output directory for figures")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ABC2VEC DATASET FIGURE GENERATION")
    print("="*80)
    print(f"\nData directory:   {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n📂 Loading data...")
    train_df, val_df, test_df, all_df = load_data(args.data_dir)
    print(f"  Loaded {len(all_df):,} total tunes")

    # Print summary
    print_dataset_summary(train_df, val_df, test_df, all_df)

    # Generate figures
    print("📊 Generating figures...")
    print()

    plot_dataset_statistics(
        all_df, train_df, val_df, test_df,
        str(output_path / "dataset_statistics.pdf")
    )

    plot_key_mode_distribution(
        all_df,
        str(output_path / "key_mode_distribution.pdf")
    )

    plot_bar_patching_stats(
        all_df,
        str(output_path / "bar_patching_stats.pdf")
    )

    print("\n" + "="*80)
    print("✓ All figures generated successfully!")
    print("="*80)
    print(f"\nOutput files:")
    for ext in ['pdf', 'png']:
        print(f"  - dataset_statistics.{ext}")
        print(f"  - key_mode_distribution.{ext}")
        print(f"  - bar_patching_stats.{ext}")
    print()


if __name__ == "__main__":
    main()
