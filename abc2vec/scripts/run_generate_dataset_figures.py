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
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 22

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.6)


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
    """Create comprehensive dataset statistics figure (6 subplots, 3x2 grid)."""
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.06)

    # Larger font sizes for readability
    title_fs = 20
    label_fs = 18
    tick_fs = 16
    annot_fs = 15

    # 1. Bar length distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_df['num_bars'].clip(0, 64), bins=40, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Number of Bars', fontsize=label_fs)
    ax1.set_ylabel('Frequency', fontsize=label_fs)
    ax1.set_title('(a) Bar Length Distribution', fontweight='bold', fontsize=title_fs)
    ax1.axvline(all_df['num_bars'].median(), color='red', linestyle='--', linewidth=1.5, label=f"Median: {all_df['num_bars'].median():.0f}")
    ax1.legend(fontsize=annot_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Character length distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(all_df['char_length'].clip(0, 800), bins=40, color='#e67e22', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Character Length', fontsize=label_fs)
    ax2.set_ylabel('Frequency', fontsize=label_fs)
    ax2.set_title('(b) Tune Length (Characters)', fontweight='bold', fontsize=title_fs)
    ax2.axvline(all_df['char_length'].median(), color='red', linestyle='--', linewidth=1.5, label=f"Median: {all_df['char_length'].median():.0f}")
    ax2.legend(fontsize=annot_fs)
    ax2.tick_params(labelsize=tick_fs)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Tune type distribution
    ax3 = fig.add_subplot(gs[1, 0])
    tune_type_counts = all_df[all_df['tune_type'].str.strip() != '']['tune_type'].value_counts().head(8)
    colors_tune = sns.color_palette("husl", len(tune_type_counts))
    bars = ax3.barh(range(len(tune_type_counts)), tune_type_counts.values, color=colors_tune, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(range(len(tune_type_counts)))
    ax3.set_yticklabels(tune_type_counts.index, fontsize=tick_fs)
    ax3.set_xlabel('Number of Tunes', fontsize=label_fs)
    ax3.set_title('(c) Tune Type Distribution', fontweight='bold', fontsize=title_fs, pad=10)
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_xlim(0, max(tune_type_counts.values) * 1.15)
    ax3.tick_params(labelsize=tick_fs)

    for i, (bar, val) in enumerate(zip(bars, tune_type_counts.values)):
        ax3.text(val + max(tune_type_counts.values)*0.01, i, f'{int(val):,}', va='center', fontsize=annot_fs, ha='left')

    # 4. Mode distribution (horizontal bar chart)
    ax4 = fig.add_subplot(gs[1, 1])
    mode_counts = all_df['mode'].value_counts().head(4)
    colors_mode = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(mode_counts)]
    bars_mode = ax4.barh(range(len(mode_counts)), mode_counts.values, color=colors_mode, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_yticks(range(len(mode_counts)))
    ax4.set_yticklabels(mode_counts.index, fontsize=tick_fs)
    ax4.set_xlabel('Number of Tunes', fontsize=label_fs)
    ax4.set_title('(d) Mode Distribution', fontweight='bold', fontsize=title_fs, pad=10)
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_xlim(0, max(mode_counts.values) * 1.20)
    ax4.tick_params(labelsize=tick_fs)

    for i, (bar, val) in enumerate(zip(bars_mode, mode_counts.values)):
        pct = val / mode_counts.values.sum() * 100
        ax4.text(val + max(mode_counts.values)*0.01, i, f'{int(val):,} ({pct:.1f}%)', va='center', fontsize=annot_fs, ha='left')

    # 5. Key root distribution
    ax5 = fig.add_subplot(gs[2, 0])
    root_counts = all_df['root_note'].value_counts().head(10)
    colors_root = sns.color_palette("Spectral", len(root_counts))
    bars = ax5.bar(range(len(root_counts)), root_counts.values, color=colors_root, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax5.set_xticks(range(len(root_counts)))
    ax5.set_xticklabels(root_counts.index, fontsize=tick_fs)
    ax5.set_ylabel('Number of Tunes', fontsize=label_fs)
    ax5.set_xlabel('Root Note', fontsize=label_fs)
    ax5.set_title('(e) Key Root Distribution', fontweight='bold', fontsize=title_fs, pad=10)
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_ylim(0, max(root_counts.values) * 1.15)
    ax5.tick_params(labelsize=tick_fs)

    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(root_counts.values)*0.01,
                f'{int(height/1000):.0f}k' if height >= 1000 else f'{int(height)}',
                ha='center', va='bottom', fontsize=annot_fs, rotation=0)

    # 6. Meter distribution
    ax6 = fig.add_subplot(gs[2, 1])
    meter_counts = all_df['meter'].value_counts().head(6)
    colors_meter = sns.color_palette("muted", len(meter_counts))
    bars = ax6.bar(range(len(meter_counts)), meter_counts.values, color=colors_meter, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax6.set_xticks(range(len(meter_counts)))
    ax6.set_xticklabels(meter_counts.index, rotation=45, ha='right', fontsize=tick_fs)
    ax6.set_ylabel('Number of Tunes', fontsize=label_fs)
    ax6.set_xlabel('Meter Signature', fontsize=label_fs)
    ax6.set_title('(f) Meter Distribution', fontweight='bold', fontsize=title_fs, pad=10)
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_ylim(0, max(meter_counts.values) * 1.15)
    ax6.tick_params(labelsize=tick_fs)

    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(meter_counts.values)*0.01,
                f'{int(height/1000):.0f}k' if height >= 1000 else f'{int(height)}',
                ha='center', va='bottom', fontsize=annot_fs)

    plt.suptitle('ABC2Vec Dataset Statistics', fontsize=22, fontweight='bold', y=0.985)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_key_mode_distribution(all_df, output_path):
    """100% stacked bar charts: key root × mode and tune type × mode."""
    mode_palette = {
        'major': '#3498db', 'minor': '#e74c3c',
        'dorian': '#2ecc71', 'mixolydian': '#f39c12',
        'phrygian': '#9b59b6', 'lydian': '#1abc9c', 'locrian': '#e67e22',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'wspace': 0.40})

    def stacked_pct_bar(ax, cross, title, ylabel):
        # Normalise to 100%
        pct = cross.div(cross.sum(axis=1), axis=0) * 100
        modes = pct.columns.tolist()
        bottoms = np.zeros(len(pct))
        for mode in modes:
            color = mode_palette.get(mode, '#95a5a6')
            bars = ax.bar(range(len(pct)), pct[mode], bottom=bottoms,
                          color=color, label=mode, edgecolor='white', linewidth=0.4)
            # Label segments ≥ 8%
            for i, (b, h) in enumerate(zip(bottoms, pct[mode])):
                if h >= 8:
                    ax.text(i, b + h / 2, f'{h:.0f}%', ha='center', va='center',
                            fontsize=13, color='white', fontweight='bold')
            bottoms += pct[mode].values

        ax.set_xticks(range(len(pct)))
        ax.set_xticklabels(pct.index, fontsize=15, rotation=30, ha='right')
        ax.set_ylabel('Proportion (%)', fontsize=17)
        ax.set_xlabel(ylabel, fontsize=17)
        ax.set_title(title, fontweight='bold', fontsize=18, pad=10)
        ax.set_ylim(0, 105)
        ax.tick_params(labelsize=15)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # (a) Key root × mode — top 8 roots
    key_mode_cross = pd.crosstab(all_df['root_note'], all_df['mode'])
    key_mode_cross = key_mode_cross.loc[
        key_mode_cross.sum(axis=1).sort_values(ascending=False).head(8).index
    ]
    # Keep only modes present in palette / with data
    key_mode_cross = key_mode_cross.loc[:, key_mode_cross.sum(axis=0) > 0]
    stacked_pct_bar(axes[0], key_mode_cross,
                    '(a) Key Root by Mode', 'Root Note')

    # (b) Tune type × mode — top 7 types (no empty labels)
    type_mode_cross = pd.crosstab(all_df['tune_type'], all_df['mode'])
    top_types = (all_df[all_df['tune_type'].str.strip() != '']['tune_type']
                 .value_counts().head(7).index)
    type_mode_cross = type_mode_cross.loc[type_mode_cross.index.isin(top_types)]
    type_mode_cross = type_mode_cross.loc[
        type_mode_cross.sum(axis=1).sort_values(ascending=False).index
    ]
    type_mode_cross = type_mode_cross.loc[:, type_mode_cross.sum(axis=0) > 0]
    stacked_pct_bar(axes[1], type_mode_cross,
                    '(b) Tune Type by Mode', 'Tune Type')

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=mode_palette.get(m, '#95a5a6'))
               for m in key_mode_cross.columns]
    labels = list(key_mode_cross.columns)
    fig.legend(handles, labels, title='Mode', title_fontsize=15,
               fontsize=14, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.14), frameon=True)

    plt.suptitle('Key and Mode Relationships', fontsize=20, fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf', pad_inches=0.2)
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_bar_patching_stats(all_df, output_path):
    """Create bar patching statistics: 2x2 grid, large fonts, no embedded table."""
    title_fs = 20
    label_fs = 18
    tick_fs = 16
    annot_fs = 15

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.35})

    # (a) Bars per tune distribution
    ax1 = axes[0, 0]
    bars_clipped = all_df['num_bars'].clip(0, 64)
    ax1.hist(bars_clipped, bins=32, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(all_df['num_bars'].mean(), color='red', linestyle='--', linewidth=1.8,
                label=f'Mean: {all_df["num_bars"].mean():.1f}')
    ax1.axvline(all_df['num_bars'].median(), color='orange', linestyle='--', linewidth=1.8,
                label=f'Median: {all_df["num_bars"].median():.0f}')
    ax1.set_xlabel('Number of Bars per Tune', fontsize=label_fs)
    ax1.set_ylabel('Frequency', fontsize=label_fs)
    ax1.set_title('(a) Bars per Tune', fontweight='bold', fontsize=title_fs, pad=10)
    ax1.legend(fontsize=annot_fs)
    ax1.tick_params(labelsize=tick_fs)
    ax1.grid(axis='y', alpha=0.3)

    # (b) Sections per tune
    ax2 = axes[0, 1]
    sections_clipped = all_df['num_sections'].clip(0, 10)
    ax2.hist(sections_clipped, bins=10, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    mean_sec = all_df['num_sections'].mean()
    ax2.axvline(mean_sec, color='red', linestyle='--', linewidth=1.8, label=f'Mean: {mean_sec:.1f}')
    ax2.set_xlabel('Number of Sections', fontsize=label_fs)
    ax2.set_ylabel('Frequency', fontsize=label_fs)
    ax2.set_title('(b) Sections per Tune', fontweight='bold', fontsize=title_fs, pad=10)
    ax2.legend(fontsize=annot_fs)
    ax2.tick_params(labelsize=tick_fs)
    ax2.grid(axis='y', alpha=0.3)

    # (c) Bars vs character length scatter
    ax3 = axes[1, 0]
    sample = all_df.sample(min(5000, len(all_df)), random_state=42)
    ax3.scatter(sample['num_bars'], sample['char_length'], alpha=0.3, s=12, color='#9b59b6')
    corr = all_df['num_bars'].corr(all_df['char_length'])
    ax3.set_xlabel('Number of Bars', fontsize=label_fs)
    ax3.set_ylabel('Character Length', fontsize=label_fs)
    ax3.set_title('(c) Bars vs Character Length', fontweight='bold', fontsize=title_fs, pad=10)
    ax3.tick_params(labelsize=tick_fs)
    ax3.grid(alpha=0.3)
    ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
             verticalalignment='top', fontsize=annot_fs,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # (d) Tune length categories
    ax4 = axes[1, 1]
    bar_categories = pd.cut(all_df['num_bars'], bins=[0, 16, 32, 48, 64, 128],
                            labels=['0--16', '17--32', '33--48', '49--64', '64+'])
    cat_counts = bar_categories.value_counts().sort_index()
    colors = sns.color_palette('rocket', len(cat_counts))
    bars_plot = ax4.bar(range(len(cat_counts)), cat_counts.values, color=colors,
                        alpha=0.85, edgecolor='black', linewidth=0.5)
    ax4.set_xticks(range(len(cat_counts)))
    ax4.set_xticklabels(cat_counts.index, fontsize=tick_fs)
    ax4.set_ylabel('Number of Tunes', fontsize=label_fs)
    ax4.set_xlabel('Bar Count Range', fontsize=label_fs)
    ax4.set_title('(d) Tune Length Categories', fontweight='bold', fontsize=title_fs, pad=10)
    ax4.tick_params(labelsize=tick_fs)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, max(cat_counts.values) * 1.18)
    for bar in bars_plot:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., h + max(cat_counts.values) * 0.01,
                 f'{int(h/1000):.0f}k' if h >= 1000 else f'{int(h)}',
                 ha='center', va='bottom', fontsize=annot_fs)

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
