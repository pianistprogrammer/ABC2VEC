#!/usr/bin/env python
"""
Generate ABC2Vec architecture diagram for the paper.

Creates a publication-quality figure showing the complete model architecture
from ABC notation input to final embeddings with training objectives.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'serif'

def draw_abc2vec_architecture(output_path):
    """Draw the complete ABC2Vec architecture."""

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme
    color_input = '#ecf0f1'
    color_tokenize = '#3498db'
    color_embed = '#9b59b6'
    color_transformer = '#e74c3c'
    color_pool = '#2ecc71'
    color_output = '#f39c12'
    color_objective = '#1abc9c'

    # ========== INPUT: ABC NOTATION ==========
    y_start = 9.0

    # ABC notation example
    abc_text = [
        'X:1',
        'T:The Kesh Jig',
        'M:6/8',
        'K:G',
        '|:GAG GAB|AGA ABd|...'
    ]

    abc_box = FancyBboxPatch((0.5, y_start-0.5), 2.5, 1.0,
                              boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor=color_input,
                              linewidth=2)
    ax.add_patch(abc_box)

    for i, line in enumerate(abc_text):
        ax.text(1.75, y_start - 0.1 - i*0.15, line, ha='center', va='center',
                fontsize=7, fontfamily='monospace', fontweight='bold')

    ax.text(1.75, y_start + 0.7, 'ABC Notation', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # ========== STEP 1: BAR PATCHIFICATION ==========
    y_patch = 7.5

    # Arrow
    arrow1 = FancyArrowPatch((1.75, y_start-0.5), (1.75, y_patch+0.8),
                             arrowstyle='->', mutation_scale=20, linewidth=2,
                             color='black')
    ax.add_patch(arrow1)

    ax.text(2.5, 8.2, 'Split by bars', ha='left', va='center',
            fontsize=8, style='italic', color='gray')

    # Bar patches
    bars_text = ['GAG GAB', 'AGA ABd', 'edd gdB', 'AGG ABd']
    bar_width = 0.55
    bar_spacing = 0.65
    bar_start_x = 0.2

    for i, bar in enumerate(bars_text):
        x_pos = bar_start_x + i * bar_spacing

        # Bar box
        bar_box = FancyBboxPatch((x_pos, y_patch), bar_width, 0.6,
                                  boxstyle="round,pad=0.03",
                                  edgecolor=color_tokenize,
                                  facecolor='white',
                                  linewidth=2)
        ax.add_patch(bar_box)

        ax.text(x_pos + bar_width/2, y_patch + 0.3, bar,
                ha='center', va='center', fontsize=7,
                fontfamily='monospace', color=color_tokenize, fontweight='bold')

        ax.text(x_pos + bar_width/2, y_patch - 0.25, f'Bar {i+1}',
                ha='center', va='center', fontsize=7, color='gray')

    # Continuation dots
    ax.text(bar_start_x + 4*bar_spacing + 0.2, y_patch + 0.3, '...', ha='center', va='center',
            fontsize=12, fontweight='bold', color=color_tokenize)

    ax.text(bar_start_x + 4*bar_spacing + 0.2, y_patch - 0.25, f'Bar K', ha='center', va='center',
            fontsize=7, color='gray')

    ax.text(1.75, y_patch + 1.0, 'Bar Patchification', ha='center', va='center',
            fontsize=10, fontweight='bold', bbox=dict(boxstyle='round',
            facecolor=color_tokenize, alpha=0.3, edgecolor=color_tokenize, linewidth=2))

    # ========== STEP 2: CHARACTER EMBEDDING ==========
    y_embed = 6.0

    # Arrows from bars to embeddings
    for i in range(4):
        x_pos = bar_start_x + bar_width/2 + i * bar_spacing
        arrow = FancyArrowPatch((x_pos, y_patch), (x_pos, y_embed+0.5),
                               arrowstyle='->', mutation_scale=15, linewidth=1.5,
                               color=color_embed, alpha=0.7)
        ax.add_patch(arrow)

    ax.text(bar_start_x + 4*bar_spacing + 0.5, 6.8, 'Char Embed', ha='left', va='center',
            fontsize=7, style='italic', color='gray')
    ax.text(bar_start_x + 4*bar_spacing + 0.5, 6.5, '+ Mean Pool', ha='left', va='center',
            fontsize=7, style='italic', color='gray')

    # Embedding vectors
    for i in range(4):
        x_pos = bar_start_x + i * bar_spacing

        # Embedding as matrix
        emb_box = Rectangle((x_pos, y_embed), bar_width, 0.4,
                            edgecolor=color_embed, facecolor=color_embed,
                            linewidth=2, alpha=0.7)
        ax.add_patch(emb_box)

        # Grid lines
        for j in range(4):
            ax.plot([x_pos, x_pos + bar_width],
                   [y_embed + j*0.1, y_embed + j*0.1],
                   color='white', linewidth=0.5, alpha=0.5)

        ax.text(x_pos + bar_width/2, y_embed - 0.25, f'h_{i+1}',
                ha='center', va='center', fontsize=8,
                style='italic', color=color_embed, fontweight='bold')

    ax.text(bar_start_x + 4*bar_spacing + 0.2, y_embed + 0.2, '...', ha='center', va='center',
            fontsize=12, fontweight='bold', color=color_embed)

    ax.text(bar_start_x + 4*bar_spacing + 0.2, y_embed - 0.25, f'h_K', ha='center', va='center',
            fontsize=8, style='italic', color=color_embed, fontweight='bold')

    ax.text(1.75, y_embed + 0.8, 'Bar Embeddings + Positional Encoding',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color_embed, alpha=0.3,
            edgecolor=color_embed, linewidth=2))

    # ========== STEP 3: TRANSFORMER ENCODER ==========
    y_trans = 3.5

    # Arrow to transformer
    arrow3 = FancyArrowPatch((1.75, y_embed), (1.75, y_trans+1.3),
                             arrowstyle='->', mutation_scale=20, linewidth=2,
                             color='black')
    ax.add_patch(arrow3)

    # Transformer block
    trans_width = 3.0
    trans_height = 1.2
    trans_x = 0.2
    trans_center_x = trans_x + trans_width/2

    trans_box = FancyBboxPatch((trans_x, y_trans), trans_width, trans_height,
                                boxstyle="round,pad=0.1",
                                edgecolor=color_transformer,
                                facecolor=color_transformer,
                                linewidth=3, alpha=0.2)
    ax.add_patch(trans_box)

    # Transformer layers (stacked)
    layer_y = [y_trans + 0.2, y_trans + 0.5, y_trans + 0.8]
    for i, ly in enumerate(layer_y):
        layer_box = Rectangle((trans_x + 0.2, ly), trans_width - 0.4, 0.2,
                              edgecolor=color_transformer,
                              facecolor='white',
                              linewidth=2)
        ax.add_patch(layer_box)

        # Multi-head attention icon (centered)
        num_heads = 6
        head_spacing = (trans_width - 0.6) / num_heads
        for j in range(num_heads):
            head_x = trans_x + 0.3 + j * head_spacing
            circle = Circle((head_x, ly + 0.1), 0.05,
                           facecolor=color_transformer, edgecolor='white',
                           linewidth=0.5)
            ax.add_patch(circle)

        if i == 1:
            ax.text(trans_x + trans_width + 0.2, ly + 0.1,
                   f'Layer {i+1}', ha='left', va='center',
                   fontsize=7, color=color_transformer)

    ax.text(trans_center_x, y_trans + trans_height + 0.3,
            'Transformer Encoder', ha='center', va='center',
            fontsize=11, fontweight='bold', color=color_transformer)

    ax.text(trans_center_x, y_trans - 0.3,
            '(L=6 layers, H=8 heads, d=128)', ha='center', va='center',
            fontsize=8, color='gray', style='italic')

    # ========== STEP 4: MEAN POOLING ==========
    y_pool = 2.0

    # Arrow to pooling
    arrow4 = FancyArrowPatch((1.75, y_trans), (1.75, y_pool+0.4),
                             arrowstyle='->', mutation_scale=20, linewidth=2,
                             color='black')
    ax.add_patch(arrow4)

    ax.text(2.5, 2.7, 'Mean Pooling', ha='left', va='center',
            fontsize=8, style='italic', color='gray')

    # Pooling visualization
    pool_box = FancyBboxPatch((0.75, y_pool), 2.0, 0.3,
                               boxstyle="round,pad=0.05",
                               edgecolor=color_pool, facecolor=color_pool,
                               linewidth=2, alpha=0.7)
    ax.add_patch(pool_box)

    ax.text(1.75, y_pool + 0.15, 'z ∈ ℝ¹²⁸', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    ax.text(1.75, y_pool + 0.65, 'Tune Embedding', ha='center', va='center',
            fontsize=10, fontweight='bold', color=color_pool)

    # ========== STEP 5: OUTPUT ==========
    y_out = 0.5

    # Arrow to output
    arrow5 = FancyArrowPatch((1.75, y_pool), (1.75, y_out+0.5),
                             arrowstyle='->', mutation_scale=20, linewidth=2,
                             color='black')
    ax.add_patch(arrow5)

    # Output embedding
    out_box = FancyBboxPatch((0.75, y_out), 2.0, 0.4,
                              boxstyle="round,pad=0.05",
                              edgecolor=color_output, facecolor=color_output,
                              linewidth=2, alpha=0.3)
    ax.add_patch(out_box)

    ax.text(1.75, y_out + 0.2, 'Dense Representation', ha='center', va='center',
            fontsize=10, fontweight='bold', color=color_output)

    # ========== TRAINING OBJECTIVES (RIGHT SIDE) ==========

    # Title
    ax.text(8.5, 9.5, 'Self-Supervised Training Objectives', ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='lightgray', edgecolor='black', linewidth=2))

    # === OBJECTIVE 1: Masked Music Modeling ===
    y_mmm = 7.5

    mmm_box = FancyBboxPatch((5.5, y_mmm-0.2), 6.0, 2.0,
                              boxstyle="round,pad=0.15",
                              edgecolor=color_objective, facecolor=color_objective,
                              linewidth=2, alpha=0.15)
    ax.add_patch(mmm_box)

    ax.text(8.5, y_mmm + 1.6, '(1) Masked Music Modeling (MMM)',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=color_objective)

    # Show masking process - aligned with consistent spacing
    bars_mmm = ['GAG', '[MASK]', 'edd', 'AGG']
    mmm_bar_width = 0.9
    mmm_start_x = 5.8
    mmm_spacing = 1.3

    for i, bar in enumerate(bars_mmm):
        x_pos = mmm_start_x + i * mmm_spacing

        bar_box = Rectangle((x_pos, y_mmm + 0.8), mmm_bar_width, 0.4,
                            edgecolor='black' if bar != '[MASK]' else 'red',
                            facecolor='white' if bar != '[MASK]' else '#ffcccc',
                            linewidth=2 if bar != '[MASK]' else 3)
        ax.add_patch(bar_box)

        ax.text(x_pos + mmm_bar_width/2, y_mmm + 1.0, bar, ha='center', va='center',
                fontsize=8, fontfamily='monospace',
                fontweight='bold' if bar == '[MASK]' else 'normal',
                color='red' if bar == '[MASK]' else 'black')

    # Arrow to prediction - centered
    pred_center_x = mmm_start_x + 1.5 * mmm_spacing
    arrow_mmm = FancyArrowPatch((pred_center_x, y_mmm + 0.7), (pred_center_x, y_mmm + 0.3),
                               arrowstyle='->', mutation_scale=15, linewidth=2,
                               color='red')
    ax.add_patch(arrow_mmm)

    # Predicted bar - centered
    pred_box = Rectangle((pred_center_x - 0.7, y_mmm - 0.1), 1.4, 0.35,
                         edgecolor='green', facecolor='#ccffcc',
                         linewidth=2, linestyle='--')
    ax.add_patch(pred_box)

    ax.text(pred_center_x, y_mmm + 0.07, 'AGA ABd', ha='center', va='center',
            fontsize=8, fontfamily='monospace', fontweight='bold', color='green')

    ax.text(mmm_start_x + 4*mmm_spacing - 0.3, y_mmm + 0.5, 'Reconstruct\nmasked bars', ha='left', va='center',
            fontsize=7, style='italic', color='gray')

    ax.text(8.5, y_mmm - 0.5, 'ℒ_MMM = -log p(bar_masked | context)',
            ha='center', va='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', linewidth=1))

    # === OBJECTIVE 2: Transposition Invariance ===
    y_ti = 4.5

    ti_box = FancyBboxPatch((5.5, y_ti-0.3), 6.0, 2.5,
                             boxstyle="round,pad=0.15",
                             edgecolor=color_objective, facecolor=color_objective,
                             linewidth=2, alpha=0.15)
    ax.add_patch(ti_box)

    ax.text(8.5, y_ti + 2.0, '(2) Transposition Invariance (TI)',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=color_objective)

    # Original tune
    ax.text(6.5, y_ti + 1.4, 'Original (D major)', ha='center', va='center',
            fontsize=8, fontweight='bold')

    orig_emb = FancyBboxPatch((5.8, y_ti + 0.8), 1.4, 0.4,
                               boxstyle="round,pad=0.05",
                               edgecolor='blue', facecolor='lightblue',
                               linewidth=2)
    ax.add_patch(orig_emb)

    ax.text(6.5, y_ti + 1.0, 'z', ha='center', va='center',
            fontsize=10, fontweight='bold', style='italic')

    # Transposed tune
    ax.text(10.5, y_ti + 1.4, 'Transposed (G major)', ha='center', va='center',
            fontsize=8, fontweight='bold')

    trans_emb = FancyBboxPatch((9.8, y_ti + 0.8), 1.4, 0.4,
                                boxstyle="round,pad=0.05",
                                edgecolor='blue', facecolor='lightblue',
                                linewidth=2)
    ax.add_patch(trans_emb)

    ax.text(10.5, y_ti + 1.0, 'z⁺', ha='center', va='center',
            fontsize=10, fontweight='bold', style='italic')

    # Connection line showing similarity
    ax.plot([7.2, 9.8], [y_ti + 1.0, y_ti + 1.0],
            color='green', linewidth=3, linestyle='-', marker='o',
            markersize=8, markerfacecolor='green')

    ax.text(8.5, y_ti + 1.25, 'High Similarity', ha='center', va='center',
            fontsize=8, fontweight='bold', color='green')

    # Negative examples - properly aligned
    neg_width = 0.6
    neg_height = 0.3
    neg_y = y_ti + 0.1
    neg_start_x = 6.5
    neg_spacing = 0.8

    for idx, label in enumerate(['z₁⁻', 'z₂⁻', 'z₃⁻']):
        x_pos = neg_start_x + idx * neg_spacing

        neg_emb = FancyBboxPatch((x_pos, neg_y), neg_width, neg_height,
                                   boxstyle="round,pad=0.03",
                                   edgecolor='gray', facecolor='lightgray',
                                   linewidth=1.5, alpha=0.5)
        ax.add_patch(neg_emb)

        ax.text(x_pos + neg_width/2, neg_y + neg_height/2, label, ha='center', va='center',
                fontsize=7, style='italic', color='gray')

    ax.text(neg_start_x + 1.5*neg_spacing, neg_y - 0.25, 'Negative samples (other tunes)',
            ha='center', va='center', fontsize=7, color='gray', style='italic')

    # Repulsion lines - from original embedding to negatives
    orig_emb_center_x = 6.5
    orig_emb_center_y = y_ti + 0.8

    for idx in range(3):
        neg_center_x = neg_start_x + neg_width/2 + idx * neg_spacing
        neg_center_y = neg_y + neg_height

        ax.plot([orig_emb_center_x, neg_center_x],
                [orig_emb_center_y, neg_center_y],
                color='red', linewidth=1.5, linestyle='--', alpha=0.5)

    ax.text(neg_start_x - 0.8, neg_y + neg_height/2, 'Low\nSimilarity',
            ha='center', va='center', fontsize=7, color='red', fontweight='bold')

    # Loss formula
    ax.text(8.5, y_ti - 0.55, 'ℒ_TI = -log[exp(sim(z,z⁺)/τ) / Σ exp(sim(z,z_j)/τ)]',
            ha='center', va='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', linewidth=1))

    # === Combined Loss ===
    y_loss = 1.5

    loss_box = FancyBboxPatch((5.5, y_loss), 6.0, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#ffffcc',
                               linewidth=2)
    ax.add_patch(loss_box)

    ax.text(8.5, y_loss + 0.55, 'Combined Training Objective',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.text(8.5, y_loss + 0.25, 'ℒ = λ_MMM · ℒ_MMM + λ_TI · ℒ_TI',
            ha='center', va='center', fontsize=10, fontweight='bold',
            style='italic')

    ax.text(8.5, y_loss - 0.25, '(λ_MMM = 1.0, λ_TI = 0.5)',
            ha='center', va='center', fontsize=8, color='gray', style='italic')

    # ========== ANNOTATIONS ==========

    # Main flow annotation - adjusted position
    ax.annotate('', xy=(4.2, 5.0), xytext=(5.3, 5.0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='gray', linestyle='--'))

    ax.text(4.75, 5.4, 'Training', ha='center', va='center',
            fontsize=9, color='gray', style='italic', fontweight='bold')

    # Dataset info - aligned with output
    ax.text(trans_center_x, 0.05, '211,524 Irish folk tunes', ha='center', va='center',
            fontsize=8, color='gray', style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf', pad_inches=0.2)
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate ABC2Vec architecture diagram")
    parser.add_argument("--output_dir", type=str, default="./figures",
                       help="Output directory for figure")

    args = parser.parse_args()

    from pathlib import Path
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("ABC2VEC ARCHITECTURE DIAGRAM GENERATION")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}\n")

    draw_abc2vec_architecture(str(output_path / "architecture_diagram.pdf"))

    print("\n" + "="*80)
    print("✓ Architecture diagram generated successfully!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - architecture_diagram.pdf")
    print(f"  - architecture_diagram.png")
    print()


if __name__ == "__main__":
    main()
