#!/usr/bin/env python
"""
Generate ABC2Vec architecture diagram - clean horizontal layout.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_diagram():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Colors
    color_blue = '#b3d9ff'
    color_purple = '#9b8bc4'
    color_red = '#d9534f'
    color_red_light = '#e57373'
    color_green = '#5cb85c'
    color_border = '#2c3e50'

    # ========== ABC NOTATION ==========
    abc_x, abc_y = 0.3, 3.5
    ax.add_patch(FancyBboxPatch((abc_x, abc_y), 2.2, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor=color_blue, edgecolor=color_border, lw=2.5))

    ax.text(abc_x + 1.1, abc_y + 2.25, 'ABC Notation', ha='center', va='center',
            fontsize=12, fontweight='bold', color=color_border)

    abc_lines = ['X:1', 'T:The Kesh Jig', 'M:6/8', 'K:G', '|:GAG GAB|AGA ABd|', '|...']
    for i, line in enumerate(abc_lines):
        ax.text(abc_x + 0.2, abc_y + 1.8 - i*0.3, line, ha='left', va='center',
                fontsize=9, fontfamily='monospace', color=color_border)

    # Arrow to Bar Patchification
    ax.annotate("", xy=(2.8, abc_y + 1.25), xytext=(abc_x + 2.2, abc_y + 1.25),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='black'))

    # ========== BAR PATCHIFICATION ==========
    patch_x, patch_y = 2.8, 4.8

    ax.text(patch_x + 1.8, patch_y + 0.6, 'Bar Patchification', ha='center', va='center',
            fontsize=11, fontweight='bold', color=color_border)

    # Bar boxes
    bar_labels = ['Bar 1', 'Bar 2', 'Bar 3', 'Bar K']
    bar_width = 0.85
    bar_spacing = 0.9

    for i, label in enumerate(bar_labels):
        b_x = patch_x + i * bar_spacing
        ax.add_patch(FancyBboxPatch((b_x, patch_y - 0.15), bar_width, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor='white', edgecolor=color_border, lw=2))
        ax.text(b_x + bar_width/2, patch_y + 0.15, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color_border)

    # Character Embeddings label
    ax.text(patch_x + 1.8, patch_y - 0.75, 'Character Embeddings', ha='center', va='center',
            fontsize=10, fontweight='bold', color=color_border)
    ax.text(patch_x + 1.8, patch_y - 1.05, '+ Positional Encoding', ha='center', va='center',
            fontsize=9, style='italic', color='gray')

    # h1, h2, h3, hK boxes
    h_y = patch_y - 1.65
    for i, label in enumerate(['h₁', 'h₂', 'h₃', 'h_K']):
        h_x = patch_x + i * bar_spacing
        ax.add_patch(FancyBboxPatch((h_x, h_y), bar_width, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color_purple, edgecolor=color_border, lw=2))
        ax.text(h_x + bar_width/2, h_y + 0.25, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', style='italic')

    # Arrow to Transformer
    ax.annotate("", xy=(6.5, abc_y + 1.25), xytext=(patch_x + 3.6, abc_y + 1.25),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='black'))

    # ========== TRANSFORMER ENCODER ==========
    trans_x, trans_y = 6.5, 3.2
    trans_w, trans_h = 2.5, 3.0

    ax.add_patch(FancyBboxPatch((trans_x, trans_y), trans_w, trans_h,
                                boxstyle="round,pad=0.1",
                                facecolor=color_red, edgecolor=color_border, lw=2.5))

    ax.text(trans_x + trans_w/2, trans_y + trans_h - 0.3, 'Transformer Encoder',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Self-Attention layers (stacked)
    layer_colors = [color_red_light, color_red_light, color_red_light]
    for i, lc in enumerate(layer_colors):
        ly = trans_y + 0.3 + i * 0.7
        ax.add_patch(FancyBboxPatch((trans_x + 0.3, ly), trans_w - 0.6, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=lc, edgecolor='white', lw=2))
        ax.text(trans_x + trans_w/2, ly + 0.25, 'Self-Attention', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        # Downward arrow between layers
        if i < len(layer_colors) - 1:
            ax.annotate("", xy=(trans_x + trans_w/2, ly + 0.5),
                       xytext=(trans_x + trans_w/2, ly + 0.65),
                       arrowprops=dict(arrowstyle="->", lw=2, color='white'))

    # Arrow from embeddings to transformer (going up)
    arrow_start_x = patch_x + 1.8
    ax.plot([arrow_start_x, arrow_start_x, trans_x + trans_w/2, trans_x + trans_w/2],
            [h_y, trans_y - 0.3, trans_y - 0.3, trans_y],
            color='black', lw=2.5)
    ax.annotate("", xy=(trans_x + trans_w/2, trans_y),
               xytext=(trans_x + trans_w/2, trans_y - 0.2),
               arrowprops=dict(arrowstyle="->", lw=2.5, color='black'))

    # Arrow to Mean Pooling
    ax.annotate("", xy=(9.5, abc_y + 1.25), xytext=(trans_x + trans_w, abc_y + 1.25),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='black'))

    # ========== MEAN POOLING ==========
    pool_x, pool_y = 9.5, 4.2
    ax.add_patch(FancyBboxPatch((pool_x, pool_y), 1.8, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=color_green, edgecolor=color_border, lw=2.5))

    ax.text(pool_x + 0.9, pool_y + 0.75, 'Mean Pooling', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Arrow to Training Objectives
    ax.annotate("", xy=(11.8, abc_y + 1.25), xytext=(pool_x + 1.8, abc_y + 1.25),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='black'))

    # ========== TRAINING OBJECTIVES ==========
    obj_x, obj_y = 11.8, 2.5

    # Container box
    ax.add_patch(FancyBboxPatch((obj_x, obj_y), 3.8, 4.5,
                                boxstyle="round,pad=0.15",
                                facecolor='white', edgecolor=color_border, lw=2.5))

    ax.text(obj_x + 1.9, obj_y + 4.2, 'Training Objectives', ha='center', va='center',
            fontsize=12, fontweight='bold', color=color_border)

    # Masked Music Modeling box
    mmm_y = obj_y + 2.5
    ax.add_patch(FancyBboxPatch((obj_x + 0.3, mmm_y), 3.2, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=color_green, edgecolor=color_border, lw=2.5))

    ax.text(obj_x + 1.9, mmm_y + 0.75, 'Masked Music', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(obj_x + 1.9, mmm_y + 0.4, 'Modeling', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Transposition Invariance box
    ti_y = obj_y + 0.8
    ax.add_patch(FancyBboxPatch((obj_x + 0.3, ti_y), 3.2, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=color_green, edgecolor=color_border, lw=2.5))

    ax.text(obj_x + 1.9, ti_y + 0.75, 'Transposition', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(obj_x + 1.9, ti_y + 0.4, 'Invariance', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    plt.tight_layout()
    return fig

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate ABC2Vec architecture diagram")
    parser.add_argument("--output_dir", type=str, default="./figures",
                       help="Output directory for figure")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("ABC2VEC ARCHITECTURE DIAGRAM GENERATION")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}\n")

    fig = draw_diagram()

    pdf_path = output_path / "architecture_diagram.pdf"
    png_path = output_path / "architecture_diagram.png"

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf', pad_inches=0.2)
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    print("\n" + "="*80)
    print("✓ Architecture diagram generated successfully!")
    print("="*80)
    print()

if __name__ == "__main__":
    main()
