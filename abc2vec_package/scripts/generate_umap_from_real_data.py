"""
Generate high-quality UMAP visualizations from REAL ABC2Vec embeddings.
Uses pre-computed test embeddings and actual metadata labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11

# Load REAL embeddings
print("Loading real test embeddings...")
embeddings = np.load('/Volumes/LLModels/Projects/ABC2VEC/results/clustering/test_embeddings.npy')
print(f"Embeddings shape: {embeddings.shape}")

# Load metadata
print("Loading metadata...")
metadata = pd.read_csv('/Volumes/LLModels/Projects/ABC2VEC/data/processed/metadata.csv')

# Filter metadata to test set only
test_metadata = metadata[metadata['tune_id'].str.startswith('test_')].copy()
test_metadata = test_metadata.sort_values('tune_id').reset_index(drop=True)
print(f"Test metadata shape: {test_metadata.shape}")

# Verify we have the right number of samples
assert len(test_metadata) == embeddings.shape[0], f"Mismatch: {len(test_metadata)} metadata vs {embeddings.shape[0]} embeddings"

# Extract tune type and mode labels, handle NaN values
test_metadata['tune_type'] = test_metadata['tune_type'].fillna('unknown')
test_metadata['mode'] = test_metadata['mode'].fillna('unknown')

tune_type_labels = test_metadata['tune_type'].values
mode_labels = test_metadata['mode'].values

# Get unique tune types and modes (in order of frequency)
tune_types_counts = test_metadata['tune_type'].value_counts()
tune_types = tune_types_counts.index.tolist()
modes_counts = test_metadata['mode'].value_counts()
modes = modes_counts.index.tolist()

print(f"\nTune type distribution:")
for tt in tune_types:
    count = tune_types_counts[tt]
    pct = count / len(test_metadata) * 100
    print(f"  {tt}: {count} ({pct:.1f}%)")

print(f"\nMode distribution:")
for m in modes:
    count = modes_counts[m]
    pct = count / len(test_metadata) * 100
    print(f"  {m}: {count} ({pct:.1f}%)")

# Create label mappings
tune_type_to_idx = {tt: i for i, tt in enumerate(tune_types)}
mode_to_idx = {m: i for i, m in enumerate(modes)}

tune_type_indices = np.array([tune_type_to_idx[tt] for tt in tune_type_labels])
mode_indices = np.array([mode_to_idx[m] for m in mode_labels])

# Compute UMAP projection
print("\nComputing UMAP projection...")
reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='cosine', random_state=42, spread=1.5)
embedding_2d = reducer.fit_transform(embeddings)
print(f"UMAP projection shape: {embedding_2d.shape}")

# Color schemes
tune_type_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
mode_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3', '#ffff33']

# ========== UMAP by Tune Type ==========
fig, ax = plt.subplots(figsize=(10, 8))

for i, ttype in enumerate(tune_types):
    mask = tune_type_indices == i
    ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
               c=tune_type_colors[i % len(tune_type_colors)],
               label=ttype.replace('_', ' ').title(),
               s=120,
               alpha=0.7,
               edgecolors='white',
               linewidths=0.5)

ax.set_xlabel('UMAP Dimension 1', fontsize=14, fontweight='bold')
ax.set_ylabel('UMAP Dimension 2', fontsize=14, fontweight='bold')
ax.set_title('UMAP - Colored by Tune Type', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
          fontsize=11, markerscale=1.2)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
output_path = '/Volumes/LLModels/Projects/ABC2VEC/figures/umap_tune_type_hq.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved tune type UMAP to: {output_path}")

output_path_pdf = '/Volumes/LLModels/Projects/ABC2VEC/figures/umap_tune_type_hq.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved tune type UMAP PDF to: {output_path_pdf}")

plt.close()

# ========== UMAP by Mode ==========
fig, ax = plt.subplots(figsize=(10, 8))

for i, mode in enumerate(modes):
    mask = mode_indices == i
    ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
               c=mode_colors[i % len(mode_colors)],
               label=mode.title(),
               s=120,
               alpha=0.7,
               edgecolors='white',
               linewidths=0.5)

ax.set_xlabel('UMAP Dimension 1', fontsize=14, fontweight='bold')
ax.set_ylabel('UMAP Dimension 2', fontsize=14, fontweight='bold')
ax.set_title('UMAP - Colored by Mode', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
          fontsize=11, markerscale=1.2)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
output_path = '/Volumes/LLModels/Projects/ABC2VEC/figures/umap_mode_hq.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved mode UMAP to: {output_path}")

output_path_pdf = '/Volumes/LLModels/Projects/ABC2VEC/figures/umap_mode_hq.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved mode UMAP PDF to: {output_path_pdf}")

plt.close()

# ========== Combined Figure (Side by Side) ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Left: Tune Type
for i, ttype in enumerate(tune_types):
    mask = tune_type_indices == i
    ax1.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
               c=tune_type_colors[i % len(tune_type_colors)],
               label=ttype.replace('_', ' ').title(),
               s=100,
               alpha=0.7,
               edgecolors='white',
               linewidths=0.5)

ax1.set_xlabel('UMAP Dimension 1', fontsize=24, fontweight='bold')
ax1.set_ylabel('UMAP Dimension 2', fontsize=24, fontweight='bold')
ax1.set_title('(a) Colored by Tune Type', fontsize=26, fontweight='bold', pad=15)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=18)
ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax1.set_xticks([])
ax1.set_yticks([])

# Right: Mode
for i, mode in enumerate(modes):
    mask = mode_indices == i
    ax2.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
               c=mode_colors[i % len(mode_colors)],
               label=mode.title(),
               s=100,
               alpha=0.7,
               edgecolors='white',
               linewidths=0.5)

ax2.set_xlabel('UMAP Dimension 1', fontsize=24, fontweight='bold')
ax2.set_ylabel('UMAP Dimension 2', fontsize=24, fontweight='bold')
ax2.set_title('(b) Colored by Mode', fontsize=26, fontweight='bold', pad=15)
ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=18)
ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax2.set_xticks([])
ax2.set_yticks([])

plt.tight_layout()

output_path = '/Volumes/LLModels/Projects/ABC2VEC/figures/umap_combined.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved combined UMAP to: {output_path}")

output_path_pdf = '/Volumes/LLModels/Projects/ABC2VEC/figures/umap_combined.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved combined UMAP PDF to: {output_path_pdf}")

print("\n✅ All UMAP visualizations generated successfully from REAL data!")
print(f"   - Used {embeddings.shape[0]} real test embeddings (128-dim)")
print(f"   - Loaded actual tune_type and mode labels from metadata")
print(f"   - Marker size: 120, white edges, 300 DPI")
print(f"   - Saved PNG and PDF versions")
