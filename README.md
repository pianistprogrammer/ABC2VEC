# ABC2Vec - Self-Supervised Representation Learning for Folk Music

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A transformer-based model for learning melodic representations from ABC notation using self-supervised learning. Achieves 78.4% tune type classification and 80.8% with optimal objective combination.

## Overview

ABC2Vec learns dense vector embeddings of folk music tunes that capture melodic similarity, structure, and musical properties through bar-level transformer encoding.

**Key Results:**
- **Original Model**: 78.4% tune type, 78.5% mode classification
- **Best Ablation (MMM+SCL)**: 80.8% tune type (outperforms full model!)
- **Ablation Range**: 43.8% (TI-only) to 80.8% (MMM+SCL)

## Quick Start

### Installation

```bash
git clone https://github.com/pianistprogrammer/ABC2VEC.git
cd ABC2VEC/abc2vec

# Install dependencies
uv venv && source .venv/bin/activate  # Or: python -m venv .venv
uv pip install -e ".[all]"            # Or: pip install -e ".[all]"
```

### Basic Usage

```python
from abc2vec import ABC2VecModel, ABCVocabulary, BarPatchifier
import torch

# Load vocabulary and model (IMPORTANT: Always load saved vocab!)
vocab = ABCVocabulary.load('data/processed/vocab.json')  # 98 tokens
patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)
model = ABC2VecModel.load_pretrained('checkpoints/best_model.pt')

# Encode a tune
abc_tune = "D2 EF | G2 AB | c2 dc | BAGF |"
patches = patchifier.patchify(abc_tune)

with torch.no_grad():
    embedding = model.get_embedding(
        patches['bar_indices'].unsqueeze(0),
        patches['char_mask'].unsqueeze(0),
        patches['bar_mask'].unsqueeze(0)
    )
```

## Training & Evaluation

### 1. Data Preparation

```bash
cd abc2vec
python scripts/run_data_pipeline.py --output_dir data/processed
```

This creates:
- `train.parquet` (198,893 tunes)
- `val.parquet` (6,463 tunes)
- `test.parquet` (2,162 tunes)
- `vocab.json` (98-token vocabulary)

### 2. Model Training

```bash
# Full model (MMM + TI + SCL)
python scripts/run_training.py \
    --data_dir data/processed \
    --output_dir checkpoints \
    --epochs 40 \
    --batch_size 32 \
    --grad_accum 4 \
    --lr 1e-4 \
    --lambda_mmm 1.0 \
    --lambda_ti 0.5 \
    --lambda_scl 0.5

# Or train specific ablation variant (see Ablation Study section)
```

### 3. Evaluation

#### Complete Ablation Study (Recommended)

```bash
python scripts/run_ablation_study.py \
    --data_dir data/processed \
    --device mps \
    --output_dir results/ablation_study
```

Evaluates all models and generates comparison tables/figures.

#### Individual Evaluations

```bash
# Tune type classification
python scripts/run_evaluate_tune_type.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data/processed

# Mode classification
python scripts/run_evaluate_mode.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data/processed

# Clustering quality
python scripts/run_evaluate_clustering.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data/processed

# Linear probing (all properties)
python scripts/run_evaluate_linear_probing.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data/processed
```

## Ablation Study Results

### Pre-training Objectives

1. **MMM (Masked Music Modeling)**: Reconstruction objective, predicts masked bars
2. **TI (Transposition Invariance)**: Contrastive learning for pitch-invariant representations
3. **SCL (Section Contrastive Learning)**: Distinguishes tune sections (A vs B)

### Performance by Objective Combination

| Model | Tune Type | Mode | Key Root | Clustering |
|-------|-----------|------|----------|------------|
| **MMM+SCL** | **80.8%** ✅ | 80.7% | 81.0% | 0.061 |
| TI+SCL | 77.3% | 78.9% | 51.4% | 0.021 |
| MMM+TI+SCL (Full) | 78.1% | 78.1% | 57.0% | 0.024 |
| MMM-only | 66.8% | 81.4% | 76.8% | 0.063 |
| SCL-only | 66.4% | 79.4% | 72.5% | **0.112** ✅ |
| MMM+TI | 59.9% | 77.9% | 33.6% | 0.013 |
| TI-only | 43.8% ❌ | 78.0% | 28.3% | 0.011 |

### Key Findings

1. **SCL is critical** - All top performers include SCL
2. **MMM+SCL is optimal** for tune type classification (80.8% > 78.1% full model)
3. **TI has mixed effects** - Helps with SCL, hurts with MMM
4. **More objectives ≠ better** - Careful selection based on task is essential
5. **Clustering vs classification trade-off** - SCL-only has best clustering but moderate accuracy

### Model Selection Guide

- **Tune type/rhythm tasks**: Use MMM+SCL (80.8%)
- **Cross-key retrieval**: Use TI+SCL (77.3%, low key dependence)
- **Balanced performance**: Use full model MMM+TI+SCL (78.1%)
- **Exploratory clustering**: Use SCL-only (silhouette 0.112)

## Training Ablation Models

```bash
# MMM+SCL (best for tune type)
bash train_ablation_4_mmm_scl.sh

# TI+SCL (best for key-invariance)
bash train_ablation_5_ti_scl.sh

# Full model
bash train_ablation_3_mmm_ti_scl.sh

# Individual objectives
bash train_ablation_1_mmm_only.sh
bash train_ablation_2_ti_only.sh
bash train_ablation_6_scl_only.sh
bash train_ablation_7_mmm_ti.sh
```

All scripts use identical hyperparameters (40 epochs, max 40,000 steps).

## ⚠️ Critical: Vocabulary Loading

**ALWAYS load the saved vocabulary** when evaluating models:

```python
# ✅ CORRECT (98 tokens)
vocab = ABCVocabulary.load("data/processed/vocab.json")

# ❌ WRONG
vocab = ABCVocabulary()  # Default vocab
```

All official evaluation scripts automatically load the correct vocabulary.

**Verification:**
```python
print(vocab.size)  # Should output: 98
assert vocab.size == 98, "Wrong vocabulary!"
```

If you get ~46% tune type accuracy, the vocabulary is WRONG!

## Model Architecture

- **Encoder**: 6 layers, 256-dim, 8 attention heads
- **Input**: Bar-level patches (max 64 bars × 64 chars)
- **Output**: 128-dim embedding
- **Vocabulary**: 98 ABC notation tokens
- **Training**: 40 epochs, ~40,000 steps, AdamW optimizer

## Project Structure

```
ABC2VEC/
├── abc2vec/
│   ├── core/
│   │   ├── data/              # Data loading
│   │   ├── model/             # Transformer model
│   │   ├── tokenizer/         # ABC tokenization
│   │   └── training/          # Training loop
│   ├── scripts/
│   │   ├── run_training.py              # Model training
│   │   ├── run_ablation_study.py        # Complete ablation evaluation
│   │   ├── run_evaluate_tune_type.py    # Tune type eval
│   │   ├── run_evaluate_mode.py         # Mode eval
│   │   ├── run_evaluate_clustering.py   # Clustering eval
│   │   ├── run_evaluate_linear_probing.py  # Multi-property eval
│   │   └── run_generate_ablation_figures.py  # Figure generation
│   └── pyproject.toml
├── checkpoints/
│   ├── best_model.pt          # Original full model (78.4% tune type)
│   ├── model_config.json
│   └── ablation/
│       ├── mmm_only/
│       ├── ti_only/
│       ├── scl_only/
│       ├── mmm_ti/
│       ├── mmm_scl/          # Best variant (80.8%)
│       ├── ti_scl/
│       └── mmm_ti_scl/       # Full model
├── data/processed/
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   └── vocab.json            # 98-token vocabulary (ESSENTIAL!)
├── results/ablation_study/
│   ├── ablation_comparison.csv
│   └── ablation_comparison.json
├── Paper/
│   ├── abc2vec_paper.tex
│   └── figures/
└── train_ablation_*.sh       # Training scripts for each variant
```

## Reproducing Paper Results

### Section 4: Main Results (Original Model)

```bash
# Train original model
python abc2vec/scripts/run_training.py \
    --data_dir data/processed \
    --output_dir checkpoints \
    --epochs 40 \
    --lambda_mmm 1.0 --lambda_ti 0.5 --lambda_scl 0.5

# Evaluate
python abc2vec/scripts/run_evaluate_tune_type.py --checkpoint checkpoints/best_model.pt
python abc2vec/scripts/run_evaluate_clustering.py --checkpoint checkpoints/best_model.pt
```

**Expected Results:**
- Tune type: 78.4% ± 1.2%
- Mode: 78.5% ± 1.6%
- Clustering (silhouette): 0.025

### Section 5: Ablation Study

```bash
# Train all 7 variants
bash train_ablation_1_mmm_only.sh
bash train_ablation_2_ti_only.sh
bash train_ablation_3_mmm_ti_scl.sh
bash train_ablation_4_mmm_scl.sh
bash train_ablation_5_ti_scl.sh
bash train_ablation_6_scl_only.sh
bash train_ablation_7_mmm_ti.sh

# Evaluate all variants + generate figures
python abc2vec/scripts/run_ablation_study.py \
    --data_dir data/processed \
    --device mps \
    --output_dir results/ablation_study

# Regenerate paper figures
python abc2vec/scripts/run_generate_ablation_figures.py \
    --results_dir results/ablation_study \
    --output_dir Paper/figures
```

## Citation

```bibtex
@article{abimbola2026abc2vec,
  title={ABC2Vec: Self-Supervised Representation Learning for Folk Music via Bar-Level Transformers},
  author={Abimbola, Jeremiah and Kostrzewa, Daniel and Benecki, Paweł},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details.

## Authors

- Jeremiah Abimbola (jeremiah.abimbola@polsl.pl)
- Daniel Kostrzewa (daniel.kostrzewa@polsl.pl)
- Paweł Benecki (pawel.benecki@polsl.pl)

Silesian University of Technology, Gliwice, Poland

## Acknowledgments

We thank The Session community for curating and maintaining the Irish traditional music archive that made this work possible.

---

## Troubleshooting

**Q: I get 46% tune type accuracy instead of 78%**
A: You're using the wrong vocabulary. Use `ABCVocabulary.load("data/processed/vocab.json")` not `ABCVocabulary()`

**Q: Which model should I use?**
A: For tune type classification, use `checkpoints/ablation/mmm_scl/best_model.pt` (80.8%). For general use, use `checkpoints/best_model.pt` (78.4%).

**Q: How do I train only MMM+SCL?**
A: Run `bash train_ablation_4_mmm_scl.sh` or use `--lambda_mmm 1.0 --lambda_ti 0.0 --lambda_scl 0.5`

**Q: Can I use different hyperparameters?**
A: Yes, but results may differ. Our ablation study uses standardized hyperparameters for fair comparison.

---

⭐ **If you find this project useful, please consider giving it a star!**
