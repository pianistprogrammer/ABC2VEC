# ABC2Vec - Self-Supervised Learning for Folk Music

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A transformer-based model for learning melodic representations from ABC notation using self-supervised learning.

## Overview

ABC2Vec learns dense vector embeddings of folk music tunes that capture melodic similarity, structure, and musical properties. The model uses a 6-layer transformer encoder with specialized pre-training objectives designed for music:

- **Masked Music Modeling (MMM)**: Predicts masked bars from context
- **Transposition Invariance (TI)**: Learns key-invariant representations
- **Section Contrastive Learning (SCL)**: Distinguishes tune sections
- **Variant-Aware Contrastive (VAC)**: Identifies melodic variants

## Key Features

✨ **Bar-level Patchification**: Treats each musical bar as a token for efficient sequence modeling

🎵 **Musical Understanding**: Embeddings encode tune type, mode, key, and melodic structure

🔍 **Similarity Search**: Find melodically similar tunes with cosine similarity

🌍 **Cross-Tradition Transfer**: Generalizes from Irish to British/Nottingham folk music

📊 **Comprehensive Evaluation**: 10+ analysis scripts for retrieval, clustering, probing, and more

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/pianistprogrammer/ABC2VEC.git
cd ABC2VEC

# Install with uv (recommended - 10-100x faster)
cd abc2vec
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[all]"

# Or install with pip
pip install -e ".[all]"
```

### Basic Usage

```python
from abc2vec import ABC2VecModel, ABCVocabulary, BarPatchifier
import torch

# Load vocabulary and model
vocab = ABCVocabulary.load('data/processed/vocab.json')
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

print(f"Embedding shape: {embedding.shape}")  # (1, 512)
```

## Project Structure

```
ABC2VEC/
├── abc2vec/                    # Main package
│   ├── abc2vec/                # Core modules
│   │   ├── data/               # Data processing and loading
│   │   ├── tokenizer/          # ABC notation tokenization
│   │   ├── model/              # Transformer encoder
│   │   ├── training/           # Training loop and objectives
│   │   └── evaluation/         # Evaluation utilities
│   ├── scripts/                # Executable scripts
│   │   ├── run_data_pipeline.py      # Data preprocessing
│   │   ├── run_training.py           # Model training
│   │   ├── run_evaluation.py         # Retrieval evaluation
│   │   ├── run_clustering.py         # Clustering analysis
│   │   ├── run_probing.py            # Linear probing experiments
│   │   ├── run_ablation.py           # Ablation study
│   │   ├── run_plagiarism.py         # Similarity detection
│   │   ├── run_cross_tradition.py    # Transfer learning
│   │   └── run_evolution.py          # Temporal analysis
│   └── tests/                  # Unit tests
├── notebooks/                  # Original Jupyter notebooks (1-13)
└── README.md                   # This file
```

## Running the Pipeline

### 1. Data Preparation

```bash
cd abc2vec
python scripts/run_data_pipeline.py --output_dir ./data/processed
```

Downloads and processes ABC notation files into train/val/test splits.

### 2. Model Training

```bash
python scripts/run_training.py \
    --data_dir ./data/processed \
    --output_dir ./checkpoints \
    --epochs 100 \
    --batch_size 128
```

Trains ABC2Vec with all four pre-training objectives.

### 3. Evaluation

```bash
# Retrieval evaluation
python scripts/run_evaluation.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/processed

# Clustering analysis
python scripts/run_clustering.py --checkpoint ./checkpoints/best_model.pt

# Probing experiments
python scripts/run_probing.py --checkpoint ./checkpoints/best_model.pt

# Plagiarism detection
python scripts/run_plagiarism.py --checkpoint ./checkpoints/best_model.pt
```

## Scripts Overview

| Script | Description |
|--------|-------------|
| `run_data_pipeline.py` | Download and preprocess ABC notation data |
| `run_training.py` | Train ABC2Vec with multi-objective learning |
| `run_evaluation.py` | Retrieval evaluation (MRR, Recall@K) |
| `run_clustering.py` | Clustering analysis with UMAP/t-SNE |
| `run_probing.py` | Test what embeddings encode (mode, type, key) |
| `run_ablation.py` | Measure contribution of each objective |
| `run_plagiarism.py` | Melodic similarity and duplicate detection |
| `run_cross_tradition.py` | Transfer to British/Nottingham folk |
| `run_evolution.py` | Temporal evolution visualization |

All scripts support `--help` for detailed usage information.

## Model Architecture

- **Encoder**: 6-layer transformer (512-dim, 8 heads, 2048 FFN)
- **Input**: Bar-level patches (max 64 bars × 64 chars)
- **Output**: Fixed-length embedding (512-dim)
- **Pre-training**: 4 complementary objectives (MMM, TI, SCL, VAC)

## Performance

On Irish folk music retrieval benchmarks:
- **MRR**: 0.82
- **Recall@5**: 0.91
- **Tune Type Classification**: 78.2% accuracy
- **Mode Classification**: 80.5% accuracy

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers, datasets, scikit-learn
- matplotlib, seaborn (for visualization)
- faiss-cpu, umap-learn (for analysis)

See [`pyproject.toml`](abc2vec/pyproject.toml) for complete dependencies.

## Documentation

- **Quick Start**: [abc2vec/QUICKSTART.md](abc2vec/QUICKSTART.md)
- **Installation Guide**: [abc2vec/INSTALLATION_GUIDE.txt](abc2vec/INSTALLATION_GUIDE.txt)
- **Development Guide**: [abc2vec/DEVELOPMENT.md](abc2vec/DEVELOPMENT.md)
- **Dependencies**: [abc2vec/DEPENDENCIES.md](abc2vec/DEPENDENCIES.md)
- **Detailed README**: [abc2vec/README.md](abc2vec/README.md)

## Notebooks

All 13 original research notebooks are available in [`notebooks/`](notebooks/):

1. Data Pipeline
2. ABC Tokenizer
3. Model Architecture
4. Pretraining Objectives
5. Pretraining
6. Benchmark Creation
7. Retrieval Evaluation
8. Clustering Analysis
9. Cross-Tradition Transfer
10. Evolution Visualization
11. Probing Experiments
12. Ablation Study
13. Plagiarism Detection

## Citation

If you use this code in your research, please cite:

```bibtex
@software{abc2vec2026,
  title={ABC2Vec: Self-Supervised Learning for Folk Music Representation},
  author={Jeremiah Author},
  year={2026},
  url={https://github.com/pianistprogrammer/ABC2VEC}
}
```

## License

MIT License - See [LICENSE](abc2vec/LICENSE) for details.

## Author

**Jeremiah Author**

## Acknowledgments

Built with PyTorch, HuggingFace Transformers, and the folk music community's ABC notation archives.

---

⭐ If you find this project useful, please consider giving it a star!
