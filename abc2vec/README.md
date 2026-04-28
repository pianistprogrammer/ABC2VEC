# ABC2Vec Package - Self-Supervised Folk Music Representation Learning

A production-ready implementation of ABC2Vec for learning dense embeddings of folk music from ABC notation using transformer-based self-supervised learning.

## Quick Start

### Installation

```bash
cd abc2vec

# With uv (recommended - 10-100x faster)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[all]"

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

### Basic Usage

```python
from abc2vec import ABC2VecModel, ABCVocabulary, BarPatchifier
import torch

# Load vocabulary (CRITICAL: Always load saved vocab!)
vocab = ABCVocabulary.load('data/processed/vocab.json')  # 98 tokens
patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)

# Load model
model = ABC2VecModel.load_pretrained('checkpoints/best_model.pt')
model.eval()

# Encode a tune
abc_tune = "D2 EF | G2 AB | c2 dc | BAGF |"
patches = patchifier.patchify(abc_tune)

with torch.no_grad():
    embedding = model.get_embedding(
        patches['bar_indices'].unsqueeze(0),
        patches['char_mask'].unsqueeze(0),
        patches['bar_mask'].unsqueeze(0)
    )

print(f"Embedding shape: {embedding.shape}")  # (1, 128)
```

## Core Components

### 1. Data Pipeline

```python
from abc2vec.data import load_processed_data

# Load train/val/test splits
train_df = load_processed_data('data/processed', 'train')  # 198,893 tunes
val_df = load_processed_data('data/processed', 'val')      # 6,463 tunes
test_df = load_processed_data('data/processed', 'test')    # 2,162 tunes
```

### 2. Tokenization

```python
from abc2vec.tokenizer import ABCVocabulary, BarPatchifier

# Load vocabulary (98 ABC notation tokens)
vocab = ABCVocabulary.load('data/processed/vocab.json')

# Create patchifier
patchifier = BarPatchifier(
    vocab,
    max_bar_length=64,  # Max chars per bar
    max_bars=64         # Max bars per tune
)

# Patchify ABC notation
abc_body = "D2 EF | G2 AB | c2 dc | BAGF |"
patches = patchifier.patchify(abc_body)
# Returns: {'bar_indices': Tensor, 'char_mask': Tensor, 'bar_mask': Tensor}
```

### 3. Model Architecture

```python
from abc2vec.model import ABC2VecModel, ABC2VecConfig

# Create model from config
config = ABC2VecConfig(
    vocab_size=98,
    d_model=256,
    n_layers=6,
    n_heads=8,
    d_embed=128,
    max_bars=64,
    max_bar_length=64
)

model = ABC2VecModel(config)

# Or load pretrained
model = ABC2VecModel.load_pretrained('checkpoints/best_model.pt')
```

### 4. Training

```python
from abc2vec.training import train_model

# Train with multi-objective learning
train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir='checkpoints',
    lambda_mmm=1.0,   # Masked Music Modeling
    lambda_ti=0.5,    # Transposition Invariance
    lambda_scl=0.5,   # Section Contrastive Learning
    epochs=40,
    batch_size=32,
    learning_rate=1e-4
)
```

## Command-Line Scripts

### Data Processing

```bash
python scripts/data_pipeline.py \
    --output_dir data/processed \
    --num_workers 8
```

### Model Training

```bash
# Full model (MMM + TI + SCL)
python scripts/training.py \
    --data_dir data/processed \
    --output_dir checkpoints \
    --epochs 40 \
    --batch_size 32 \
    --grad_accum 4 \
    --lambda_mmm 1.0 \
    --lambda_ti 0.5 \
    --lambda_scl 0.5

# Train all 7 ablation variants
python scripts/train_all_ablations.py \
    --data_dir data/processed \
    --device mps \
    --output_dir checkpoints/ablation

# Train specific ablation models
python scripts/train_all_ablations.py \
    --models mmm_scl ti_scl \
    --device mps
```

### Evaluation

```bash
# Complete ablation study (evaluates all models)
python scripts/ablation_study.py \
    --data_dir data/processed \
    --output_dir results/ablation_study

# Individual evaluations
python scripts/evaluate_tune_type.py --checkpoint checkpoints/best_model.pt
python scripts/evaluate_mode.py --checkpoint checkpoints/best_model.pt
python scripts/evaluate_clustering.py --checkpoint checkpoints/best_model.pt
python scripts/evaluate_linear_probing.py --checkpoint checkpoints/best_model.pt
```

### Figure Generation

```bash
python scripts/generate_ablation_figures.py \
    --results_dir results/ablation_study \
    --output_dir ../Paper/figures
```

## Pre-training Objectives

### 1. Masked Music Modeling (MMM)
Reconstruction objective - predicts masked bars from context.

### 2. Transposition Invariance (TI)
Contrastive learning - enforces pitch-invariant representations.

### 3. Section Contrastive Learning (SCL)
Structure learning - distinguishes tune sections (A vs B).

## Ablation Study Results

| Model | Tune Type | Mode | Key Root | Use Case |
|-------|-----------|------|----------|----------|
| **MMM+SCL** | **80.8%** | 80.7% | 81.0% | Rhythm/tune type tasks |
| TI+SCL | 77.3% | 78.9% | 51.4% | Cross-key retrieval |
| MMM+TI+SCL | 78.1% | 78.1% | 57.0% | Balanced performance |
| MMM-only | 66.8% | 81.4% | 76.8% | Simple baseline |
| SCL-only | 66.4% | 79.4% | 72.5% | Clustering/visualization |
| MMM+TI | 59.9% | 77.9% | 33.6% | - |
| TI-only | 43.8% | 78.0% | 28.3% | - |

**Key Finding:** MMM+SCL outperforms the full three-objective model for tune type classification!

## Model Selection

```python
# For tune type classification (best: 80.8%)
model = ABC2VecModel.load_pretrained('checkpoints/ablation/mmm_scl/best_model.pt')

# For cross-key retrieval (key-invariant)
model = ABC2VecModel.load_pretrained('checkpoints/ablation/ti_scl/best_model.pt')

# For general purpose (balanced)
model = ABC2VecModel.load_pretrained('checkpoints/best_model.pt')
```

## ⚠️ Critical: Vocabulary Loading

**ALWAYS load the saved vocabulary:**

```python
# ✅ CORRECT (98 tokens)
vocab = ABCVocabulary.load("data/processed/vocab.json")

# ❌ WRONG (5 tokens - gives ~46% accuracy!)
vocab = ABCVocabulary()  # Default vocabulary
```

All official scripts automatically use the correct vocabulary.

**Verify:**
```python
assert vocab.size == 98, "Wrong vocabulary loaded!"
```

## Project Structure

```
abc2vec/
├── core/
│   ├── data/                    # Data loading & processing
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Download & preprocess
│   │   ├── dataset.py           # PyTorch datasets
│   │   └── utils.py
│   ├── tokenizer/               # ABC tokenization
│   │   ├── __init__.py
│   │   ├── vocabulary.py        # 98-token vocab
│   │   ├── patchifier.py        # Bar-level patching
│   │   └── transposer.py
│   ├── model/                   # Transformer model
│   │   ├── __init__.py
│   │   ├── encoder.py           # ABC2VecModel
│   │   ├── embedding.py
│   │   └── objectives.py        # MMM, TI, SCL losses
│   ├── training/                # Training logic
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── evaluation/              # Evaluation utilities
│       ├── __init__.py
│       └── metrics.py
├── scripts/                     # Executable scripts
│   ├── data_pipeline.py
│   ├── training.py
│   ├── ablation_study.py
│   ├── evaluate_tune_type.py
│   ├── evaluate_mode.py
│   ├── evaluate_clustering.py
│   ├── evaluate_linear_probing.py
│   └── generate_ablation_figures.py
├── tests/                       # Unit tests
│   ├── test_data.py
│   ├── test_tokenizer.py
│   └── test_model.py
├── pyproject.toml               # Package dependencies
└── README.md                    # This file
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black core/ scripts/ tests/

# Check types
mypy core/

# Lint
ruff check core/ scripts/
```

## Dependencies

Core:
- PyTorch 2.0+
- transformers
- pandas
- scikit-learn

Visualization:
- matplotlib
- seaborn
- umap-learn

Optional:
- faiss-cpu (fast similarity search)

See `pyproject.toml` for complete list.

## Performance

**Original Model (MMM+TI+SCL):**
- Tune Type: 78.4% ± 1.2%
- Mode: 78.5% ± 1.6%
- Key Root: 60.1%
- Clustering: 0.025 silhouette

**Best Ablation (MMM+SCL):**
- Tune Type: 80.8% ± 1.0%
- Mode: 80.7% ± 1.6%
- Key Root: 81.0%
- Clustering: 0.061 silhouette

## Citation

```bibtex
@article{abimbola2026abc2vec,
  title={ABC2Vec: Self-Supervised Representation Learning for Folk Music via Bar-Level Transformers},
  author={Abimbola, Jeremiah and Kostrzewa, Daniel and Benecki, Paweł},
  year={2026}
}
```

## License

MIT License

## Authors

- Jeremiah Abimbola (jeremiah.abimbola@polsl.pl)
- Daniel Kostrzewa
- Paweł Benecki

Silesian University of Technology, Gliwice, Poland

---

## Troubleshooting

**Q: Getting ~46% accuracy instead of 78%?**
A: You're using the wrong vocabulary. Use `ABCVocabulary.load("data/processed/vocab.json")`

**Q: Which model file should I use?**
A: For tune type: `checkpoints/ablation/mmm_scl/best_model.pt` (80.8%). For general: `checkpoints/best_model.pt` (78.4%)

**Q: How to train only specific objectives?**
A: Set unwanted lambdas to 0.0: `--lambda_mmm 1.0 --lambda_ti 0.0 --lambda_scl 0.5`

**Q: Where are the evaluation results?**
A: `results/ablation_study/ablation_comparison.csv` contains all metrics for all models

**Q: How to regenerate paper figures?**
A: `python scripts/generate_ablation_figures.py --results_dir results/ablation_study --output_dir ../Paper/figures`
