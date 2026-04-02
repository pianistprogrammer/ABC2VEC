# ABC2Vec - Music Representation Learning

A clean, production-ready implementation of ABC2Vec - a self-supervised learning system for folk music representation from ABC notation.

## Overview

ABC2Vec learns dense vector representations of folk music tunes from ABC notation using:
- **Character-level tokenization** with bar-level patching
- **Transformer-based encoder** architecture
- **Multi-objective pre-training**: Masked Bar Modeling (MBM), Transposition Invariance (TIL), and Section Contrastive Learning (SCL)

## Project Structure

```
abc2vec/
├── abc2vec/              # Core library
│   ├── __init__.py
│   ├── data/             # Data processing and loading
│   │   ├── __init__.py
│   │   ├── pipeline.py   # Data download, cleaning, normalization
│   │   ├── dataset.py    # PyTorch datasets
│   │   └── utils.py      # Data utilities
│   ├── tokenizer/        # Tokenization and patching
│   │   ├── __init__.py
│   │   ├── vocabulary.py # Character vocabulary
│   │   ├── patchifier.py # Bar patchifier
│   │   └── transposer.py # ABC transposition
│   ├── model/            # Model architecture
│   │   ├── __init__.py
│   │   ├── encoder.py    # Main encoder
│   │   ├── embedding.py  # Patch embeddings
│   │   └── objectives.py # Pre-training objectives
│   ├── training/         # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py    # Training loop
│   │   └── config.py     # Configuration
│   └── evaluation/       # Evaluation and analysis
│       ├── __init__.py
│       ├── retrieval.py  # Retrieval evaluation
│       ├── clustering.py # Clustering analysis
│       └── visualization.py # Plotting utilities
├── scripts/              # Runnable scripts
│   ├── run_data_pipeline.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── run_benchmark.py
├── tests/                # Unit tests
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_data.py
├── requirements.txt      # Python dependencies
├── setup.py             # Package installation
└── README.md            # This file
```

## Installation

### 1. Clone and Install

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .
```

### 2. Quick Start

```bash
# Step 1: Download and process data
python scripts/run_data_pipeline.py

# Step 2: Train the model
python scripts/run_training.py

# Step 3: Evaluate on retrieval benchmarks
python scripts/run_evaluation.py
```

## Usage

### As a Python Library

```python
from abc2vec import ABC2VecModel, ABCVocabulary, BarPatchifier
from abc2vec.data import load_processed_data
import torch

# Load vocabulary and create patchifier
vocab = ABCVocabulary.load('data/processed/vocab.json')
patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)

# Load pre-trained model
model = ABC2VecModel.load_pretrained('checkpoints/best_model.pt')
model.eval()

# Encode a tune
abc_body = "D2 EF | G2 AB | c2 dc | BAGF |"
patches = patchifier.patchify(abc_body)

with torch.no_grad():
    embedding = model.encode(
        patches['bar_indices'].unsqueeze(0),
        patches['char_mask'].unsqueeze(0),
        patches['bar_mask'].unsqueeze(0)
    )

print(f"Tune embedding shape: {embedding.shape}")  # (1, d_model)
```

### Command Line Scripts

```bash
# Data pipeline with custom output directory
python scripts/run_data_pipeline.py --output_dir ./data/processed

# Training with custom config
python scripts/run_training.py \
    --d_model 512 \
    --n_layers 8 \
    --batch_size 128 \
    --epochs 50

# Evaluation on specific benchmark
python scripts/run_evaluation.py --checkpoint ./checkpoints/epoch_50.pt
```

## Features

- **Clean, modular code**: Organized into logical modules with clear separation of concerns
- **Type hints**: Full type annotations for better IDE support and code clarity
- **Comprehensive docstrings**: Google-style docstrings for all classes and functions
- **Configuration management**: YAML-based config with CLI overrides
- **Logging**: Structured logging with rich formatting
- **Testing**: Unit tests for core components
- **Reproducibility**: Fixed random seeds and deterministic operations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- HuggingFace transformers & datasets
- pandas, numpy, scikit-learn
- And more...

All dependencies are managed in `pyproject.toml`. See [DEPENDENCIES.md](DEPENDENCIES.md) for details.

Install with:
```bash
uv pip install -e .           # Core dependencies
uv pip install -e ".[dev]"    # With dev tools
uv pip install -e ".[all]"    # Everything
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{abc2vec2026,
  title={ABC2Vec: Self-Supervised Learning for Folk Music Representation},
  author={Your Name},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions, please open an issue on the GitHub repository.
