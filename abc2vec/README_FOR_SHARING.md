# ABC2Vec - Self-Supervised Learning for Folk Music 🎵

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Production-ready implementation of ABC2Vec: learning dense vector representations of folk music tunes from ABC notation using self-supervised learning.

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Process data
python scripts/run_data_pipeline.py

# Train model
python scripts/run_training.py --epochs 10

# Use it
python scripts/example_usage.py
```

## ✨ Features

- 🎼 **ABC Notation Support** - Character-level tokenization with bar patching
- 🤖 **Transformer Architecture** - 6-layer encoder with multi-head attention
- 📊 **Multi-Objective Training** - MMM, SCL, TI, and VAC objectives
- 🔍 **Tune Retrieval** - Find similar tunes by melodic content
- 📈 **214K Training Tunes** - Trained on IrishMAN folk music dataset
- ⚡ **GPU Accelerated** - Efficient training with PyTorch
- 🧪 **Fully Tested** - Comprehensive unit tests
- 📝 **Well Documented** - Type hints and docstrings throughout

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/abc2vec.git
cd abc2vec

# Quick setup
./setup.sh

# Or manual install
python -m venv venv
source venv/bin/activate
pip install -e .
```

## 🎯 Usage Example

```python
from abc2vec import ABC2VecModel, ABCVocabulary, BarPatchifier
import torch

# Setup
vocab = ABCVocabulary.load('data/processed/vocab.json')
model = ABC2VecModel.load_pretrained('checkpoints/best_model.pt')
patchifier = BarPatchifier(vocab)

# Encode a tune
abc_tune = "D2 EF | G2 AB | c2 dc | BAGF |"
patches = patchifier.patchify(abc_tune)

with torch.no_grad():
    embedding = model.get_embedding(
        patches['bar_indices'].unsqueeze(0),
        patches['char_mask'].unsqueeze(0),
        patches['bar_mask'].unsqueeze(0)
    )

# embedding shape: (1, 128) - ready for similarity search!
```

## 📊 Results

- **Tune Type Retrieval**: Recall@10 > 85%
- **Transposition Invariance**: Same tune different keys = high similarity
- **Section Recognition**: A/B parts of same tune cluster together

## 🏗️ Architecture

```
ABC Notation → Bar Patchifier → Patch Embedding → Transformer → Pooling → Embedding
     |              |                  |               |           |          |
  "D2 EF|"      [bars]         [char vectors]    [attention]  [mean]    [128-d]
```

## 📚 Documentation

- [**START_HERE.md**](START_HERE.md) - Comprehensive overview
- [**QUICKSTART.md**](QUICKSTART.md) - 5-minute quick start
- [**PACKAGE_SUMMARY.md**](PACKAGE_SUMMARY.md) - Technical details
- [**CONTRIBUTING.md**](CONTRIBUTING.md) - How to contribute

## 🛠️ Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Format code
make format

# Check style
make lint
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on PyTorch and HuggingFace Transformers
- Dataset: IrishMAN (214K folk tunes)
- Inspired by CLaMP and MelodyT5 architectures

## 📮 Citation

```bibtex
@software{abc2vec2026,
  title = {ABC2Vec: Self-Supervised Learning for Folk Music Representation},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/abc2vec}
}
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## ⭐ Star Us!

If you find this project useful, please star it on GitHub!

---

Made with ❤️ for the folk music and ML communities
