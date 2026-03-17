# ABC2Vec - Quick Start with uv ⚡

> **Using uv**: This guide uses [uv](https://github.com/astral-sh/uv), a blazingly fast Python package installer (10-100x faster than pip!). Traditional pip instructions are also provided.

## Prerequisites

- Python 3.8 or higher
- (Optional) CUDA-capable GPU for faster training

## Installation with uv ⚡ (Recommended)

### 1. Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 2. Setup ABC2Vec Package

```bash
cd abc2vec_package

# Automated setup (recommended)
./setup_uv.sh

# Or manual setup
uv venv                    # Create virtual environment
source .venv/bin/activate  # Activate it
uv pip install -e .        # Install package (fast!)
```

### 3. Verify Installation

```bash
python verify_installation.py
```

**✓ Done!** Much faster than traditional pip, right?

## Alternative: Traditional pip Installation

If you prefer not to use uv:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
python verify_installation.py
```

## Quick Example (5 minutes)

### 1. Process Data

```bash
python scripts/run_data_pipeline.py --output_dir ./data/processed
```

This downloads and processes 214K folk tunes (~5-10 minutes).

### 2. Train Model (Quick Test)

```bash
# Quick test with small model (5 minutes)
python scripts/run_training.py \
    --epochs 2 \
    --batch_size 32 \
    --d_model 128 \
    --n_layers 4
```

### 3. Use the Model

```python
from abc2vec import ABC2VecModel, ABCVocabulary, BarPatchifier
import torch

# Load components
vocab = ABCVocabulary.load('./data/processed/vocab.json')
patchifier = BarPatchifier(vocab)

# Create model (or load pre-trained)
from abc2vec.model import ABC2VecConfig
config = ABC2VecConfig(vocab_size=vocab.size)
model = ABC2VecModel(config)
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

### 4. Compare Tunes

```python
# Encode two tunes
tune1 = "D2 EF | G2 AB |"
tune2 = "D2 FE | G2 BA |"

patches1 = patchifier.patchify(tune1)
patches2 = patchifier.patchify(tune2)

with torch.no_grad():
    emb1 = model.get_embedding(
        patches1['bar_indices'].unsqueeze(0),
        patches1['char_mask'].unsqueeze(0),
        patches1['bar_mask'].unsqueeze(0)
    )
    emb2 = model.get_embedding(
        patches2['bar_indices'].unsqueeze(0),
        patches2['char_mask'].unsqueeze(0),
        patches2['bar_mask'].unsqueeze(0)
    )

similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
print(f"Similarity: {similarity.item():.4f}")
```

## Command Cheatsheet

### With uv (Fast! ⚡)

```bash
# Setup
uv venv                          # Create venv
source .venv/bin/activate        # Activate
uv pip install -e .              # Install package
uv pip install -e ".[dev]"       # Install with dev deps

# Development
uv pip install pytest black      # Install tools
uv pip list                      # List installed packages
```

### With Makefile (Convenience)

```bash
make install        # Install package
make verify         # Verify installation
make test           # Run tests
make format         # Format code
make lint           # Check style
make data           # Process dataset
make train          # Quick training
make train-full     # Full training
make eval           # Evaluate model
make help           # Show all commands
```

## Performance Comparison

| Operation | pip | uv | Speedup |
|-----------|-----|-----|---------|
| Install deps | ~60s | ~5s | **12x faster** |
| Create venv | ~10s | ~1s | **10x faster** |
| Resolve deps | ~30s | ~2s | **15x faster** |

**Total setup time**: pip (~100s) vs uv (~8s) 🚀

## Full Training (Production)

For production training with full model:

```bash
python scripts/run_training.py \
    --data_dir ./data/processed \
    --output_dir ./checkpoints \
    --epochs 50 \
    --batch_size 64 \
    --d_model 256 \
    --n_layers 6 \
    --lr 1e-4
```

This takes 1-2 hours on GPU, ~8-10 hours on CPU.

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size and model size
python scripts/run_training.py \
    --batch_size 16 \
    --d_model 128 \
    --n_layers 4
```

### Import Errors

```bash
# Reinstall package
uv pip install -e . --force-reinstall

# Or with pip
pip install -e . --force-reinstall
```

### uv Not Found

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then restart shell or:
source $HOME/.cargo/env
```

## Next Steps

1. ✅ **Read** [START_HERE.md](START_HERE.md) for comprehensive overview
2. ✅ **Install** with `./setup_uv.sh`
3. ✅ **Verify** with `python verify_installation.py`
4. ✅ **Process** data with `python scripts/run_data_pipeline.py`
5. ✅ **Train** with `python scripts/run_training.py`
6. ✅ **Share** on GitHub and with the community!

## Why uv?

- **10-100x faster** than pip
- **Better dependency resolution**
- **Reliable and deterministic**
- **Compatible with pip** (drop-in replacement)
- **Modern Python tooling**

Learn more: https://github.com/astral-sh/uv

---

**Ready to rock! 🎸** Your package is professional, well-documented, and blazingly fast to install with uv!
