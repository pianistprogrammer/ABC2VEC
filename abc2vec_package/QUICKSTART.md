# ABC2Vec - Quick Start with uv ⚡

> **Fast Setup**: This guide uses [uv](https://github.com/astral-sh/uv) - a blazingly fast Python package manager (10-100x faster than pip!)

## Why uv?

- ⚡ **10-100x faster** than pip
- 🎯 **Better dependency resolution**
- 🔒 **Reliable and reproducible**
- 🔄 **Drop-in pip replacement**
- 💪 **Production-ready**

Installation that takes 2 minutes with pip takes **10 seconds with uv**!

## Quick Installation (30 seconds) ⚡

### Step 1: Install uv (once)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip/pipx
pip install uv
```

### Step 2: Setup ABC2Vec

```bash
cd abc2vec_package

# Automated setup (recommended)
./setup_uv.sh

# Or manual
uv venv                      # Create venv (1 second!)
source .venv/bin/activate    # Activate
uv pip install -e .          # Install (10 seconds!)
```

### Step 3: Verify

```bash
python verify_installation.py
```

**✅ Done!** You're ready to use ABC2Vec.

## Quick Usage Example

### Process Data (5-10 minutes)

```bash
python scripts/run_data_pipeline.py
```

Downloads and processes 214K Irish folk tunes.

### Train Model (Quick Test - 5 minutes)

```bash
python scripts/run_training.py \
    --epochs 2 \
    --batch_size 32 \
    --d_model 128 \
    --n_layers 4
```

### Use in Code

```python
from abc2vec import ABC2VecModel, ABCVocabulary, BarPatchifier
import torch

# Load
vocab = ABCVocabulary.load('./data/processed/vocab.json')
patchifier = BarPatchifier(vocab)

# Encode tune
tune = "D2 EF | G2 AB | c2 dc | BAGF |"
patches = patchifier.patchify(tune)

# Get embedding
with torch.no_grad():
    embedding = model.get_embedding(
        patches['bar_indices'].unsqueeze(0),
        patches['char_mask'].unsqueeze(0),
        patches['bar_mask'].unsqueeze(0)
    )
```

## Makefile Commands (Super Convenient)

```bash
make quick-uv       # Install with uv + verify
make install-uv     # Install with uv
make verify         # Check installation
make test           # Run tests
make data           # Process dataset
make train          # Quick training test
make eval           # Evaluate model
make help           # Show all commands
```

## Full Training (Production)

```bash
# Full model (1-2 hours on GPU)
python scripts/run_training.py \
    --epochs 50 \
    --batch_size 64 \
    --d_model 256 \
    --n_layers 6
```

## Development Setup (with uv)

```bash
# Install with dev tools (pytest, black, mypy, etc.)
uv pip install -e ".[dev]"

# Or with make
make install-uv-dev

# Run checks
make format      # Auto-format code
make lint        # Check style
make test        # Run tests
make check-all   # All checks
```

## uv vs pip Comparison

| Task | pip | uv | Speedup |
|------|-----|-----|---------|
| Create venv | ~10s | ~1s | **10x** |
| Install deps | ~60s | ~5s | **12x** |
| Resolve deps | ~30s | ~2s | **15x** |
| **Total setup** | **~100s** | **~8s** | **12x** ⚡ |

## Traditional pip Installation

If you prefer pip:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

Or use: `make install`

## Troubleshooting

### uv not found

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

### Import errors

```bash
# Reinstall with all dependencies
uv pip install -e .

# Or with visualization tools
uv pip install -e ".[viz]"

# Or everything
uv pip install -e ".[all]"
```

### Out of memory

```bash
# Use smaller batch size
python scripts/run_training.py --batch_size 16
```

### Tests failing

```bash
# Install dev dependencies
uv pip install -e ".[dev]"
pytest tests/ -v
```

## Next Steps

1. ✅ Read [START_HERE.md](START_HERE.md) for comprehensive overview
2. ✅ Read [README.md](README.md) for full documentation
3. ✅ Check [PACKAGE_SUMMARY.md](PACKAGE_SUMMARY.md) for technical details
4. ✅ See [DEVELOPMENT.md](DEVELOPMENT.md) for contributing

## Learn More About uv

- GitHub: https://github.com/astral-sh/uv
- Docs: https://docs.astral.sh/uv/
- Why uv is fast: Written in Rust, parallel downloads, smart caching

---

**Ready to go!** 🚀 With uv, you'll be up and running in seconds instead of minutes!
