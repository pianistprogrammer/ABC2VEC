# Dependencies

ABC2Vec uses `pyproject.toml` for dependency management (modern Python standard).

## Core Dependencies

Defined in `[project.dependencies]` section of `pyproject.toml`:

- **torch>=2.0.0** - PyTorch for neural networks
- **transformers>=4.30.0** - HuggingFace transformers
- **datasets>=2.14.0** - HuggingFace datasets
- **tokenizers>=0.13.0** - Fast tokenizers
- **numpy>=1.24.0** - Numerical computing
- **pandas>=2.0.0** - Data manipulation
- **scikit-learn>=1.3.0** - Machine learning utilities
- **scipy>=1.10.0** - Scientific computing
- **tqdm>=4.65.0** - Progress bars
- **pyyaml>=6.0** - YAML configuration
- **pyarrow>=12.0.0** - Parquet file support
- **Levenshtein>=0.21.0** - String similarity

## Optional Dependencies

### Development Tools (`[dev]`)

```bash
uv pip install -e ".[dev]"
```

- **pytest>=7.3.0** - Testing framework
- **pytest-cov>=4.1.0** - Code coverage
- **black>=23.0.0** - Code formatter
- **isort>=5.12.0** - Import sorter
- **flake8>=6.0.0** - Linter
- **mypy>=1.3.0** - Type checker

### Visualization (`[viz]`)

```bash
uv pip install -e ".[viz]"
```

- **matplotlib>=3.7.0** - Plotting
- **seaborn>=0.12.0** - Statistical visualization
- **plotly>=5.14.0** - Interactive plots

### Extra Features (`[extra]`)

```bash
uv pip install -e ".[extra]"
```

- **music21>=9.1.0** - Music analysis
- **faiss-cpu>=1.7.4** - Vector similarity search
- **umap-learn>=0.5.3** - Dimensionality reduction
- **einops>=0.6.1** - Tensor operations
- **rich>=13.0.0** - Rich terminal output
- **accelerate>=0.20.0** - Training acceleration
- **wandb>=0.15.0** - Experiment tracking

### All Dependencies (`[all]`)

```bash
uv pip install -e ".[all]"
```

Installs everything: core + dev + viz + extra

## Installation Commands

### With uv (Recommended - Fast!)

```bash
# Core only
uv pip install -e .

# With dev tools
uv pip install -e ".[dev]"

# With visualization
uv pip install -e ".[viz]"

# Everything
uv pip install -e ".[all]"
```

### With pip (Traditional)

```bash
# Core only
pip install -e .

# With dev tools
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

## Why pyproject.toml?

Modern Python packaging standard (PEP 621):

✅ Single source of truth for dependencies
✅ Works with uv, pip, poetry, pdm
✅ Includes metadata (version, author, license)
✅ Tool configurations (black, pytest, isort)
✅ No separate requirements.txt needed

## Dependency Management

### Adding New Dependency

Edit `pyproject.toml`:

```toml
[project]
dependencies = [
    "torch>=2.0.0",
    "your-new-package>=1.0.0",  # Add here
]
```

Then reinstall:

```bash
uv pip install -e .
```

### Checking Installed Packages

```bash
uv pip list           # List all packages
uv pip show abc2vec   # Show abc2vec info
```

### Updating Dependencies

```bash
uv pip install -e . --upgrade
```

## Platform-Specific Notes

### GPU Support (CUDA)

For CUDA support, install PyTorch with CUDA:

```bash
# CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### macOS (Apple Silicon)

PyTorch with MPS (Metal Performance Shaders):

```bash
uv pip install torch torchvision torchaudio
```

### Minimal Install (No ML)

If you only need the data processing utilities:

```bash
uv pip install pandas numpy pyyaml pyarrow
```

## Troubleshooting

### Dependency Conflicts

```bash
# Force reinstall
uv pip install -e . --reinstall

# Clear cache
rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### Missing Packages

Check which packages are missing:

```bash
python -c "import abc2vec; print('✓ abc2vec works')"
```

If import fails, reinstall:

```bash
uv pip install -e ".[all]"
```

## Migration from requirements.txt

If you have an old `requirements.txt`, dependencies are now in `pyproject.toml`.

To migrate:
1. ✅ Already done! Dependencies moved to pyproject.toml
2. ✅ requirements.txt removed
3. ✅ All scripts updated

## Learn More

- **pyproject.toml spec**: https://peps.python.org/pep-0621/
- **uv documentation**: https://docs.astral.sh/uv/
- **Python packaging**: https://packaging.python.org/
