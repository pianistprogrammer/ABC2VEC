# Development Guide

## Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/abc2vec.git
cd abc2vec

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Code Style

This project follows PEP 8 style guidelines with the following tools:

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run formatting and checks:

```bash
# Format code
black abc2vec scripts tests

# Sort imports
isort abc2vec scripts tests

# Lint
flake8 abc2vec scripts tests

# Type check
mypy abc2vec
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=abc2vec --cov-report=html

# Run specific test file
pytest tests/test_tokenizer.py -v
```

## Project Structure

```
abc2vec_package/
├── abc2vec/                  # Main package
│   ├── data/                 # Data processing
│   │   ├── pipeline.py       # Data download and normalization
│   │   └── dataset.py        # PyTorch datasets
│   ├── tokenizer/            # Tokenization
│   │   ├── vocabulary.py     # Character vocabulary
│   │   ├── patchifier.py     # Bar patchification
│   │   └── transposer.py     # ABC transposition
│   ├── model/                # Model architecture
│   │   ├── embedding.py      # Patch embeddings
│   │   ├── encoder.py        # Transformer encoder
│   │   └── objectives.py     # Training objectives
│   ├── training/             # Training logic
│   │   └── trainer.py        # Trainer class
│   ├── evaluation/           # Evaluation metrics
│   └── utils/                # Utilities
│       └── config.py         # Configuration management
├── scripts/                  # Entry point scripts
│   ├── run_data_pipeline.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── example_usage.py
├── tests/                    # Unit tests
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_data.py
└── docs/                     # Documentation
```

## Adding New Features

1. **New Data Source**: Add processing logic to `abc2vec/data/pipeline.py`
2. **New Model Component**: Add to appropriate file in `abc2vec/model/`
3. **New Objective**: Add to `abc2vec/model/objectives.py`
4. **New Evaluation**: Add to `abc2vec/evaluation/`

Always:
- Add type hints
- Write docstrings (Google style)
- Add unit tests
- Update README if user-facing

## Documentation Style

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Short description.

    Longer description if needed, explaining behavior,
    algorithms, or important notes.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided

    Example:
        >>> example_function("test", 42)
        True
    """
    pass
```

## Continuous Integration

The project uses GitHub Actions for:
- Running tests on push/PR
- Code style checks
- Type checking
- Coverage reports

See `.github/workflows/` for configuration.

## Releasing

1. Update version in `setup.py` and `abc2vec/__init__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.1.0`
4. Push: `git push origin v0.1.0`
5. GitHub Actions will build and publish to PyPI

## Getting Help

- Open an issue on GitHub
- Check existing issues and discussions
- Read the full documentation at docs/

## Code Review Guidelines

When reviewing PRs, check for:
- [ ] Tests added for new functionality
- [ ] Type hints on new functions/methods
- [ ] Docstrings following Google style
- [ ] Code formatted with black
- [ ] Imports sorted with isort
- [ ] No new linter warnings
- [ ] Backwards compatibility maintained
