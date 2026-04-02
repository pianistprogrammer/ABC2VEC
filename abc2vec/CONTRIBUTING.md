# Contributing to ABC2Vec

Thank you for your interest in contributing to ABC2Vec! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/abc2vec.git
   cd abc2vec
   ```
3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Workflow

1. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style guidelines

3. **Add tests** for new functionality

4. **Run tests** to ensure nothing breaks:
   ```bash
   pytest tests/ -v
   ```

5. **Format your code**:
   ```bash
   black abc2vec scripts tests
   isort abc2vec scripts tests
   flake8 abc2vec scripts tests
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## Code Style Guidelines

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** for all function signatures
- Maximum line length: **88 characters** (black default)
- Use **Google-style docstrings**

Example:

```python
def process_tune(abc_text: str, normalize: bool = True) -> Dict[str, Any]:
    """
    Process ABC notation tune.

    Args:
        abc_text: Raw ABC notation string
        normalize: Whether to normalize the notation

    Returns:
        Dictionary containing processed tune data with keys:
            - 'abc_clean': Normalized ABC text
            - 'metadata': Extracted metadata dict

    Raises:
        ValueError: If ABC text is invalid

    Example:
        >>> result = process_tune("X:1\\nK:D\\nD2 EF|")
        >>> print(result['abc_clean'])
        X:1
        K:D
        D2 EF|
    """
    pass
```

### Docstring Requirements

All public functions, classes, and methods must have docstrings with:

- **Short description** (one line)
- **Args**: All parameters with types and descriptions
- **Returns**: Return value description
- **Raises**: Any exceptions that may be raised
- **Example** (optional): Usage example for complex functions

### Type Hints

All function signatures should include type hints:

```python
from typing import List, Dict, Optional, Tuple

def encode_tunes(
    tunes: List[str],
    vocab: ABCVocabulary,
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode ABC tunes to tensors."""
    pass
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use pytest fixtures for common setup
- Aim for >80% code coverage

Example test:

```python
import pytest
from abc2vec import ABCVocabulary


class TestABCVocabulary:
    """Test cases for ABCVocabulary."""

    @pytest.fixture
    def vocab(self):
        """Create vocabulary fixture."""
        vocab = ABCVocabulary()
        vocab.build_from_corpus(["ABC DEF"], verbose=False)
        return vocab

    def test_encode_decode(self, vocab):
        """Test encode/decode roundtrip."""
        text = "ABC"
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        assert decoded == text
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_tokenizer.py -v

# With coverage
pytest tests/ --cov=abc2vec --cov-report=html

# Fast fail
pytest tests/ -x
```

## Pull Request Process

1. **Update documentation** if you're changing user-facing functionality

2. **Add/update tests** for your changes

3. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v
   ```

4. **Format code**:
   ```bash
   black abc2vec scripts tests
   isort abc2vec scripts tests
   ```

5. **Check for linter issues**:
   ```bash
   flake8 abc2vec scripts tests
   ```

6. **Update CHANGELOG.md** with your changes

7. **Write a clear PR description**:
   - What problem does this solve?
   - What changes were made?
   - How to test the changes?

8. **Request review** from maintainers

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
Add feature: bar-level attention masking

- Implement selective attention masking for bars
- Add tests for masking functionality
- Update documentation

Closes #123
```

Format: `<type>: <subject>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes (formatting)
- `chore`: Maintenance tasks

## What to Contribute

### Good First Issues

- Documentation improvements
- Adding more tests
- Fixing typos or formatting
- Adding examples

### Feature Contributions

Before starting work on a major feature:

1. **Open an issue** describing the feature
2. **Discuss the approach** with maintainers
3. **Get approval** before implementing
4. **Follow the development workflow**

### Bug Reports

When reporting bugs, include:

- **Clear description** of the bug
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, package versions)
- **Error messages** and stack traces

### Feature Requests

When requesting features:

- **Describe the problem** you're trying to solve
- **Explain why** this feature would be useful
- **Suggest implementation** if you have ideas
- **Volunteer to implement** if possible!

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Give credit where it's due
- Help others learn and grow

## Questions?

- Open an issue with the `question` label
- Check existing issues and discussions
- Read the documentation

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- CHANGELOG.md for their contributions
- GitHub contributors page

Thank you for making ABC2Vec better! 🎵
