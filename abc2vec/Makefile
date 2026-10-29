.PHONY: help install install-dev install-uv test lint format clean verify train eval

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install:  ## Install package with pip
	pip install -e .

install-dev:  ## Install package with development dependencies (pip)
	pip install -e ".[dev]"

install-uv:  ## Install package with uv (fast! ⚡)
	@if command -v uv >/dev/null 2>&1; then \
		uv venv .venv && \
		. .venv/bin/activate && \
		uv pip install -e . && \
		echo "✓ Installed with uv (much faster!)"; \
	else \
		echo "uv not found. Install it with:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "Or use: make install (with pip)"; \
	fi

install-uv-dev:  ## Install with dev dependencies using uv
	@if command -v uv >/dev/null 2>&1; then \
		uv venv .venv && \
		. .venv/bin/activate && \
		uv pip install -e ".[dev]" && \
		echo "✓ Installed with dev dependencies using uv"; \
	else \
		echo "uv not found. Install it first."; \
	fi

verify:  ## Verify installation
	python verify_installation.py

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=abc2vec --cov-report=html --cov-report=term

lint:  ## Check code style
	flake8 abc2vec scripts tests
	mypy abc2vec --ignore-missing-imports || true

format:  ## Format code with black and isort
	black abc2vec scripts tests
	isort abc2vec scripts tests

clean:  ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/
	rm -rf .venv/ venv/

# Data and training targets
data:  ## Process data pipeline
	python scripts/run_data_pipeline.py --output_dir ./data/processed

train:  ## Train model (quick test)
	python scripts/run_training.py --epochs 5 --batch_size 32 --d_model 128 --n_layers 4

train-full:  ## Train full model
	python scripts/run_training.py --epochs 50 --batch_size 64

eval:  ## Evaluate model
	@if [ -f "./checkpoints/best_model.pt" ]; then \
		python scripts/run_evaluation.py --checkpoint ./checkpoints/best_model.pt; \
	else \
		echo "No checkpoint found. Train model first with: make train"; \
	fi

example:  ## Run example workflow
	python scripts/complete_workflow.py

# Development targets
docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

check-all: format lint test  ## Run all checks (format, lint, test)

release:  ## Build distribution packages
	python setup.py sdist bdist_wheel

# Quick commands
quick-uv: install-uv verify  ## Quick setup with uv: install and verify

quick: install verify  ## Quick setup with pip: install and verify

all: install-dev verify test  ## Full setup: install, verify, and test

all-uv: install-uv-dev verify test  ## Full setup with uv

# Info targets
info:  ## Show package info
	@echo "ABC2Vec Package Information"
	@echo "=========================="
	@echo "Python modules: $$(find abc2vec -name '*.py' | wc -l)"
	@echo "Scripts:        $$(find scripts -name '*.py' | wc -l)"
	@echo "Tests:          $$(find tests -name '*.py' | wc -l)"
	@echo "Lines of code:  $$(find . -name '*.py' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $$1}')"
	@echo ""
	@echo "Installation method:"
	@if [ -d ".venv" ]; then \
		echo "  ✓ uv virtual environment (.venv)"; \
	elif [ -d "venv" ]; then \
		echo "  ✓ pip virtual environment (venv)"; \
	else \
		echo "  ✗ Not installed yet"; \
	fi
