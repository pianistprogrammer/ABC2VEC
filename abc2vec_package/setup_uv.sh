#!/bin/bash
# Quick setup script for ABC2Vec using uv (fast Python package manager)

echo "============================================"
echo "ABC2Vec Package Setup (using uv)"
echo "============================================"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "  ✓ uv installed"
    echo ""
    echo "Please restart your shell or run: source $HOME/.cargo/env"
    echo "Then run this script again."
    exit 0
fi

echo "✓ uv is installed"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python $python_version"
echo ""

# Create virtual environment with uv
echo "Creating virtual environment with uv..."
uv venv
echo "  ✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "  ✓ Activated"
echo ""

# Install package with uv (much faster than pip!)
echo "Installing ABC2Vec package with uv..."
uv pip install -e .
echo "  ✓ Package installed"
echo ""

# Install dev dependencies (optional)
read -p "Install development dependencies? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv pip install -e ".[dev]"
    echo "  ✓ Dev dependencies installed"
fi
echo ""

# Verify installation
echo "Verifying installation..."
python verify_installation.py

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Virtual environment created at: .venv"
echo ""
echo "To activate in future sessions:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. python scripts/run_data_pipeline.py"
echo "  2. python scripts/run_training.py"
echo ""
echo "See QUICKSTART.md for detailed instructions."
