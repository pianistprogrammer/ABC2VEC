#!/bin/bash
# Quick setup script for ABC2Vec

echo "============================================"
echo "ABC2Vec Package Setup"
echo "============================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "  ✓ pip upgraded"
echo ""

# Install package
echo "Installing ABC2Vec package..."
pip install -e . -q
echo "  ✓ Package installed"
echo ""

# Install dev dependencies (optional)
read -p "Install development dependencies? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[dev]" -q
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
echo "Next steps:"
echo "  1. source venv/bin/activate  (if not already activated)"
echo "  2. python scripts/run_data_pipeline.py"
echo "  3. python scripts/run_training.py"
echo ""
echo "See QUICKSTART.md for detailed instructions."
