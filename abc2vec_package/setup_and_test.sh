#!/bin/bash
# Complete setup and test workflow for ABC2Vec using uv

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "                    ABC2VEC COMPLETE SETUP WITH UV ⚡"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "  ✓ uv installed"
    echo ""
    echo "⚠️  Please restart your shell or run:"
    echo "    source \$HOME/.cargo/env"
    echo ""
    echo "Then run this script again: ./setup_and_test.sh"
    exit 0
fi

echo "✓ uv is installed"
echo ""

# Create virtual environment
echo "📂 Creating virtual environment..."
uv venv .venv
echo "  ✓ Virtual environment created at .venv"
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate
echo "  ✓ Activated"
echo ""

# Install package
echo "📥 Installing ABC2Vec package..."
uv pip install -e .
echo "  ✓ Package installed"
echo ""

# Install dev dependencies
echo "🛠️  Installing development dependencies..."
uv pip install -e ".[dev]"
echo "  ✓ Dev dependencies installed"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
python verify_installation.py
echo ""

# Run tests
echo "🧪 Running unit tests..."
pytest tests/ -v --tb=short
echo ""

# Show package info
echo "════════════════════════════════════════════════════════════════════════════════"
echo "                              ✅ SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📦 Package Information:"
echo "   • Python modules:  $(find abc2vec -name '*.py' | wc -l | xargs)"
echo "   • Scripts:         $(find scripts -name '*.py' | wc -l | xargs)"
echo "   • Tests:           $(find tests -name '*.py' | wc -l | xargs)"
echo "   • Documentation:   $(ls *.md | wc -l | xargs) guides"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "   1. Process data (5-10 minutes):"
echo "      $ python scripts/run_data_pipeline.py"
echo ""
echo "   2. Train model (quick test - 5 minutes):"
echo "      $ python scripts/run_training.py --epochs 2 --batch_size 32"
echo ""
echo "   3. Run full workflow:"
echo "      $ python scripts/complete_workflow.py"
echo ""
echo "   4. Use in your code:"
echo "      $ python scripts/example_usage.py --help"
echo ""
echo "📚 Documentation:"
echo "   • START_HERE.md      - Comprehensive overview"
echo "   • QUICKSTART.md      - Quick start guide (with uv!)"
echo "   • README.md          - Full documentation"
echo "   • INSTALLATION_GUIDE.txt - Step-by-step install"
echo ""
echo "💡 Tips:"
echo "   • Activate venv:  source .venv/bin/activate"
echo "   • Run tests:      make test"
echo "   • Format code:    make format"
echo "   • Show commands:  make help"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "                    🎵 READY TO USE ABC2VEC! 🎵"
echo "════════════════════════════════════════════════════════════════════════════════"
