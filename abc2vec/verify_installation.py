#!/usr/bin/env python
"""
Installation verification script.

Checks that all components are properly installed and working.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (requires >= 3.8)")
        return False


def check_imports():
    """Check that all required packages can be imported."""
    print("\nChecking package imports...")

    imports_to_check = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
    ]

    all_ok = True
    for module, name in imports_to_check:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
            all_ok = False

    return all_ok


def check_abc2vec_imports():
    """Check ABC2Vec package imports."""
    print("\nChecking ABC2Vec imports...")

    try:
        from core import (
            ABCVocabulary,
            BarPatchifier,
            ABCTransposer,
            ABC2VecModel,
        )

        print("  ✓ Core components")

        from core.data import ABC2VecDataset
        print("  ✓ Data components")

        from core.model import ABC2VecConfig, ABC2VecLoss
        print("  ✓ Model components")

        from core.training import Trainer
        print("  ✓ Training components")

        return True

    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("\nMake sure you installed the package:")
        print("  pip install -e .")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Devices: {torch.cuda.device_count()}")
            print(f"    Current: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available (CPU mode only)")

    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")


def run_basic_test():
    """Run a basic functionality test."""
    print("\nRunning basic functionality test...")

    try:
        from core import ABCVocabulary, BarPatchifier
        import torch

        # Create vocabulary
        vocab = ABCVocabulary()
        corpus = ["D2 EF | G2 AB | c2 dc | BAGF |"]
        vocab.build_from_corpus(corpus, verbose=False)
        print("  ✓ Vocabulary creation")

        # Create patchifier
        patchifier = BarPatchifier(vocab, max_bar_length=64, max_bars=64)
        print("  ✓ Patchifier creation")

        # Test patchification
        patches = patchifier.patchify(corpus[0])
        assert patches["bar_indices"].shape == (64, 64)
        print("  ✓ Patchification")

        # Test model creation
        from core.model import ABC2VecConfig, ABC2VecModel

        config = ABC2VecConfig(vocab_size=vocab.size, d_model=128, n_layers=2)
        model = ABC2VecModel(config)
        print("  ✓ Model creation")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            embedding = model.get_embedding(
                patches["bar_indices"].unsqueeze(0),
                patches["char_mask"].unsqueeze(0),
                patches["bar_mask"].unsqueeze(0),
            )
        assert embedding.shape == (1, config.d_embed)
        print("  ✓ Forward pass")

        return True

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("ABC2Vec Installation Verification")
    print("=" * 80 + "\n")

    checks = [
        ("Python version", check_python_version),
        ("Required packages", check_imports),
        ("ABC2Vec package", check_abc2vec_imports),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, False))

    # Optional checks
    check_cuda()

    # Basic functionality test
    test_result = run_basic_test()
    results.append(("Basic functionality", test_result))

    # Summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80 + "\n")

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_passed = False

    print()

    if all_passed:
        print("=" * 80)
        print("✓ All checks passed! ABC2Vec is ready to use.")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Process data:  python scripts/run_data_pipeline.py")
        print("  2. Train model:   python scripts/run_training.py")
        print("  3. Run examples:  python scripts/example_usage.py --help")
        print("\nSee QUICKSTART.md for detailed instructions.")
        return 0
    else:
        print("=" * 80)
        print("✗ Some checks failed. Please fix the issues above.")
        print("=" * 80)
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Install package: pip install -e .")
        print("  - Check Python version: python --version (need >= 3.8)")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
