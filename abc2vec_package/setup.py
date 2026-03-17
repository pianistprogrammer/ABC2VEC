"""Setup script for ABC2Vec package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Dependencies are now managed in pyproject.toml
# setup.py is kept for compatibility but defers to pyproject.toml
requirements = []

setup(
    name="abc2vec",
    version="0.1.0",
    author="ABC2Vec Team",
    author_email="",
    description="Self-supervised learning for folk music representation from ABC notation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/abc2vec",
    packages=find_packages(exclude=["tests", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    # Dependencies are managed in pyproject.toml
    # This file is kept for backward compatibility
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "abc2vec-train=scripts.run_training:main",
            "abc2vec-eval=scripts.run_evaluation:main",
            "abc2vec-process=scripts.run_data_pipeline:main",
        ],
    },
)
