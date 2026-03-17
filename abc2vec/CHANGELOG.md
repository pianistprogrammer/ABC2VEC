# Changelog

All notable changes to ABC2Vec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-17

### Added

- Initial release of ABC2Vec package
- Character-level tokenization with bar-level patching
- ABC notation normalization and cleaning pipeline
- Transformer-based encoder architecture
- Multi-objective pre-training framework:
  - Masked Music Modeling (MMM)
  - Section Contrastive Loss (SCL)
  - Transposition Invariance (TI)
  - Variant-Aware Contrastive (VAC)
- ABC transposition utility for data augmentation
- Data processing pipeline for IrishMAN dataset (214K tunes)
- PyTorch datasets for training and evaluation
- Training scripts with progress tracking
- Evaluation scripts for retrieval benchmarks
- Comprehensive unit tests
- Type hints throughout codebase
- Google-style docstrings
- Example scripts and workflow demonstrations

### Documentation

- README with installation and usage guide
- QUICKSTART guide for rapid onboarding
- DEVELOPMENT guide for contributors
- CONTRIBUTING guidelines
- Code examples and scripts
- Inline documentation with docstrings

### Package Structure

- Modular design with clear separation of concerns
- Clean import hierarchy
- Configuration management system
- Trainer class with checkpoint handling
- Evaluation utilities for retrieval metrics

## [Unreleased]

### Planned Features

- Pre-trained model weights on HuggingFace
- FAISS-based efficient similarity search
- Web demo for tune similarity search
- Extended benchmarks (clustering, classification)
- Support for other ABC notation datasets
- Attention visualization tools
- Model interpretability utilities

---

## Version History

- **0.1.0** (2026-03-17): Initial release with core functionality
