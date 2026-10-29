# ABC2Vec Scripts

This directory contains all scripts used in the ABC2Vec research paper. All scripts follow the `run_*.py` naming convention for clarity and consistency.

## Training

### `run_training.py`
Train the ABC2Vec model from scratch.

```bash
python run_training.py \
    --data_dir ../data/processed \
    --output_dir ../checkpoints \
    --epochs 40 \
    --batch_size 256 \
    --device mps
```

**Outputs:**
- `checkpoints/best_model.pt` - Best model checkpoint
- `checkpoints/model_config.json` - Model configuration
- `checkpoints/training_history.json` - Training metrics

## Data Preparation

### `run_data_pipeline.py`
Preprocess raw ABC notation data and create train/val/test splits.

```bash
python run_data_pipeline.py \
    --input_dir ../data/raw \
    --output_dir ../data/processed
```

**Outputs:**
- `data/processed/train.parquet` - Training split
- `data/processed/val.parquet` - Validation split
- `data/processed/test.parquet` - Test split
- `data/processed/vocab.json` - Vocabulary

## Evaluation Scripts

### `run_evaluate_tune_type.py`
Evaluate tune type classification (6 classes: jig, reel, polka, waltz, slide, slip jig).

```bash
python run_evaluate_tune_type.py \
    --checkpoint ../checkpoints/best_model.pt \
    --data_dir ../data/processed \
    --device mps
```

**Outputs:**
- `checkpoints/tune_type_classification_results.json`
- **Paper Result:** 78.4% ± 1.2% accuracy

### `run_evaluate_mode.py`
Evaluate mode classification (4 classes: major, minor, dorian, mixolydian).

```bash
python run_evaluate_mode.py \
    --checkpoint ../checkpoints/best_model.pt \
    --data_dir ../data/processed \
    --device mps
```

**Outputs:**
- `checkpoints/mode_classification_results.json`
- **Paper Result:** 78.8% ± 1.6% accuracy

### `run_evaluate_linear_probing.py`
Linear probing analysis for property disentanglement.

```bash
python run_evaluate_linear_probing.py \
    --checkpoint ../checkpoints/best_model.pt \
    --data_dir ../data/processed \
    --device mps
```

**Outputs:**
- `checkpoints/linear_probing_results.json`

**Paper Results:**
- Tune Type: 79.1% ± 1.2%
- Mode: 80.8% ± 0.5%
- Key Root: 62.3% ± 0.9%
- Tune Length: 89.5% ± 0.7%

### `run_evaluate_clustering.py`
Clustering quality analysis (K-means with silhouette, NMI, ARI metrics).

```bash
python run_evaluate_clustering.py \
    --checkpoint ../checkpoints/best_model.pt \
    --data_dir ../data/processed \
    --device mps
```

**Outputs:**
- `checkpoints/clustering_results.json`

**Paper Results:**
- Silhouette scores: 0.025-0.026 (positive but low)
- NMI: 0.004-0.037
- ARI: 0.001-0.019

## Figure Generation Scripts

### `run_generate_dataset_figures.py`
Generate dataset statistics figure (6 subplots showing distribution of tune properties).

```bash
python run_generate_dataset_figures.py \
    --data_dir ../data/processed \
    --output_dir ../figures
```

**Outputs:**
- `figures/dataset_statistics.pdf`
- `figures/dataset_statistics.png`

### `run_generate_architecture_diagram.py`
Generate model architecture diagram.

```bash
python run_generate_architecture_diagram.py \
    --output_dir ../figures
```

**Outputs:**
- `figures/architecture_diagram.pdf`
- `figures/architecture_diagram.png`

### `run_generate_training_curves.py`
Generate training loss curves (MMM, TI, SCL, and combined loss).

```bash
python run_generate_training_curves.py \
    --checkpoint ../checkpoints/best_model.pt \
    --output_dir ../figures
```

**Outputs:**
- `figures/training_curves_hq.pdf`
- `figures/training_curves_hq.png`

### `run_generate_confusion_matrix.py`
Generate confusion matrix for tune type classification.

```bash
python run_generate_confusion_matrix.py \
    --results_file ../checkpoints/tune_type_classification_results.json \
    --output_dir ../figures
```

**Outputs:**
- `figures/tune_type_confusion.pdf`
- `figures/tune_type_confusion.png`

### `run_generate_umap.py`
Generate UMAP visualization colored by tune type and mode.

```bash
python run_generate_umap.py \
    --checkpoint ../checkpoints/best_model.pt \
    --data_dir ../data/processed \
    --device mps \
    --output_dir ../figures
```

**Outputs:**
- `figures/umap_combined.pdf`
- `figures/umap_combined.png`

## Quick Start: Reproduce All Results

To reproduce all results from the paper:

```bash
# 1. Train the model
python run_training.py --data_dir ../data/processed --output_dir ../checkpoints --device mps

# 2. Run all evaluations
python run_evaluate_tune_type.py --checkpoint ../checkpoints/best_model.pt --data_dir ../data/processed
python run_evaluate_mode.py --checkpoint ../checkpoints/best_model.pt --data_dir ../data/processed
python run_evaluate_linear_probing.py --checkpoint ../checkpoints/best_model.pt --data_dir ../data/processed
python run_evaluate_clustering.py --checkpoint ../checkpoints/best_model.pt --data_dir ../data/processed

# 3. Generate all figures
python run_generate_dataset_figures.py --data_dir ../data/processed --output_dir ../figures
python run_generate_training_curves.py --checkpoint ../checkpoints/best_model.pt --output_dir ../figures
python run_generate_confusion_matrix.py --results_file ../checkpoints/tune_type_classification_results.json --output_dir ../figures
python run_generate_umap.py --checkpoint ../checkpoints/best_model.pt --data_dir ../data/processed --output_dir ../figures
```

## Requirements

All scripts require the dependencies specified in `requirements.txt`:
- torch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- umap-learn
- tqdm

## Paper Reference

These scripts were used to generate all results in:

**"ABC2Vec: Self-Supervised Representation Learning for Irish Folk Music"**

- Dataset: https://huggingface.co/datasets/pianistprogrammer/abc2vec-irish-folk-dataset
- Model: https://huggingface.co/pianistprogrammer/abc2vec-model
- Code: https://github.com/pianistprogrammer/ABC2VEC
