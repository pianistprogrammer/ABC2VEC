# Ablation Study - Training Guide

This document describes the ablation study setup for ABC2Vec, which measures the contribution of each pre-training objective.

## Overview

The ablation study trains 7 model variants:

| Model | MMM | TI | SCL | Status | Output Directory |
|-------|-----|----|----|--------|------------------|
| **0. MMM + TI** (current) | ✓ | ✓ | ✗ | ✅ Trained | `checkpoints/best_model.pt` |
| 1. MMM only | ✓ | ✗ | ✗ | ⏳ Pending | `checkpoints/ablation/mmm_only/` |
| 2. TI only | ✗ | ✓ | ✗ | ⏳ Pending | `checkpoints/ablation/ti_only/` |
| 3. MMM + TI + SCL (full) | ✓ | ✓ | ✓ | ⏳ Pending | `checkpoints/ablation/mmm_ti_scl/` |
| 4. MMM + SCL | ✓ | ✗ | ✓ | ⏳ Pending | `checkpoints/ablation/mmm_scl/` |
| 5. TI + SCL | ✗ | ✓ | ✓ | ⏳ Pending | `checkpoints/ablation/ti_scl/` |
| 6. SCL only | ✗ | ✗ | ✓ | ⏳ Pending | `checkpoints/ablation/scl_only/` |

**Objectives:**
- **MMM**: Masked Music Modeling (predicts masked bars)
- **TI**: Transposition Invariance (contrastive learning across keys)
- **SCL**: Section Contrastive Learning (distinguishes tune sections)

## Training Options

### Option 1: Train All Models (Automated)

Run the batch script to train all 6 models sequentially:

```bash
./train_ablation_models.sh
```

**Time estimate:** ~18 hours per model × 6 = ~108 hours (~4.5 days) on Apple M4 Mac

### Option 2: Train Individual Models

Train models one at a time using the commands below. Useful for:
- Running models in parallel on multiple GPUs
- Resuming after interruption
- Training only specific variants

#### 1. MMM only
```bash
python abc2vec/scripts/run_training.py \
    --data_dir ./abc2vec/data/processed \
    --output_dir ./checkpoints/ablation/mmm_only \
    --d_model 256 --n_layers 6 --n_heads 8 \
    --batch_size 32 --grad_accum 4 --epochs 40 \
    --lr 1e-4 --lambda_mmm 1.0 --lambda_ti 0.0 --lambda_scl 0.0 \
    --num_workers 8 --device mps --amp
```

#### 2. TI only
```bash
python abc2vec/scripts/run_training.py \
    --data_dir ./abc2vec/data/processed \
    --output_dir ./checkpoints/ablation/ti_only \
    --d_model 256 --n_layers 6 --n_heads 8 \
    --batch_size 32 --grad_accum 4 --epochs 40 \
    --lr 1e-4 --lambda_mmm 0.0 --lambda_ti 1.0 --lambda_scl 0.0 \
    --num_workers 8 --device mps --amp
```

#### 3. MMM + TI + SCL (full model)
```bash
python abc2vec/scripts/run_training.py \
    --data_dir ./abc2vec/data/processed \
    --output_dir ./checkpoints/ablation/mmm_ti_scl \
    --d_model 256 --n_layers 6 --n_heads 8 \
    --batch_size 32 --grad_accum 4 --epochs 40 \
    --lr 1e-4 --lambda_mmm 1.0 --lambda_ti 0.5 --lambda_scl 0.5 \
    --num_workers 8 --device mps --amp
```

#### 4. MMM + SCL
```bash
python abc2vec/scripts/run_training.py \
    --data_dir ./abc2vec/data/processed \
    --output_dir ./checkpoints/ablation/mmm_scl \
    --d_model 256 --n_layers 6 --n_heads 8 \
    --batch_size 32 --grad_accum 4 --epochs 40 \
    --lr 1e-4 --lambda_mmm 1.0 --lambda_ti 0.0 --lambda_scl 0.5 \
    --num_workers 8 --device mps --amp
```

#### 5. TI + SCL
```bash
python abc2vec/scripts/run_training.py \
    --data_dir ./abc2vec/data/processed \
    --output_dir ./checkpoints/ablation/ti_scl \
    --d_model 256 --n_layers 6 --n_heads 8 \
    --batch_size 32 --grad_accum 4 --epochs 40 \
    --lr 1e-4 --lambda_mmm 0.0 --lambda_ti 1.0 --lambda_scl 0.5 \
    --num_workers 8 --device mps --amp
```

#### 6. SCL only
```bash
python abc2vec/scripts/run_training.py \
    --data_dir ./abc2vec/data/processed \
    --output_dir ./checkpoints/ablation/scl_only \
    --d_model 256 --n_layers 6 --n_heads 8 \
    --batch_size 32 --grad_accum 4 --epochs 40 \
    --lr 1e-4 --lambda_mmm 0.0 --lambda_ti 0.0 --lambda_scl 1.0 \
    --num_workers 8 --device mps --amp
```

## Monitoring Training

Each model saves:
- `best_model.pt` - Best checkpoint based on validation loss
- `training_history.json` - Loss curves for plotting
- `checkpoint_epochN.pt` - Epoch checkpoints
- `checkpoint_stepN.pt` - Step checkpoints (every 5000 steps)

Monitor progress:
```bash
# Watch training history
tail -f checkpoints/ablation/mmm_only/training_history.json

# Check validation loss
cat checkpoints/ablation/mmm_only/training_history.json | jq '.val_losses[-1]'
```

## After Training: Evaluation

Once all models are trained, evaluate them on downstream tasks:

```bash
# Tune type classification
for variant in mmm_only ti_only mmm_ti_scl mmm_scl ti_scl scl_only; do
    python abc2vec/scripts/run_evaluate_tune_type.py \
        --checkpoint checkpoints/ablation/$variant/best_model.pt \
        --data_dir abc2vec/data/processed \
        --output_dir checkpoints/ablation/$variant
done

# Mode classification
for variant in mmm_only ti_only mmm_ti_scl mmm_scl ti_scl scl_only; do
    python abc2vec/scripts/run_evaluate_mode.py \
        --checkpoint checkpoints/ablation/$variant/best_model.pt \
        --data_dir abc2vec/data/processed \
        --output_dir checkpoints/ablation/$variant
done

# Linear probing (all tasks)
for variant in mmm_only ti_only mmm_ti_scl mmm_scl ti_scl scl_only; do
    python abc2vec/scripts/run_evaluate_linear_probing.py \
        --checkpoint checkpoints/ablation/$variant/best_model.pt \
        --data_dir abc2vec/data/processed \
        --output_dir checkpoints/ablation/$variant
done
```

## Expected Results

The ablation study will reveal:
1. **MMM contribution**: Compare MMM+TI vs TI-only
2. **TI contribution**: Compare MMM+TI vs MMM-only
3. **SCL contribution**: Compare full model (MMM+TI+SCL) vs MMM+TI
4. **Objective interactions**: Compare pairs vs single objectives

Key metrics to compare:
- Tune type classification accuracy
- Mode classification accuracy
- Key detection accuracy
- Clustering quality (silhouette score)
- Retrieval performance (MRR, Recall@K)

## Hardware Requirements

- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: ~4GB per model × 7 = ~28GB total
- **Compute**: 
  - Apple Silicon (M4): ~18 hours per model
  - NVIDIA GPU (RTX 3090): ~8-10 hours per model
  - CPU only: Not recommended (100+ hours per model)

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size or gradient accumulation:
```bash
--batch_size 16 --grad_accum 8  # Same effective batch size
```

### SCL Dataset Missing
If you see "section_pairs.json not found", SCL will be automatically disabled.
Generate section pairs first (if needed):
```bash
python abc2vec/scripts/generate_section_pairs.py \
    --data_dir abc2vec/data/processed
```

### Resume Interrupted Training
```bash
python abc2vec/scripts/run_training.py \
    --resume checkpoints/ablation/mmm_only/checkpoint_epoch10.pt \
    --output_dir checkpoints/ablation/mmm_only \
    # ... other args same as original
```

## Notes

- All models use identical architecture (6 layers, 256-dim, 8 heads)
- Training hyperparameters are kept constant across variants
- Only the loss function weights (lambda_mmm, lambda_ti, lambda_scl) vary
- Random seed is controlled for reproducibility (set in training script)
