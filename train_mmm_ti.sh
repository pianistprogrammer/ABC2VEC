#!/bin/bash

# Train MMM+TI ablation model
# This trains a model with only Masked Music Modeling and Transposition Invariance objectives
# (no Section Contrastive Learning)

echo "=========================================="
echo "Training MMM+TI Ablation Model"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - MMM weight (λ_MMM): 1.0"
echo "  - TI weight (λ_TI): 0.5"
echo "  - SCL weight (λ_SCL): 0.0 (disabled)"
echo "  - Output: checkpoints/ablation/mmm_ti/"
echo ""
echo "Estimated time: ~18 hours"
echo "=========================================="
echo ""

python abc2vec/scripts/run_train.py \
    --data_dir data/processed \
    --checkpoint_dir checkpoints/ablation/mmm_ti \
    --batch_size 128 \
    --epochs 40 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --max_steps 40000 \
    --device mps \
    --lambda_mmm 1.0 \
    --lambda_ti 0.5 \
    --lambda_scl 0.0 \
    --save_every 2000 \
    --log_every 100 \
    --seed 42

echo ""
echo "=========================================="
echo "Training complete!"
echo "Model saved to: checkpoints/ablation/mmm_ti/"
echo ""
echo "Next steps:"
echo "  1. Run ablation evaluation:"
echo "     python abc2vec/scripts/run_ablation_study.py"
echo "  2. Regenerate figures:"
echo "     python abc2vec/scripts/run_generate_ablation_figures.py"
echo "=========================================="
