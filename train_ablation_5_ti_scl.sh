#!/bin/bash
# Ablation Model 5: TI + SCL
# Tests Transposition Invariance + Section Contrastive Learning

set -e

DATA_DIR="./data/processed"
OUTPUT_DIR="./checkpoints/ablation/ti_scl"
EPOCHS=40
BATCH_SIZE=32
GRAD_ACCUM=4
LR=1e-4
D_MODEL=256
N_LAYERS=6
N_HEADS=8
NUM_WORKERS=8
DEVICE="mps"  # Change to "cuda" if using NVIDIA GPU

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Training Ablation Model 5/6: TI + SCL"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo ""

# Check for existing checkpoints to resume from
LATEST_EPOCH=$(ls -t "$OUTPUT_DIR"/checkpoint_epoch*.pt 2>/dev/null | head -1)
LATEST_STEP=$(ls -t "$OUTPUT_DIR"/checkpoint_step*.pt 2>/dev/null | head -1)

RESUME_ARG=""
if [ -n "$LATEST_STEP" ]; then
    echo "Found existing checkpoint: $LATEST_STEP"
    read -p "Resume from this checkpoint? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESUME_ARG="--resume $LATEST_STEP"
        echo "Resuming training from $LATEST_STEP"
    fi
elif [ -n "$LATEST_EPOCH" ]; then
    echo "Found existing checkpoint: $LATEST_EPOCH"
    read -p "Resume from this checkpoint? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESUME_ARG="--resume $LATEST_EPOCH"
        echo "Resuming training from $LATEST_EPOCH"
    fi
fi

echo ""
echo "Starting training..."
echo "Press Ctrl+C to stop (progress will be saved)"
echo ""

python abc2vec/scripts/run_training.py \
    $RESUME_ARG \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_mmm 0.0 \
    --lambda_ti 1.0 \
    --lambda_scl 0.5 \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --max_steps 40000 \
    --amp

echo ""
echo "=============================================="
echo "✓ TI + SCL training complete!"
echo "=============================================="
echo "Model saved to: $OUTPUT_DIR/best_model.pt"
echo "Training history: $OUTPUT_DIR/training_history.json"
echo ""
