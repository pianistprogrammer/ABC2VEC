#!/bin/bash
# Ablation Study Training Script
# Trains 6 model variants to measure the contribution of each objective
# Current model (MMM+TI) is already trained as best_model.pt

set -e

DATA_DIR="./data/processed"
BASE_OUTPUT_DIR="./checkpoints/ablation"
EPOCHS=40
BATCH_SIZE=32
GRAD_ACCUM=4
LR=1e-4
D_MODEL=256
N_LAYERS=6
N_HEADS=8
NUM_WORKERS=8
DEVICE="mps"  # Change to "cuda" if using NVIDIA GPU

# Create ablation directory
mkdir -p "$BASE_OUTPUT_DIR"

echo "=============================================="
echo "ABC2Vec Ablation Study - Training 6 Variants"
echo "=============================================="
echo ""
echo "Each model will be saved to: $BASE_OUTPUT_DIR/<variant_name>/"
echo ""

# ============================================
# Model 1: MMM only
# ============================================
echo "----------------------------------------"
echo "Training Model 1/6: MMM only"
echo "----------------------------------------"
python abc2vec/scripts/run_training.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT_DIR/mmm_only" \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_mmm 1.0 \
    --lambda_ti 0.0 \
    --lambda_scl 0.0 \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --amp

echo ""
echo "✓ MMM only complete!"
echo ""

# ============================================
# Model 2: TI only
# ============================================
echo "----------------------------------------"
echo "Training Model 2/6: TI only"
echo "----------------------------------------"
python abc2vec/scripts/run_training.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT_DIR/ti_only" \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_mmm 0.0 \
    --lambda_ti 1.0 \
    --lambda_scl 0.0 \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --amp

echo ""
echo "✓ TI only complete!"
echo ""

# ============================================
# Model 3: MMM + TI + SCL (full model)
# ============================================
echo "----------------------------------------"
echo "Training Model 3/6: MMM + TI + SCL (full)"
echo "----------------------------------------"
python abc2vec/scripts/run_training.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT_DIR/mmm_ti_scl" \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_mmm 1.0 \
    --lambda_ti 0.5 \
    --lambda_scl 0.5 \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --amp

echo ""
echo "✓ MMM + TI + SCL complete!"
echo ""

# ============================================
# Model 4: MMM + SCL
# ============================================
echo "----------------------------------------"
echo "Training Model 4/6: MMM + SCL"
echo "----------------------------------------"
python abc2vec/scripts/run_training.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT_DIR/mmm_scl" \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_mmm 1.0 \
    --lambda_ti 0.0 \
    --lambda_scl 0.5 \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --amp

echo ""
echo "✓ MMM + SCL complete!"
echo ""

# ============================================
# Model 5: TI + SCL
# ============================================
echo "----------------------------------------"
echo "Training Model 5/6: TI + SCL"
echo "----------------------------------------"
python abc2vec/scripts/run_training.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT_DIR/ti_scl" \
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
    --amp

echo ""
echo "✓ TI + SCL complete!"
echo ""

# ============================================
# Model 6: SCL only
# ============================================
echo "----------------------------------------"
echo "Training Model 6/6: SCL only"
echo "----------------------------------------"
python abc2vec/scripts/run_training.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$BASE_OUTPUT_DIR/scl_only" \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --lambda_mmm 0.0 \
    --lambda_ti 0.0 \
    --lambda_scl 1.0 \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --amp

echo ""
echo "✓ SCL only complete!"
echo ""

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "✓ Ablation Study Training Complete!"
echo "=============================================="
echo ""
echo "Trained models:"
echo "  1. MMM only         → $BASE_OUTPUT_DIR/mmm_only/best_model.pt"
echo "  2. TI only          → $BASE_OUTPUT_DIR/ti_only/best_model.pt"
echo "  3. MMM + TI + SCL   → $BASE_OUTPUT_DIR/mmm_ti_scl/best_model.pt"
echo "  4. MMM + SCL        → $BASE_OUTPUT_DIR/mmm_scl/best_model.pt"
echo "  5. TI + SCL         → $BASE_OUTPUT_DIR/ti_scl/best_model.pt"
echo "  6. SCL only         → $BASE_OUTPUT_DIR/scl_only/best_model.pt"
echo ""
echo "Existing baseline:"
echo "  0. MMM + TI (current) → ./checkpoints/best_model.pt"
echo ""
echo "Next steps:"
echo "  1. Run evaluation on all models to compare performance"
echo "  2. Generate ablation comparison plots"
echo ""
