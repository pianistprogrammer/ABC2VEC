# Ablation Study - Individual Training Scripts

Each model has its own training script that can be run independently, stopped, and resumed.

## Available Scripts

| Script | Model | Objectives | Output Directory |
|--------|-------|------------|------------------|
| `train_ablation_1_mmm_only.sh` | MMM only | MMM: 1.0, TI: 0.0, SCL: 0.0 | `checkpoints/ablation/mmm_only/` |
| `train_ablation_2_ti_only.sh` | TI only | MMM: 0.0, TI: 1.0, SCL: 0.0 | `checkpoints/ablation/ti_only/` |
| `train_ablation_3_mmm_ti_scl.sh` | MMM+TI+SCL (full) | MMM: 1.0, TI: 0.5, SCL: 0.5 | `checkpoints/ablation/mmm_ti_scl/` |
| `train_ablation_4_mmm_scl.sh` | MMM+SCL | MMM: 1.0, TI: 0.0, SCL: 0.5 | `checkpoints/ablation/mmm_scl/` |
| `train_ablation_5_ti_scl.sh` | TI+SCL | MMM: 0.0, TI: 1.0, SCL: 0.5 | `checkpoints/ablation/ti_scl/` |
| `train_ablation_6_scl_only.sh` | SCL only | MMM: 0.0, TI: 0.0, SCL: 1.0 | `checkpoints/ablation/scl_only/` |

**Baseline (already trained):** `checkpoints/best_model.pt` (MMM+TI)

## Usage

### Start Training a Model

Simply run any script:
```bash
./train_ablation_1_mmm_only.sh
```

Each script will:
- ✅ Create the output directory
- ✅ Check for existing checkpoints
- ✅ Ask if you want to resume (if checkpoints found)
- ✅ Start/resume training
- ✅ Save progress automatically

### Stop Training

Press **Ctrl+C** at any time. Your progress is automatically saved every:
- **1 epoch** (~1.6 hours) → `checkpoint_epochN.pt`
- **5000 steps** (~30-40 minutes) → `checkpoint_stepN.pt`

### Resume Training

Just run the same script again:
```bash
./train_ablation_1_mmm_only.sh
```

The script will:
1. Detect the latest checkpoint
2. Ask if you want to resume
3. Continue from where it left off

**Or manually resume** by editing the script to uncomment the `--resume` line.

## Training Time

- **Per model:** ~63 hours (~2.6 days)
- **All 6 models:** ~378 hours (~16 days sequential)

## Monitoring Progress

### Check current training status:
```bash
# View last 20 lines of training output
tail -20 checkpoints/ablation/mmm_only/training_history.json

# Check latest validation loss
cat checkpoints/ablation/mmm_only/training_history.json | jq '.val_losses[-1]'

# See how many epochs completed
ls checkpoints/ablation/mmm_only/checkpoint_epoch*.pt | wc -l
```

### Check checkpoint status:
```bash
# List all checkpoints with timestamps
ls -lht checkpoints/ablation/mmm_only/checkpoint_*.pt

# Find latest checkpoint
ls -t checkpoints/ablation/mmm_only/checkpoint_*.pt | head -1
```

## What Gets Saved

Each model directory contains:
```
checkpoints/ablation/mmm_only/
├── model_config.json           # Architecture config
├── training_history.json       # Loss curves (updated during training)
├── best_model.pt              # Best checkpoint (lowest val loss)
├── final_model.pt             # Final checkpoint (when complete)
├── checkpoint_epoch1.pt       # Epoch checkpoints
├── checkpoint_epoch2.pt
├── ...
├── checkpoint_step5000.pt     # Step checkpoints (every 5000 steps)
├── checkpoint_step10000.pt
└── ...
```

## Recommended Training Order

### Priority 1: Core ablation (most important)
1. `./train_ablation_1_mmm_only.sh` - Shows MMM contribution
2. `./train_ablation_2_ti_only.sh` - Shows TI contribution
3. `./train_ablation_3_mmm_ti_scl.sh` - Tests if SCL improves current model

These 3 models give you the main ablation story.

### Priority 2: Complete analysis
4. `./train_ablation_4_mmm_scl.sh`
5. `./train_ablation_5_ti_scl.sh`
6. `./train_ablation_6_scl_only.sh`

## Configuration

Each script uses these settings (edit the script to change):
- `EPOCHS=40` - Number of training epochs
- `BATCH_SIZE=32` - Batch size per GPU
- `GRAD_ACCUM=4` - Gradient accumulation steps
- `DEVICE="mps"` - Device (change to "cuda" for NVIDIA GPU)
- `NUM_WORKERS=8` - DataLoader workers

## Troubleshooting

### "Out of memory" error
Edit the script and reduce batch size:
```bash
BATCH_SIZE=16  # Was 32
GRAD_ACCUM=8   # Was 4 (keeps effective batch size same)
```

### Resume not working
Manually specify checkpoint in the script by uncommenting and editing:
```bash
RESUME_ARG="--resume checkpoints/ablation/mmm_only/checkpoint_epoch15.pt"
```

### Want to start fresh
Delete the model's directory:
```bash
rm -rf checkpoints/ablation/mmm_only/
./train_ablation_1_mmm_only.sh  # Starts from scratch
```

## After Training: Evaluation

Once models finish, evaluate them:
```bash
# Run tune type classification on all models
for model in mmm_only ti_only mmm_ti_scl mmm_scl ti_scl scl_only; do
    python abc2vec/scripts/run_evaluate_tune_type.py \
        --checkpoint checkpoints/ablation/$model/best_model.pt \
        --data_dir abc2vec/data/processed \
        --output_dir checkpoints/ablation/$model
done
```

See `ABLATION_STUDY.md` for full evaluation instructions.
