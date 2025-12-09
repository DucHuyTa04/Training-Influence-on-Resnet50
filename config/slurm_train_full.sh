#!/bin/bash
#SBATCH --account=def-gzhang-ab
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=resnet50_full_train
#SBATCH --output=%j_train.out
#SBATCH --error=%j_train.err

# Full training run with checkpoints for TracIn
# Saves checkpoints every 10 epochs for 100 epochs total (with early stopping)

# Load required modules
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

# Activate virtual environment
source venv/bin/activate

# Run full training (100 epochs max, checkpoints every 10 epochs, early stopping enabled)
python scripts/2_train.py \
  -num_epochs 100 \
  -head_epochs 3 \
  -batch_size 64 \
  -lr 1e-4 \
  -checkpoint_freq 10 \
  -early_stopping_patience 15 \
  -val_split 0.2 \
  -seed 42

# Print checkpoint summary
python - << 'PY'
import json, os
p = 'models/checkpoints/finetune_checkpoints_metadata.json'
if os.path.exists(p):
    m = json.load(open(p))
    print(f"\n✓ Saved {len(m)} checkpoints:")
    for c in m:
        print(f"  Epoch {c['epoch']}: acc={c['val_acc']:.4f}, lr={c['learning_rate']:.2e}")
else:
    print("No checkpoint metadata found.")
PY
