#!/bin/bash
#SBATCH --account=def-gzhang-ab
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=resnet50_gpu
#SBATCH --output=%j_train.out
#SBATCH --error=%j_train.err

# Load required modules
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

# Activate virtual environment (already has PyTorch 2.9.0 with CUDA 12.6)
source venv/bin/activate

# Run training
python train.py \
  -num_epochs 6 \
  -head_epochs 2 \
  -batch_size 64 \
  -lr 1e-4 \
  -checkpoint_freq 3 \
  -early_stopping_patience 10 \
  -val_split 0.2 \
  -seed 30

# Print checkpoint summary
python - << 'PY'
import json, os
p = 'models/checkpoints/finetune_checkpoints_metadata.json'
if os.path.exists(p):
    m = json.load(open(p))
    print(f"\nSaved {len(m)} checkpoints:")
    for c in m:
        print(f"epoch={c['epoch']}, lr={c['learning_rate']:.2e}, path={c['checkpoint_path']}")
else:
    print("No checkpoint metadata found.")
PY