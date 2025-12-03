#!/bin/bash
set -euo pipefail
#SBATCH --account=def-gzhang-ab
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=resnet50_short
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load python/3.10

if [ ! -d "venv" ]; then
  python -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip --no-index
pip install --no-index torch torchvision tqdm pandas numpy psutil

mkdir -p logs

python train.py \
  -num_epochs 6 \
  -head_epochs 2 \
  -batch_size 64 \
  -lr 1e-4 \
  -checkpoint_freq 3 \
  -early_stopping_patience 10 \
  -val_split 0.2 \
  -seed 30

python - << 'PY'
import json, os
p = 'models/checkpoints/finetune_checkpoints_metadata.json'
if os.path.exists(p):
    m = json.load(open(p))
    print(f"\nSaved {len(m)} checkpoints:")
    for c in m:
        print(f"epoch={c['epoch']}, lr={c['learning_rate']:.2e}, path={c['checkpoint_path']}")
else:
    print("\nNo checkpoint metadata found.")
PY