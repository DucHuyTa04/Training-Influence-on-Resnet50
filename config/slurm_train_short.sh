#!/bin/bash
#SBATCH --account=def-gzhang-ab
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=resnet50_test
#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err

module load StdEnv/2023
module load python/3.11.5
module load cuda/12.6

source venv/bin/activate

mkdir -p logs

python scripts/2_train.py \
  --num_epochs 6 \
  --head_epochs 2 \
  --batch_size 64 \
  --lr 1e-4 \
  --checkpoint_freq 3 \
  --early_stopping_patience 10 \
  --val_split 0.2 \
  --seed 30