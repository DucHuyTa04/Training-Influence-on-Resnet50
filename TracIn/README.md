# Efficient TracIn with Ghost Dot-Product

**New implementation** based on professor's requirements and Lei's optimization techniques.

## Key Features

**Ghost Dot-Product**: Compute influences using activation/error signals instead of full gradients  
**Top-K Selection**: Keep only K most influential samples per test point (massive memory savings)  
**Multi-Checkpoint**: Aggregate influences across multiple training checkpoints  
**Tile-and-Stitch**: Process large datasets in chunks to avoid OOM  
**GPU Optimized**: Runs efficiently on NVIDIA H100  

**Memory Improvement**: Instead of 26k × 2.6k matrix (676M values), we get 2.6k × 100 (260k values) → **2600x reduction!**

---

## Quick Start

### 1. Train Model with Checkpoints
```bash
# First, train a model that saves checkpoints
sbatch slurm_train_full.sh  # 30 epochs, saves every 5 epochs
```

### 2. Test TracIn on Small Subset
```bash
# Quick test (~2 min on login node)
bash test_tracin.sh
```

### 3. Run Full TracIn Analysis
```bash
# After checkpoints are ready
python TracIn/efficient_tracin.py --top_k 100
```

---

## Command-Line Options

```bash
python TracIn/efficient_tracin.py [OPTIONS]

Options:
  --data_dir PATH           Data directory (default: data/processed)
  --checkpoint_dir PATH     Checkpoint directory (default: models/checkpoints)
  --output_dir PATH         Output directory (default: TracIn/results)
  --top_k INT               Top influences per test sample (default: 100)
  --batch_size INT          Batch size (default: 32)
  --train_subset INT        Limit train samples (default: None = all)
  --test_subset INT         Limit test samples (default: None = all)
```

**Examples**:
```bash
# Small test
python TracIn/efficient_tracin.py --train_subset 100 --test_subset 10 --top_k 20

# Medium analysis
python TracIn/efficient_tracin.py --train_subset 1000 --test_subset 100 --top_k 50

# Full dataset with top-100
python TracIn/efficient_tracin.py --top_k 100
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/top_k_influences_values.npy` | Influence scores [num_test, k] |
| `results/top_k_influences_indices.npy` | Train sample indices [num_test, k] |
| `results/top_k_influences.csv` | Human-readable CSV format |

---

## Understanding Results

**Dashboard shows**:
- Overview statistics (dataset size, health score)
- Positive/negative influence distributions with mean values
- Average influence per test sample (green=good, red=bad)
- Top 10 most helpful training samples
- Key findings & recommendations

**Health Score**:
- **>95%**: EXCELLENT - high quality training data
- **80-95%**: GOOD - minor issues possible
- **60-80%**: FAIR - review training data
- **<60%**: NEEDS ATTENTION - check for mislabeled samples

**What to do**:
1. Open `analysis_dashboard.png` for overview
2. Check `top_influences/` folder for actual influential images
3. Review samples in `negative/` folder if health score is low

---

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_path` | `models/Resnet50_animals10_val_0_9796_0_5963.pth` | Trained model path |
| `--data_dir` | `data/processed` | Data directory |
| `--output_dir` | `results` | Output directory |
| `--train_subset` | None (all) | Number of training samples |
| `--test_subset` | None (all) | Number of test samples |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 1e-4 | Learning rate for scaling |