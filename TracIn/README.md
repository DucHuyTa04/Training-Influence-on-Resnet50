# TracIn Implementation for ResNet50 Animals-10

Implementation of **TracIn (Tracing Influence)** for analyzing training data influence on the ResNet50 Animals-10 model.

**Paper**: [Estimating Training Data Influence by Tracing Gradient Descent](https://arxiv.org/abs/2002.08484)

## What is TracIn?

TracIn quantifies how much each training sample influences model predictions by computing gradient similarity. This helps:
- Find **helpful training samples** (positive influence)
- Identify **harmful training samples** (negative influence - potentially mislabeled)
- Debug predictions and improve dataset quality

---

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision numpy pandas matplotlib seaborn tqdm pillow
```

### 2. Compute Influence Scores
```bash
# Quick test (1-2 min)
python TracIn_Resnet50.py --train_subset 100 --test_subset 10

# Recommended (5-10 min)
python TracIn_Resnet50.py --train_subset 500 --test_subset 50

# Full dataset (several hours)
python TracIn_Resnet50.py
```

### 3. Analyze Results
```bash
python analyze_results.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/analysis_dashboard.png` | **Main visualization** - overview, distributions, top samples |
| `results/top_influences/{class}/` | **Top 10 helpful/harmful images** per class |
| `results/{class}_summary.png` | Per-class detailed analysis |
| `results/influence_scores.csv` | Raw influence matrix for custom analysis |
| `results/influence_summary.csv` | Per-test summary statistics |

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