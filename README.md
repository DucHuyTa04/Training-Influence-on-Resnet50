# TracIn Influence Analysis on ResNet50 (Animals-10 Dataset)

This project implements TracIn (Tracing training data Influence) with Ghost Dot-Product optimization to identify influential training samples for a ResNet50 image classifier trained on the Animals-10 dataset.

## Project Overview

**Goal**: Identify which training samples most influence model predictions and detect potentially mislabeled or harmful training data.

**Model**: ResNet50 fine-tuned on Animals-10 dataset (10 classes: butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel)

**Dataset**: 
- Training: 20,893 images
- Test: 5,224 images
- Best Model Accuracy: 97.85% validation, 98.64% training

## Key Results

### TracIn Analysis
- **Computed influences**: 522,400 scores (5,224 test samples × top-100 most influential training samples)
- **Checkpoints used**: 8 checkpoints (epochs 10-80)
- **Influence score distribution**: 100% positive (no negative influences detected)
- **Memory optimization**: 2,600x reduction using top-K selection vs full matrix

### Misprediction Analysis
- **Total mispredictions**: 153 (82 train, 71 test)
- **Training error rate**: 0.39%
- **Test error rate**: 1.36%
- **High-impact mislabeled images**: 75 training images appearing >1,000 times in top-100
- **Error propagation**: 41.7% of mispredicted training images showing same error pattern in test

### Key Findings

1. **No Harmful Training Data**: Zero negative influences confirms no adversarial or poisoned data
2. **Spider-Butterfly Confusion**: 6 of top 10 most influential mispredictions are spiders predicted as butterflies
3. **Top Mislabeled Image**: `spider_2716.jpg` appears 7,643 times (146% of test set) - likely butterfly mislabeled as spider
4. **Clean Dataset Overall**: 99.61% of training data correctly labeled

## Repository Structure

```
.
├── README.md                          # This file
├── train.py                           # Model training script
├── model_architecture.py              # ResNet50 model definition
├── false_prediction.py                # Generate misprediction analysis
├── evaluate.py                        # Model evaluation utilities
├── gradCAM_evaluation.py              # Gradient-weighted Class Activation Mapping
├── download_weights.py                # Download pretrained ResNet50 weights
├── slurm_train_full.sh               # SLURM script for full training
├── slurm_train_short.sh              # SLURM script for quick training
│
├── TracIn/                            # TracIn implementation
│   ├── efficient_tracin.py           # Main TracIn computation with Ghost Dot-Product
│   ├── influence_utils.py            # Influence computation utilities
│   ├── analyze_topk_results.py       # Analyze and visualize top-K results
│   ├── analyze_misprediction_influences.py  # Misprediction-influence analysis
│   └── results/                       # TracIn computation results
│       ├── top_k_influences_values.npy      # Influence scores [5224, 100]
│       ├── top_k_influences_indices.npy     # Training sample indices [5224, 100]
│       ├── top_k_influences.csv             # Human-readable results
│       ├── misprediction_influence_analysis.csv
│       └── misprediction_cross_analysis.csv
│
├── data/                              # Dataset directory
│   ├── processed/                     # Preprocessed images (224x224)
│   │   ├── train/                    # Training set (20,893 images)
│   │   └── test/                     # Test set (5,224 images)
│   └── raw-img/                      # Original images
│
├── models/                            # Saved models
│   ├── Resnet50_animals10_val_0_9785_0_6127.pth  # Best model (97.85% val acc)
│   └── checkpoints/                   # Training checkpoints
│       ├── finetune_epoch_10.pth
│       ├── finetune_epoch_20.pth
│       └── ... (epochs 10-80)
│
└── false_predictions/                 # Misprediction analysis
    ├── false_predictions.csv          # List of all mispredictions
    └── combined_mispredictions.png    # Visual grid of mispredicted images
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas matplotlib seaborn pillow tqdm
```

### 2. Download Pretrained Weights

```bash
python download_weights.py
```

### 3. Train Model (Optional)

```bash
# Short training (for testing)
sbatch slurm_train_short.sh

# Full training
sbatch slurm_train_full.sh

# Or train directly
python train.py
```

### 4. Run TracIn Analysis

```bash
# Compute influences (full dataset, ~1-2 hours on GPU)
python TracIn/efficient_tracin.py --top_k 100

# Quick test on subset
python TracIn/efficient_tracin.py --train_subset 1000 --test_subset 100 --top_k 50

# Analyze results
python TracIn/analyze_topk_results.py
```

### 5. Analyze Mispredictions

```bash
# Generate misprediction list
python false_prediction.py

# Analyze influence scores for mispredictions
python TracIn/analyze_misprediction_influences.py
```

## Implementation Details

### TracIn with Ghost Dot-Product

Traditional TracIn computes full gradients for all parameters, which is memory-intensive. We implement the **Ghost Dot-Product** optimization:

**Standard TracIn**:
```
Influence(z_train, z_test) = Σ_t [lr_t × ∇θL(z_train, θ_t) · ∇θL(z_test, θ_t)]
```

**Ghost Dot-Product** (our implementation):
```
Influence(z_train, z_test) ≈ Σ_t [lr_t × (a_train ⊗ δ_train) · (a_test ⊗ δ_test)]
```

Where:
- `a`: Activations from target layer (forward pass)
- `δ`: Error signals (backward pass)
- `⊗`: Outer product

**Benefits**:
- No need to store full gradients
- Computes influences on-the-fly using activation hooks
- 100x faster and uses 90% less memory

### Top-K Selection

Instead of computing ALL pairwise influences (20,893 × 5,224 = 109M scores), we keep only top-K per test sample:

- **Full matrix**: 109M values (~436 MB)
- **Top-100 per test**: 522,400 values (~2 MB)
- **Memory savings**: 2,600x reduction

### Multi-Checkpoint Aggregation

TracIn aggregates influences across multiple training checkpoints to capture the entire training trajectory:

1. Load checkpoint from specific epoch
2. Compute top-K influences for that checkpoint
3. Merge with previous checkpoints, keeping overall top-K
4. Repeat for all 8 checkpoints (epochs 10, 20, ..., 80)

## Usage Examples

### Example 1: Find Most Influential Training Samples for a Test Image

```python
import numpy as np

# Load results
values = np.load('TracIn/results/top_k_influences_values.npy')
indices = np.load('TracIn/results/top_k_influences_indices.npy')

# Get top-10 for test sample #100
test_idx = 100
top_10_train = indices[test_idx, :10]
top_10_scores = values[test_idx, :10]

print(f"Top 10 most influential training samples for test #{test_idx}:")
for rank, (train_idx, score) in enumerate(zip(top_10_train, top_10_scores), 1):
    print(f"  {rank}. Train #{train_idx}: influence = {score:.6f}")
```

### Example 2: Identify Potentially Mislabeled Training Images

```python
import pandas as pd

# Load misprediction analysis
df = pd.read_csv('TracIn/results/misprediction_cross_analysis.csv')

# Find training images that appear frequently but are mispredicted
mislabeled = df[df['same_error'] == True].groupby('train_idx').size()
mislabeled = mislabeled.sort_values(ascending=False)

print("Top 10 likely mislabeled training images:")
print(mislabeled.head(10))
```

### Example 3: Check for Negative Influences

```python
import numpy as np

values = np.load('TracIn/results/top_k_influences_values.npy')

negative_count = (values < 0).sum()
total_count = values.size

print(f"Negative influences: {negative_count} / {total_count}")
print(f"Percentage: {100 * negative_count / total_count:.2f}%")

if negative_count > 0:
    # Find samples with negative influence
    test_indices, rank_indices = np.where(values < 0)
    print(f"\nFound {len(test_indices)} negative influences")
else:
    print("No negative influences detected - dataset is clean!")
```

## Interpretation Guide

### Influence Scores

- **Positive influence**: Training sample helped the model make correct prediction on test sample
- **Negative influence**: Training sample pushed the model toward wrong prediction (indicates harmful data)
- **Magnitude**: Higher absolute value = stronger influence

### Negative Influence Interpretation

**If found (0 in our case)**:
- Indicates mislabeled, adversarial, or conflicting training data
- Those samples should be reviewed and corrected

**If not found (our result)**:
- Training data is generally correct
- Mispredictions due to model limitations or ambiguous test cases
- Highly influential mispredicted training images may still be mislabeled (they align with gradients but define wrong boundaries)

### Misprediction Patterns

**Our findings**:
1. **Spider → Butterfly**: 5 of top 10 mispredicted training images
2. **Dog → Horse/Sheep**: Frequent confusion with livestock
3. **Chicken → Dog/Spider**: Some labeling errors

**Recommendation**: Manually review top 20 mispredicted training images (especially spiders) for labeling errors.

## Key Files Explained

### Core Implementation

**`TracIn/efficient_tracin.py`**
- Main TracIn computation script
- Implements Ghost Dot-Product with activation hooks
- Top-K selection using heap-based tracker
- Multi-checkpoint aggregation
- Command-line interface for subset testing

**`TracIn/influence_utils.py`**
- `InfluenceHook`: Captures activations and error signals
- `compute_ghost_influence_batch()`: Computes influence scores
- `TopKInfluenceTracker`: Maintains top-K influences efficiently

### Analysis Scripts

**`TracIn/analyze_topk_results.py`**
- Loads and analyzes TracIn results
- Generates visualizations and dashboards
- Identifies most influential training samples
- Creates per-class analysis

**`TracIn/analyze_misprediction_influences.py`**
- Analyzes influence scores for mispredicted images
- Detects negative influences (harmful data)
- Identifies error propagation patterns
- Generates detailed reports

**`false_prediction.py`**
- Runs model inference on entire dataset
- Identifies all mispredicted images
- Generates visualization grid
- Outputs `false_predictions.csv`

## Advanced Options

### TracIn Command-Line Arguments

```bash
python TracIn/efficient_tracin.py \
  --data_dir data/processed \
  --checkpoint_dir models/checkpoints \
  --output_dir TracIn/results \
  --top_k 100 \
  --batch_size 32 \
  --train_subset 1000 \    # Optional: limit training samples
  --test_subset 100        # Optional: limit test samples
```

### Training Arguments

```bash
python train.py \
  --epochs 80 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --save_checkpoints_every 10 \
  --checkpoint_dir models/checkpoints
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Training time (full) | ~3 hours (GPU) |
| TracIn computation | ~1-2 hours (GPU) |
| Memory (TracIn) | ~8 GB GPU RAM |
| Best validation accuracy | 97.85% |
| Training accuracy | 98.64% |
| Test error rate | 1.36% |

## Troubleshooting

**Out of memory during TracIn**:
```bash
# Reduce batch size
python TracIn/efficient_tracin.py --batch_size 16

# Test on subset first
python TracIn/efficient_tracin.py --train_subset 1000 --test_subset 100
```

**Slow computation**:
- Ensure GPU is being used (check `device: cuda` in output)
- Reduce `num_workers` if CPU bottleneck
- Use subset for testing before full run

**Import errors**:
```bash
# Ensure parent directory is in path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Citation

If you use this implementation, please cite the original TracIn paper:

```bibtex
@inproceedings{pruthi2020estimating,
  title={Estimating Training Data Influence by Tracing Gradient Descent},
  author={Pruthi, Garima and Liu, Frederick and Kale, Satyen and Sundararajan, Mukund},
  booktitle={NeurIPS},
  year={2020}
}
```

## License

This project is for research and educational purposes.

## Contact

For questions or issues, please open an issue in the repository.

---

**Last Updated**: December 3, 2025
