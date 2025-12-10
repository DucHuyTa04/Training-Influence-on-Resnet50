# Training Influence Analysis on ResNet50

A comprehensive pipeline for training ResNet50 on the Animals-10 dataset and using TracIn (Training Data Attribution using Influence Functions) to identify mislabeled images through influence analysis.

## Project Overview

This project implements a complete machine learning workflow:

1. **Model Training**: Fine-tune ResNet50 (ImageNet pretrained) on Animals-10 dataset
2. **Misprediction Detection**: Identify images the model misclassifies  
3. **Influence Computation**: Use TracIn with Ghost Dot-Product optimization to compute training data influence scores
4. **Mislabel Discovery**: Cross-reference mispredictions with influence scores to find likely mislabeled training images
5. **Visual Inspection**: Generate detailed visualizations for human review

**Key Innovation**: Efficient TracIn implementation using Ghost Dot-Product for scalable influence computation on large datasets.

## Repository Structure

```
Training-Influence-on-Resnet50/
│
├── scripts/                          # All executable Python scripts (numbered by execution order)
│   ├── 1_download_weights.py        # Download ImageNet pretrained ResNet50 weights
│   ├── 2_train.py                   # Two-stage fine-tuning (head-only → full fine-tuning)
│   ├── 3_detect_mispredictions.py   # Identify model mispredictions on test set
│   ├── 4_compute_influence.py       # TracIn influence score computation (Ghost Dot-Product)
│   ├── 5a_generate_dashboards.py    # Create influence analysis visualizations
│   ├── 5b_cross_reference_analysis.py # Cross-reference mispredictions with influences
│   ├── 6_inspect_mislabeled.py      # Generate visual inspection grid for mislabeled candidates
│   ├── 7_inspect_influential.py     # Find and visualize most helpful/harmful training images
│   └── utils/                       # Shared utilities
│       ├── model_architecture.py    # ResNet50 model definition
│       ├── version_manager.py       # Model versioning system
│       └── influence_utils.py       # TracIn computation utilities
│
├── config/                          # Configuration files
│   ├── slurm_train_full.sh         # SLURM job: full training (100 epochs)
│   └── slurm_train_short.sh        # SLURM job: quick test (6 epochs)
│
├── models/                          # Model storage
│   ├── best/                        # Best performing models (organized by version)
│   │   ├── v1/
│   │   │   └── model_v1.pth
│   │   └── v2/
│   │       └── model_v2.pth
│   ├── checkpoints/                 # Training checkpoints (every N epochs, by version)
│   │   ├── v1/
│   │   │   ├── head_only_epoch_*.pth
│   │   │   ├── finetune_epoch_*.pth
│   │   │   └── finetune_metadata.json
│   │   └── v2/
│   ├── pretrained/                  # Pretrained weights (ImageNet)
│   │   └── resnet50_imagenet1k_v1.pth
│   └── version_registry.json        # Version tracking metadata
│
├── outputs/                         # All generated outputs (organized by version)
│   ├── v1/
│   │   ├── mispredictions/              # Misprediction detection results
│   │   │   ├── false_predictions.csv    # List of all mispredictions
│   │   │   └── mispredictions_grid.png  # Visual grid of mispredicted images
│   │   ├── influence_analysis/          # TracIn results
│   │   │   ├── top_k_influences_values.npy
│   │   │   ├── top_k_influences_indices.npy
│   │   │   ├── top_k_influences.csv
│   │   │   ├── overall_dashboard.png
│   │   │   ├── per_class/               # Per-class influence visualizations
│   │   │   ├── misprediction_influence_analysis.csv
│   │   │   └── misprediction_cross_analysis.csv
│   │   └── inspection/                  # Visual inspection outputs
│   │       ├── mislabeled_candidates.png
│   │       ├── top_helpful_images.png
│   │       ├── top_harmful_images.png
│   │       └── detailed/
│   │           ├── helpful/
│   │           └── harmful/
│   └── v2/
│       └── ...
│
├── data/                            # Dataset storage
│   ├── raw/                         # Original downloaded images (10 class folders)
│   └── processed/                   # Preprocessed images (224x224, normalized)
│       ├── train/                   # Training split (80%)
│       └── test/                    # Test split (20%)
│
└── preprocess_data.ipynb            # Jupyter notebook for data preprocessing
```

## Quick Start

### Prerequisites

```bash
# Python 3.8+ with CUDA support
module load python/3.11.5 cuda/12.6

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download Animals-10 dataset and place in `data/raw/` with 10 class folders:
   - `butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel`

2. Preprocess images:
   ```bash
   jupyter notebook preprocess_data.ipynb
   ```

### Pipeline

```bash
# Step 1: Download pretrained weights
python scripts/1_download_weights.py

# Step 2: Train model
python scripts/2_train.py \
  --version 1 \                      # Model version (auto-assigned if omitted)
  --num_epochs 100 \                 # Fine-tuning epochs
  --head_epochs 5 \                  # Head-only training epochs
  --batch_size 64 \                  # Batch size
  --lr 1e-4 \                        # Learning rate
  --checkpoint_freq 10 \             # Checkpoint frequency (epochs)
  --early_stopping_patience 15 \     # Early stopping patience
  --val_split 0.2 \                  # Validation split ratio
  --seed 30                          # Random seed

# Step 3: Detect mispredictions
python scripts/3_detect_mispredictions.py \
  --version 1 \                      # Model version to evaluate
  --model_path models/best/v1/model_v1.pth  # Or direct path to model (overrides version)

# Step 4: Compute TracIn influence scores
python scripts/4_compute_influence.py \
  --version 1 \                      # Model version number
  --top_k 100 \                      # Top influences per test sample
  --batch_size 32 \                  # Batch size
  --data_dir data/processed \        # Data directory
  --train_subset None \              # Train subset (None = all)
  --test_subset None                 # Test subset (None = all)

# Step 5a: Generate analysis dashboards
python scripts/5a_generate_dashboards.py \
  --results_dir outputs/v1/influence_analysis  # Results directory

# Step 5b: Cross-reference analysis
python scripts/5b_cross_reference_analysis.py \
  --mispredictions_csv outputs/v1/mispredictions/false_predictions.csv \  # Mispredictions file
  --influence_dir outputs/v1/influence_analysis                           # Influence results directory

# Step 6: Inspect mislabeled candidates
python scripts/6_inspect_mislabeled.py \
  --results_dir outputs/v1/influence_analysis \  # Results directory
  --top_n 20 \                                   # Number of top candidates to inspect
  --threshold 1000 \                             # High-priority threshold
  --output outputs/inspection/mislabeled_candidates.png  # Output image path

# Step 7: Inspect most influential images
python scripts/7_inspect_influential.py \
  --results_dir outputs/v1/influence_analysis \      # Results directory
  --top_n 20 \                                       # Number of top images to show
  --output_helpful outputs/inspection/top_helpful_images.png \  # Output for helpful images
  --output_harmful outputs/inspection/top_harmful_images.png    # Output for harmful images
```

### SLURM Batch Jobs

```bash
# Full training (10 hours, 100 epochs)
sbatch config/slurm_train_full.sh

# Quick test (1 hour, 6 epochs)
sbatch config/slurm_train_short.sh
```

## Model Versioning System

The training pipeline automatically assigns version numbers to each training run and organizes all outputs in version-specific directories.

### Automatic Versioning

```bash
# Train new model (automatically assigns next version)
python scripts/2_train.py --num_epochs 100

# Output:
# [VERSION] Model version: 1
# models/best/v1/model_v1.pth
# models/checkpoints/v1/*.pth
# Entry added to models/version_registry.json
```

### Working with Specific Versions

```bash
# Use specific version for analysis
python scripts/3_detect_mispredictions.py --version 1
python scripts/4_compute_influence.py --version 1 --top_k 100

# Or use custom model path
python scripts/3_detect_mispredictions.py --model_path models/best/v2/model_v2.pth
```

### Version Registry

The `models/version_registry.json` file tracks all trained models:

```json
{
  "1": {
    "timestamp": "2024-12-09T10:30:00",
    "version": 1,
    "val_accuracy": 0.9816,
    "val_loss": 0.6165,
    "num_epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "seed": 42,
    "device": "cuda",
    "model_path": "models/best/v1/model_v1.pth",
    "checkpoint_dir": "models/checkpoints/v1",
    "output_dir": "outputs/v1"
  }
}
```

### Benefits

1. **Organization**: All outputs for a model are in one place (`outputs/vX/`)
2. **Reproducibility**: Easy to track which outputs came from which training run
3. **Comparison**: Compare different model versions side-by-side
4. **No Conflicts**: Multiple training runs won't overwrite each other

## Training Details

### Two-Stage Training
1. **Head-only fine-tuning**: Train only final classification layer (few epochs)
2. **Full fine-tuning**: Unfreeze all layers with small learning rate

### Features
- **ReduceLROnPlateau**: Adaptive learning rate based on validation performance
- **Early Stopping**: Prevent overfitting with patience mechanism
- **Checkpoint Saving**: Save model every N epochs for TracIn
- **Class Weighting**: Handle imbalanced classes
- **Data Augmentation**: Random crops, flips, rotations, color jitter


## TracIn Methodology

**Training Data Attribution** identifies which training examples most influenced a model's prediction on a test example.

### How It Works

1. **Gradient Computation**: For each checkpoint during training, compute:
   - Test sample gradient: ∇L(θ, x_test)
   - Training sample gradients: ∇L(θ, x_train_i)

2. **Influence Score**: 
   ```
   Influence(x_train_i, x_test) = Σ_checkpoints ∇L(θ, x_train_i) · ∇L(θ, x_test)
   ```

3. **Interpretation**:
   - **Positive influence**: Training sample helped correct prediction
   - **Negative influence**: Training sample pushed toward wrong prediction

### Mislabel Detection Strategy

For mispredicted test images:
- **High negative self-influence** → Likely mislabeled in training set
- **Consistent harmful influences** → Systemic labeling issues in that class

### Efficient Implementation

- **Ghost Dot-Product**: Memory-efficient gradient computation
- **Top-K Selection**: Track only most influential training samples
- **Tile-based Processing**: Handle large datasets in manageable chunks

## Command Referencelts

**Influence Score Interpretation:**
- **Positive values**: Training sample helped correct prediction
- **Negative values**: Training sample pushed toward wrong prediction
- **High appearances (>1000)**: Very influential sample, investigate if mispredicted

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size or top-k
python scripts/4_compute_influence.py --batch_size 16 --top_k 50
```

### Import Errors
```bash
# Always run scripts from project root
cd /path/to/Training-Influence-on-Resnet50
python scripts/2_train.py
```

## Workflow Summary

```
1. Download Weights  →  2. Train Model  →  3. Detect Mispredictions
                                ↓
                        4. Compute TracIn Influences
                                ↓
        ┌───────────────────────┴───────────────────────┐
        ↓                                               ↓
5a. Generate Dashboards                    5b. Cross-Reference Analysis
        ↓                                               ↓
        └───────────────────────┬───────────────────────┘
                                ↓
                    6. Inspect Mislabeled Candidates
                                ↓
                    7. Inspect Influential Samples
                                ↓
                        Human Review & Correction
                                ↓
                        Retrain with Clean Data
```

## References

1. **TracIn Paper**: [Estimating Training Data Influence by Tracing Gradient Descent](https://arxiv.org/abs/2002.08484)
2. **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
3. **Animals-10 Dataset**: [Kaggle Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
