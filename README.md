# Training Influence Analysis on ResNet50

A comprehensive pipeline for training ResNet50 on the Animals-10 dataset and using **TracIn (Training Data Attribution using Influence Functions)** to identify mislabeled images through influence analysis.

## 🎯 Project Overview

This project implements a complete ML workflow:

1. **Model Training**: Fine-tune ResNet50 (ImageNet pretrained) on Animals-10 dataset
2. **Misprediction Detection**: Identify images the model misclassifies  
3. **Influence Computation**: Use TracIn with Ghost Dot-Product optimization to compute training data influence scores
4. **Mislabel Discovery**: Cross-reference mispredictions with influence scores to find likely mislabeled training images
5. **Visual Inspection**: Generate detailed visualizations for human review

**Key Innovation**: Efficient TracIn implementation using Ghost Dot-Product for scalable influence computation on large datasets.

---

## 📁 Repository Structure

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
│       └── influence_utils.py       # TracIn computation utilities
│
├── config/                          # Configuration files
│   ├── slurm_train_full.sh         # SLURM job: full training (100 epochs)
│   └── slurm_train_short.sh        # SLURM job: quick test (6 epochs)
│
├── models/                          # Model storage
│   ├── best/                        # Best performing models
│   ├── checkpoints/                 # Training checkpoints (every N epochs)
│   └── pretrained/                  # Pretrained weights (ImageNet)
│
├── outputs/                         # All generated outputs
│   ├── mispredictions/              # Misprediction detection results
│   │   ├── false_predictions.csv    # List of all mispredictions
│   │   └── mispredictions_grid.png  # Visual grid of mispredicted images
│   │
│   ├── influence_analysis/          # TracIn results
│   │   ├── influence_scores.npy/csv # Influence score matrices
│   │   ├── influence_indices.npy    # Index mapping for influence scores
│   │   ├── overall_dashboard.png    # Summary visualization
│   │   ├── per_class/               # Per-class influence visualizations
│   │   ├── misprediction_influence_analysis.csv
│   │   └── misprediction_cross_analysis.csv
│   │
│   ├── inspection/                  # Visual inspection outputs
│   │   ├── mislabeled_candidates.png          # Grid of likely mislabeled images
│   │   ├── top_helpful_images.png             # Most helpful training images
│   │   ├── top_harmful_images.png             # Most harmful training images
│   │   └── detailed/                          # Individual image inspections
│   │       ├── helpful/
│   │       └── harmful/
│   │
│   └── logs/                        # Training logs
│
├── data/                            # Dataset storage
│   ├── raw/                         # Original downloaded images (10 class folders)
│   └── processed/                   # Preprocessed images (224x224, normalized)
│       ├── train/                   # Training split (80%)
│       └── test/                    # Test split (20%)
│
└── preprocess_data.ipynb            # Jupyter notebook for data preprocessing
```

---

## 🚀 Quick Start

### 1️⃣ Prerequisites

```bash
# Python 3.11+ with CUDA support
module load python/3.11.5 cuda/12.6

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy pandas matplotlib seaborn scikit-learn tqdm Pillow
```

### 2️⃣ Data Preparation

1. Download Animals-10 dataset and place in `data/raw/` with 10 class folders:
   - `butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel`

2. Preprocess images:
   ```bash
   jupyter notebook preprocess_data.ipynb  # Or run preprocessing script
   ```

### 3️⃣ Training Pipeline

**Option A: Interactive Execution**

```bash
# Step 1: Download pretrained weights
python scripts/1_download_weights.py

# Step 2: Train model
python scripts/2_train.py \
  -num_epochs 100 \
  -head_epochs 3 \
  -batch_size 64 \
  -lr 1e-4 \
  -checkpoint_freq 10 \
  -early_stopping_patience 15

# Step 3: Detect mispredictions
python scripts/3_detect_mispredictions.py \
  --model_path models/best/Resnet50_animals10_val_*.pth \
  --data_dir data/processed \
  --output_dir outputs/mispredictions

# Step 4: Compute TracIn influence scores
python scripts/4_compute_influence.py \
  --checkpoint_dir models/checkpoints \
  --test_data_dir data/processed/test \
  --train_data_dir data/processed/train \
  --output_dir outputs/influence_analysis \
  --top_k 100 \
  --batch_size 32

# Step 5a: Generate influence dashboards
python scripts/5a_generate_dashboards.py \
  --results_dir outputs/influence_analysis \
  --data_dir data/processed \
  --output_dir outputs/influence_analysis

# Step 5b: Cross-reference with mispredictions
python scripts/5b_cross_reference_analysis.py \
  --mispredictions_csv outputs/mispredictions/false_predictions.csv \
  --influence_dir outputs/influence_analysis \
  --output_dir outputs/influence_analysis

# Step 6: Inspect mislabeled candidates
python scripts/6_inspect_mislabeled.py \
  --influence_dir outputs/influence_analysis \
  --data_dir data/processed \
  --output_dir outputs/inspection

# Step 7: Inspect most influential images
python scripts/7_inspect_influential.py \
  --influence_dir outputs/influence_analysis \
  --data_dir data/processed \
  --output_dir outputs/inspection \
  --top_n 20
```

**Option B: SLURM Batch Jobs**

```bash
# Full training (10 hours, 100 epochs)
sbatch config/slurm_train_full.sh

# Quick test (1 hour, 6 epochs)
sbatch config/slurm_train_short.sh
```

---

## 📊 Key Features

### Two-Stage Training
1. **Head-only fine-tuning** (few epochs): Train only final classification layer
2. **Full fine-tuning**: Unfreeze all layers with small learning rate

### Advanced Scheduling
- **ReduceLROnPlateau**: Adaptive learning rate based on validation performance
- **Early Stopping**: Prevent overfitting with patience mechanism
- **Checkpoint Saving**: Save model every N epochs for TracIn

### Efficient TracIn Implementation
- **Ghost Dot-Product**: Memory-efficient gradient computation
- **Top-K Selection**: Track only most influential training samples
- **Tile-based Processing**: Handle large datasets in manageable chunks

### Comprehensive Analysis
- **Per-class dashboards**: Influence distributions for each category
- **Cross-reference analysis**: Link mispredictions to training influences
- **Visual inspection grids**: Human-friendly image grids for review

---

## 🔬 TracIn Methodology

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

---

## 📈 Expected Results

### Training Performance
- **Validation Accuracy**: ~98%+ (on Animals-10)
- **Training Time**: ~8-10 hours on single GPU (100 epochs)

### Influence Analysis
- **Top-K influences**: 100 most influential samples per test image
- **Mislabel detection rate**: ~5-10% of training data flagged for review
- **False positive rate**: ~20-30% (requires human verification)

---

## 🛠️ Customization

### Modify Training Parameters

Edit `scripts/2_train.py` or pass CLI arguments:

```python
# Learning rate
-lr 1e-4

# Batch size (adjust based on GPU memory)
-batch_size 64

# Training epochs
-num_epochs 100
-head_epochs 3

# Checkpoint frequency (for TracIn)
-checkpoint_freq 10

# Early stopping
-early_stopping_patience 15
```

### Adjust TracIn Settings

Edit `scripts/4_compute_influence.py`:

```python
# Number of top influences to track
--top_k 100

# Tile size for memory efficiency
--tile_size 5000

# Batch processing size
--batch_size 32
```

---

## 📝 Output Files Explained

### `outputs/mispredictions/false_predictions.csv`
Columns: `Image Path`, `True Label`, `Predicted Label`, `Confidence`, `Top-3 Predictions`

### `outputs/influence_analysis/influence_scores.npy`
Shape: `(num_test_samples, top_k)`  
Contains influence scores for top-k training samples per test sample

### `outputs/influence_analysis/influence_indices.npy`
Shape: `(num_test_samples, top_k)`  
Training sample indices corresponding to scores

### `outputs/inspection/mislabeled_candidates.png`
Visual grid showing training images flagged as likely mislabeled

---

## 🔧 Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
python scripts/4_compute_influence.py --batch_size 16

# Reduce top-k
python scripts/4_compute_influence.py --top_k 50

# Reduce tile size
python scripts/4_compute_influence.py --tile_size 2000
```

### Training Not Converging
```bash
# Lower learning rate
python scripts/2_train.py -lr 5e-5

# Increase head-only epochs
python scripts/2_train.py -head_epochs 5

# Check data preprocessing
jupyter notebook preprocess_data.ipynb
```

### Import Errors
```bash
# Ensure you're in project root
cd /path/to/Training-Influence-on-Resnet50

# Scripts automatically add utils/ to path
# No manual PYTHONPATH needed
```

---

## 📚 References

1. **TracIn Paper**: [Estimating Training Data Influence by Tracing Gradient Descent](https://arxiv.org/abs/2002.08484)
2. **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
3. **Animals-10 Dataset**: [Kaggle Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

---

## 🤝 Workflow Summary

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

---

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in publications.

---

## ✨ Acknowledgments

- **PyTorch** for deep learning framework
- **TracIn authors** for influence function methodology  
- **Animals-10 dataset creators** for the benchmark dataset
- **ResNet authors** for the foundational architecture

---

**For questions or issues, please refer to the individual script docstrings or open an issue.**
