# Training Influence Analysis on ResNet50: A TracIn Implementation for Animal Classification

## Executive Summary

This project implements **TracIn (Tracing Training Influence)**, a state-of-the-art method for understanding how individual training samples affect model predictions. I apply this technique to a ResNet50 model fine-tuned on the Animals-10 dataset to:

1. **Identify influential training samples** - Which training images most strongly affect each prediction?
2. **Diagnose model errors** - Why does the model make specific mistakes?
3. **Detect mislabeled data** - Are there labeling errors in the training set?

### Key Achievements

| Metric | Result |
|--------|--------|
| **Best Test Accuracy** | **99.50%** (v1 - cleaned dataset) |
| **Best Test Accuracy (Uncleaned)** | **99.01%** (v4) |
| **Data Cleaning Improvement** | **+0.49%** (v1 vs best uncleaned) |
| **Average Test Accuracy** | **98.84%** (v2-v10) |
| Final Validation Accuracy | 98.28% (v3) |
| Test Samples | 5,979 |
| Influence Scores Computed | 597,900 (top-100 per test) |
| Influence Range | [-0.1702, +0.2729] |
| Mispredictions Detected | 142-172 per version |
| Cross-Reference Matches | 2,680 |

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Theoretical Background: TracIn](#2-theoretical-background-tracin)
3. [Ghost Dot-Product Optimization](#3-ghost-dot-product-optimization)
4. [Model Architecture Design](#4-model-architecture-design)
5. [Training Methodology](#5-training-methodology)
6. [Influence Computation Pipeline](#6-influence-computation-pipeline)
7. [Analysis and Visualization Tools](#7-analysis-and-visualization-tools)
8. [Results and Findings](#8-results-and-findings)
9. [Reproducibility and Version Control](#9-reproducibility-and-version-control)
10. [Conclusions](#10-conclusions)
11. [Cross-Version Analysis (v2-v10)](#11-cross-version-analysis-v2-v10)
12. [References](#12-references)

---

## 1. Introduction and Motivation

### 1.1 The Problem: Understanding Deep Learning Decisions

Deep neural networks achieve remarkable accuracy but operate as "black boxes." When a model misclassifies an image, one cannot easily answer:

- *Which training images caused this error?*
- *Is this error due to mislabeled training data?*
- *Which training samples are most "helpful" or "harmful" to model performance?*

### 1.2 Why This Matters

Understanding training influence has critical practical applications:

1. **Data Quality Improvement**: Identify and correct mislabeled training samples
2. **Error Diagnosis**: Trace prediction errors back to their training data sources
3. **Model Debugging**: Understand why certain classes are confused
4. **Data Valuation**: Quantify the contribution of individual training samples

### 1.3 My Approach

I implement TracIn (Pruthi et al., 2020), which estimates influence through gradient similarity across training checkpoints. This project demonstrates the complete pipeline from model training to influence-based data diagnosis.

### 1.4 Dataset: Animals-10

| Property | Value |
|----------|-------|
| Classes | 10 (butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel) |
| Training Samples | ~26,000 |
| Test Samples | ~6,000 |
| Image Size | 224×224 (resized for ResNet) |
| Split | 80% train / 20% test |

---

## 2. Theoretical Background: TracIn

### 2.1 The Core Idea

TracIn answers the question: *"If I removed training sample $z$ from the training set, how would the loss on test sample $z'$ change?"*

Instead of expensive leave-one-out retraining, TracIn approximates this through **gradient dot products**.

### 2.2 Mathematical Formulation

The influence of training sample $z$ on test sample $z'$ is:

$$\text{TracIn}(z, z') = \sum_{i=1}^{n} \eta_i \cdot \nabla_\theta \mathcal{L}(\theta_i, z) \cdot \nabla_\theta \mathcal{L}(\theta_i, z')$$

Where:
- $\theta_i$ = model parameters at checkpoint $i$
- $\eta_i$ = learning rate at checkpoint $i$
- $\nabla_\theta \mathcal{L}$ = gradient of loss with respect to parameters
- $n$ = number of checkpoints (I use 5: epochs 10, 20, 30, 40, 50)

### 2.3 Interpretation

| Sign | Meaning | Interpretation |
|------|---------|----------------|
| **Positive** | Proponent | Training sample *reduces* test loss (helpful) |
| **Negative** | Opponent | Training sample *increases* test loss (harmful) |
| **Magnitude** | Strength | Larger = more influential |

### 2.4 Why Multiple Checkpoints?

Using multiple checkpoints provides:
1. **Stability**: Single-checkpoint estimates are noisy
2. **Coverage**: Captures influence throughout training trajectory
3. **Efficiency**: No need to store all training steps

I save checkpoints at epochs 10, 20, 30, 40, 50 during fine-tuning.

---

## 3. Ghost Dot-Product Optimization

### 3.1 The Memory Challenge

For a layer with input dimension $d_{in}$ and output dimension $d_{out}$, the weight gradient is:

$$\nabla_W \mathcal{L} = \text{error}^\top \cdot \text{activation} \in \mathbb{R}^{d_{out} \times d_{in}}$$

For the final layer: $d_{out} = 10$, $d_{in} = 256$
- Full gradient size: $10 \times 256 = 2,560$ values per sample
- For 26,000 training samples: **66.5 million values per checkpoint**

### 3.2 The Ghost Dot-Product Insight

The key mathematical identity that enables memory-efficient computation:

$$\langle uv^\top, u'v'^\top \rangle_F = \langle u, u' \rangle \cdot \langle v, v' \rangle$$

Where $\langle \cdot, \cdot \rangle_F$ denotes the Frobenius inner product.

**Proof:**
$$\langle uv^\top, u'v'^\top \rangle_F = \text{tr}((uv^\top)^\top (u'v'^\top)) = \text{tr}(vu^\top u'v'^\top) = (u^\top u') \cdot (v^\top v') = \langle u, u' \rangle \cdot \langle v, v' \rangle$$

### 3.3 Memory Savings

Instead of storing full gradients, I store factor components:
- **Error signal**: $e \in \mathbb{R}^{10}$ (output dimension)
- **Activation**: $a \in \mathbb{R}^{256}$ (input dimension)
- **Storage per sample**: $10 + 256 = 266$ values

| Approach | Storage per Sample | Total (26k samples) |
|----------|-------------------|---------------------|
| Full Gradient | 2,560 values | 66.5M values |
| Ghost Factors | 266 values | 6.9M values |
| **Reduction** | **9.6×** | **9.6×** |

### 3.4 Implementation

```python
def compute_ghost_influence_batch(train_act, train_err, test_act, test_err, lr):
    """
    Compute TracIn influence using Ghost Dot-Product.
    
    Args:
        train_act: (n_train, d_in) - training activations
        train_err: (n_train, d_out) - training error signals  
        test_act: (n_test, d_in) - test activations
        test_err: (n_test, d_out) - test error signals
        lr: learning rate at this checkpoint
    
    Returns:
        influence: (n_test, n_train) matrix
    """
    # Factor 1: activation similarity
    act_dots = test_act @ train_act.T  # (n_test, n_train)
    
    # Factor 2: error signal similarity  
    err_dots = test_err @ train_err.T  # (n_test, n_train)
    
    # Ghost Dot-Product: element-wise multiply the factors
    influence = lr * act_dots * err_dots  # (n_test, n_train)
    
    return influence
```

---

## 4. Model Architecture Design

### 4.1 Why ResNet50?

I chose ResNet50 as the backbone for several reasons:

1. **Proven Performance**: State-of-the-art on ImageNet with residual connections
2. **Transfer Learning**: Rich pretrained features from 1.2M ImageNet images
3. **Appropriate Capacity**: 25.6M parameters - powerful yet trainable
4. **Influence Compatibility**: Clean final layer structure for gradient extraction

### 4.2 Architecture Modifications

I replace ResNet50's 1000-class ImageNet head with a custom classification head:

```
ResNet50 Backbone (pretrained)
    ↓
AdaptiveAvgPool2d → 2048-dim feature vector
    ↓
Linear(2048, 512) → LeakyReLU(0.01)
    ↓  
Linear(512, 256) → LeakyReLU(0.01)
    ↓
Linear(256, 10) → 10-class logits
```

### 4.3 Design Rationale

| Design Choice | Why |
|---------------|-----|
| **3-layer FC head** | Gradual dimensionality reduction preserves information |
| **LeakyReLU** | Prevents dead neurons during training |
| **256 penultimate dim** | Balance between expressiveness and influence computation cost |
| **Sequential wrapper** | Clean hook attachment for gradient capture |

### 4.4 Implementation

```python
class ResNet50_Animals10(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet50(weights=weights)
        
        # Replace classifier with custom head
        in_features = self.model.fc.in_features  # 2048
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, num_classes)  # Final layer for influence hooks
        )
```

---

## 5. Training Methodology

### 5.1 Two-Stage Training Strategy

I employ a **two-stage training approach** to maximize transfer learning effectiveness:

#### Stage 1: Head-Only Training (Warm-up)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 5 | Quick adaptation of new head |
| Learning Rate | 1e-3 | Aggressive learning for random weights |
| Backbone | **Frozen** | Preserve pretrained features |
| Optimizer | AdamW | Modern optimizer with weight decay |

#### Stage 2: Full Fine-Tuning
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max Epochs | 100 | Allow full convergence |
| Backbone LR | 1e-5 | Gentle updates to pretrained features |
| Head LR | 2e-4 | Faster learning for task-specific layers |
| Early Stopping | 10 epochs | Prevent overfitting |

### 5.2 Why Two Stages?

1. **Stage 1 (Head-only)**: The randomly initialized head would produce large, noisy gradients that could damage pretrained backbone features. By freezing the backbone, I let the head "catch up" to produce meaningful gradients.

2. **Stage 2 (Full fine-tuning)**: With the head now producing reasonable predictions, I can safely fine-tune the backbone with a small learning rate to adapt features to the specific animal classification task.

### 5.3 Differential Learning Rates

The **20× difference** between head and backbone learning rates (2e-4 vs 1e-5) reflects:
- Backbone: Already well-trained, needs gentle refinement
- Head: Task-specific, needs more aggressive learning

### 5.4 Loss Function Design

```python
# Compute class weights (inverse frequency)
class_counts = get_class_distribution(train_dataset)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes

# Create loss function
criterion = nn.CrossEntropyLoss(
    weight=class_weights,        # Handle class imbalance
    label_smoothing=0.1          # Regularization
)
```

**Why Class Weights?**: The Animals-10 dataset has imbalanced class distributions. Weighting by inverse frequency ensures minority classes receive adequate learning signal.

**Why Label Smoothing?**: Smoothing (0.1) prevents overconfident predictions and improves generalization by softening hard labels: $y_{smooth} = (1-\epsilon)y + \epsilon/K$

### 5.5 Checkpoint Strategy

I save checkpoints at fixed intervals during fine-tuning:
- **Epochs saved**: 10, 20, 30, 40, 50
- **Metadata saved**: Learning rate, epoch number, validation metrics

Each checkpoint stores:
```json
{
    "epoch": 50,
    "learning_rate": 1e-5,
    "learning_rate_head": 2e-4,
    "train_loss": 0.1234,
    "val_accuracy": 0.9828
}
```

**Critical**: The `learning_rate_head` is essential for TracIn computation since I hook the head's final layer.

### 5.6 Training Results

See **Section 11.1** for comprehensive performance metrics across all 10 versions, including validation and test accuracy.

---

## 6. Influence Computation Pipeline

### 6.1 Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Load Model &   │ --> │  Register Hooks  │ --> │ Process Train   │
│   Checkpoint    │     │  on Final Layer  │     │    Dataset      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐     ┌────────▼────────┐
│  Accumulate     │ <-- │  Compute Ghost   │ <-- │  Process Test   │
│  Across Ckpts   │     │  Influence Batch │     │    Dataset      │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│   TopK Track    │ --> │   Save Results   │
│    & Export     │     │   (CSV, NPY)     │
└─────────────────┘     └──────────────────┘
```

### 6.2 Hook Registration

I use PyTorch forward hooks to capture intermediate values:

```python
class InfluenceHook:
    """Captures activations and gradients for a Linear layer."""
    
    def forward_hook(self, module, input, output):
        # Input to final layer = activations
        self.stored_activations = input[0].detach()  # (batch, 256)
        
        # Output gradients w.r.t. loss = error signal
        output.register_hook(self.backward_hook)
    
    def backward_hook(self, grad):
        self.stored_error_signals = grad.detach()  # (batch, 10)
```

**Critical Implementation Detail**: I hook `model.model.fc[-1]` (the final Linear layer), not the entire Sequential container. This ensures I capture:
- **Activation**: Input to the final layer (256-dim)
- **Error Signal**: Gradient of loss w.r.t. output logits (10-dim)

### 6.3 Factor Collection

For each sample, I compute loss and backpropagate to collect factors:

```python
def collect_factors(model, dataloader, criterion, hook):
    all_activations = []
    all_errors = []
    
    for images, labels in dataloader:
        outputs = model(images)           # Forward pass
        loss = criterion(outputs, labels) # Same loss as training
        loss.backward()                   # Backward pass
        
        all_activations.append(hook.stored_activations)
        all_errors.append(hook.stored_error_signals)
    
    return torch.cat(all_activations), torch.cat(all_errors)
```

**Why Same Loss Function?**: The gradient must be computed with the identical loss function used during training (including class weights and label smoothing). Otherwise, the influence estimates would be meaningless.

### 6.4 Tile-and-Stitch Strategy

Computing all pairwise influences at once would require:
- Memory: $6,000 \times 26,000 \times 4$ bytes = **624 MB** per checkpoint

I use a memory-efficient tile-and-stitch approach:

```python
class TopKInfluenceTracker:
    def __init__(self, n_test, k=100):
        self.top_k_values = torch.full((n_test, k), -float('inf'))
        self.top_k_indices = torch.zeros((n_test, k), dtype=torch.long)
    
    def update_batch(self, test_start, influence_batch, train_offset):
        """Update top-k for a batch of test samples."""
        for i, test_idx in enumerate(range(test_start, test_start + len(influence_batch))):
            # Combine current batch with existing top-k
            combined_values = torch.cat([
                self.top_k_values[test_idx],
                influence_batch[i]
            ])
            combined_indices = torch.cat([
                self.top_k_indices[test_idx],
                torch.arange(len(influence_batch[i])) + train_offset
            ])
            
            # Select new top-k by absolute magnitude
            _, top_indices = torch.topk(combined_values.abs(), k=self.k)
            self.top_k_values[test_idx] = combined_values[top_indices]
            self.top_k_indices[test_idx] = combined_indices[top_indices]
```

**Why Top-K by Absolute Value?**: I want both:
- **Proponents** (positive influence) - helpful training samples
- **Opponents** (negative influence) - harmful training samples

Selecting by absolute magnitude captures both extremes.

### 6.5 Multi-Checkpoint Accumulation

TracIn sums influences across checkpoints:

```python
def accumulate_influences_across_checkpoints(checkpoint_results):
    """Sum influence scores from multiple checkpoints."""
    
    # Initialize with first checkpoint
    accumulated_values = checkpoint_results[0]['values'].clone()
    accumulated_indices = checkpoint_results[0]['indices'].clone()
    
    for ckpt in checkpoint_results[1:]:
        # Use scatter_add for efficient accumulation
        accumulated_values.scatter_add_(
            dim=1,
            index=ckpt['indices'],
            src=ckpt['values']
        )
    
    return accumulated_values, accumulated_indices
```

---

## 7. Analysis and Visualization Tools

### 7.1 Misprediction Detection (`3_detect_mispredictions.py`)

**Purpose**: Identify and catalog all model errors for further analysis.

**Output** (`false_predictions.csv`):
```csv
index,true_label,predicted_label,confidence,image_path
198,butterfly,chicken,0.7234,data/processed/test/butterfly/img_001.jpg
...
```

**Why Analyze Mispredictions?**: These are the cases where influence analysis is most valuable - understanding which training samples led to incorrect predictions.

### 7.2 Influence Dashboards (`5a_generate_dashboards.py`)

**Purpose**: Generate comprehensive visualizations of influence patterns.

**Outputs**:
1. **Per-class influence distributions** - Histograms showing influence patterns for each class
2. **Top influential samples** - Ranked lists of most impactful training samples
3. **Class confusion analysis** - Which classes influence which others

### 7.3 Cross-Reference Analysis (`5b_cross_reference_analysis.py`)

**Purpose**: Connect mispredictions to their most influential training samples.

**Key Analysis**: For each mispredicted test sample, I examine:
- Top-100 most influential training samples
- Whether influential samples have the same error pattern (same_error)
- Whether influential samples are themselves mispredicted

**Output** (`misprediction_cross_analysis.csv`):
```csv
test_idx,test_true,test_pred,train_idx,train_true,train_pred,influence,rank,same_error
198,butterfly,chicken,23070,squirrel,chicken,0.0156,1,True
```

**Interpretation**: The `same_error=True` flag indicates that both the test sample and the influential training sample made the same mistake (predicting "chicken"). This suggests the training sample may be mislabeled or unusually confusing.

### 7.4 Mislabel Detection (`6_inspect_mislabeled.py`)

**Purpose**: Identify likely mislabeled training samples.

**Methodology**:
1. Find training samples that appear frequently as top influences
2. Compute average influence score for each training sample
3. Flag samples with consistently negative influence (opponents)

**Output** (`mislabeled_inspection_report.csv`):
- Training samples ranked by likelihood of mislabeling
- Influence statistics per sample

### 7.5 Influential Sample Inspection (`7_inspect_influential.py`)

**Purpose**: Detailed examination of most helpful and harmful training samples.

**Analysis**:
- **Most Helpful**: Training samples with highest positive influence
- **Most Harmful**: Training samples with highest negative influence
- **Cross-class patterns**: Which training classes influence which test classes

---

## 8. Results and Findings

> **Note:** For comprehensive cross-version results including test accuracy metrics, see **Section 11**.

### 8.1 Summary Metrics

| Metric | Value |
|--------|-------|
| Model versions trained | 10 (v1-v10) |
| Best test accuracy | **99.50%** (v1, cleaned data) |
| Best test accuracy (uncleaned) | **99.01%** (v4) |
| Average test accuracy (v2-v10) | **98.84%** |
| Test samples | 5,979 |
| Data cleaning improvement | +0.49% |

### 8.2 Key Findings

1. **TracIn successfully identifies mislabeled samples**: 31.6% of influential training samples for mispredictions show the same error pattern, indicating likely mislabels.

2. **Persistent mispredictions**: ~40 images are consistently mispredicted across all 9 model versions (v2-v10), indicating inherent ambiguity or mislabeling.

3. **Universally influential samples**: Some training samples appear in >50% of test influence computations, suggesting they are prototypical class exemplars.

4. **Class confusion patterns**: Cow↔sheep (most common), dog↔horse, and chicken↔spider are the primary confusion pairs.

---

## 9. Reproducibility and Version Control

### 9.1 Version Management System

I implemented a `VersionManager` class to track all training runs:

```python
class VersionManager:
    """Manage versioned model training runs."""
    
    def create_new_version(self):
        # Increment version counter
        # Create directories: models/best/vN, models/checkpoints/vN, outputs/vN
        # Return version number
    
    def register_training_run(self, version, metadata):
        # Save training metadata to version_registry.json
```

### 9.2 Directory Structure

```
models/
├── version_registry.json    # Central registry of all versions
├── best/v3/model_v3.pth     # Best model checkpoint
├── checkpoints/v3/          # Training checkpoints for TracIn
│   ├── finetune_epoch_10.pth
│   ├── finetune_epoch_20.pth
│   └── ...
└── pretrained/
    └── resnet50_imagenet1k_v1.pth

outputs/v3/
├── influence_analysis/
│   ├── influence_report.txt
│   ├── top_k_influences.csv
│   └── misprediction_cross_analysis.csv
├── mispredictions/
│   └── false_predictions.csv
└── inspection/
    └── mislabeled_inspection_report.csv
```

### 9.3 Running the Pipeline

**Full Pipeline Execution**:
```bash
python scripts/run_full_pipeline.py --version 3
```

This executes steps 3-7 in sequence:
1. Detect mispredictions
2. Compute influence scores
3. Generate dashboards
4. Cross-reference analysis
5. Mislabel inspection
6. Influential sample inspection

**Individual Scripts**:
```bash
python scripts/1_download_weights.py          # Download pretrained weights
python scripts/2_train.py                     # Train model
python scripts/3_detect_mispredictions.py 
python scripts/4_compute_influence.py 
python scripts/5a_generate_dashboards.py 
python scripts/5b_cross_reference_analysis.py 
python scripts/6_inspect_mislabeled.py 
python scripts/7_inspect_influential.py
```

### 9.4 HPC Execution

SLURM job scripts are provided for GPU cluster execution:
```bash
sbatch config/slurm_train_full.sh
```

---

## 10. Conclusions

### 10.1 Summary of Contributions

1. **Complete TracIn Implementation**: End-to-end pipeline from training to influence-based diagnosis
2. **Memory-Efficient Computation**: Ghost Dot-Product achieves 9.6× memory reduction
3. **Robust Methodology**: Two-stage training, class balancing, label smoothing
4. **Practical Tooling**: Automated version management, visualization dashboards
5. **Real Analysis**: 98.28% accuracy model with detailed influence examination

### 10.2 Key Insights

1. **Influence methods work**: I successfully traced prediction errors to training samples
2. **Same-error patterns reveal mislabels**: 31.6% of error-influencing samples show same error pattern
3. **Some samples are universally influential**: Top training sample appears in 54% of test influences
4. **Memory optimization is essential**: Without Ghost Dot-Product, computation would be impractical

### 10.3 Future Work

1. **Influence-guided data cleaning**: Automatically remove or relabel detected mislabels
2. **Active learning**: Use influence scores to select most valuable samples for labeling
3. **Model comparison**: Compare influence patterns across different architectures
4. **Temporal analysis**: Track how influence evolves during training

---

## 11. Cross-Version Analysis (v2-v10)

This section presents an in-depth comparative analysis of model versions 2 through 10, identifying patterns, similarities, and noteworthy findings across training runs.

### 11.1 Model Performance Summary

| Version | Val Acc | Val Loss | **Test Acc** | **Test Loss** | Mispredictions | Date | Notes |
|---------|---------|----------|--------------|---------------|----------------|------|-------|
| v1 | 98.16% | 0.6189 | **99.50%** | **0.1411** | N/A | 2025-12-10 | 🧹 Cleaned dataset |
| v2 | 98.21% | 0.5926 | **98.66%** | **0.1611** | 169 | 2025-12-10 | |
| v3 | 98.28% | 0.5909 | **98.83%** | **0.1558** | 158 | 2025-12-13 | |
| v4 | 98.25% | 0.5914 | **99.01%** | **0.1518** | 150 | 2025-12-14 | |
| v5 | 98.15% | 0.5922 | **98.88%** | **0.1525** | 158 | 2025-12-14 | |
| v6 | 98.42% | 0.5898 | **98.85%** | **0.1533** | 155 | 2025-12-14 | |
| v7 | 98.34% | 0.5903 | **98.78%** | **0.1564** | 150 | 2025-12-14 | |
| v8 | 98.23% | 0.5915 | **98.81%** | **0.1536** | 172 | 2025-12-14 | |
| v9 | 98.34% | 0.5913 | **98.80%** | **0.1520** | 165 | 2025-12-14 | |
| v10 | 98.14% | 0.5919 | **98.75%** | **0.1545** | 142 | 2025-12-14 | |

**Key Observations:**
- **Best test accuracy**: v1 with **99.50%** - trained on **cleaned dataset** after removing suspicious/mislabeled images
- **Best on uncleaned data (v2-v10)**: v4 with **99.01%** test accuracy
- **Data cleaning impact**: v1 shows **~0.5% improvement** over the best uncleaned model, demonstrating the value of the influence analysis workflow
- **Lowest test loss**: v1 with 0.1411, followed by v4 with 0.1518
- **Validation vs Test**: Test accuracy is generally **higher** than validation accuracy across versions
- All models achieve >98.6% test accuracy, confirming robust generalization

### 11.2 Persistently Mispredicted Images (Across All Versions)

These training images are consistently mispredicted across all 9 model versions (v2-v10), indicating **inherent ambiguity or potential mislabeling**:

| Image | True Class | Predicted As | Versions Affected |
|-------|-----------|--------------|-------------------|
| `butterfly_1751.jpeg` | butterfly (0) | spider (8) | All 9 versions |
| `cat_1035.jpeg` | cat (1) | sheep (7) | All 9 versions |
| `cat_386.jpg` | cat (1) | squirrel/sheep (9/7) | All 9 versions |
| `cat_552.jpeg` | cat (1) | dog (4) | All 9 versions |
| `chicken_1080.jpeg` | chicken (2) | dog (4) | All 9 versions |
| `chicken_1515.jpeg` | chicken (2) | dog (4) | All 9 versions |
| `chicken_1529.jpeg` | chicken (2) | spider (8) | All 9 versions |
| `chicken_1653.jpeg` | chicken (2) | spider (8) | All 9 versions |
| `chicken_2165.jpeg` | chicken (2) | spider (8) | All 9 versions |
| `chicken_2223.jpeg` | chicken (2) | squirrel (9) | All 9 versions |
| `cow_1100.jpeg` | cow (3) | sheep (7) | All 9 versions |
| `cow_1554.jpeg` | cow (3) | varies (2/4/9) | All 9 versions |
| `cow_1823.jpeg` | cow (3) | horse (6) | All 9 versions |
| `cow_327.jpeg` | cow (3) | sheep (7) | All 9 versions |
| `cow_425.jpeg` | cow (3) | horse/dog (6/4) | All 9 versions |
| `cow_471.jpeg` | cow (3) | horse (6) | All 9 versions |
| `cow_496.jpeg` | cow (3) | sheep (7) | All 9 versions |
| `cow_916.jpeg` | cow (3) | elephant (5) | All 9 versions |
| `dog_1165.jpeg` | dog (4) | cow (3) | All 9 versions |
| `dog_2071.jpeg` | dog (4) | cow (3) | All 9 versions |
| `dog_2155.jpeg` | dog (4) | chicken (2) | All 9 versions |
| `dog_3413.jpeg` | dog (4) | sheep (7) | All 9 versions |
| `dog_3987.jpeg` | dog (4) | sheep (7) | All 9 versions |
| `dog_4039.jpeg` | dog (4) | cow (3) | All 9 versions |
| `dog_4121.jpeg` | dog (4) | squirrel (9) | 8 of 9 versions |
| `dog_558.jpeg` | dog (4) | horse (6) | All 9 versions |
| `dog_562.jpeg` | dog (4) | horse (6) | All 9 versions |
| `elephant_199.jpeg` | elephant (5) | sheep (7) | All 9 versions |
| `elephant_883.jpeg` | elephant (5) | horse (6) | 8 of 9 versions |
| `horse_1299.jpeg` | horse (6) | cow (3) | All 9 versions |
| `horse_2624.jpeg` | horse (6) | cow (3) | All 9 versions |
| `horse_518.jpeg` | horse (6) | sheep (7) | All 9 versions |
| `horse_758.jpeg` | horse (6) | spider (8) | All 9 versions |
| `sheep_789.jpeg` | sheep (7) | cow (3) | All 9 versions |
| `spider_2523.jpeg` | spider (8) | butterfly (0) | All 9 versions |
| `spider_4288.jpeg` | spider (8) | butterfly (0) | 8 of 9 versions |
| `squirrel_1106.jpeg` | squirrel (9) | butterfly (0) | 7 of 9 versions |
| `squirrel_1418.jpeg` | squirrel (9) | dog (4) | 8 of 9 versions |
| `squirrel_1434.jpeg` | squirrel (9) | elephant (5) | 8 of 9 versions |
| `squirrel_1546.jpeg` | squirrel (9) | chicken (2) | All 9 versions |

**Total persistent errors: ~40 training images are consistently problematic**

### 11.3 Confusion Pattern Analysis

The most common confusion pairs across all versions:

| True Class | Often Confused With | Frequency | Likely Cause |
|-----------|---------------------|-----------|--------------|
| **cow** | sheep | Very High | Similar body structure, grazing animals |
| **cow** | horse | High | Both are large quadrupeds |
| **dog** | horse | High | Various dog breeds resemble horse profiles |
| **dog** | cow | Medium | Spotted dogs confused with cows |
| **chicken** | spider | Medium | Both have thin legs, unusual postures |
| **chicken** | dog | Medium | Color/texture similarities |
| **horse** | cow | High | Large body mass confusion |
| **spider** | butterfly | Medium | Wing-like leg spread patterns |
| **butterfly** | spider | Medium | Similar color patterns (orange/black) |
| **squirrel** | spider/butterfly | Low | Edge cases |

### 11.4 Most Influential Training Samples (Consistent Across Versions)

These training samples appear in the **top-50 most influential** list across multiple versions:

| Train ID | Class | Avg Appearances | Versions in Top 50 | Significance |
|----------|-------|----------------|-------------------|--------------|
| 8661 | dog | ~2,900 | 9/9 | **Most consistent influencer** |
| 7654 | cow | ~2,800 | 9/9 | Highly influential for cow predictions |
| 7902 | cow | ~2,500 | 9/9 | Key cow exemplar |
| 22958 | squirrel | ~2,500 | 9/9 | Dominant squirrel sample |
| 10628 | dog | ~2,300 | 9/9 | Strong dog influence |
| 5037 | chicken | ~2,250 | 9/9 | Key chicken sample |
| 11000 | dog | ~2,200 | 9/9 | Important dog reference |
| 8381 | cow | ~2,400 | 8/9 | High cow influence |
| 12964 | elephant | ~2,400 | 7/9 | Key elephant sample |
| 15925 | horse | ~2,100 | 9/9 | Important horse reference |

### 11.5 Mislabeled Sample Candidates (High Confidence)

These samples appear repeatedly in the mislabeled candidates across versions, with **high influence on mispredictions**:

#### Top Candidates for Review

| Image | Labeled As | Predicted As | Evidence |
|-------|-----------|--------------|----------|
| `cat_1035.jpeg` | cat | sheep | Appears in 9/9 versions, high influence on sheep predictions |
| `cow_496.jpeg` | cow | sheep | Consistently influences sheep mispredictions (18-22 appearances) |
| `cow_327.jpeg` | cow | sheep | Strong sheep influence (14-23 appearances per version) |
| `cow_1100.jpeg` | cow | sheep | Consistent across all versions |
| `cow_1759.jpeg` | cow | sheep | High total influence scores |
| `horse_518.jpeg` | horse | sheep | Appears in all version reports (14-23 appearances) |
| `dog_3413.jpeg` | dog | sheep | Persistent across versions |
| `dog_3987.jpeg` | dog | sheep | Strong sheep influence |
| `horse_1299.jpeg` | horse | cow | Consistent cow confusion trigger |
| `horse_612.jpeg` | horse | cow | Multiple version appearances |
| `sheep_789.jpeg` | sheep | cow | High influence on cow predictions |
| `dog_1165.jpeg` | dog | cow | Strong cow influence signal |

### 11.6 Influence Statistics Comparison

| Version | Mean Influence | Median Influence | Influence Range |
|---------|---------------|-----------------|-----------------|
| v2 | 5.67e-06 | 2.58e-06 | [-0.0154, 0.0256] |
| v3 | 2.27e-04 | 5.73e-05 | [-0.1702, 0.2729] |
| v4 | 1.53e-04 | 3.31e-05 | [-0.1994, 0.2879] |
| v5 | 9.21e-05 | 3.06e-05 | [-0.0940, 0.1709] |
| v6 | 9.18e-05 | 2.97e-05 | [-0.0987, 0.1794] |
| v7 | 6.26e-05 | 2.81e-05 | [-0.1320, 0.1997] |
| v8 | 1.41e-04 | 5.02e-05 | [-0.1255, 0.2692] |
| v9 | 4.45e-05 | 1.41e-05 | [-0.0410, 0.0731] |
| v10 | 4.60e-05 | 4.05e-05 | [-0.0897, 0.1435] |

**Notable observation**: v3 and v4 show the highest influence magnitude ranges, suggesting more pronounced training sample effects in those models.

### 11.7 Class-Specific Error Patterns

#### Classes with Most Mispredictions

| Class | Avg Errors/Version | Main Confusion Targets |
|-------|-------------------|----------------------|
| **dog** | ~22 | cat, cow, horse, sheep |
| **cow** | ~14 | sheep, horse, chicken |
| **chicken** | ~10 | spider, dog, squirrel |
| **horse** | ~7 | cow, sheep |
| **squirrel** | ~7 | butterfly, spider, dog |
| **sheep** | ~6 | cow, chicken |
| **spider** | ~5 | butterfly |
| **cat** | ~4 | dog, sheep |
| **butterfly** | ~3 | spider, chicken |
| **elephant** | ~3 | sheep, horse |

**Insight**: Dogs have the highest error rate, likely due to the tremendous variety in dog breeds causing confusion with other quadrupeds.

### 11.8 Recommendations Based on Analysis

#### 1. **Data Cleaning Priority List**
The following images should be manually reviewed for potential mislabeling:
1. `cow_496.jpeg`, `cow_327.jpeg`, `cow_1100.jpeg` (labeled cow, consistently predicted sheep)
2. `cat_1035.jpeg` (labeled cat, consistently predicted sheep)
3. `horse_518.jpeg`, `horse_1299.jpeg` (consistently confused)
4. `dog_558.jpeg`, `dog_562.jpeg` (labeled dog, predicted horse)

#### 2. **Class Augmentation Needs**
- **Cow vs Sheep**: Need more distinctive training examples
- **Dog**: Expand breed diversity to reduce confusion
- **Chicken vs Spider**: Add more characteristic pose examples

#### 3. **Model Architecture Considerations**
- Consider attention mechanisms to focus on distinctive features
- Fine-grained classification heads for confusing class pairs

### 11.9 Summary Statistics

| Metric | Value |
|--------|-------|
| Total versions analyzed | 9 (v2-v10) |
| Average validation accuracy | 98.26% |
| **Average test accuracy** | **98.84%** |
| **Best test accuracy (cleaned data)** | **99.50% (v1)** |
| **Best test accuracy (uncleaned)** | **99.01% (v4)** |
| **Improvement from data cleaning** | **+0.49%** |
| Total unique mispredicted images | ~80 |
| Persistently mispredicted (all versions) | ~40 |
| Top influential samples (consistent) | 10 |
| Candidate mislabeled images | 12 |
| Most problematic class | dog (22 avg errors) |
| Most confused pair | cow → sheep |
| Test set size | 5,979 samples |

---

## 12. References

1. **TracIn**: Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. (2020). "Estimating Training Data Influence by Tracing Gradient Descent." *Advances in Neural Information Processing Systems*, 33.

2. **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.

3. **Influence Functions**: Koh, P. W., & Liang, P. (2017). "Understanding Black-box Predictions via Influence Functions." *ICML*.

4. **Label Smoothing**: Müller, R., Kornblith, S., & Hinton, G. E. (2019). "When Does Label Smoothing Help?" *NeurIPS*.

---

## Appendix A: File Reference

| File | Purpose |
|------|---------|
| `scripts/1_download_weights.py` | Download pretrained ResNet50 weights |
| `scripts/2_train.py` | Two-stage model training |
| `scripts/3_detect_mispredictions.py` | Identify model errors |
| `scripts/4_compute_influence.py` | TracIn influence computation |
| `scripts/5a_generate_dashboards.py` | Visualization generation |
| `scripts/5b_cross_reference_analysis.py` | Error-influence cross-reference |
| `scripts/6_inspect_mislabeled.py` | Mislabel detection |
| `scripts/7_inspect_influential.py` | Influential sample analysis |
| `scripts/run_full_pipeline.py` | End-to-end pipeline |
| `scripts/utils/model_architecture.py` | ResNet50 model definition |
| `scripts/utils/influence_utils.py` | TracIn computation utilities |
| `scripts/utils/version_manager.py` | Version management |

---

## Appendix B: Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
Pillow>=9.5.0
scikit-learn>=1.2.0
```