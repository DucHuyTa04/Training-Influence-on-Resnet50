"""
TracIn influence computation with Ghost Dot-Product optimization.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'utils'))

from model_architecture import ResNet50_Animals10
from version_manager import VersionManager
from influence_utils import (
    InfluenceHook,
    compute_ghost_influence_batch,
    TopKInfluenceTracker,
    accumulate_influences_across_checkpoints
)


def setup_data_loaders(
    data_dir: str,
    batch_size: int,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    if train_subset is not None:
        indices = torch.randperm(len(train_dataset))[:train_subset]
        train_dataset = Subset(train_dataset, indices)
    
    if test_subset is not None:
        indices = torch.randperm(len(test_dataset))[:test_subset]
        test_dataset = Subset(test_dataset, indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


def compute_influence_features_batch(
    model: nn.Module,
    hook: InfluenceHook,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute influence features for a batch using Ghost Dot-Product.
    
    Returns activations and error signals separately to enable the
    Ghost Dot-Product optimization: <uv^T, u'v'^T> = <u, u'> * <v, v'>
    
    Returns:
        activations: [batch_size, in_features] - input to target layer
        error_signals: [batch_size, out_features] - gradient w.r.t. layer output
    """
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    
    batch_activations = []
    batch_errors = []
    
    for i in range(len(inputs)):
        model.zero_grad()
        hook.clear()
        
        single_input = inputs[i:i+1]
        single_label = labels[i:i+1]
        
        output = model(single_input)
        loss = criterion(output, single_label)
        loss.backward()
        
        activations, errors = hook.get_activations_and_errors()
        batch_activations.append(activations)
        batch_errors.append(errors)
    
    return torch.cat(batch_activations, dim=0), torch.cat(batch_errors, dim=0)


def precompute_test_features(
    model: nn.Module,
    test_loader: DataLoader,
    hook: InfluenceHook,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute influence features for all test samples.
    
    Returns activations and error signals separately for Ghost Dot-Product.
    
    Returns:
        test_activations: [num_test, in_features]
        test_errors: [num_test, out_features]
    """
    model.eval()
    all_test_activations = []
    all_test_errors = []
    
    print("[COMPUTE] Precomputing test influence features...")
    for inputs, labels in tqdm(test_loader, desc="Test features", leave=False):
        with torch.set_grad_enabled(True):  # Need gradients for backward
            activations, errors = compute_influence_features_batch(
                model, hook, inputs, labels, criterion, device
            )
            all_test_activations.append(activations.cpu())
            all_test_errors.append(errors.cpu())
    
    return torch.cat(all_test_activations, dim=0), torch.cat(all_test_errors, dim=0)


def compute_top_k_influences_single_checkpoint(
    model: nn.Module,
    train_loader: DataLoader,
    test_activations: torch.Tensor,
    test_errors: torch.Tensor,
    hook: InfluenceHook,
    criterion: nn.Module,
    device: torch.device,
    top_k: int,
    learning_rate: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-K influences using tile-and-stitch approach with Ghost Dot-Product.
    
    Uses the identity: <uv^T, u'v'^T> = <u, u'> * <v, v'>
    to compute gradient dot products without materializing full gradients.
    
    Returns:
        top_k_values: [num_test, top_k]
        top_k_indices: [num_test, top_k]
    """
    model.eval()
    num_test = test_activations.size(0)
    test_activations = test_activations.to(device)
    test_errors = test_errors.to(device)
    
    tracker = TopKInfluenceTracker(k=top_k, num_test=num_test, device=device)
    
    print(f"[COMPUTE] Computing influences (lr={learning_rate:.2e})...")
    
    for inputs, labels in tqdm(train_loader, desc="Train batches", leave=False):
        with torch.set_grad_enabled(True):
            train_activations, train_errors = compute_influence_features_batch(
                model, hook, inputs, labels, criterion, device
            )
            
            # True Ghost Dot-Product: influence = lr * (a_train · a_test) * (δ_train · δ_test)
            influences = compute_ghost_influence_batch(
                train_activations, train_errors,
                test_activations, test_errors,
                learning_rate
            )
            
            tracker.add_tile(influences, train_batch_size=len(inputs))
    
    return tracker.get_top_k()


def compute_tracin_multi_checkpoint(
    checkpoint_paths: List[str],
    data_dir: str,
    output_dir: str,
    top_k: int = 100,
    batch_size: int = 32,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    device: Optional[torch.device] = None
):
    """Main TracIn computation with multiple checkpoints."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Top-K: {top_k}")
    print(f"Batch size: {batch_size}")
    
    # Setup data
    train_loader, test_loader, train_dataset, test_dataset = setup_data_loaders(
        data_dir, batch_size, train_subset, test_subset
    )
    
    num_train = len(train_dataset)
    num_test = len(test_dataset)
    
    print(f"Train samples: {num_train}")
    print(f"Test samples: {num_test}")
    print(f"Checkpoints: {len(checkpoint_paths)}")
    
    model = ResNet50_Animals10(num_animal_classes=10, pretrained=False)
    model = model.to(device)
    
    # Hook the FINAL Linear layer of the classifier head
    # model.model.fc is a Sequential: [Linear(2048,512), LeakyReLU, Linear(512,256), LeakyReLU, Linear(256,10)]
    # We want the last Linear layer (index -1) for proper Ghost Dot-Product
    target_layer = model.model.fc[-1]  # Final Linear(256, 10)
    hook = InfluenceHook(target_layer)
    hook.register_hooks()
    
    # Use the SAME loss function as training for theoretical correctness
    # Training uses class weights and label smoothing
    # Compute class weights based on training data distribution
    train_dir = Path(data_dir) / 'train'
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    counts = np.array([len(list((train_dir / c).glob('*'))) for c in class_names])
    class_weights = torch.tensor(
        counts.sum() / (len(class_names) * counts),
        dtype=torch.float32
    ).to(device)
    
    # Match training loss: CrossEntropyLoss with class weights and label smoothing
    # Note: We use reduction='mean' here since we compute per-sample gradients
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Collect influences from all checkpoints
    checkpoint_influences = []
    
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n{'='*60}")
        print(f"Processing checkpoint {checkpoint_idx + 1}/{len(checkpoint_paths)}")
        print(f"Path: {checkpoint_path}")
        print('='*60)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Look for metadata file (training saves as 'finetune_metadata.json')
        metadata_path = Path(checkpoint_path).parent / 'finetune_metadata.json'
        learning_rate = 1e-4  # Default fallback
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                for entry in metadata:
                    if str(checkpoint_path) == entry['checkpoint_path'] or \
                       Path(checkpoint_path).name == Path(entry['checkpoint_path']).name:
                        # Use head LR if available (since we hook the final FC layer)
                        # The head LR is typically 2x the base LR in training
                        if 'learning_rate_head' in entry:
                            learning_rate = entry['learning_rate_head']
                            print(f"Learning rate (head): {learning_rate:.2e}")
                        else:
                            # Backward compatibility: for older checkpoints without learning_rate_head,
                            # the stored learning_rate is the backbone LR.
                            # Head LR was 2x base LR in training, so estimate it
                            base_lr = entry.get('learning_rate', 1e-4)
                            learning_rate = base_lr * 2  # Head was trained with 2x LR
                            print(f"Learning rate (estimated head LR): {learning_rate:.2e}")
                            print(f"  (backbone LR was {base_lr:.2e}, head LR = 2x backbone)")
                        break
        else:
            print(f"[WARN] Metadata file not found, using default LR: {learning_rate:.2e}")
        
        # Precompute test features (activations and error signals)
        test_activations, test_errors = precompute_test_features(
            model, test_loader, hook, criterion, device
        )
        
        top_k_values, top_k_indices = compute_top_k_influences_single_checkpoint(
            model, train_loader, test_activations, test_errors, hook, criterion,
            device, top_k, learning_rate
        )
        
        # Store checkpoint influences for accumulation
        checkpoint_influences.append((top_k_values.cpu(), top_k_indices.cpu()))
    
    hook.remove_hooks()
    
    # Accumulate influences across all checkpoints
    print(f"\n{'='*60}")
    print("Accumulating influences across checkpoints...")
    print('='*60)
    
    aggregated_values, aggregated_indices = accumulate_influences_across_checkpoints(
        checkpoint_influences, k=top_k, num_train=num_train
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Saving results...")
    print('='*60)
    
    # Save as numpy arrays
    np.save(
        os.path.join(output_dir, 'top_k_influences_values.npy'),
        aggregated_values.numpy()
    )
    np.save(
        os.path.join(output_dir, 'top_k_influences_indices.npy'),
        aggregated_indices.numpy()
    )
    
    # Save as CSV for easy inspection
    results_df = create_results_dataframe(
        aggregated_values.numpy(),
        aggregated_indices.numpy(),
        test_dataset,
        train_dataset
    )
    results_df.to_csv(os.path.join(output_dir, 'top_k_influences.csv'), index=False)
    
    print(f"Results saved to {output_dir}")
    print(f"  - top_k_influences_values.npy: shape {aggregated_values.shape}")
    print(f"  - top_k_influences_indices.npy: shape {aggregated_indices.shape}")
    print(f"  - top_k_influences.csv")
    
    return aggregated_values, aggregated_indices


def create_results_dataframe(
    values: np.ndarray,
    indices: np.ndarray,
    test_dataset: Dataset,
    train_dataset: Dataset
) -> pd.DataFrame:
    """Create a readable DataFrame from top-K results with class information."""
    
    num_test, k = values.shape
    
    # Get class names from the underlying dataset
    if hasattr(train_dataset, 'dataset'):
        # It's a Subset
        base_train = train_dataset.dataset
        train_indices = train_dataset.indices
    else:
        base_train = train_dataset
        train_indices = None
    
    if hasattr(test_dataset, 'dataset'):
        base_test = test_dataset.dataset
        test_indices = test_dataset.indices
    else:
        base_test = test_dataset
        test_indices = None
    
    class_names = base_train.classes if hasattr(base_train, 'classes') else None
    
    rows = []
    for test_idx in range(num_test):
        # Get actual test index if using subset
        actual_test_idx = test_indices[test_idx] if test_indices is not None else test_idx
        test_label = base_test.targets[actual_test_idx] if hasattr(base_test, 'targets') else None
        test_class = class_names[test_label] if class_names and test_label is not None else None
        
        for rank in range(k):
            train_idx = int(indices[test_idx, rank])
            influence = float(values[test_idx, rank])
            
            # Get train label if using subset
            if train_indices is not None:
                actual_train_idx = train_indices[train_idx] if train_idx < len(train_indices) else train_idx
            else:
                actual_train_idx = train_idx
            
            train_label = base_train.targets[actual_train_idx] if hasattr(base_train, 'targets') and actual_train_idx < len(base_train.targets) else None
            train_class = class_names[train_label] if class_names and train_label is not None else None
            
            row = {
                'test_idx': test_idx,
                'test_class': test_class,
                'rank': rank + 1,
                'train_idx': train_idx,
                'train_class': train_class,
                'influence': influence,
                'influence_type': 'proponent' if influence > 0 else 'opponent'
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Compute TracIn influence scores')
    
    parser.add_argument('--version', type=int, help='Model version number')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory (overrides version)')
    parser.add_argument('--output_dir', type=str, help='Output directory (auto if version specified)')
    parser.add_argument('--top_k', type=int, default=100, help='Top influences per test sample')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_subset', type=int, default=None, help='Train subset (None = all)')
    parser.add_argument('--test_subset', type=int, default=None, help='Test subset (None = all)')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / args.data_dir
    
    vm = VersionManager(script_dir)
    
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        output_dir = args.output_dir if args.output_dir else 'outputs/influence_analysis'
        version_num = None
    elif args.version:
        version_num = args.version
        checkpoint_dir = script_dir / 'models' / 'checkpoints' / f'v{version_num}'
        output_dir = script_dir / 'outputs' / f'v{version_num}' / 'influence_analysis'
    else:
        latest_version = vm.get_latest_version()
        if latest_version is None:
            raise ValueError("No trained models found. Use --version or train a model first.")
        version_num = latest_version
        checkpoint_dir = script_dir / 'models' / 'checkpoints' / f'v{version_num}'
        output_dir = script_dir / 'outputs' / f'v{version_num}' / 'influence_analysis'
    
    checkpoint_paths = sorted(checkpoint_dir.glob('finetune_epoch_*.pth'))
    
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    
    checkpoint_paths = [str(p) for p in checkpoint_paths]
    
    print("=" * 70)
    print("TracIn Influence Computation")
    print("=" * 70)
    if version_num:
        print(f"[VERSION] Model: v{version_num}")
    print(f"[CHECKPOINTS] {len(checkpoint_paths)} found")
    print(f"[CONFIG] Top-K: {args.top_k} | Batch: {args.batch_size}")
    print(f"[OUTPUT] {output_dir}")
    print("=" * 70)
    
    compute_tracin_multi_checkpoint(
        checkpoint_paths=checkpoint_paths,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        top_k=args.top_k,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset
    )
    
    print("\n[DONE] Influence computation complete")


if __name__ == '__main__':
    main()
