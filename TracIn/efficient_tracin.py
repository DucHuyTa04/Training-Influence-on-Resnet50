"""
Efficient TracIn with Ghost Dot-Product and Top-K Selection

Usage:
    # Test on small subset
    python efficient_tracin.py --train_subset 100 --test_subset 10 --top_k 20
    
    # Recommended
    python efficient_tracin.py --train_subset 1000 --test_subset 100 --top_k 100
    
    # Full dataset
    python efficient_tracin.py --top_k 100
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

# Add parent directory to import model
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from model_architecture import ResNet50_Animals10
from TracIn.influence_utils import (
    InfluenceHook,
    compute_ghost_influence_batch,
    TopKInfluenceTracker
)


def setup_data_loaders(
    data_dir: str,
    batch_size: int,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """Setup train and test data loaders."""
    
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
) -> torch.Tensor:
    """
    Compute influence features for a batch using ghost dot-product.
    
    Returns:
        influence_features: [batch_size, feature_dim]
    """
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    
    batch_features = []
    
    for i in range(len(inputs)):
        model.zero_grad()
        hook.clear()
        
        single_input = inputs[i:i+1]
        single_label = labels[i:i+1]
        
        output = model(single_input)
        loss = criterion(output, single_label)
        loss.backward()
        
        features = hook.get_influence_features()
        batch_features.append(features)
    
    return torch.cat(batch_features, dim=0)


def precompute_test_features(
    model: nn.Module,
    test_loader: DataLoader,
    hook: InfluenceHook,
    criterion: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """
    Precompute influence features for all test samples.
    
    Returns:
        test_features: [num_test, feature_dim]
    """
    model.eval()
    all_test_features = []
    
    print("Precomputing test influence features...")
    for inputs, labels in tqdm(test_loader, desc="Test features"):
        with torch.set_grad_enabled(True):  # Need gradients for backward
            features = compute_influence_features_batch(
                model, hook, inputs, labels, criterion, device
            )
            all_test_features.append(features.cpu())
    
    return torch.cat(all_test_features, dim=0)


def compute_top_k_influences_single_checkpoint(
    model: nn.Module,
    train_loader: DataLoader,
    test_features: torch.Tensor,
    hook: InfluenceHook,
    criterion: nn.Module,
    device: torch.device,
    top_k: int,
    learning_rate: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-K influences using tile-and-stitch approach.
    
    Returns:
        top_k_values: [num_test, top_k]
        top_k_indices: [num_test, top_k]
    """
    model.eval()
    num_test = test_features.size(0)
    test_features = test_features.to(device)
    
    tracker = TopKInfluenceTracker(k=top_k, num_test=num_test, device=device)
    
    print(f"Computing influences (learning_rate={learning_rate:.2e})...")
    
    for inputs, labels in tqdm(train_loader, desc="Processing train batches"):
        with torch.set_grad_enabled(True):
            train_features = compute_influence_features_batch(
                model, hook, inputs, labels, criterion, device
            )
            
            influences = compute_ghost_influence_batch(
                train_features, test_features, learning_rate
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
    
    target_layer = model.model.fc
    hook = InfluenceHook(target_layer)
    hook.register_hooks()
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    aggregated_values = None
    aggregated_indices = None
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n{'='*60}")
        print(f"Processing checkpoint {checkpoint_idx + 1}/{len(checkpoint_paths)}")
        print(f"Path: {checkpoint_path}")
        print('='*60)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        metadata_path = Path(checkpoint_path).parent / 'finetune_checkpoints_metadata.json'
        learning_rate = 1e-4
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                for entry in metadata:
                    if entry['checkpoint_path'] == checkpoint_path:
                        learning_rate = entry['learning_rate']
                        print(f"Learning rate: {learning_rate:.2e}")
                        break
        
        test_features = precompute_test_features(
            model, test_loader, hook, criterion, device
        )
        
        top_k_values, top_k_indices = compute_top_k_influences_single_checkpoint(
            model, train_loader, test_features, hook, criterion,
            device, top_k, learning_rate
        )
        
        if aggregated_values is None:
            aggregated_values = top_k_values.cpu()
            aggregated_indices = top_k_indices.cpu()
        else:
            from TracIn.influence_utils import merge_top_k_tiles
            aggregated_values, aggregated_indices = merge_top_k_tiles(
                [(aggregated_values, aggregated_indices), 
                 (top_k_values.cpu(), top_k_indices.cpu())],
                k=top_k
            )
    
    hook.remove_hooks()
    
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
    """Create a readable DataFrame from top-K results."""
    
    num_test, k = values.shape
    
    rows = []
    for test_idx in range(num_test):
        for rank in range(k):
            train_idx = indices[test_idx, rank]
            influence = values[test_idx, rank]
            
            rows.append({
                'test_idx': test_idx,
                'rank': rank + 1,
                'train_idx': int(train_idx),
                'influence': float(influence)
            })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Efficient TracIn with Ghost Dot-Product')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--output_dir', type=str, default='TracIn/results',
                        help='Output directory for results')
    
    # Computation parameters
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of top influences to keep per test sample')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    
    # Subset options for testing
    parser.add_argument('--train_subset', type=int, default=None,
                        help='Number of train samples to use (None = all)')
    parser.add_argument('--test_subset', type=int, default=None,
                        help='Number of test samples to use (None = all)')
    
    args = parser.parse_args()
    
    # Find all checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob('finetune_epoch_*.pth'))
    
    if len(checkpoint_paths) == 0:
        print(f"ERROR: No checkpoints found in {checkpoint_dir}")
        print("Please train a model first with checkpoint saving enabled.")
        return
    
    checkpoint_paths = [str(p) for p in checkpoint_paths]
    
    print("="*60)
    print("Efficient TracIn with Ghost Dot-Product")
    print("="*60)
    print(f"Found {len(checkpoint_paths)} checkpoints")
    
    # Run TracIn
    compute_tracin_multi_checkpoint(
        checkpoint_paths=checkpoint_paths,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset
    )
    
    print("\nTracIn computation complete!")


if __name__ == '__main__':
    main()
