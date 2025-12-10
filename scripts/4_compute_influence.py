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
    
    print("[COMPUTE] Precomputing test influence features...")
    for inputs, labels in tqdm(test_loader, desc="Test features", leave=False):
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
    
    print(f"[COMPUTE] Computing influences (lr={learning_rate:.2e})...")
    
    for inputs, labels in tqdm(train_loader, desc="Train batches", leave=False):
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
                        learning_rate = entry['learning_rate']
                        print(f"Learning rate: {learning_rate:.2e}")
                        break
        else:
            print(f"[WARN] Metadata file not found, using default LR: {learning_rate:.2e}")
        
        test_features = precompute_test_features(
            model, test_loader, hook, criterion, device
        )
        
        top_k_values, top_k_indices = compute_top_k_influences_single_checkpoint(
            model, train_loader, test_features, hook, criterion,
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
