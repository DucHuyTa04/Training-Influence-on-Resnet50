"""
TracIn Implementation for ResNet50 Animals-10

Computes training sample influence on test predictions using gradient similarity.
Reference: https://arxiv.org/abs/2002.08484

Usage:
  # Quick test
  python TracIn_Resnet50.py --train_subset 100 --test_subset 10
  
  # Medium analysis
  python TracIn_Resnet50.py --train_subset 500 --test_subset 50
  
  # Full dataset
  python TracIn_Resnet50.py
  
  # After computation, analyze results
  python analyze_results.py
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import pandas as pd

# Add parent directory to path to import model architecture
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from model_architecture import ResNet50_Animals10


def compute_sample_gradients(
    model: nn.Module,
    criterion: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Compute loss gradients for each sample in a batch."""
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    
    # Forward pass
    outputs = model(inputs)
    losses = criterion(outputs, labels)
    
    # Compute gradient for each sample
    batch_gradients = []
    for i in range(len(losses)):
        model.zero_grad()
        losses[i].backward(retain_graph=True)
        
        # Concatenate all parameter gradients into single vector
        grad_vector = torch.cat([
            param.grad.reshape(-1) 
            for param in model.parameters() 
            if param.requires_grad
        ]).detach()
        
        batch_gradients.append(grad_vector)
    
    return torch.stack(batch_gradients)  # [batch_size, num_params]


def compute_influence_via_einsum(
    train_grads: torch.Tensor,
    test_grads: torch.Tensor,
    learning_rate: float
) -> torch.Tensor:
    """Compute influence scores: influence[i,j] = lr * dot(train_grad[i], test_grad[j])"""
    return learning_rate * torch.einsum('ip,jp->ij', train_grads, test_grads)


def precompute_test_gradients(
    model: nn.Module,
    criterion: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    desc: str = "Computing test gradients"
) -> torch.Tensor:
    """Precompute test gradients once, reuse for all training batches."""
    all_test_grads = []
    
    for inputs, labels in tqdm(test_dataloader, desc=desc):
        batch_grads = compute_sample_gradients(
            model=model,
            criterion=criterion,
            inputs=inputs,
            labels=labels,
            device=device
        )
        all_test_grads.append(batch_grads)
    
    return torch.cat(all_test_grads, dim=0)  # [total_test, num_params]


def calculate_tracin_scores(
    model: nn.Module,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    learning_rate: float,
    device: torch.device
) -> np.ndarray:
    """Calculate TracIn influence scores [num_train, num_test]."""
    num_train = len(train_dataloader.dataset)
    num_test = len(test_dataloader.dataset)
    
    print(f"\nTracIn Computation:")
    print(f"  Training samples: {num_train}")
    print(f"  Test samples: {num_test}")
    print(f"  Learning rate: {learning_rate}")
    
    # Initialize score matrix
    score_matrix = np.zeros((num_train, num_test), dtype=np.float32)
    
    # Precompute test gradients (done once, reused for all training batches)
    test_gradients = precompute_test_gradients(
        model=model,
        criterion=criterion,
        test_dataloader=test_dataloader,
        device=device,
        desc="1/2: Precomputing test gradients"
    )
    
    print(f"  Test gradients shape: {test_gradients.shape}")
    
    # Process training data batch by batch
    train_start_time = time.time()
    train_batch_size = train_dataloader.batch_size
    
    for train_idx, (inputs, labels) in enumerate(tqdm(
        train_dataloader, 
        desc="2/2: Computing train influences"
    )):
        # Compute gradients for current training batch
        train_grads = compute_sample_gradients(
            model=model,
            criterion=criterion,
            inputs=inputs,
            labels=labels,
            device=device
        )
        
        # Compute influence scores for this training batch
        influence = compute_influence_via_einsum(
            train_grads=train_grads,
            test_grads=test_gradients,
            learning_rate=learning_rate
        ).cpu().numpy()
        
        # Update score matrix
        train_start = train_idx * train_batch_size
        train_end = train_start + train_grads.shape[0]
        score_matrix[train_start:train_end, :] = influence
    
    elapsed = time.time() - train_start_time
    print(f"\n  Training influence computation time: {elapsed:.2f}s")
    print(f"  Score matrix shape: {score_matrix.shape}")
    print(f"  Score range: [{score_matrix.min():.4f}, {score_matrix.max():.4f}]")
    
    return score_matrix


class TracInResNet50:
    """TracIn wrapper for ResNet50 Animals-10 model."""
    
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        print("="*60)
        print("TracIn for ResNet50 Animals-10")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model: {self.model_path}")
        print(f"Data: {self.data_dir}")
        print(f"Batch size: {self.batch_size}")
        print("="*60)
        
        self.model = self._load_model()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def _load_model(self) -> nn.Module:
        """Load trained ResNet50 model from checkpoint."""
        print("\nLoading model...")
        model = ResNet50_Animals10(num_animal_classes=10, pretrained=False)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ✓ Model loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
    
    def load_data(
        self,
        train_subset_size: Optional[int] = None,
        test_subset_size: Optional[int] = None,
        train_indices: Optional[List[int]] = None,
        test_indices: Optional[List[int]] = None
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
        """Load training and test datasets with optional subsetting."""
        print("\nLoading datasets...")
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'train'),
            transform=self.train_transform
        )
        
        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'test'),
            transform=self.test_transform
        )
        
        class_names = train_dataset.classes
        if train_indices is not None:
            train_dataset = Subset(train_dataset, train_indices)
            print(f"Using {len(train_indices)} specified training samples")
        elif train_subset_size is not None:
            train_indices = list(range(min(train_subset_size, len(train_dataset))))
            train_dataset = Subset(train_dataset, train_indices)
            print(f"Using {train_subset_size} training samples")
        else:
            print(f"Using all {len(train_dataset)} training samples")
        
        if test_indices is not None:
            test_dataset = Subset(test_dataset, test_indices)
            print(f"Using {len(test_indices)} specified test samples")
        elif test_subset_size is not None:
            test_indices = list(range(min(test_subset_size, len(test_dataset))))
            test_dataset = Subset(test_dataset, test_indices)
            print(f"Using {test_subset_size} test samples")
        else:
            print(f"Using all {len(test_dataset)} test samples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Important: don't shuffle for TracIn
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader, class_names
    
    def compute_influence_scores(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 1e-4
    ) -> np.ndarray:
        """Compute TracIn influence scores [num_train, num_test]."""
        return calculate_tracin_scores(
            model=self.model,
            criterion=self.criterion,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            learning_rate=learning_rate,
            device=self.device
        )
    
    def save_results(
        self,
        score_matrix: np.ndarray,
        output_dir: str,
        class_names: List[str],
        train_loader: DataLoader,
        test_loader: DataLoader
    ):
        """Save influence scores to CSV and NPY files."""
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'influence_scores.npy'), score_matrix)
        print(f"\nSaved influence scores to {output_dir}/influence_scores.npy")
        df = pd.DataFrame(score_matrix)
        df.to_csv(os.path.join(output_dir, 'influence_scores.csv'), index=False)
        print(f"Saved CSV format to {output_dir}/influence_scores.csv")
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.numpy())
        train_labels = np.array(train_labels)
        
        test_labels = []
        for _, labels in test_loader:
            test_labels.extend(labels.numpy())
        test_labels = np.array(test_labels)
        summary = {
            'test_idx': [],
            'test_class': [],
            'most_helpful_train_idx': [],
            'most_helpful_train_class': [],
            'max_positive_influence': [],
            'most_harmful_train_idx': [],
            'most_harmful_train_class': [],
            'max_negative_influence': [],
            'avg_influence': []
        }
        
        for test_idx in range(score_matrix.shape[1]):
            scores = score_matrix[:, test_idx]
            most_helpful_idx = np.argmax(scores)
            most_harmful_idx = np.argmin(scores)
            
            summary['test_idx'].append(test_idx)
            summary['test_class'].append(class_names[test_labels[test_idx]])
            summary['most_helpful_train_idx'].append(int(most_helpful_idx))
            summary['most_helpful_train_class'].append(class_names[train_labels[most_helpful_idx]])
            summary['max_positive_influence'].append(float(scores[most_helpful_idx]))
            summary['most_harmful_train_idx'].append(int(most_harmful_idx))
            summary['most_harmful_train_class'].append(class_names[train_labels[most_harmful_idx]])
            summary['max_negative_influence'].append(float(scores[most_harmful_idx]))
            summary['avg_influence'].append(float(np.mean(scores)))
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, 'influence_summary.csv'), index=False)
        print(f"Saved influence summary to {output_dir}/influence_summary.csv")
        
        print(f"\nAll results saved to {output_dir}/")
        print("\n" + "="*60)
        print("Next step: Run analysis")
        print("="*60)
        print("python TracIn/analyze_results.py")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Compute TracIn influence scores')
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/Resnet50_animals10_val_0_9796_0_5963.pth',
        help='Path to trained model checkpoint (relative to project root)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Path to processed data directory (relative to project root)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='TracIn/results',
        help='Directory to save results (relative to project root)'
    )
    
    parser.add_argument(
        '--train_subset',
        type=int,
        default=None,
        help='Number of training samples (None = all)'
    )
    
    parser.add_argument(
        '--test_subset',
        type=int,
        default=None,
        help='Number of test samples (None = all)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for data loading'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for scaling influence scores'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent.parent
    model_path = script_dir / args.model_path
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir
    
    print("\n" + "="*60)
    print("TracIn for ResNet50 Animals-10")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Training subset: {args.train_subset or 'all'}")
    print(f"Test subset: {args.test_subset or 'all'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*60)
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print(f"   Please check the path or train a model first.")
        return
    
    if not data_dir.exists():
        print(f"\nError: Data directory not found at {data_dir}")
        print(f"   Run preprocess_data.ipynb first.")
        return
    
    start_time = time.time()
    
    tracin = TracInResNet50(
        model_path=str(model_path),
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_loader, test_loader, class_names = tracin.load_data(
        train_subset_size=args.train_subset,
        test_subset_size=args.test_subset
    )
    score_matrix = tracin.compute_influence_scores(
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate
    )
    tracin.save_results(
        score_matrix=score_matrix,
        output_dir=str(output_dir),
        class_names=class_names,
        train_loader=train_loader,
        test_loader=test_loader
    )
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✓ TracIn computation completed successfully!")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
