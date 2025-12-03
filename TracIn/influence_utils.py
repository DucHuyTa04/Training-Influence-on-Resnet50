"""
Utility functions for efficient TracIn influence computation.

Implements Ghost Dot-Product and Top-K selection as described in Lei's slides.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class InfluenceHook:
    """
    Hook to capture activations and error signals for Ghost Dot-Product.
    
    Captures activations (a) during forward pass and error signals (δ) during backward pass
    to compute influence = lr * dot(a_train * δ_train, a_test * δ_test).
    """
    
    def __init__(self, target_layer: nn.Module):
        self.target_layer = target_layer
        self.activations = None
        self.error_signals = None
        self.forward_handle = None
        self.backward_handle = None
        
    def register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        
        def forward_hook(module, input, output):
            self.activations = input[0].detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.error_signals = grad_output[0].detach()
        
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove registered hooks."""
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()
    
    def get_influence_features(self) -> torch.Tensor:
        """
        Compute the influence feature vector: a ⊗ δ (outer product flattened).
        
        Returns:
            Tensor of shape [batch_size, features_in * features_out]
        """
        if self.activations is None or self.error_signals is None:
            raise RuntimeError("Must run forward and backward pass before getting features")
        
        batch_size = self.activations.size(0)
        
        influence_features = torch.bmm(
            self.activations.unsqueeze(2),
            self.error_signals.unsqueeze(1)
        )
        
        return influence_features.view(batch_size, -1)
    
    def clear(self):
        self.activations = None
        self.error_signals = None


def compute_ghost_influence_batch(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    learning_rate: float = 1.0
) -> torch.Tensor:
    """
    Compute influence scores using ghost dot-product.
    
    Args:
        train_features: [num_train, feature_dim]
        test_features: [num_test, feature_dim]
        learning_rate: Scaling factor
    
    Returns:
        influence_scores: [num_train, num_test]
    """
    return learning_rate * torch.mm(train_features, test_features.t())


def select_top_k_influences(
    influences: torch.Tensor,
    k: int,
    return_indices: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-K most influential training samples for each test sample.
    
    Args:
        influences: [num_train, num_test] influence matrix
        k: Number of top influences to keep per test sample
    
    Returns:
        top_k_values: [num_test, k]
        top_k_indices: [num_test, k]
    """
    top_k_values, top_k_indices = torch.topk(
        influences,
        k=min(k, influences.size(0)),
        dim=0,
        largest=True,
        sorted=True
    )
    
    return top_k_values.t(), top_k_indices.t()


def merge_top_k_tiles(
    tiles: List[Tuple[torch.Tensor, torch.Tensor]],
    k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge top-K results from multiple tiles to get global top-K.
    
    Args:
        tiles: List of (values, indices) tuples, each of shape [num_test, k_tile]
        k: Final number of top influences to keep
    
    Returns:
        final_values: [num_test, k]
        final_indices: [num_test, k]
    """
    if len(tiles) == 0:
        raise ValueError("No tiles to merge")
    
    if len(tiles) == 1:
        return tiles[0]
    
    all_values = torch.cat([tile[0] for tile in tiles], dim=1)
    all_indices = torch.cat([tile[1] for tile in tiles], dim=1)
    
    top_k_values, top_k_positions = torch.topk(
        all_values,
        k=min(k, all_values.size(1)),
        dim=1,
        largest=True,
        sorted=True
    )
    
    top_k_indices = torch.gather(all_indices, 1, top_k_positions)
    
    return top_k_values, top_k_indices


class TopKInfluenceTracker:
    """
    Efficiently track top-K influences across multiple training batches using tile-and-stitch.
    """
    
    def __init__(self, k: int, num_test: int, device: torch.device):
        self.k = k
        self.num_test = num_test
        self.device = device
        self.tiles = []
        self.train_offset = 0
    
    def add_tile(self, influences: torch.Tensor, train_batch_size: int):
        tile_values, tile_indices = select_top_k_influences(influences, self.k)
        tile_indices = tile_indices + self.train_offset
        self.tiles.append((tile_values, tile_indices))
        self.train_offset += train_batch_size
    
    def get_top_k(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return merge_top_k_tiles(self.tiles, self.k)
    
    def clear(self):
        self.tiles = []
        self.train_offset = 0
