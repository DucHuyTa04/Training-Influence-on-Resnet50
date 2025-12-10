"""
Utility functions for efficient TracIn influence computation.

Implements Ghost Dot-Product and Top-K selection.
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
        """Compute the influence feature vector: a ⊗ δ (outer product flattened)."""

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
    """Compute influence scores using ghost dot-product.
    
    Returns positive influence for proponents (helpful) and negative for opponents (harmful).
    We negate because gradients point in direction of loss increase.
    TracIn measures loss REDUCTION, so we flip the sign.
    """
    return -learning_rate * torch.mm(train_features, test_features.t())


def select_top_k_influences(
    influences: torch.Tensor,
    k: int,
    return_indices: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-K most influential training samples by absolute magnitude.
    
    This includes both:
    - Proponents: Large positive values (helpful, reduce loss)
    - Opponents: Large negative values (harmful, increase loss)
    
    We select by largest absolute value to capture the most influential examples
    regardless of whether they help or hurt.
    """
    # Get absolute values for ranking
    abs_influences = torch.abs(influences)
    
    # Select top-K by absolute magnitude
    top_k_abs, top_k_indices = torch.topk(
        abs_influences,
        k=min(k, influences.size(0)),
        dim=0,
        largest=True,
        sorted=True
    )
    
    # Get the actual signed values (not absolute)
    top_k_values = torch.gather(influences, 0, top_k_indices)
    
    return top_k_values.t(), top_k_indices.t()


def merge_top_k_tiles(
    tiles: List[Tuple[torch.Tensor, torch.Tensor]],
    k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge top-K results from multiple tiles by absolute magnitude.
    
    This properly handles both proponents (positive) and opponents (negative)
    by selecting based on absolute influence magnitude.
    """
    if len(tiles) == 0:
        raise ValueError("No tiles to merge")
    
    if len(tiles) == 1:
        return tiles[0]
    
    all_values = torch.cat([tile[0] for tile in tiles], dim=1)
    all_indices = torch.cat([tile[1] for tile in tiles], dim=1)
    
    # Select top-K by absolute magnitude
    abs_values = torch.abs(all_values)
    top_k_abs, top_k_positions = torch.topk(
        abs_values,
        k=min(k, all_values.size(1)),
        dim=1,
        largest=True,
        sorted=True
    )
    
    # Get actual signed values and corresponding indices
    top_k_values = torch.gather(all_values, 1, top_k_positions)
    top_k_indices = torch.gather(all_indices, 1, top_k_positions)
    
    return top_k_values, top_k_indices


class TopKInfluenceTracker:
    """Efficiently track top-K influences across multiple training batches using tile-and-stitch.
    
    This maintains top-K by absolute magnitude across all training samples,
    properly handling both positive (proponent) and negative (opponent) influences.
    """
    
    def __init__(self, k: int, num_test: int, device: torch.device):
        self.k = k
        self.num_test = num_test
        self.device = device
        self.tiles = []
        self.train_offset = 0
    
    def add_tile(self, influences: torch.Tensor, train_batch_size: int):
        """Add a batch of training influences to the tracker."""
        tile_values, tile_indices = select_top_k_influences(influences, self.k)
        # Adjust indices to reflect global training set position
        tile_indices = tile_indices + self.train_offset
        self.tiles.append((tile_values, tile_indices))
        self.train_offset += train_batch_size
    
    def get_top_k(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get final top-K influences by merging all tiles."""
        return merge_top_k_tiles(self.tiles, self.k)
    
    def clear(self):
        """Reset the tracker for a new computation."""
        self.tiles = []
        self.train_offset = 0


def accumulate_influences_across_checkpoints(
    checkpoint_influences: List[Tuple[torch.Tensor, torch.Tensor]],
    k: int,
    num_train: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate influence scores across multiple checkpoints.
    
    TracIn requires SUMMING influences across checkpoints for each (train, test) pair.
    This function properly accumulates scores and then selects top-K.
    
    Args:
        checkpoint_influences: List of (values, indices) tuples from each checkpoint
        k: Number of top influences to return
        num_train: Total number of training samples
        
    Returns:
        top_k_values: [num_test, k] accumulated influence scores
        top_k_indices: [num_test, k] training sample indices
    """
    if len(checkpoint_influences) == 0:
        raise ValueError("No checkpoint influences provided")
    
    if len(checkpoint_influences) == 1:
        # Only one checkpoint, return as-is but ensure correct k
        values, indices = checkpoint_influences[0]
        if values.size(1) > k:
            abs_values = torch.abs(values)
            top_k_abs, top_k_positions = torch.topk(abs_values, k, dim=1, largest=True)
            top_k_values = torch.gather(values, 1, top_k_positions)
            top_k_indices = torch.gather(indices, 1, top_k_positions)
            return top_k_values, top_k_indices
        return values, indices
    
    # Get dimensions
    num_test = checkpoint_influences[0][0].size(0)
    
    # Create a dense accumulator for summing influences
    # Shape: [num_test, num_train]
    accumulated_influences = torch.zeros(num_test, num_train)
    
    # Accumulate influences from each checkpoint
    for values, indices in checkpoint_influences:
        # Scatter add the influences to their correct positions
        accumulated_influences.scatter_add_(1, indices, values)
    
    # Select top-K by absolute magnitude from accumulated influences
    abs_accumulated = torch.abs(accumulated_influences)
    top_k_abs, top_k_indices = torch.topk(
        abs_accumulated,
        k=min(k, num_train),
        dim=1,
        largest=True,
        sorted=True
    )
    
    # Get actual signed values
    top_k_values = torch.gather(accumulated_influences, 1, top_k_indices)
    
    return top_k_values, top_k_indices
