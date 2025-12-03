#!/usr/bin/env python3
"""
Download ResNet50 ImageNet pretrained weights to local cache.
Run this on a login node (which has internet access) before training.

Usage:
    python download_weights.py
"""

import torch
import torchvision
from pathlib import Path

def download_weights():
    """Download ResNet50 ImageNet weights to models/weights_cache/"""
    
    # Create cache directory
    cache_dir = Path(__file__).parent / 'models' / 'weights_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = cache_dir / 'resnet50_imagenet1k_v1.pth'
    
    if weights_path.exists():
        print(f"Weights already exist: {weights_path}")
        print(f"Size: {weights_path.stat().st_size / 1024**2:.1f} MB")
        return
    
    print("Downloading ResNet50 ImageNet weights...")
    print(f"Target: {weights_path}")
    
    # Load pretrained model to trigger download
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Save the state dict
    torch.save(model.state_dict(), weights_path)
    
    print(f"Download complete!")
    print(f"Saved to: {weights_path}")
    print(f"Size: {weights_path.stat().st_size / 1024**2:.1f} MB")

if __name__ == '__main__':
    download_weights()
