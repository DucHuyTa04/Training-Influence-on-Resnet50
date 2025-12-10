#!/usr/bin/env python3
"""
Download ResNet50 ImageNet pretrained weights.
"""

import torch
import torchvision
from pathlib import Path


def download_weights():
    cache_dir = Path(__file__).parent.parent / 'models' / 'pretrained'
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / 'resnet50_imagenet1k_v1.pth'
    
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / 1024**2
        print(f"[INFO] Weights already exist: {weights_path.name} ({size_mb:.1f} MB)")
        return
    
    print("[INFO] Downloading ResNet50 ImageNet weights...")
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )
    torch.save(model.state_dict(), weights_path)
    
    size_mb = weights_path.stat().st_size / 1024**2
    print(f"[DONE] Weights saved: {weights_path.name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    download_weights()
