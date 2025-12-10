"""
ResNet50 training script with model versioning.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import copy
import numpy as np
from tqdm.auto import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from model_architecture import ResNet50_Animals10
from version_manager import VersionManager


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, is_training=True):
    """Train or validate for one epoch."""
    model.train() if is_training else model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    phase = "Train" if is_training else "Val"
    pbar = tqdm(dataloader, desc=f"{phase}", leave=False)
    
    with torch.set_grad_enabled(is_training):
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if is_training:
                optimizer.zero_grad()
            
            if is_training and scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                if is_training:
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            current_acc = (running_corrects.float() / total_samples).item()
            pbar.set_postfix({'loss': f'{running_loss/total_samples:.4f}', 'acc': f'{current_acc:.4f}'})
    
    epoch_loss = running_loss / total_samples
    epoch_acc = (running_corrects.float() / total_samples).item()
    
    return epoch_loss, epoch_acc


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
                device, num_epochs, early_stopping_patience, checkpoint_dir, 
                checkpoint_freq, stage_name, version_num):
    """Main training loop."""
    start_time = time.time()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    checkpoint_metadata = []
    
    print(f"\n[TRAIN] Starting {stage_name} training")
    print(f"[INFO] Checkpoints will be saved every {checkpoint_freq} epochs to: {checkpoint_dir}")
    
    for epoch in range(num_epochs):
        print(f"\n[EPOCH] {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        for phase in ['train', 'val']:
            is_training = (phase == 'train')
            loss, acc = train_epoch(
                model, dataloaders[phase], criterion, optimizer, 
                scaler, device, is_training
            )
            
            print(f"[{phase.upper()}] Loss: {loss:.4f} | Accuracy: {acc:.4f}")
            
            if phase == 'train':
                history['train_loss'].append(loss)
                history['train_acc'].append(acc)
            else:
                history['val_loss'].append(loss)
                history['val_acc'].append(acc)
                
                if hasattr(scheduler, 'step'):
                    if 'metrics' in scheduler.step.__code__.co_varnames:
                        scheduler.step(loss)
                    else:
                        scheduler.step()
                
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_val_acc = acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improve = 0
                    print(f"[BEST] New best model | Val Acc: {best_val_acc:.4f} | Val Loss: {best_val_loss:.4f}")
                else:
                    epochs_no_improve += 1
        
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f'{stage_name}_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            
            checkpoint_metadata.append({
                'epoch': epoch + 1,
                'checkpoint_path': str(checkpoint_path),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'val_loss': history['val_loss'][-1],
                'val_acc': history['val_acc'][-1],
                'stage': stage_name
            })
            print(f"[CHECKPOINT] Saved: {checkpoint_path.name}")
        
        if epochs_no_improve >= early_stopping_patience:
            print(f"[EARLY STOP] Triggered after {epoch+1} epochs (best: epoch {best_epoch+1})")
            break
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"[RESULT] Best Val Accuracy: {best_val_acc:.4f} | Best Val Loss: {best_val_loss:.4f}")
    
    model.load_state_dict(best_model_wts)
    
    if checkpoint_metadata:
        metadata_path = checkpoint_dir / f'{stage_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        print(f"[INFO] Checkpoint metadata saved: {metadata_path.name}")
    
    return model, history, best_val_acc, best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 on Animals-10')
    parser.add_argument('--version', type=int, help='Model version number (auto-assigned if not provided)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Fine-tuning epochs')
    parser.add_argument('--head_epochs', type=int, default=5, help='Head-only training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=30, help='Random seed')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint frequency (epochs)')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data' / 'processed'
    
    vm = VersionManager(script_dir)
    version_num = args.version if args.version else vm.get_next_version()
    version_dirs = vm.create_version_dirs(version_num)
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    print("=" * 70)
    print(f"ResNet-50 Training on Animals-10 Dataset")
    print("=" * 70)
    print(f"[VERSION] Model version: {version_num}")
    print(f"[CONFIG] Batch size: {args.batch_size} | LR: {args.lr} | Epochs: {args.num_epochs}")
    print(f"[CONFIG] Val split: {args.val_split} | Seed: {args.seed} | Device: {device}")
    print(f"[PATHS] Data: {data_dir}")
    print(f"[PATHS] Checkpoints: {version_dirs['checkpoints']}")
    print(f"[PATHS] Model output: {version_dirs['model']}")
    print("=" * 70)
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    full_train_dataset = datasets.ImageFolder(
        root=str(data_dir / 'train'), 
        transform=data_transforms['train']
    )
    
    if len(full_train_dataset) == 0:
        raise RuntimeError(f"No training images found in: {data_dir / 'train'}")
    
    val_size = int(args.val_split * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, 
                          shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=4, pin_memory=True),
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = full_train_dataset.classes
    
    print(f"\n[DATA] Train: {dataset_sizes['train']} | Val: {dataset_sizes['val']}")
    print(f"[DATA] Classes: {', '.join(class_names)}")
    
    counts = np.array([len(list((data_dir / 'train' / c).glob('*'))) for c in class_names])
    class_weights = torch.tensor(
        counts.sum() / (len(class_names) * counts), 
        dtype=torch.float32
    ).to(device)
    
    model = ResNet50_Animals10(num_animal_classes=10, pretrained=True, freeze_backbone=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    print("\n" + "=" * 70)
    print("STAGE 1: Head-Only Training (Backbone Frozen)")
    print("=" * 70)
    
    head_params = model.model.fc.parameters()
    optimizer_head = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    scheduler_head = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_head, T_0=5, T_mult=1)
    
    model, _, _, _ = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer_head, scheduler_head,
        device, args.head_epochs, 999, version_dirs['checkpoints'], 
        args.checkpoint_freq, 'head_only', version_num
    )
    
    print("\n" + "=" * 70)
    print("STAGE 2: Full Fine-Tuning (Backbone Unfrozen)")
    print("=" * 70)
    
    for param in model.model.parameters():
        param.requires_grad = True
    
    backbone_params = [p for n, p in model.model.named_parameters() if 'fc' not in n]
    head_params = [p for n, p in model.model.named_parameters() if 'fc' in n]
    
    optimizer_ft = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr * 2}
    ], weight_decay=1e-4)
    
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    model, history, best_val_acc, best_val_loss = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer_ft, scheduler_ft,
        device, args.num_epochs, args.early_stopping_patience, 
        version_dirs['checkpoints'], args.checkpoint_freq, 'finetune', version_num
    )
    
    model_path = version_dirs['model'] / f'model_v{version_num}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n[SAVED] Final model: {model_path}")
    
    metadata = {
        'version': version_num,
        'val_accuracy': best_val_acc,
        'val_loss': best_val_loss,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'device': str(device),
        'model_path': str(model_path),
        'checkpoint_dir': str(version_dirs['checkpoints']),
        'output_dir': str(version_dirs['outputs'])
    }
    
    vm.register_version(version_num, metadata)
    print(f"[REGISTERED] Version {version_num} registered in version registry")
    print(f"\n[INFO] All outputs for this model will be saved in: outputs/v{version_num}/")


if __name__ == '__main__':
    main()
