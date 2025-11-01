# Tristan Mirolla 40112168, Melissa Ananian 40112159, Duc Huy Ta (40232735)
# October 17, 2025
# Project: Explainable Classification using GRAD-CAM
# Script to train and evaluate ResNet50 on Animals-10 dataset
# Modified to use a train/validation split to keep the test set strictly for final evaluation

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import copy
from tqdm.auto import tqdm

# Load the custom Resnet50 model for Animals10
from resnet50_animals10_model import ResNet50_Animals10

def train_model(model, criterion, optimizer, scheduler, num_epochs, early_stopping_patience, save_model=True):
    since = time.time()
    
    # Enable mixed precision training if using CUDA
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_epoch = 0
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            seen_samples = 0
            
            data_iter = dataloaders[phase]
            pbar = tqdm(
                data_iter,
                desc=f"{phase.capitalize()} [{epoch+1}/{num_epochs}]",
                total=len(data_iter),
                unit='batch',
                leave=False
            )
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision training for speed + memory efficiency
                if phase == 'train' and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                seen_samples += inputs.size(0)

                if seen_samples:
                    avg_loss = running_loss / seen_samples
                    avg_acc = (running_corrects.float() / seen_samples).item()
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")
            try:
                pbar.close()
            except Exception:
                pass
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                # Early stopping on validation loss
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_val_acc = epoch_acc.item()
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improve = 0
                    print(f'*** New best model found! Val Accuracy: {best_val_acc:.4f} | Loss: {best_val_loss:.4f} ***')
                else:
                    epochs_no_improve += 1
                    print(f'No improvement in val loss for {epochs_no_improve} epoch(s).')

        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs. Best epoch: {best_epoch+1}')
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Accuracy: {best_val_acc:.4f}')
    
    model.load_state_dict(best_model_wts)

    # Save best model only if requested (do not save after intermediate head-only stage)
    if save_model:
        os.makedirs("models", exist_ok=True)
        acc_str = f"{best_val_acc:.4f}".replace(".", "_")
        loss_str = f"{best_val_loss:.4f}".replace(".", "_")
        model_save_path = os.path.join("models", f"Resnet50_animals10_val_{acc_str}_{loss_str}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f'\nBest model saved to: {model_save_path}')

    try:
        pbar.close()
    except Exception:
        pass

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-num_epochs", type=int, default=100, dest="num_epochs", help="Number of epochs for fine-tuning")
    parser.add_argument("-head_epochs", type=int, default=10, dest="head_epochs", help="Number of epochs for head-only training")
    parser.add_argument("-batch_size", type=int, default=64, dest="batch_size", help="Batch size")
    parser.add_argument("-lr", type=float, default=1e-4, dest="lr", help="Learning rate")
    parser.add_argument("-early_stopping_patience", type=int, default=10, dest="early_stopping_patience", help="Early stopping patience")
    parser.add_argument("-val_split", type=float, default=0.2, dest="val_split", help="Fraction of train data to use for validation (0-1)")
    parser.add_argument("-seed", type=int, default=30, dest="seed", help="Random seed for train/val split")

    args = parser.parse_args()
    num_epochs = args.num_epochs
    head_epochs = args.head_epochs
    batch_size = args.batch_size
    lr = args.lr
    early_stopping_patience = args.early_stopping_patience
    val_split = args.val_split
    seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print("Parameter Settings")
    print("------------Train-----------")
    print("lr: {}".format(lr))
    print("batch_size: {}".format(batch_size))
    print("head_epochs: {}".format(head_epochs))
    print("finetune_epochs: {}".format(num_epochs))
    print("early_stopping_patience: {}".format(early_stopping_patience))
    print("val_split: {}".format(val_split))
    print("seed: {}".format(seed))
    print("device: {}".format(device))
    print("\n" + "="*60)
    print("Starting ResNet-50 Training on Animals-10 Dataset")
    print("="*60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', 'processed')

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

    full_train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=data_transforms['train'])

    # Split train into train and val
    total_train = len(full_train_dataset)
    if total_train == 0:
        raise RuntimeError("No training images found in: {}".format(os.path.join(data_dir, 'train')))
    val_size = int(val_split * total_train)
    train_size = total_train - val_size
    if val_size <= 0:
        raise RuntimeError("val_split too small or dataset too small; resulting validation set size <= 0")
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    }

    dataset_sizes = { 'train': len(train_dataset), 'val': len(val_dataset) }
    # class names from the original ImageFolder
    class_names = full_train_dataset.classes

    # create model (architecture is defined in resnet50_animals10_model.py)
    model = ResNet50_Animals10(num_animal_classes=10, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # Compute class weights for imbalanced classes
    import numpy as np
    counts = np.array([len(os.listdir(os.path.join(data_dir, 'train', c))) for c in class_names])
    class_weights = counts.sum() / (len(class_names) * counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    print(f"\n{'='*60}")
    print("STAGE 1: Training classifier head only (backbone frozen)")
    print(f"{'='*60}")
    
    # Stage A: train head only with higher LR
    head_params = model.model.fc.parameters()
    optimizer_head = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    scheduler_head = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_head, T_0=5, T_mult=1)

    model, history = train_model(model, criterion, optimizer_head, scheduler_head, num_epochs=head_epochs, early_stopping_patience=early_stopping_patience, save_model=False)

    print(f"\n{'='*60}")
    print("STAGE 2: Fine-tuning entire model (backbone unfrozen)")
    print(f"{'='*60}")
    
    # After Stage A completes: Unfreeze backbone
    for param in model.model.parameters():
        param.requires_grad = True

    # Set different learning rates for backbone vs head (discriminative learning)
    backbone_params = [p for n,p in model.model.named_parameters() if 'fc' not in n]
    head_params = [p for n,p in model.model.named_parameters() if 'fc' in n]

    optimizer_ft = optim.AdamW([
        {'params': backbone_params, 'lr': 5e-6},  # Lower LR for pretrained backbone
        {'params': head_params, 'lr': 5e-5}       # Higher LR for head
    ], weight_decay=1e-4)

    # Use OneCycleLR for faster convergence
    from torch.optim.lr_scheduler import OneCycleLR
    steps_per_epoch = max(1, len(dataloaders['train']))
    scheduler_ft = OneCycleLR(optimizer_ft, max_lr=[5e-5, 2e-4], steps_per_epoch=steps_per_epoch, epochs=num_epochs, pct_start=0.3)

    model, history = train_model(model, criterion, optimizer_ft, scheduler_ft, num_epochs=num_epochs, early_stopping_patience=early_stopping_patience)
