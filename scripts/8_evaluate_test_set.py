"""
Evaluate trained models on the test set to get final test accuracy and loss.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from model_architecture import ResNet50_Animals10
from version_manager import VersionManager


def get_test_transform():
    """Standard test-time transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    model = ResNet50_Animals10(num_animal_classes=10, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_on_test_set(model, test_loader, criterion, device):
    """
    Evaluate model on test set.
    
    Returns:
        test_loss (float): Average test loss
        test_accuracy (float): Test accuracy
        num_samples (int): Total number of test samples
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    test_loss = running_loss / total_samples
    test_accuracy = (running_corrects.float() / total_samples).item()
    
    return test_loss, test_accuracy, total_samples


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--version', type=int, help='Model version to evaluate (evaluates all if not specified)')
    parser.add_argument('--model_path', type=str, help='Direct path to model file')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data' / 'processed'
    test_dir = data_dir / 'test'
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load test dataset
    test_transform = get_test_transform()
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print("=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)
    print(f"[DEVICE] {device}")
    print(f"[DATA] Test samples: {len(test_dataset)}")
    print(f"[DATA] Classes: {len(test_dataset.classes)}")
    print("=" * 70)
    
    # Use unweighted CrossEntropyLoss for fair evaluation
    criterion = nn.CrossEntropyLoss()
    
    vm = VersionManager(script_dir)
    results = {}
    
    # Determine which models to evaluate
    if args.model_path:
        # Evaluate single model from path
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"\n[EVALUATING] {model_path.name}")
        model = load_model(model_path, device)
        test_loss, test_acc, num_samples = evaluate_on_test_set(model, test_loader, criterion, device)
        
        print(f"[RESULT] Test Loss: {test_loss:.6f}")
        print(f"[RESULT] Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        results['custom'] = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'num_test_samples': num_samples,
            'model_path': str(model_path)
        }
        
    elif args.version:
        # Evaluate specific version
        version_info = vm.get_version_info(args.version)
        if version_info is None:
            raise ValueError(f"Version {args.version} not found in registry")
        
        model_path = Path(version_info['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"\n[EVALUATING] Version {args.version}")
        print(f"[MODEL] {model_path}")
        
        model = load_model(model_path, device)
        test_loss, test_acc, num_samples = evaluate_on_test_set(model, test_loader, criterion, device)
        
        print(f"[RESULT] Test Loss: {test_loss:.6f}")
        print(f"[RESULT] Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        results[f'v{args.version}'] = {
            'version': args.version,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'val_accuracy': version_info.get('val_accuracy'),
            'num_test_samples': num_samples,
            'model_path': str(model_path)
        }
        
        # Update version registry with test metrics
        version_info['test_loss'] = test_loss
        version_info['test_accuracy'] = test_acc
        vm.register_version(args.version, version_info)
        print(f"[UPDATED] Version registry updated with test metrics")
        
    else:
        # Evaluate all versions
        registry = vm._load_registry()
        if not registry:
            raise ValueError("No models found in version registry")
        all_versions = {int(k): v for k, v in registry.items()}
        
        print(f"\n[INFO] Found {len(all_versions)} model versions")
        print("=" * 70)
        
        for version_num in sorted(all_versions.keys()):
            version_info = all_versions[version_num]
            model_path = Path(version_info['model_path'])
            
            if not model_path.exists():
                print(f"\n[SKIP] Version {version_num}: Model not found at {model_path}")
                continue
            
            print(f"\n[EVALUATING] Version {version_num}")
            print(f"[MODEL] {model_path.name}")
            print(f"[VAL ACC] {version_info.get('val_accuracy', 'N/A'):.4f}" if version_info.get('val_accuracy') else "[VAL ACC] N/A")
            
            model = load_model(model_path, device)
            test_loss, test_acc, num_samples = evaluate_on_test_set(model, test_loader, criterion, device)
            
            print(f"[RESULT] Test Loss: {test_loss:.6f}")
            print(f"[RESULT] Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            
            results[f'v{version_num}'] = {
                'version': version_num,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'val_accuracy': version_info.get('val_accuracy'),
                'val_loss': version_info.get('val_loss'),
                'num_test_samples': num_samples,
                'timestamp': version_info.get('timestamp'),
                'model_path': str(model_path)
            }
            
            # Update version registry with test metrics
            version_info['test_loss'] = test_loss
            version_info['test_accuracy'] = test_acc
            vm.register_version(version_num, version_info)
        
        print("\n" + "=" * 70)
        print("SUMMARY: All Model Versions")
        print("=" * 70)
        print(f"{'Version':<10} {'Val Acc':<12} {'Test Acc':<12} {'Test Loss':<12}")
        print("-" * 70)
        for version_num in sorted(all_versions.keys()):
            if f'v{version_num}' in results:
                r = results[f'v{version_num}']
                val_acc_str = f"{r['val_accuracy']:.4f}" if r.get('val_accuracy') else "N/A"
                print(f"v{version_num:<9} {val_acc_str:<12} {r['test_accuracy']:.4f}      {r['test_loss']:.6f}")
        print("=" * 70)
    
    # Save results to JSON
    output_file = script_dir / 'test_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results saved to: {output_file}")
    
    print("\n[DONE] Evaluation complete!")


if __name__ == '__main__':
    main()
