"""Detect model mispredictions on test and training data."""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from model_architecture import ResNet50_Animals10
from version_manager import VersionManager


def get_data_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_model(model_path, device):
    model = ResNet50_Animals10(num_animal_classes=10, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def collect_mispredictions(model, data_dir, split, device, transform):
    split_dir = data_dir / split
    if not split_dir.is_dir():
        print(f"[WARN] Directory not found: {split_dir}")
        return []
    
    dataset = datasets.ImageFolder(root=str(split_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    mispredictions = []
    sample_index = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            
            for i in range(len(labels_np)):
                if preds[i] != labels_np[i]:
                    img_path, _ = dataset.samples[sample_index + i]
                    mispredictions.append({
                        'path': img_path,
                        'true': int(labels_np[i]),
                        'pred': int(preds[i]),
                        'split': split
                    })
            sample_index += len(labels_np)
    
    return mispredictions


def save_mispredictions_csv(mispredictions, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'split', 'true', 'pred'])
        writer.writeheader()
        writer.writerows(mispredictions)
    print(f"[SAVED] Mispredictions CSV: {output_path}")


def create_misprediction_grid(mispredictions, class_names, output_path, max_images=500):
    if len(mispredictions) == 0:
        print("[INFO] No mispredictions to visualize")
        return
    
    mis_subset = mispredictions[:max_images]
    n = len(mis_subset)
    cols = 10
    rows = (n + cols - 1) // cols
    
    thumb_size = 100
    label_height = 20
    padding = 5
    
    canvas_width = cols * (thumb_size + padding) + padding
    canvas_height = rows * (thumb_size + label_height + padding) + padding
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
    
    for idx, entry in enumerate(mis_subset):
        try:
            img = Image.open(entry['path']).convert('RGB')
        except:
            continue
        
        img = img.resize((thumb_size, thumb_size))
        row = idx // cols
        col = idx % cols
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + label_height + padding)
        
        canvas.paste(img, (x, y))
        
        true_name = class_names[entry['true']]
        pred_name = class_names[entry['pred']]
        label = f"{idx+1}. T:{true_name} P:{pred_name}"
        draw.text((x + 2, y + thumb_size + 2), label, fill=(0, 0, 0), font=font)
    
    canvas.save(output_path)
    print(f"[SAVED] Misprediction grid ({len(mis_subset)} images): {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Detect model mispredictions')
    parser.add_argument('--version', type=int, help='Model version to evaluate')
    parser.add_argument('--model_path', type=str, help='Direct path to model file (overrides version)')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data' / 'processed'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vm = VersionManager(script_dir)
    
    if args.model_path:
        model_path = Path(args.model_path)
        version_num = None
        output_dir = script_dir / 'outputs' / 'mispredictions'
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.version:
        version_num = args.version
        version_info = vm.get_version_info(version_num)
        if not version_info:
            raise ValueError(f"Version {version_num} not found in registry")
        model_path = Path(version_info['model_path'])
        output_dir = script_dir / 'outputs' / f'v{version_num}' / 'mispredictions'
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        latest_version = vm.get_latest_version()
        if latest_version is None:
            raise ValueError("No trained models found. Please specify --model_path or train a model first.")
        version_num = latest_version
        version_info = vm.get_version_info(version_num)
        model_path = Path(version_info['model_path'])
        output_dir = script_dir / 'outputs' / f'v{version_num}' / 'mispredictions'
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Misprediction Detection")
    print("=" * 70)
    if version_num:
        print(f"[VERSION] Model version: {version_num}")
    print(f"[MODEL] {model_path}")
    print(f"[DEVICE] {device}")
    print(f"[OUTPUT] {output_dir}")
    print("=" * 70)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    transform = get_data_transform()
    model = load_model(model_path, device)
    
    print("\n[DETECTING] Scanning train split...")
    mis_train = collect_mispredictions(model, data_dir, 'train', device, transform)
    print(f"[RESULT] Train mispredictions: {len(mis_train)}")
    
    print("\n[DETECTING] Scanning test split...")
    mis_test = collect_mispredictions(model, data_dir, 'test', device, transform)
    print(f"[RESULT] Test mispredictions: {len(mis_test)}")
    
    all_mispredictions = mis_train + mis_test
    print(f"\n[TOTAL] {len(all_mispredictions)} mispredictions")
    
    csv_path = output_dir / 'false_predictions.csv'
    save_mispredictions_csv(all_mispredictions, csv_path)
    
    train_dir = data_dir / 'train'
    if train_dir.is_dir():
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    else:
        class_names = [str(i) for i in range(10)]
    
    if all_mispredictions:
        grid_path = output_dir / 'mispredictions_grid.png'
        create_misprediction_grid(all_mispredictions, class_names, grid_path, max_images=500)


if __name__ == '__main__':
    main()
