"""
Run a trained ResNet50 model over the entire dataset (train + test) and display/save all mispredicted images.
Outputs:
 - false_predictions.csv in repo root (columns: image_path,true_class,pred_class)
 - saved thumbnails in ./false_predictions/
 - shows paged matplotlib figures with images and labels

Set MODEL_PATH below if needed.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import csv

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from model_architecture import ResNet50_Animals10

# Path to the model (can be relative). Edit if needed.
MODEL_PATH = r"models\Resnet50_animals10_val_0_9796_0_5963.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms for model input (must match training)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Repo/data locations
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data', 'processed')

# Resolve model path (try case-insensitive match in models/ if not exact)
def resolve_model_path(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    p = os.path.join(script_dir, path) if not os.path.isabs(path) else path
    if os.path.exists(p):
        return p
    # try case-insensitive search in models/
    models_dir = os.path.join(script_dir, 'models')
    if os.path.isdir(models_dir):
        candidates = os.listdir(models_dir)
        target_lower = os.path.basename(path).lower()
        for c in candidates:
            if c.lower() == target_lower:
                return os.path.join(models_dir, c)
    raise FileNotFoundError(f"Model file not found: {path}. Checked: {p} and models/ directory.")

MODEL_PATH = resolve_model_path(MODEL_PATH)
print(f"Using model: {MODEL_PATH}")

# Load model
model = ResNet50_Animals10(num_animal_classes=10, pretrained=False, freeze_backbone=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Helper to collect mispredictions for a split
def collect_mispredictions(split):
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"Warning: {split_dir} does not exist, skipping")
        return []
    dataset = datasets.ImageFolder(root=split_dir, transform=data_transform)
    # Use num_workers=0 on Windows to avoid multiprocessing spawn issues
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    mis = []
    sample_index = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            batch_size = labels_np.shape[0]
            for i in range(batch_size):
                if preds[i] != labels_np[i]:
                    img_path, _ = dataset.samples[sample_index + i]
                    mis.append({'path': img_path, 'true': labels_np[i], 'pred': int(preds[i]), 'split': split})
            sample_index += batch_size
    return mis

# Run over train and test
mis_train = collect_mispredictions('train')
mis_test = collect_mispredictions('test')
mis_all = mis_train + mis_test

print(f"Total mispredictions: {len(mis_all)} (train: {len(mis_train)}, test: {len(mis_test)})")

# Attempt to map each mispredicted image to the original image under data/raw-img (match by filename)
# The raw images live at <script_dir>/data/raw-img (not under data/processed)
raw_dir = os.path.join(script_dir, 'data', 'raw-img')
raw_map = {}
if os.path.isdir(raw_dir):
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            raw_map.setdefault(fn, []).append(os.path.join(root, fn))
else:
    print(f"Warning: raw-img directory not found at {raw_dir}. Keeping processed image paths.")

# Replace the stored path with the raw-img path when a match is found (match by basename).
not_found = 0
ambiguous = 0
for entry in mis_all:
    if raw_map:
        basename = os.path.basename(entry['path'])
        matches = raw_map.get(basename)
        if not matches:
            not_found += 1
            # keep original processed path if no raw match
        elif len(matches) == 1:
            entry['path'] = matches[0]
        else:
            ambiguous += 1
            # multiple raw files with same name: pick first (could be adjusted to disambiguate)
            entry['path'] = matches[0]

if raw_map:
    matched = len(mis_all) - not_found
    print(f"Raw-img matches: {matched}, not found in raw-img: {not_found}, ambiguous matches: {ambiguous}")

# Save CSV (paths will point to raw-img when a match was found)
os.makedirs(script_dir, exist_ok=True)
csv_path = os.path.join(script_dir, 'false_predictions.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path','split','true_class_index','pred_class_index'])
    for r in mis_all:
        writer.writerow([r['path'], r['split'], r['true'], r['pred']])
print(f"Saved mispredictions list to {csv_path}")

# Create folder with thumbnails
out_dir = os.path.join(script_dir, 'false_predictions')
os.makedirs(out_dir, exist_ok=True)

# Function to display pages of mispredictions
def show_mispredictions(mis_list, class_names, page_size=12):
    if not mis_list:
        print('No mispredictions to show.')
        return
    total = len(mis_list)
    pages = (total + page_size - 1) // page_size
    for p in range(pages):
        start = p * page_size
        end = min(start + page_size, total)
        n = end - start
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = np.array(axes).reshape(-1)
        for i, idx in enumerate(range(start, end)):
            entry = mis_list[idx]
            img = Image.open(entry['path']).convert('RGB')
            axes[i].imshow(img)
            axes[i].axis('off')
            true_name = class_names[entry['true']] if entry['true'] < len(class_names) else str(entry['true'])
            pred_name = class_names[entry['pred']] if entry['pred'] < len(class_names) else str(entry['pred'])
            axes[i].set_title(f"True: {true_name}\nPred: {pred_name}\n{os.path.basename(entry['path'])}", fontsize=9)
        # hide remaining axes
        for j in range(n, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        # Save page to file instead of showing to avoid blocking the script
        page_file = os.path.join(out_dir, f'page_{p+1}.png')
        fig.savefig(page_file)
        plt.close(fig)
        print(f"Saved misprediction page: {page_file}")
def save_all_mispredictions_png(mis_list, class_names, out_path, thumb_size=(224,224), cols=6, max_images=None, label_height=30, pad=10):
    """Create a single PNG that contains all mispredicted images in a grid with labels.
    Caps at max_images if provided to avoid extremely large files.
    """
    if not mis_list:
        print('No mispredictions to save as single PNG.')
        return
    total = len(mis_list)
    if max_images is not None:
        total = min(total, max_images)
    cols = min(cols, total)
    rows = (total + cols - 1) // cols

    thumb_w, thumb_h = thumb_size
    label_h = label_height
    canvas_w = cols * (thumb_w + pad) + pad
    canvas_h = rows * (thumb_h + label_h + pad) + pad

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255,255,255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    for idx in range(total):
        entry = mis_list[idx]
        try:
            img = Image.open(entry['path']).convert('RGB')
        except Exception as e:
            print(f"Failed to open {entry['path']}: {e}")
            continue
        img = img.resize((thumb_w, thumb_h))
        row = idx // cols
        col = idx % cols
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h + pad)
        canvas.paste(img, (x, y))
        true_name = class_names[entry['true']] if entry['true'] < len(class_names) else str(entry['true'])
        pred_name = class_names[entry['pred']] if entry['pred'] < len(class_names) else str(entry['pred'])
        label = f"{idx+1}. T:{true_name} P:{pred_name}"
        text_x = x + 2
        text_y = y + thumb_h + 2
        draw.text((text_x, text_y), label, fill=(0,0,0), font=font)

    canvas.save(out_path)
    print(f"Saved combined mispredictions image: {out_path}")

# Load class names from dataset root train folder (if exists)
train_root = os.path.join(data_dir, 'train')
if os.path.isdir(train_root):
    tmp = datasets.ImageFolder(root=train_root)
    class_names = tmp.classes
else:
    class_names = [str(i) for i in range(10)]

# Show mispredictions
show_mispredictions(mis_all, class_names, page_size=12)

# Also create a single combined PNG (capped to avoid extremely large files)
combined_path = os.path.join(out_dir, 'combined_mispredictions.png')
if len(mis_all) == 0:
    print('No mispredictions to save as combined image.')
else:
    # cap to first N images to avoid creating massive mega-images on large datasets
    CAP = 500
    if len(mis_all) > CAP:
        print(f"Large number of mispredictions ({len(mis_all)}). Saving first {CAP} into combined image {combined_path}.")
        save_all_mispredictions_png(mis_all, class_names, combined_path, max_images=CAP)
    else:
        save_all_mispredictions_png(mis_all, class_names, combined_path)

print('Done.')
