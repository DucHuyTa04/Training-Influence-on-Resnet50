# Perform gradCAM/torchCAM evaluation on ResNet-50 model trained on Animals-10 dataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from resnet50_animals10_model import ResNet50_Animals10

from torchcam.methods import GradCAM
import cv2
import random

import argparse

parser = argparse.ArgumentParser(description="Grad-CAM script")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model in the 'models' directory (.pth file)")
args = parser.parse_args()
# Example usage: python gradCAM_evaluation.py --model_path models/resnet50_animals10_0_9721_0_1134.pth

MODEL_PATH = args.model_path

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Data transforms (must match training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data', 'processed')
test_dir = os.path.join(data_dir, 'test')

# Load test set
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
class_names = test_dataset.classes

# Create directory to save the grad-CAM images
plot_dir = os.path.join(script_dir, 'gradcam_figures')
os.makedirs(plot_dir, exist_ok=True)  # This line is safe even if folder exists

# Generate and save Grad-CAM heatmaps for a single image
def generate_heatmap(image, label, cam_extractor, class_names, img_idx):
    # Ensure the model is in eval mode
    model.eval()

    # Add batch dimension to image (expected shape of the model: [1, 3, 224, 224])
    input_tensor = image.unsqueeze(0).to(device)

    # Enable gradient computation temporarily for class activation map (CAM) extraction
    with torch.set_grad_enabled(True):
        # Forward pass
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        # Computes activation map for the predicted class
        # Identifies the logit/score corresponding to the predicted class
        # Calls backward on that score to compute gradients
        # Computes the Grad-CAM weights and combines them with the activations to produce the CAM -- returning the raw activation map
        activation_map = cam_extractor(pred_class, output)

    torch.set_grad_enabled(False) # Disable gradients again -- Just to ensure gradients are disabled, even though they should be outside of the with block

    # Extract the CAM from the first element and convert to numpy for transformations (the raw activation map (size: 1xHxW))
    heatmap = activation_map[0].squeeze().cpu().numpy() # shape: HxW

    # Convert original image back to [0, 1] RGB for visualization 
    inverse_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    img_np = inverse_normalize(image).permute(1, 2, 0).cpu().numpy() # Permute channel order to HxWxC for plotting
    img_np = np.clip(img_np, 0, 1) # Ensure values are within [0, 1] (inverse normalization is an approximation)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    # Normalize heatmap to [0, 1] and apply colormap
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8) # min-max normalization
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET) # OpenCV (and in turn COLORMAP_JET) uses BGR format
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for plotting

    # Overlay heatmap on original image
    overlaid_img = cv2.addWeighted(np.uint8(img_np * 255), 0.6, heatmap_colored, 0.4, 0) # 0.6 and 0.4 are weights for the original image and heatmap respectively -- 60% original image, 40% heatmap

    # Plot orignal image and heatmap overlaid image side-by-side
    # Includes true label over original and predicted label over heatmap
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"Original\n(True: {class_names[label]})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlaid_img)
    plt.title(f"GradCAM\n(Pred: {class_names[pred_class]})")
    plt.axis('off')

    plt.tight_layout()

    # Save figure into directory
    gradcam_img_filename = os.path.join(plot_dir, f'gradcam_image_{label}_{img_idx}.png')
    plt.savefig(gradcam_img_filename)

    plt.show()

# Randomly selects a number of images from the test dataset and generates Grad-CAM heatmaps for those images
# Currently set to 5 images (in the __main__ section)
def evaluate_gradcam(model, dataloader, cam_extractor, class_names, num_images):
    # Ensure the model is in eval mode
    model.eval()

    # Randomly select a number of images from the test dataset (currently set to 5)
    ran_selected_indices = random.sample(range(len(dataloader.dataset)), num_images)

    # Iterate through the randomly selected indices
    for idx in ran_selected_indices:
        # Get the image and label for the selected index
        img, label = dataloader.dataset[idx]

        # Generate and display the heatmap
        generate_heatmap(img, label, cam_extractor, class_names, idx)

if __name__ == '__main__':
    num_images = 5  # Number of random test images to generate Grad-CAM heatmaps for
    print("\n" + "="*61)
    print(f"Applying Grad-CAM to {num_images} random images from Animals-10 Test Set")
    print("="*61)

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = ResNet50_Animals10(num_animal_classes=10, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()  # Set to eval mode

    # Initialize Grad-CAM extractor
    target_layer = model.model.layer4[-1] # last conv layer of ResNet50
    cam_extractor = GradCAM(model=model, target_layer=target_layer) # Computes raw activation maps and registers forward and backward hooks to capture activations and gradients respectively

    # Evaluate and generate heatmaps for 5 random test images
    evaluate_gradcam(model, test_loader, cam_extractor, class_names, num_images=num_images)
