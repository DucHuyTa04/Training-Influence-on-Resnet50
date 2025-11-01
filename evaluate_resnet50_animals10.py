# Evaluate ResNet50 on Animals-10 test set and plot confusion matrix
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from resnet50_animals10_model import ResNet50_Animals10

# Set the model path here 
MODEL_PATH = "models\Resnet50_animals10_val_0_9796_0_5963.pth"

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

# Evaluate the model and compute metrics
def evaluate_model(model, dataloader, criterion, class_names):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    per_class_correct = np.zeros(len(class_names), dtype=int)
    per_class_total = np.zeros(len(class_names), dtype=int)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy stats
            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                per_class_total[label] += 1
                if label == pred:
                    per_class_correct[label] += 1

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.float() / len(dataloader.dataset)
    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
    return total_acc.item(), total_loss, per_class_acc, np.array(all_labels), np.array(all_preds)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Evaluating ResNet-50 on Animals-10 Test Set")
    print("="*60)

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = ResNet50_Animals10(num_animal_classes=10, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate
    acc, loss, per_class_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion, class_names)

    # Print overall metrics
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")

    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i, cname in enumerate(class_names):
        print(f"{cname:12s}: {per_class_acc[i]*100:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title('Confusion Matrix - Animals-10 Test Set')
    plt.tight_layout()
    plt.show()
