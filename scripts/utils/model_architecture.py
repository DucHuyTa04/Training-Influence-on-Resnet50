import torch
import torchvision
import torch.nn as nn
from pathlib import Path

class ResNet50_Animals10(nn.Module):
    def __init__(self, num_animal_classes=10, pretrained=True, freeze_backbone=True):
        super(ResNet50_Animals10, self).__init__()
        
        if pretrained:
            # Load from local weights cache (downloaded by 1_download_weights.py)
            weights_cache = Path(__file__).parent.parent.parent / 'models' / 'pretrained' / 'resnet50_imagenet1k_v1.pth'
            if weights_cache.exists():
                print(f"Loading ResNet50 ImageNet weights from: {weights_cache}")
                self.model = torchvision.models.resnet50(weights=None)
                state_dict = torch.load(str(weights_cache), map_location='cpu')
                self.model.load_state_dict(state_dict)
                print(f"Successfully loaded ImageNet pretrained weights")
            else:
                # Fallback: try to download from torchvision (requires internet)
                try:
                    print("Weights cache not found, attempting to download ImageNet weights...")
                    self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
                    print(f"Downloaded and loaded ImageNet weights")
                except Exception as e:
                    print(f"Could not load pretrained weights: {e}")
                    print("Training with random initialization")
                    self.model = torchvision.models.resnet50(weights=None)
        else:
            self.model = torchvision.models.resnet50(weights=None)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        fc_in_features = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_features, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_animal_classes)
        )

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
