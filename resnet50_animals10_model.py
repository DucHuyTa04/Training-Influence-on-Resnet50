# Tristan Mirolla 40112168, Melissa Ananian 40112159, Duc Huy Ta (40232735)
# October 16, 2025
# Project: Explainable Classification using GRAD-CAM

import torch
import torchvision
import torch.nn as nn

# Creating a custom class for transfer learning on Animals-10 dataset
class ResNet50_Animals10(nn.Module):
    def __init__(self, num_animal_classes=10, pretrained=True, freeze_backbone=True):
        super(ResNet50_Animals10, self).__init__()
        
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = torchvision.models.resnet50(weights=weights)
        
        # Freeze backbone for initial training phase
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get input features
        fc_in_features = self.model.fc.in_features
        
        # Optimized classifier head for 10-class Animals dataset
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_features, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_animal_classes)
        )

        # Ensuring that the newly added layers are not frozen, since we intend on fine-tuning them
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


