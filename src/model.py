import torch
import torch.nn as nn
from torchvision import models

from . import config


class DefectClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def build_model(task: str, pretrained: bool = True) -> DefectClassifier:
    if task == "codebrim":
        return DefectClassifier(num_classes=len(config.CODEBRIM_CLASSES), pretrained=pretrained)
    if task == "sdnet":
        return DefectClassifier(num_classes=1, pretrained=pretrained)
    raise ValueError(f"Unknown task: {task}")
