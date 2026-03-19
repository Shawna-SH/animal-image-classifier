import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True):
    """
    Build a ResNet18 model for binary animal classification.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model