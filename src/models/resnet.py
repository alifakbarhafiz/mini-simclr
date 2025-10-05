# src/models/resnet.py
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """
    ResNet backbone that returns pooled feature vectors (batch, feature_dim).
    Uses all children layers except the final fc classifier.
    """
    def __init__(self, backbone="resnet18", pretrained=False):
        super().__init__()
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone {backbone} not supported")

        # children except the final fc layer (global pool included)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # until adaptive avg pool
        self.feature_dim = resnet.fc.in_features

    def forward(self, x):
        x = self.features(x)  # shape (B, C, 1, 1)
        x = x.view(x.size(0), -1)
        return x

def get_resnet(backbone="resnet18", pretrained=False):
    model = ResNetBackbone(backbone=backbone, pretrained=pretrained)
    return model, model.feature_dim
