import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LeafResNet18(nn.Module):
    def __init__(self, num_classes: int, d_o_p: float):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(d_o_p),
            nn.Linear(in_features, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.backbone.fc.train(mode)
        return self

    def forward(self, x):
        return self.backbone(x)
