import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# cumulative: "layer3" unfreezes layer3 AND layer4, not layer3 alone
UNFREEZE_CHOICES = ("none", "layer4", "layer3")


class LeafResNet18(nn.Module):
    def __init__(self, num_classes: int, d_o_p: float, unfreeze: str = "none"):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(d_o_p),
            nn.Linear(in_features, num_classes),
        )

        self.unfrozen_backbone_modules = []
        if (unfreeze == "layer4"):
            self.unfrozen_backbone_modules.append(self.backbone.layer4)
        elif (unfreeze == "layer3"):
            self.unfrozen_backbone_modules.append(self.backbone.layer3)
            self.unfrozen_backbone_modules.append(self.backbone.layer4)

        for module in self.unfrozen_backbone_modules:
            for param in module.parameters():
                param.requires_grad = True

        self.trainable_modules = (
            [self.backbone.fc] + self.unfrozen_backbone_modules
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # BN running stats stay frozen everywhere, unfrozen blocks included:
        # small target datasets + batch size 32 make recomputed BN stats
        # noisy, so only weights (not BN stats) train in unfrozen blocks.
        self.backbone.eval()
        for module in self.trainable_modules:
            module.train(mode)
        return self

    def head_parameters(self):
        return self.backbone.fc.parameters()

    def backbone_parameters(self):
        params = []
        for module in self.unfrozen_backbone_modules:
            params.extend(module.parameters())
        return params

    def forward(self, x):
        return self.backbone(x)
