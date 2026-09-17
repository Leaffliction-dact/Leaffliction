import torch.nn as nn
from torch.profiler import record_function
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# cumulative down the list
UNFREEZE_CHOICES = ("none", "layer4", "layer3")


class LeafResNet18(nn.Module):
    def __init__(
            self,
            num_classes: int,
            d_o_p: float,
            unfreeze: str = "none"):
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
        b = self.backbone

        with record_function("[ResNet18] stem"):          # conv1+bn1+relu+pool
            x = b.maxpool(b.relu(b.bn1(b.conv1(x))))

        with record_function("[ResNet18] layer1"):
            x = b.layer1(x)

        with record_function("[ResNet18] layer2"):
            x = b.layer2(x)

        with record_function("[ResNet18] layer3"):
            x = b.layer3(x)

        with record_function("[ResNet18] layer4"):
            x = b.layer4(x)

        with record_function("[ResNet18] head"):          # avgpool + fc
            x = b.avgpool(x)
            x = x.flatten(1)
            return b.fc(x)
