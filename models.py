from leafcnn import LeafCNN
from resnetcnn import LeafResNet18

ARCH_CHOICES = ("leafcnn", "resnet18")


def build_model(
        arch: str,
        num_classes: int,
        dropout: float,
        unfreeze: str = "none"):
    if (arch == "resnet18"):
        return LeafResNet18(
            num_classes=num_classes,
            d_o_p=dropout,
            unfreeze=unfreeze,
        )
    else:
        return LeafCNN(num_classes=num_classes, d_o_p=dropout)
