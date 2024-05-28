
from typing import Tuple
from torchvision.models import resnet50
from torch.nn import Identity
import torchvision.transforms as T

from lightly.utils.benchmarking import LinearClassifier


class ResNet50Classifier(LinearClassifier):
    def __init__(
        self,
        batch_size_per_device: int,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
        freeze_model: bool = False,
    ) -> None:
        model = resnet50()
        model.fc = Identity()
        super().__init__(
            model=model,
            batch_size_per_device=batch_size_per_device,
            feature_dim=feature_dim,
            num_classes=num_classes,
            topk=topk,
            freeze_model=freeze_model,
        )


transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=45),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])