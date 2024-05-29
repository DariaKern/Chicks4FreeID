import argparse
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import Tensor
from torch.nn import (
    CrossEntropyLoss,
    Identity,
    Linear,
    Module,
    ModuleList,
)
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchmetrics.classification import MulticlassAveragePrecision

from datasets import Dataset, load_dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from lightly.data import LightlyDataset
from lightly.loss.swav_loss import SwaVLoss
from lightly.models.modules import (
    SwaVProjectionHead,
    SwaVPrototypes,
)
from lightly.models.modules.memory_bank import MemoryBankModule
from lightly.models.utils import get_weight_decay_parameters
from lightly.transforms import SwaVTransform
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.benchmarking import (
    MetricCallback,
    OnlineLinearClassifier,
)
from lightly.utils.dist import print_rank_zero
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

import timm


@dataclass
class Config:
    batch_size_per_device: int = 128
    epochs: int = 500
    num_workers: int = 4
    log_dir: Path = Path("benchmark_logs")
    checkpoint_path: Optional[Path] = None
    num_classes: int = 50
    skip_embedding_training: bool = False
    skip_knn_eval: bool = False
    skip_linear_eval: bool = False
    methods: Optional[List[str]] = None
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    test_run: bool = False
    check_val_every_n_epoch: int = 5


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration_seconds = end_time - start_time
        duration_timedelta = timedelta(seconds=duration_seconds)
        print(f"Duration: {duration_timedelta}")
        return result
    return wrapper


def knn_predict(
    feature: Tensor,
    feature_bank: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
) -> Tensor:
    """
    [Modified version from lightly, which also returns the scores.]

    Run kNN predictions on features based on a feature bank

    This method is commonly used to monitor performance of self-supervised
    learning methods.

    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    # code for kNN prediction from here:
    # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb

    Args:
        feature:
            Tensor with shape (B, D) for which you want predictions.
        feature_bank:
            Tensor of shape (D, N) of a database of features used for kNN.
        feature_labels:
            Labels with shape (N,) for the features in the feature_bank.
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10).
        knn_k:
            Number of k neighbors used for kNN.
        knn_t:
            Temperature parameter to reweights similarities for kNN.

    Returns:
        A tensor containing the kNN predictions

    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     targets_bank,
        >>>     num_classes=10,
        >>> )
    """
    # compute cos similarity between each feature vector and feature bank ---> (B, N)
    sim_matrix = torch.mm(feature, feature_bank)
    # (B, K)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # (B, K)
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # (B*K, C)
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> (B, C)
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels, pred_scores


class KNNClassifier(LightningModule):
    """
    A lightly KNN Classifier modified to log mean average precision metric.
    """
    def __init__(
        self,
        model: Module,
        num_classes: int,
        knn_k: int = 200,
        knn_t: float = 0.1,
        topk: Tuple[int, ...] = (1, 5),
        feature_dtype: torch.dtype = torch.float32,
        normalize: bool = True,
    ):
        """KNN classifier for benchmarking.

        Settings based on InstDisc [0]. Code adapted from MoCo [1].

        - [0]: InstDisc, 2018, https://arxiv.org/pdf/1805.01978v1.pdf
        - [1]: MoCo, 2019, https://github.com/facebookresearch/moco

        Args:
            model:
                Model used for feature extraction. Must define a forward(images) method
                that returns a feature tensor.
            num_classes:
                Number of classes in the dataset.
            knn_k:
                Number of neighbors used for KNN search.
            knn_t:
                Temperature parameter to reweights similarities.
            topk:
                Tuple of integers defining the top-k accuracy metrics to compute.
            feature_dtype:
                Torch data type of the features used for KNN search. Reduce to float16
                for memory-efficient KNN search.
            normalize:
                Whether to normalize the features for KNN search.

        Examples:
            >>> from pytorch_lightning import Trainer
            >>> from torch import nn
            >>> import torchvision
            >>> from lightly.models import LinearClassifier
            >>> from lightly.modles.modules import SimCLRProjectionHead
            >>>
            >>> class SimCLR(nn.Module):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.backbone = torchvision.models.resnet18()
            >>>         self.backbone.fc = nn.Identity() # Ignore classification layer
            >>>         self.projection_head = SimCLRProjectionHead(512, 512, 128)
            >>>
            >>>     def forward(self, x):
            >>>         # Forward must return image features.
            >>>         features = self.backbone(x).flatten(start_dim=1)
            >>>         return features
            >>>
            >>> # Initialize a model.
            >>> model = SimCLR()
            >>>
            >>>
            >>> # Wrap it with a KNNClassifier.
            >>> knn_classifier = KNNClassifier(resnet, num_classes=10)
            >>>
            >>> # Extract features and evaluate.
            >>> trainer = Trainer(max_epochs=1)
            >>> trainer.fit(knn_classifier, train_dataloder, val_dataloader)

        """
        super().__init__()
        self.save_hyperparameters(
            {
                "num_classes": num_classes,
                "knn_k": knn_k,
                "knn_t": knn_t,
                "topk": topk,
                "feature_dtype": str(feature_dtype),
            }
        )
        self.model = model
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.topk = topk
        self.feature_dtype = feature_dtype
        self.normalize = normalize

        self._train_features = []
        self._train_targets = []
        self._train_features_tensor: Optional[Tensor] = None
        self._train_targets_tensor: Optional[Tensor] = None

        # Initialize metric for mean average precision.
        self.map_metric = MulticlassAveragePrecision(num_classes=num_classes)

    @torch.no_grad()
    def training_step(self, batch, batch_idx) -> None:
        images, targets = batch[0], batch[1]
        features = self.model.forward(images).flatten(start_dim=1)
        if self.normalize:
            features = F.normalize(features, dim=1)
        features = features.to(self.feature_dtype)
        self._train_features.append(features.cpu())
        self._train_targets.append(targets.cpu())

    def validation_step(self, batch, batch_idx) -> None:
        if self._train_features_tensor is None or self._train_targets_tensor is None:
            return

        images, targets = batch[0], batch[1]
        features = self.model.forward(images).flatten(start_dim=1)
        if self.normalize:
            features = F.normalize(features, dim=1)
        features = features.to(self.feature_dtype)
        predicted_classes, pred_scores = knn_predict(
            feature=features,
            feature_bank=self._train_features_tensor,
            feature_labels=self._train_targets_tensor,
            num_classes=self.num_classes,
            knn_k=self.knn_k,
            knn_t=self.knn_t,
        )
        topk = mean_topk_accuracy(
            predicted_classes=predicted_classes, targets=targets, k=self.topk
        )
        self.map_metric(pred_scores, targets)
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        log_dict["val_mAP"] = self.map_metric.compute()
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=len(targets))

    def on_validation_epoch_start(self) -> None:
        if self._train_features and self._train_targets:
            # Features and targets have size (world_size, batch_size, dim) and
            # (world_size, batch_size) after gather. For non-distributed training,
            # features and targets have size (batch_size, dim) and (batch_size,).
            features = self.all_gather(torch.cat(self._train_features, dim=0))
            self._train_features = []
            targets = self.all_gather(torch.cat(self._train_targets, dim=0))
            self._train_targets = []
            # Reshape to (dim, world_size * batch_size)
            features = features.flatten(end_dim=-2).t().contiguous()
            self._train_features_tensor = features.to(self.device)
            # Reshape to (world_size * batch_size,)
            targets = targets.flatten().t().contiguous()
            self._train_targets_tensor = targets.to(self.device)

    def on_train_epoch_start(self) -> None:
        # Set model to eval mode to disable norm layer updates.
        self.model.eval()

        # Reset features and targets.
        self._train_features = []
        self._train_targets = []
        self._train_features_tensor = None
        self._train_targets_tensor = None

    def configure_optimizers(self) -> None:
        # configure_optimizers must be implemented for PyTorch Lightning. Returning None
        # means that no optimization is performed.
        pass


class LinearClassifier(LightningModule):
    """
    A lightly Linear Classifier, modified to log the mean average precision
    """
    def __init__(
        self,
        model: Module,
        batch_size_per_device: int,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
        freeze_model: bool = False,
    ) -> None:
        """Linear classifier for benchmarking.

        Settings based on SimCLR [0].

        - [0]: https://arxiv.org/abs/2002.05709

        Args:
            model:
                Model used for feature extraction. Must define a forward(images) method
                that returns a feature tensor.
            batch_size_per_device:
                Batch size per device.
            feature_dim:
                Dimension of features returned by forward method of model.
            num_classes:
                Number of classes in the dataset.
            topk:
                Tuple of integers defining the top-k accuracy metrics to compute.
            freeze_model:
                If True, the model is frozen and only the classification head is
                trained. This corresponds to the linear eval setting. Set to False for
                finetuning.

        Examples:

            >>> from pytorch_lightning import Trainer
            >>> from torch import nn
            >>> import torchvision
            >>> from lightly.models import LinearClassifier
            >>> from lightly.modles.modules import SimCLRProjectionHead
            >>>
            >>> class SimCLR(nn.Module):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.backbone = torchvision.models.resnet18()
            >>>         self.backbone.fc = nn.Identity() # Ignore classification layer
            >>>         self.projection_head = SimCLRProjectionHead(512, 512, 128)
            >>>
            >>>     def forward(self, x):
            >>>         # Forward must return image features.
            >>>         features = self.backbone(x).flatten(start_dim=1)
            >>>         return features
            >>>
            >>> # Initialize a model.
            >>> model = SimCLR()
            >>>
            >>> # Wrap it with a LinearClassifier.
            >>> linear_classifier = LinearClassifier(
            >>>     model,
            >>>     batch_size=256,
            >>>     num_classes=10,
            >>>     freeze_model=True, # linear evaluation, set to False for finetune
            >>> )
            >>>
            >>> # Train the linear classifier.
            >>> trainer = Trainer(max_epochs=90)
            >>> trainer.fit(linear_classifier, train_dataloader, val_dataloader)

        """
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.batch_size_per_device = batch_size_per_device
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.topk = topk
        self.freeze_model = freeze_model

        self.classification_head = Linear(feature_dim, num_classes)
        self.criterion = CrossEntropyLoss()

        # Initialize metric for mean average precision.
        self.map_metric = MulticlassAveragePrecision(num_classes=num_classes)

    def forward(self, images: Tensor) -> Tensor:
        if self.freeze_model:
            with torch.no_grad():
                features = self.model.forward(images).flatten(start_dim=1)
        else:
            features = self.model.forward(images).flatten(start_dim=1)
        output: Tensor = self.classification_head(features)
        return output

    def shared_step(
        self, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
        self.map_metric.update(predictions, targets)
        mAP = self.map_metric.compute()
        return loss, topk, mAP

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, topk, mAP = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        log_dict["train_mAP"] = mAP
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, topk, mAP = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        log_dict["val_mAP"] = mAP
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.classification_head.parameters())
        if not self.freeze_model:
            parameters += self.model.parameters()
        optimizer = SGD(
            parameters,
            lr=0.1 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def on_train_epoch_start(self) -> None:
        if self.freeze_model:
            # Set model to eval mode to disable norm layer updates.
            self.model.eval()


class OnlineLinearClassifier(LightningModule):
    """
    A lightly Online Linear Classifier, modified to log the mean average precision
    """
    def __init__(
        self,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.topk = topk

        self.classification_head = Linear(feature_dim, num_classes, dtype=dtype)
        self.criterion = CrossEntropyLoss()

        # Initialize metric for mean average precision.
        self.map_metric = MulticlassAveragePrecision(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classification_head(x.detach().flatten(start_dim=1))

    def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        features, targets = batch[0], batch[1]
        predictions = self.forward(features)
        loss = self.criterion(predictions, targets)
        _, predicted_classes = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_classes, targets, k=self.topk)
        self.map_metric.update(predictions, targets)
        mAP = self.map_metric.compute()
        return loss, topk, mAP

    def training_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk, mAP = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"train_online_cls_loss": loss}
        log_dict.update({f"train_online_cls_top{k}": acc for k, acc in topk.items()})
        log_dict["train_online_cls_mAP"] = mAP
        return loss, log_dict

    def validation_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk, mAP = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"val_online_cls_loss": loss}
        log_dict.update({f"val_online_cls_top{k}": acc for k, acc in topk.items()})
        log_dict["val_online_cls_mAP"] = mAP
        return loss, log_dict


class SwAV(LightningModule):
    """
    A lightly SwAV model, modified to log the mean average precision
    """
    CROP_COUNTS: Tuple[int, int] = (2, 6)

    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = SwaVProjectionHead()
        self.prototypes = SwaVPrototypes(n_steps_frozen_prototypes=1)
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True)
        self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)

        # Use a queue for small batch sizes (<= 256).
        self.start_queue_at_epoch = 15
        self.n_batches_in_queue = 15
        self.queues = ModuleList(
            [
                MemoryBankModule(
                    size=(self.n_batches_in_queue * self.batch_size_per_device, 128)
                )
                for _ in range(SwAV.CROP_COUNTS[0])
            ]
        )

        # Initialize metric for mean average precision.
        self.map_metric = MulticlassAveragePrecision(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def project(self, x: Tensor) -> Tensor:
        x = self.projection_head(x)
        return F.normalize(x, dim=1, p=2)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        # Normalize the prototypes so they are on the unit sphere.
        self.prototypes.normalize()

        # The dataloader returns a list of image crops where the
        # first few items are high resolution crops and the rest are low
        # resolution crops.
        multi_crops, targets = batch[0], batch[1]

        # Forward pass through backbone and projection head.
        multi_crop_features = [
            self.forward(crops).flatten(start_dim=1) for crops in multi_crops
        ]
        multi_crop_projections = [
            self.project(features) for features in multi_crop_features
        ]

        # Get the queue projections and logits.
        queue_crop_logits = None
        with torch.no_grad():
            if self.current_epoch >= self.start_queue_at_epoch:
                # Start filling the queue.
                queue_crop_projections = _update_queue(
                    projections=multi_crop_projections[: SwAV.CROP_COUNTS[0]],
                    queues=self.queues,
                )
                if batch_idx > self.n_batches_in_queue:
                    # The queue is filled, so we can start using it.
                    queue_crop_logits = [
                        self.prototypes(projections, step=self.current_epoch)
                        for projections in queue_crop_projections
                    ]

        # Get the rest of the multi-crop logits.
        multi_crop_logits = [
            self.prototypes(projections, step=self.current_epoch)
            for projections in multi_crop_projections
        ]

        # Calculate the SwAV loss.
        loss = self.criterion(
            high_resolution_outputs=multi_crop_logits[: SwAV.CROP_COUNTS[0]],
            low_resolution_outputs=multi_crop_logits[SwAV.CROP_COUNTS[0] :],
            queue_outputs=queue_crop_logits,
        )
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Calculate the classification loss.
        cls_loss, cls_log = self.online_classifier.training_step(
            (multi_crop_features[0].detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head, self.prototypes]
        )
        optimizer = LARS(
            [
                {"name": "swav", "params": params},
                {
                    "name": "swav_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            # Smaller learning rate for smaller batches: lr=0.6 for batch_size=256
            # scaled linearly by batch size to lr=4.8 for batch_size=2048.
            # See Appendix A.1. and A.6. in SwAV paper https://arxiv.org/pdf/2006.09882.pdf
            lr=0.6 * (self.batch_size_per_device * self.trainer.world_size) / 256,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
                end_value=0.0006
                * (self.batch_size_per_device * self.trainer.world_size)
                / 256,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

@torch.no_grad()
def _update_queue(
    projections: List[Tensor],
    queues: ModuleList,
):
    """
    [Lightly function from swav]
    
    Adds the high resolution projections to the queues and returns the queues."""

    if len(projections) != len(queues):
        raise ValueError(
            f"The number of queues ({len(queues)}) should be equal to the number of high "
            f"resolution inputs ({len(projections)})."
        )

    # Get the queue projections
    queue_projections = []
    for i in range(len(queues)):
        _, queue_proj = queues[i](projections[i], update=True)
        # Queue projections are in (num_ftrs X queue_length) shape, while the high res
        # projections are in (batch_size_per_device X num_ftrs). Swap the axes for interoperability.
        queue_proj = torch.permute(queue_proj, (1, 0))
        queue_projections.append(queue_proj)

    return queue_projections



class ResNet50Classifier(LinearClassifier):
    """
    A ResNet50 classifier for benchmarking the fully supervised setting.
    """
    def __init__(
        self,
        batch_size_per_device: int,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
        freeze_model: bool = False,
    ) -> None:
        """ResNet50 classifier for benchmarking fully supervised setting. Inherits from LinearClassifier 
        """
        super().__init__(
            model=None,
            feature_dim=feature_dim,
            num_classes=num_classes,
            batch_size_per_device=batch_size_per_device,
            topk=topk,
            freeze_model=freeze_model,
        )
        
        self.model = resnet50(num_classes=num_classes,)
        fc = self.model.fc 
        self.model.fc = Identity()
        self.classification_head = fc


class ResNetEmbedding(LightningModule):
    """
    Converts a ResNet50Classifier into a feature extractor.
    """
    def __init__(
        self,
        model: ResNet50Classifier,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model.model

    def forward(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.model.forward(images).flatten(start_dim=1)
        return features

    def configure_optimizers(self) -> None:
        # configure_optimizers must be implemented for PyTorch Lightning. Returning None
        # means that no optimization is performed.
        pass


class MegaDescriptorL384(LightningModule):
    """
    A pretrained-model that uses the MegaDescriptor-L-384 model from the HuggingFace model hub
    to benchmark the unsupervised setting.

    The model is not finetuned and only used to extract features.

    The model has been trained on external Animal Re-ID Data
    """
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Initialize your model and transforms here
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        model.eval()  # Set the model to evaluation mode
        self.backbone = model

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.backbone(x)
        

    def configure_optimizers(self):
        # configure_optimizers must be implemented for PyTorch Lightning. Returning None
        # means that no optimization is performed.
        pass


class ChicksVisionDataset(torchvision.datasets.VisionDataset):
    """
    Provides the Chicks4FreeID HuggingFace dataset as a Torchvision dataset.
    The dataset will return a tuple (PIL.Image, target:int)
    """
    from PIL import Image
    HF_DATASET_DICT: Dict[str, Dataset] = {}
    TRAIN_DATASET_CACHE: List[Tuple[Image.Image, int]] = []
    TEST_DATASET_CACHE: List[Tuple[Image.Image, int]] = []

    def __init__(
        self,
        root: Union[str, Path] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        resize: int = 384,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.resize = resize
        self.transform = transform
        self.target_transform = target_transform

        if not ChicksVisionDataset.HF_DATASET_DICT:
            ChicksVisionDataset.HF_DATASET_DICT = load_dataset(
                "dariakern/Chicks4FreeID", 
                "chicken-re-id-best-visibility", 
            )
        if not ChicksVisionDataset.TRAIN_DATASET_CACHE:
            print_rank_zero("Caching train images in memory...")
            ChicksVisionDataset.TRAIN_DATASET_CACHE = [
                self._load_row(data) for data in tqdm(ChicksVisionDataset.HF_DATASET_DICT["train"])
            ]  
        if not ChicksVisionDataset.TEST_DATASET_CACHE:
            print_rank_zero("Caching test images in memory...")
            ChicksVisionDataset.TEST_DATASET_CACHE = [
                self._load_row(data) for data in tqdm(ChicksVisionDataset.HF_DATASET_DICT["test"])
            ]

        self.cached_data = ChicksVisionDataset.TRAIN_DATASET_CACHE if train else ChicksVisionDataset.TEST_DATASET_CACHE
        
    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        img, target = self.cached_data[idx]
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def _load_row(self, row: Dict[str, Any]) -> Tuple[Image.Image, int]:
        img, target = row.values()
        img = img.resize((self.resize, self.resize))
        return img, target


    def __str__(self):
        return self


class BenchmarkMethod():
    """
    An abstract class that defines common methods for a benchmarking methods.

    The class runs a benchmarking pipeline that consists of:
        - embedding training
        - kNN evaluation
        - linear evaluation

    Reported metrics are:
        - Top-1 accuracy
        - Top-5 accuracy
        - Mean Average Precision (mAP)

    The benchmark can be configured by inheriting from this class and overriding specifics
    as well as passing a Config object.
    """
    embedding_train_dataset: Iterable[Tuple[Tensor, Tensor]]
    embedding_val_dataset: Iterable[Tuple[Tensor, Tensor]]
    linear_train_dataset: Iterable[Tuple[Tensor, Tensor]]
    linear_val_dataset: Iterable[Tuple[Tensor, Tensor]]
    knn_val_dataset: Iterable[Tuple[Tensor, Tensor]]
    knn_train_dataset: Iterable[Tuple[Tensor, Tensor]]
    
    normalize_transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    resize_transform = T.Resize(384)
    method_specific_augmentation = T.Compose([])
    
    name: str = ""
    cfg: Config
    model: Module
    feature_dim: int = 2048

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.name = self.name or self.__class__.__name__

        self.method_dir = self.cfg.log_dir / self.name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.method_dir = self.method_dir.resolve()

        # Transform for pretaining of the embedding
        self.embedding_train_transform = T.Compose([
            # self.resize_transform,
            T.RandomRotation(360),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            self.method_specific_augmentation
        ])

        # Transform for linear eval training and kkn training
        self.eval_train_transform = T.Compose([
            # self.resize_transform,
            T.RandomRotation(360),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            self.normalize_transform,
        ])

        # Transform for all validation datasets
        self.val_transform = T.Compose([
            # self.resize_transform,
            T.ToTensor(),
            self.normalize_transform
        ])


        self.embedding_train_dataset = ChicksVisionDataset(
            train=True,
            transform=self.embedding_train_transform,
        )

        self.knn_train_dataset = self.linear_train_dataset = ChicksVisionDataset(
            train=True,
            transform=self.eval_train_transform,
        )
   
        self.knn_val_dataset = self.linear_val_dataset  = self.embedding_val_dataset = ChicksVisionDataset(
            train=False,
            transform=self.val_transform,
        )

    @timing_decorator
    def run_benchmark(self):
        print_rank_zero(f"## Starting {self.name}... ")
        if self.cfg.checkpoint_path:
            self.model.load_state_dict(torch.load(self.cfg.checkpoint_path)["state_dict"])

        if self.cfg.skip_embedding_training or self.cfg.epochs == 0:
            print_rank_zero("Skipping training")
        else:
            self.embedding_training()
        
        if self.cfg.skip_knn_eval:
            print_rank_zero("Skipping KNN evaluation")
        else:
            self.knn_eval()

        if self.cfg.skip_linear_eval:
            print_rank_zero("Skipping linear evaluation")
        else:
            self.linear_eval()
        print_rank_zero(f"## Finished {self.name}")


    def get_embedding_model(self) -> Module:
        "Must return a model that returns features on forward pass."
        return self.model

    def knn_eval(self,) -> None:
        """Runs KNN evaluation on the given model.

        Parameters follow InstDisc [0] settings.

        The most important settings are:
            - Num nearest neighbors: 200
            - Temperature: 0.1

        References:
        - [0]: InstDict, 2018, https://arxiv.org/abs/1805.01978
        """
        print_rank_zero(f"### Running {self.name} KNN evaluation...")
        

        self.train(
            classifier = KNNClassifier(
                model=self.get_embedding_model(),
                num_classes=self.cfg.num_classes,
                feature_dtype=torch.float16,
            ),
            epochs = 1,
            train_dataset = self.knn_train_dataset,
            val_dataset = self.knn_val_dataset,
            log_name="knn_eval",
        )

    def linear_eval(self,) -> None:
        """Runs a linear evaluation on the given model.

        Parameters follow SimCLR [0] settings.

        The most important settings are:
            - Backbone: Frozen
            - Epochs: 90
            - Optimizer: SGD
            - Base Learning Rate: 0.1
            - Momentum: 0.9
            - Weight Decay: 0.0
            - LR Schedule: Cosine without warmup

        References:
            - [0]: SimCLR, 2020, https://arxiv.org/abs/2002.05709
        """
        print_rank_zero(f"### Running {self.name} linear evaluation... ")


        self.train(
            classifier = LinearClassifier(
                model=self.get_embedding_model(),
                batch_size_per_device=self.cfg.batch_size_per_device,
                feature_dim=self.feature_dim,
                num_classes=self.cfg.num_classes,
                freeze_model=True,
            ),
            epochs = 90,
            train_dataset = self.linear_train_dataset,
            val_dataset = self.linear_val_dataset,
            log_name="linear_eval",
        )
    
    def embedding_training(self):
        print_rank_zero(f"### Training {self.name} embedding model... ")
        
        self.train(
            classifier = self.model,
            epochs = self.cfg.epochs,
            train_dataset = self.embedding_train_dataset,
            val_dataset = self.embedding_val_dataset,
            log_name="embedding_training",
        )


    @timing_decorator
    def train(self, classifier, epochs, train_dataset, val_dataset, log_name):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size_per_device,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            persistent_workers=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size_per_device,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )

        metric_callback = MetricCallback()
        trainer = Trainer(
            max_epochs=epochs if not self.cfg.test_run else 1,
            accelerator=self.cfg.accelerator,
            devices=self.cfg.devices,
            logger=TensorBoardLogger(save_dir=str(self.method_dir), name=log_name),
            callbacks=[
                DeviceStatsMonitor(),
                metric_callback,
            ],
            num_sanity_val_steps=0,
            log_every_n_steps=0,
            precision=self.cfg.precision,
            check_val_every_n_epoch=self.cfg.check_val_every_n_epoch if not self.cfg.test_run else 1,
            strategy="auto"
        )

        trainer.fit(
            model=classifier,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        for metric in metric_callback.val_metrics.keys():
            print_rank_zero(f"{self.name} {log_name} {metric}: {max(metric_callback.val_metrics[metric])}")
        

class ResNet50Benchmark(BenchmarkMethod):
    method_specific_augmentation = T.Compose([
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def __init__(self, args):
        super().__init__(args)

        self.model = ResNet50Classifier(
            batch_size_per_device=self.cfg.batch_size_per_device,
            num_classes=self.cfg.num_classes,
            topk=(1, 5),
            freeze_model=False,
        )


    def get_embedding_model(self):
        return ResNetEmbedding(model=self.model) 

    
class MegaDescriptorL384Benchmark(BenchmarkMethod):
    normalize_transform = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __init__(self, args):
        super().__init__(args)

        self.model = MegaDescriptorL384()
        self.cfg.skip_embedding_training = True
        self.feature_dim = 1536


class SwAVBenchmark(BenchmarkMethod):
    method_specific_augmentation = SwaVTransform(
        rr_prob=0.5,
        rr_degrees=360,
        gaussian_blur=0.1,
        crop_counts=SwAV.CROP_COUNTS
    )

    def __init__(self, args):
        super().__init__(args)

        self.model = SwAV(
            batch_size_per_device=self.cfg.batch_size_per_device,
            num_classes=self.cfg.num_classes,
        )

        self.embedding_train_dataset = LightlyDataset.from_torch_dataset(
            ChicksVisionDataset(train=True),
            transform=self.embedding_train_transform,
        ) 

    
class Benchmark:
    methods: Dict[str, Type[BenchmarkMethod]] = {
        "resnet50": ResNet50Benchmark,
        "mega_descriptor": MegaDescriptorL384Benchmark,
        "swav": SwAVBenchmark,
    }

    @timing_decorator
    def run(self, args):
        cfg = Config(**vars(args))
        methods = cfg.methods or self.methods.keys()
        print_rank_zero(f"# Running Benchmarks: {list(methods)}...")   
        for method in methods:
            if method not in self.methods:
                raise ValueError(f"Unknown method: {method}. Available methods: {list(self.methods.keys())}")
            else:
                self.methods[method](cfg).run_benchmark()
        print_rank_zero(f"# All Benchmarks Done!")   
            

parser = argparse.ArgumentParser(description='Benchmark suite for the paper Chicks4FreeID')
parser.add_argument("--log-dir", type=Path, default=str(Config.log_dir))
parser.add_argument("--batch-size-per-device", type=int, default=Config.batch_size_per_device) #default=32) #default=128)
parser.add_argument("--epochs", type=int, default=Config.epochs)
parser.add_argument("--num-workers", type=int, default=Config.num_workers)
parser.add_argument("--checkpoint-path", type=Path, default=Config.checkpoint_path)
parser.add_argument("--methods", type=str, nargs="+", default=Config.methods)
parser.add_argument("--num-classes", type=int, default=Config.num_classes)
parser.add_argument("--skip-embedding-training", action="store_true", default=Config.skip_embedding_training)
parser.add_argument("--skip-knn-eval", action="store_true", default=Config.skip_knn_eval)
parser.add_argument("--skip-linear-eval", action="store_true", default=Config.skip_linear_eval)
parser.add_argument("--test-run", action="store_true", default=Config.test_run)


if __name__ == "__main__":
    args = parser.parse_args()
    Benchmark().run(args)

