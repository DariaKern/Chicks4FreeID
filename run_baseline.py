###
# Author: Tobias
# Description: This file contains the code for the baseline models used in the experiments
# of the paper "Chicks4FreeID"
# The code is based on the lightly benchmarks, but heavily modified
# Notable changes:
# - The code is refactored to be used in a single file
# - The mean average precision metric is added
# - Support for unsupervised frozen feature extractor methods like MegaDescriptorL384 is added
# - Support for fully supervised methods like ResNet50Classifier or ViT is added
# - All methods use the Chicks4FreeID dataset with the same train/val split and input size
# - All evaluation augmentations are the same
# - Added a Config class to manage hyperparameters
# - The code is refactored to use inheritance and composition where possible
# - The code is refactored to use the PyTorch Lightning implemenations for mAP and top-k accuracy
# - The end result is a markdown table to compare to the results of the paper
# - Code that is mostly taken from lightly is marked with a comment
# Today's Date: 2024-MAY-31

import argparse
from itertools import chain, islice
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

# General torch imports
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
    MSELoss
)
from torch.optim import SGD, Optimizer, AdamW
from torch.utils.data import DataLoader

# For writing the result table to markdown
import pandas as pd

# For fully supervised baselines
from torchvision.models import resnet50, vit_b_16, ViT_B_16_Weights
from torchvision.models.vision_transformer import VisionTransformer

# For calculating the metrics
from torchmetrics.classification import MulticlassAveragePrecision, MulticlassAccuracy

# To load the Chicks4FreeID dataset
from datasets import Dataset, load_dataset

# For the training loop
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger

# For self-supervised baselines and evaluation of the models
from lightly.data import LightlyDataset
from lightly.loss.swav_loss import SwaVLoss
from lightly.models.modules import (
    SwaVProjectionHead,
    SwaVPrototypes,
    AIMPredictionHead,
    MaskedCausalVisionTransformer
)
from lightly.transforms import SwaVTransform, AIMTransform
from lightly.utils.dist import print_rank_zero
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.models import utils
from lightly.models.utils import random_prefix_mask
from lightly.models.utils import get_weight_decay_parameters
from lightly.models.modules.memory_bank import MemoryBankModule

# For loading the state of the art re-id model MegaDescriptorL384
import timm

from wildlife_tools.train.objective import ArcFaceLoss

@dataclass
class Config:
    batch_size_per_device: int = 16
    epochs: int = 200
    num_workers: int = 4
    log_dir: Path = Path("baseline_logs")
    checkpoint_path: Optional[Path] = None
    num_classes: int = 50
    skip_embedding_training: bool = False
    skip_knn_eval: bool = False
    skip_linear_eval: bool = False
    methods: Optional[List[str]] = None
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    test_run: bool = True
    check_val_every_n_epoch: int = 5
    profile= None  # "pytorch"
    experiment_result_metrics: Optional[List[str]] = field(default_factory=lambda: [])
    baseline_id: Optional[str] = None
    aggregate_metrics: bool = True


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have an explicit empty_cache function, but you can set up custom logic if needed
        torch.mps.empty_cache()  # For now, do nothing as MPS doesn't provide an empty_cache method
    else:
        # CPU - no need to empty cache
        pass

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
    [Modified version from lightly, which returns the scores. instead of the predictions]

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
        A tensor containing the kNN scores

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
    # pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_scores


class MetricCallback(Callback):
    """Callback that collects log metrics from the LightningModule and stores them after
    every epoch.

    Attributes:
        train_metrics:
            Dictionary that stores the last logged metrics after every train epoch.
        val_metrics:
            Dictionary that stores the last logged metrics after every validation epoch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_metrics: Dict[str, List[float]] = {}
        self.val_metrics: Dict[str, List[float]] = {}

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.sanity_checking:
            self._append_metrics(metrics_dict=self.train_metrics, trainer=trainer)

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not trainer.sanity_checking:
            self._append_metrics(metrics_dict=self.val_metrics, trainer=trainer)

    def _append_metrics(
        self, metrics_dict: Dict[str, List[float]], trainer: Trainer
    ) -> None:
        for name, value in trainer.callback_metrics.items():
            if isinstance(value, Tensor) and value.numel() != 1:
                # Skip non-scalar tensors.
                print("skipping metric", name, value)
                continue
            metrics_dict.setdefault(name, []).append(float(value))



class MetricModule(LightningModule):
    enable_logging = True

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        if self.enable_logging:
            self.train_map = MulticlassAveragePrecision(num_classes=num_classes)
            self.val_map = MulticlassAveragePrecision(num_classes=num_classes)

            self.train_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
            self.train_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)

            self.val_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
            self.val_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)

    def update_train_metrics(self, pred_scores: Tensor, targets: Tensor):
        if self.enable_logging:
            self.train_map(pred_scores, targets)
            self.train_top1(pred_scores, targets)
            self.train_top5(pred_scores, targets)

    def update_val_metrics(self, pred_scores: Tensor, targets: Tensor):
        if self.enable_logging:
            self.val_map(pred_scores, targets)
            self.val_top1(pred_scores, targets)
            self.val_top5(pred_scores, targets)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.enable_logging and self.train_map.update_called:
            self.log("train_mAP", self.train_map, prog_bar=True)
            self.log("train_top1", self.train_top1, prog_bar=True)
            self.log("train_top5", self.train_top5, prog_bar=True)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.enable_logging and self.val_map.update_called:
            self.log("val_mAP", self.val_map, prog_bar=True)
            self.log("val_top1", self.val_top1, prog_bar=True)
            self.log("val_top5", self.val_top5, prog_bar=True)


class KNNClassifier(MetricModule):
    """
    A lightly KNN Classifier modified to log mean average precision metric.
    """
    def __init__(
        self,
        model: Module,
        num_classes: int,
        knn_k: int = 200,
        knn_t: float = 0.1,
        feature_dtype: torch.dtype = torch.float32,
        normalize: bool = True,
    ):
        """KNN classifier to compute baseline performance of embedding models.

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
        super().__init__(num_classes=num_classes)
        self.save_hyperparameters(
            {
                "num_classes": num_classes,
                "knn_k": knn_k,
                "knn_t": knn_t,
                "feature_dtype": str(feature_dtype),
            }
        )
        self.model = model
        self.model.eval()
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.feature_dtype = feature_dtype
        self.normalize = normalize

        self._train_features = []
        self._train_targets = []
        self._train_features_tensor: Optional[Tensor] = None
        self._train_targets_tensor: Optional[Tensor] = None

    @torch.no_grad()
    def training_step(self, batch, batch_idx) -> None:
        images, targets = batch[0], batch[1]
        features = self.model.forward(images).flatten(start_dim=1)
        if self.normalize:
            features = F.normalize(features, dim=1)
        features = features.to(self.feature_dtype)
        self._train_features.append(features.detach().cpu())
        self._train_targets.append(targets.detach().cpu())

    def validation_step(self, batch, batch_idx) -> None:
        if self._train_features_tensor is None or self._train_targets_tensor is None:
            return

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            features = self.model.forward(images).flatten(start_dim=1)
        if self.normalize:
            features = F.normalize(features, dim=1)
        features = features.to(self.feature_dtype)
        pred_scores = knn_predict(
            feature=features,
            feature_bank=self._train_features_tensor,
            feature_labels=self._train_targets_tensor,
            num_classes=self.num_classes,
            knn_k=self.knn_k,
            knn_t=self.knn_t,
        )

        self.update_val_metrics(pred_scores, targets)
        del images, targets, features, pred_scores

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

    def on_validation_end(self) -> None:
        super().on_validation_end()
        # Clear the cache after each validation epoch to prevent memory leaks.
        del self._train_features_tensor
        del self._train_targets_tensor
        del self._train_features
        del self._train_targets


class LinearClassifier(MetricModule):
    """
    A lightly Linear Classifier, modified to log the mean average precision
    """

    def __init__(
        self,
        model: Module,
        batch_size_per_device: int,
        feature_dim: int,
        num_classes: int,
        freeze_model: bool = False,
        enable_logging: bool = True,
    ) -> None:
        """Linear classifier for computing baseline performance.

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
        super().__init__(num_classes=num_classes)
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.batch_size_per_device = batch_size_per_device
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.freeze_model = freeze_model
        self.enable_logging = enable_logging

        self.classification_head = self.build_classification_head(
            feature_dim=feature_dim, num_classes=num_classes
        )
        self.criterion = self.build_critierion()

    def build_classification_head(self, feature_dim: int, num_classes: int):
        return Linear(feature_dim, num_classes)
    
    def build_critierion(self):
        return CrossEntropyLoss()
    
    def forward(self, images: Tensor) -> Tensor:
        with torch.set_grad_enabled(not self.freeze_model):
            features = self.model(images).flatten(start_dim=1)
        output = self.classification_head(features)
        del images, features
        return output


    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)

        #if self.enable_logging:
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=images.size(0))
        self.update_train_metrics(predictions, targets)

        # Clear unnecessary variables
        del batch, images, targets, predictions
        return loss  # Return the loss

    @torch.no_grad()
    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        #if self.enable_logging:
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=images.size(0))
        self.update_val_metrics(predictions, targets)

        # Clear unnecessary variables
        del batch, images, targets, predictions, loss


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


class OnlineLinearClassifier(LinearClassifier):
    """
    A lightly Online Linear Classifier, modified to log the mean average precision
    """
    def __init__(
        self,
        feature_dim,
        num_classes,
        enable_logging: bool = True
    ) -> None:
        super().__init__(
            model=None,  # Not used for online classifier
            batch_size_per_device=None, # Not used for online classifier
            feature_dim=feature_dim,
            num_classes=num_classes,
            enable_logging=enable_logging
        )
    
    def forward(self, x: Tensor) -> Tensor:
        #with torch.no_grad():
        return self.classification_head(x.detach().flatten(start_dim=1))

    def configure_optimizers(self) -> Tuple[List[Optimizer] | List[Dict[str, Any | str]]]:
        # No optimization is performed in this class.
        return None


class AIM(MetricModule):
    def __init__(
        self, 
        batch_size_per_device: int, 
        num_classes: int, 
        feature_dim: int
    ) -> None:
        super().__init__(
            num_classes=num_classes
        )
        self.save_hyperparameters()
        self.feature_dim = feature_dim
        self.batch_size_per_device = batch_size_per_device

        vit = MaskedCausalVisionTransformer(
            img_size=384,
            patch_size=16,
            num_classes=num_classes,
            embed_dim=self.feature_dim,
            depth=12,
            num_heads=12,
            qk_norm=False,
            class_token=False,
            no_embed_class=True,
        )
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=vit.pos_embed, has_class_token=vit.has_class_token
        )
        self.patch_size = vit.patch_embed.patch_size[0]
        self.num_patches = vit.patch_embed.num_patches

        self.backbone = vit
        self.projection_head = AIMPredictionHead(
            input_dim=vit.embed_dim, output_dim=3 * self.patch_size**2
        )

        self.criterion = MSELoss()

        self.online_classifier = OnlineLinearClassifier(
            feature_dim=vit.embed_dim, num_classes=num_classes
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        features = self.backbone.forward_features(x, mask=mask)
        # TODO: We use mean aggregation for simplicity. The paper uses
        # AttentionPoolingClassifier to get the class features. But this is not great
        # as it requires training an additional head.
        # https://github.com/apple/ml-aim/blob/1eaedecc4d584f2eb7c6921212d86a3a694442e1/aim/torch/layers.py#L337
        return features.mean(dim=1).flatten(start_dim=1)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        views, targets = batch[0], batch[1]
        images = views[0]  # AIM has only a single view
        batch_size = images.shape[0]

        mask = random_prefix_mask(
            size=(batch_size, self.num_patches),
            max_prefix_length=self.num_patches - 1,
            device=images.device,
        )
        features = self.backbone.forward_features(images, mask=mask)
        # Add positional embedding before head.
        features = self.backbone._pos_embed(features)
        predictions = self.projection_head(features)

        # Convert images to patches and normalize them.
        patches = utils.patchify(images, self.patch_size)
        patches = utils.normalize_mean_var(patches, dim=-1)

        loss = self.criterion(predictions, patches)

        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        # TODO: We could use AttentionPoolingClassifier instead of mean aggregation:
        # https://github.com/apple/ml-aim/blob/1eaedecc4d584f2eb7c6921212d86a3a694442e1/aim/torch/layers.py#L337
        cls_features = features.mean(dim=1).flatten(start_dim=1)
        # Calculate the classification loss.
        # with torch.no_grad():
        cls_scores = self.online_classifier.forward(cls_features.detach())
        cls_loss = self.online_classifier.criterion(cls_scores, targets)
        self.log("train_cls_loss", cls_loss, prog_bar=True, sync_dist=True, batch_size=len(targets))
        
        self.update_train_metrics(cls_scores, targets)

        del views, targets, images, mask, features, predictions, patches, cls_features, cls_scores, batch

        return loss + cls_loss

    @torch.no_grad()
    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.backbone.forward_features(images, mask=None)
        # Add positional embedding before head.
        features = self.backbone._pos_embed(features)
        cls_features = features.mean(dim=1).flatten(start_dim=1)
        cls_scores = self.online_classifier.forward(cls_features)
        cls_loss = self.online_classifier.criterion(cls_scores, targets)
        
        self.log("val_loss", cls_loss, prog_bar=True, sync_dist=True, batch_size=len(targets))
        self.update_val_metrics(cls_scores, targets)

        del images, targets, cls_features, cls_scores, cls_loss

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = utils.get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = AdamW(
            [
                {"name": "aim", "params": params},
                {
                    "name": "aim_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.001 * self.batch_size_per_device * self.trainer.world_size / 4096,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=31250 / 125000 * self.trainer.estimated_stepping_batches,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Union[int, float, None] = None,
        gradient_clip_algorithm: Union[str, None] = None,
    ) -> None:
        self.clip_gradients(
            optimizer=optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )



class SwAV(MetricModule):
    """
    A lightly SwAV model, modified to log the mean average precision
    """
    CROP_COUNTS: Tuple[int, int] = (2, 6)

    def __init__(
        self, 
        batch_size_per_device: int, 
        num_classes: int, 
        feature_dim: int
    ) -> None:
        super().__init__(
            num_classes=num_classes
        )
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        resnet = resnet50()
        resnet.fc = Identity()  # Ignore classification head
        self.backbone = resnet
        self.projection_head = SwaVProjectionHead()
        self.prototypes = SwaVPrototypes(n_steps_frozen_prototypes=1)
        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True)
        self.online_classifier = OnlineLinearClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            enable_logging=False
        )

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
        self.log("train_loss",loss,prog_bar=True,sync_dist=True,batch_size=len(targets))

        # Calculate the classification loss.
        with torch.no_grad():
            cls_scores = self.online_classifier.forward(multi_crop_features[0].detach())
            cls_loss = self.online_classifier.criterion(cls_scores, targets)
            self.log("train_cls_loss", cls_loss, prog_bar=True, sync_dist=True, batch_size=len(targets))
        
        self.update_train_metrics(cls_scores, targets)

        del multi_crops, targets, multi_crop_features, multi_crop_projections, queue_crop_logits, multi_crop_logits
        return loss + cls_loss

    @torch.no_grad()
    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_scores = self.online_classifier.forward(features)
        cls_loss = self.online_classifier.criterion(cls_scores, targets)
        
        self.log("val_loss", cls_loss, prog_bar=True, sync_dist=True, batch_size=len(targets))
        self.update_val_metrics(cls_scores, targets)

        del images, targets, features, cls_scores, cls_loss
        

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
        print_rank_zero(f"Estimated steps: {self.trainer.estimated_stepping_batches}")
        warump_steps = int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs * 10)
        print_rank_zero(f"Warmup steps: {warump_steps}")
        max_steps = self.trainer.estimated_stepping_batches
        print_rank_zero(f"Max steps: {max_steps}")

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=warump_steps,
                max_epochs=max_steps,
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
    A ResNet50 classifier to compute baseline metrics of a fully supervised setting.
    """
    def __init__(
        self,
        batch_size_per_device: int,
        feature_dim,
        num_classes,
        freeze_model: bool = False,
    ) -> None:
        super().__init__(
            model=None,
            feature_dim=feature_dim,
            num_classes=num_classes,
            batch_size_per_device=batch_size_per_device,
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
        self.model.eval()

    def forward(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.model.forward(images).flatten(start_dim=1)
        return features

    def configure_optimizers(self) -> None:
        # configure_optimizers must be implemented for PyTorch Lightning. Returning None
        # means that no optimization is performed.
        pass



class ViT_B_16Classifier(LinearClassifier):
    model: VisionTransformer
    def __init__(
        self,
        batch_size_per_device,
        feature_dim,
        num_classes,
    ) -> None:
        super().__init__(
            model=None,
            feature_dim=feature_dim,
            num_classes=num_classes,
            batch_size_per_device=batch_size_per_device,
            freeze_model=False,
        )

        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.model.heads = Identity()
        

    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.model]#, self.classification_head]
        )
        optimizer = AdamW(
            [
                {"name": "mae", "params": params},
                {
                    "name": "vit_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "classhead_classifier",
                    "params": self.classification_head.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=0.001 * self.batch_size_per_device * self.trainer.world_size / 4096,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=31250 / 125000 * self.trainer.estimated_stepping_batches,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


class ViTEmbedding(LightningModule):

    def __init__(self, model: ViT_B_16Classifier) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model.model
        self.model.eval()

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model(x)



class MegaDescriptorL384(LightningModule):
    """
    A pretrained-model that uses the MegaDescriptor-L-384 model from the HuggingFace model hub
    to compute baseline metrics of an unsupervised setting.

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



class MegaDescriptorL384FineTune(LinearClassifier):
    """
    A model that uses the MegaDescriptor-L-384 model from the HuggingFace model hub
    to compute baseline metrics of an fintetuned setting.

    The model is  finetuned on the Chick4FreeID dataset

    The model has been trained on external Animal Re-ID Data + The chicks4FreeID dataset
    """
    enable_logging: bool = False

    def __init__(
        self,
        batch_size_per_device,
        feature_dim,
        num_classes,
    ) -> None:
        super().__init__(
            model=timm.create_model('swin_large_patch4_window12_384', num_classes=0, pretrained=True),
            feature_dim=feature_dim,
            num_classes=num_classes,
            batch_size_per_device=batch_size_per_device,
            freeze_model=False,
            enable_logging=self.enable_logging
        )
        
    
    def build_critierion(self):
        return ArcFaceLoss(num_classes=self.num_classes, embedding_size=self.feature_dim, margin=0.5, scale=64)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        # Combine the parameters of the model and the criterion
        params = chain(self.model.parameters(), self.criterion.parameters())

        # Define the optimizer with specified learning rate and momentum
        optimizer = SGD(params=params, lr=0.001, momentum=0.9)

        # Calculate the minimum learning rate
        min_lr = optimizer.defaults.get("lr") * 1e-3

        # Define the scheduler with a cosine annealing learning rate strategy
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=min_lr)

        return [optimizer], [scheduler]

from PIL import Image
HF_DATASET_DICT: Dict[str, Dataset] = {}
TRAIN_DATASET_CACHE: List[Tuple[Image.Image, int]] = []
TEST_DATASET_CACHE: List[Tuple[Image.Image, int]] = []


class ChicksVisionDataset(torchvision.datasets.VisionDataset):
    """
    Provides the Chicks4FreeID HuggingFace dataset as a Torchvision dataset.
    The dataset will return a tuple (PIL.Image, target:int)
    """

    def __init__(
        self,
        root: Union[str, Path] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        resize: int = 384,
        test_run: bool = False,
    ) -> None:
        global HF_DATASET_DICT, TRAIN_DATASET_CACHE, TEST_DATASET_CACHE
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.resize = resize
        self.transform = transform
        self.target_transform = target_transform
        self.test_run = test_run

        if not HF_DATASET_DICT:
            HF_DATASET_DICT = load_dataset(
                "dariakern/Chicks4FreeID", 
                "chicken-re-id-best-visibility", 
                download_mode="reuse_cache_if_exists"
            )
        if not TRAIN_DATASET_CACHE:
            print_rank_zero("Caching train images in memory...")
            TRAIN_DATASET_CACHE = [
                self._load_row(data) for data in tqdm(self.check_test_run(HF_DATASET_DICT["train"]))
            ]  
        if not TEST_DATASET_CACHE:
            print_rank_zero("Caching test images in memory...")
            TEST_DATASET_CACHE = [
                self._load_row(data) for data in tqdm(self.check_test_run(HF_DATASET_DICT["test"]))
            ]

        self.split = TRAIN_DATASET_CACHE if train else TEST_DATASET_CACHE
        
    def __len__(self):
        return len(self.split)

    def check_test_run(self, gen):
        yield from (gen if not self.test_run else islice(gen, 0, 50))
        
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)} ")
        img, target = self.split[idx]
        
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


class BaselineMethod():
    """
    An abstract class that holds common code of our baseline methods.

    The class runs:
        - embedding training
        - kNN evaluation
        - linear evaluation

    Reported metrics are:
        - Top-1 accuracy
        - Top-5 accuracy
        - Mean Average Precision (mAP)

    The baseline method can be configured by inheriting from this class and overriding specific
    attributes or functions as well as passing a config object.
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
    
    cfg: Config                            # A config class specifying the hyperparameters
    model: Module                          # The model used for embedding training
    feature_dim: int = 2048                # Important for the linear evaluation
    skip_embedding_training: bool = False  # Overwrites self.cfg.skip_embedding_training
    
    _name: str = ""                         # Name property. Will return the class name if not set


    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.method_dir = self.cfg.log_dir / self.cfg.baseline_id / self.name 
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
            test_run=self.cfg.test_run,
        )

        self.knn_train_dataset = self.linear_train_dataset = ChicksVisionDataset(
            train=True,
            transform=self.eval_train_transform,
        )
   
        self.knn_val_dataset = self.linear_val_dataset  = self.embedding_val_dataset = ChicksVisionDataset(
            train=False,
            transform=self.val_transform,
        )

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @timing_decorator
    def run_baseline_method(self):
        print_rank_zero(f"## Starting {self.name}... ")
        loaded_checkpoint = False
        if self.cfg.checkpoint_path:
            if self.name not in str(self.cfg.checkpoint_path):
                print_rank_zero(f"Not loading checkpoint for {self.name} because checkpoint path does not contain '{self.name}'.")
            else:
                self.model.load_state_dict(torch.load(self.cfg.checkpoint_path)["state_dict"])
                loaded_checkpoint = True
                print_rank_zero(f"Loaded checkpoint for {self.name} from {self.cfg.checkpoint_path}")

        skip_embedding_training = (
            self.cfg.skip_embedding_training 
            or self.cfg.epochs == 0 
            or loaded_checkpoint 
            or self.skip_embedding_training
        )
        if skip_embedding_training:
            print_rank_zero("Skipping embedding training")
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
        del self.model
        clear_cache()
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
            log_every_n_steps=1,
            precision=self.cfg.precision,
            check_val_every_n_epoch=min(
                epochs, 
                (self.cfg.check_val_every_n_epoch if not self.cfg.test_run else 1)
            ),
            strategy="auto",
            profiler=self.cfg.profile,
        )

        trainer.fit(
            model=classifier,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Print the current run results
        for metric in metric_callback.val_metrics.keys():
            max_value = max(metric_callback.val_metrics[metric])
            print_rank_zero(f"{self.name} {log_name} {metric}: {max_value}")
        
        # Update the metric values in a markdown and csv file
        metrics = {
            metric: max(value)
            for metric, value in metric_callback.val_metrics.items()
            if "train" not in metric and "loss" not in metric
        }
        self.cfg.experiment_result_metrics.append({
            "Setting": self.name,
            "Evaluation": log_name,
            **metrics
        })
        result_metrics_dir = self.cfg.log_dir / self.cfg.baseline_id
        result_metrics = pd.DataFrame(self.cfg.experiment_result_metrics)
        result_metrics.to_csv(result_metrics_dir / "metrics.csv", index=False)
        result_metrics.to_markdown(result_metrics_dir / "metrics.md", index=False, floatfmt=".4f")
                
        

class ResNet50Baseline(BaselineMethod):
    method_specific_augmentation = T.Compose([
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feature_dim: int = 2048
    
    def __init__(self, args):
        super().__init__(args)
        self.model = ResNet50Classifier(
            batch_size_per_device=self.cfg.batch_size_per_device,
            num_classes=self.cfg.num_classes,
            feature_dim=self.feature_dim,
            freeze_model=False,
        )


    def get_embedding_model(self):
        return ResNetEmbedding(model=self.model) 

    
class MegaDescriptorL384Baseline(BaselineMethod):
    normalize_transform = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    skip_embedding_training = True
    feature_dim = 1536        
        
    def __init__(self, args):
        super().__init__(args)
        self.model = MegaDescriptorL384()



class MegaDescriptorL384FineTuneBaseline(BaselineMethod):
    method_specific_augmentation = T.Compose([
        #T.Resize(size=(384, 384)),
        T.RandAugment(num_ops=2, magnitude=20),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    #normalize_transform = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    skip_embedding_training = False
    feature_dim = 1536        
    

    def __init__(self, args):
        super().__init__(args)
        self.model = MegaDescriptorL384FineTune(
            batch_size_per_device=self.cfg.batch_size_per_device,
            num_classes=self.cfg.num_classes,
            feature_dim=self.feature_dim,
        )


class ViT_B_16Baseline(BaselineMethod):
    method_specific_augmentation = T.Compose([
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feature_dim: int = 768

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = ViT_B_16Classifier(
            batch_size_per_device=self.cfg.batch_size_per_device,
            num_classes=self.cfg.num_classes,
            feature_dim=self.feature_dim,
        )

    def get_embedding_model(self):
        return ViTEmbedding(model=self.model)



class SwAVBaseline(BaselineMethod):
    method_specific_augmentation = SwaVTransform(
        rr_prob=0.5,
        rr_degrees=360,
        gaussian_blur=0.1,
        crop_counts=SwAV.CROP_COUNTS
    )
    feature_dim: int = 2048

    def __init__(self, args):
        super().__init__(
            args
        )

        self.model = SwAV(
            batch_size_per_device=self.cfg.batch_size_per_device,
            num_classes=self.cfg.num_classes,
            feature_dim=self.feature_dim,
        )

        self.embedding_train_dataset = LightlyDataset.from_torch_dataset(
            ChicksVisionDataset(train=True),
            transform=self.embedding_train_transform,
        ) 


class AIMBaseline(BaselineMethod):
    feature_dim: int = 768    
    method_specific_augmentation = AIMTransform(
        input_size=384
    )

    def __init__(self, args):
        super().__init__(args)
        self.model = AIM(
            batch_size_per_device=self.cfg.batch_size_per_device,
            num_classes=self.cfg.num_classes,
            feature_dim=self.feature_dim,
        )

        self.embedding_train_dataset = LightlyDataset.from_torch_dataset(
            ChicksVisionDataset(train=True),
            transform=self.embedding_train_transform,
        )




class Baseline:
    methods: Dict[str, Type[BaselineMethod]] = {
        #"swav": SwAVBaseline,  # Broken
        #"aim": AIMBaseline,    # Broken
        #"resnet50": ResNet50Baseline, # Resnet worked around 90% top1, it is kinda old tho tbh so it's not further pursued
        "vit_b_16": ViT_B_16Baseline,
        "mega_descriptor_finetune": MegaDescriptorL384FineTuneBaseline,
        "mega_descriptor": MegaDescriptorL384Baseline,
    }

    @timing_decorator
    def run(self, args):
        cfg = Config(**vars(args))

        if cfg.aggregate_metrics:
            self.aggregate_metrics(cfg)
            return
        
        cfg.baseline_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        methods = cfg.methods or list(self.methods.keys())
        print_rank_zero(f"# Running: {methods}...")   
        for method in methods:
            self.methods[method](cfg).run_baseline_method()
        print_rank_zero(f"# All baselines metrics computed!")
        print_rank_zero(f"# Results saved in {cfg.log_dir / cfg.baseline_id}") 
            

    
    def calculate_mean_std(self, df: pd.DataFrame, groupby: List[str]):
        """
        Calculate the mean and standard deviation for all numerical columns grouped by two identifiers,
        and return a DataFrame with the results in the ± notation.

        Parameters:
        df (pd.DataFrame): Input DataFrame.
        identifier1 (str): First identifier column name.
        identifier2 (str): Second identifier column name.

        Returns:
        pd.DataFrame: DataFrame with mean and standard deviation in ± notation.
        """
        # Determine the value columns
        value_columns = [col for col in df.columns if col not in groupby]
        
        # Group by the identifiers
        grouped = df.groupby(groupby)

        # Calculate mean and standard deviation
        mean_df = grouped.mean().reset_index()
        std_df = grouped.std().reset_index()

        # Merge the mean and standard deviation DataFrames
        merged_df = mean_df.copy()
        for col in value_columns:
            merged_df[f"{col}_std"] = std_df[col]

        # Function to combine mean and standard deviation with ± notation
        def combine_mean_std(row, mean_col, std_col):
            return f"{row[mean_col]:.3f} ± {(row[std_col] if row[std_col] is not None else 0):.3f}"

        # Create a DataFrame to store results
        result_df = mean_df.copy()
        for col in value_columns:
            result_df[col] = merged_df.apply(lambda row: combine_mean_std(row, col, f"{col}_std"), axis=1)

        #result_df = result_df.pivot(index='Setting', columns='Evaluation')
        return result_df.reset_index()

    def aggregate_metrics(self, args):
        cfg = Config(**vars(args))
        metrics = list(cfg.log_dir.glob("**/metrics*.csv"))
        result_metrics = pd.concat([pd.read_csv(metric) for metric in metrics], ignore_index=True)
        result_metrics.dropna(inplace=True)

        result_metrics.to_csv(cfg.log_dir / "agglomerated_metrics.csv", index=False)
        result_metrics.to_markdown(cfg.log_dir / "agglomerated_metrics.md", index=False, floatfmt=".4f")

        # Aggregate the metrics with error bars
        result_metrics = self.calculate_mean_std(result_metrics, ["Setting", "Evaluation"])

        result_metrics.to_csv(cfg.log_dir / "aggregated_metrics.csv", index=False)
        result_metrics.to_markdown(cfg.log_dir / "aggregated_metrics.md", index=False, floatfmt=".4f")
        print_rank_zero(f"Aggregated metrics saved in {cfg.log_dir}")


parser = argparse.ArgumentParser(description='Baseline metrics for the paper Chicks4FreeID')
parser.add_argument("--log-dir", type=Path, default=str(Config.log_dir))
parser.add_argument("--batch-size-per-device", type=int, default=Config.batch_size_per_device) #default=32) #default=128)
parser.add_argument("--epochs", type=int, default=Config.epochs)
parser.add_argument("--num-workers", type=int, default=Config.num_workers)
parser.add_argument("--checkpoint-path", type=Path, default=Config.checkpoint_path)
parser.add_argument("--methods", type=str, nargs="+", default=Config.methods, choices=Baseline.methods.keys(), required=False)
#parser.add_argument("--num-classes", type=int, default=Config.num_classes)
parser.add_argument("--skip-embedding-training", action="store_true", default=Config.skip_embedding_training)
parser.add_argument("--skip-knn-eval", action="store_true", default=Config.skip_knn_eval)
parser.add_argument("--skip-linear-eval", action="store_true", default=Config.skip_linear_eval)
#parser.add_argument("--test-run", action="store_true", default=Config.test_run)
parser.add_argument("--aggregate-metrics", action="store_true", default=Config.aggregate_metrics)

if __name__ == "__main__":
    args = parser.parse_args()
    Baseline().run(args)

