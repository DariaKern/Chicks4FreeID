###
# Author: Tobias
# Description: This file contains the code for the baseline models used in the experiments
# of the paper "Chicks4FreeID"
# The code is inspired by the lightly benchmarks, but has significantly diverged from it over time.
# The major reasons for this are:
# - The code is to be used in a single file
# - The mean average precision metric is introduced
# - Support for unsupervised frozen feature extractor methods like MegaDescriptorL384
# - Support for fully supervised methods like ResNet50Classifier or ViT
# - Implementation of Chicks4FreeID dataset with caching 
# - Single point of dataset / dataloaded / transforms handling
# - Base classes for Metrics, Methods and Experiments
# - All evaluation augmentations are the same now
# - Introducing a Config class to manage hyperparameters and CLI
# - The code is uses inheritance and composition where possible
# - The code uses the PyTorch Lightning / torchmetrics implemenations of the metrics
# - The end result is a markdown table to compare to the results of the paper
# - Allows aggregtation of multiple metric tables into a single table with error bars and mean
# - Some code is still taken from lightly, but is either imported or marked with a comment
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
    Module
)
from torch.optim import SGD, Optimizer, AdamW
from torch.utils.data import DataLoader

# For writing the result table to markdown
import pandas as pd
from PIL import Image

# For fully supervised baselines
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.vision_transformer import VisionTransformer

# For calculating the metrics
from torchmetrics.classification import MulticlassAveragePrecision, MulticlassAccuracy

# To load the Chicks4FreeID dataset
from datasets import Dataset, load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict

# For the training loop
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from lightly.utils.dist import print_rank_zero

# Some fancy stuff for optimizing vision transformers
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.models.utils import get_weight_decay_parameters

# For loading the state of the art re-id model MegaDescriptorL384
import timm

# For the re-training of the MegaDescriptorL384 model
from wildlife_tools.train.objective import ArcFaceLoss

@dataclass
class Config:
    batch_size_per_device: int = 16
    epochs: int = 200
    num_workers: int = 4
    checkpoint_path: Optional[Path] = None
    num_classes: int = 50
    skip_embedding_training: bool = False
    skip_knn_eval: bool = False
    skip_linear_eval: bool = False
    methods: Optional[List[str]] = None
    dataset_subsets: Optional[List[str]] = None
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "16-mixed"
    test_run: bool = False
    check_val_every_n_epoch: int = 5
    profile= None  # "pytorch"
    aggregate_metrics: bool = False

    # Internal Variables
    experiment_result_metrics: Optional[List[str]] = field(default_factory=lambda: [])
    baseline_id: Optional[str] = None
    dataset_subset: str = "chicken-re-id-all-visibility"
    log_dir: Path = Path("baseline_logs")



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
    [Modified version from lightly, which returns the scores instead of the predictions]

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
    """A [Lightly] Callback that collects log metrics from the LightningModule and stores them after
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
    Also it now inherits from MetricModule and the logging logic has changed.
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
        #del self._train_features_tensor
        #del self._train_targets_tensor
        #del self._train_features
        #del self._train_targets


class LinearClassifier(MetricModule):
    """
    A lightly Linear Classifier, modified to log the mean average precision
    Also, the logging logic has changed + it now inherits from MetricModule
    Further, the LinearClassifier now also allows the instantiation of fully supervised models.
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




class ViT_B_16Classifier(LinearClassifier):
    """
    A fully supervised model that uses the Vision Transformer model from the torchvision library

    The model uses the standard ViT_B_16 model and cross entropy (as in inherited from LinearClassifier) for training
    """
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
        # Use the Identity head to get to the features
        self.model.heads = Identity()
        

    def configure_optimizers(self):
        """
        This optimizer is a inspired the optimizer used in the lightly benchmarks for their Vision Transformer backbones
        specifically the AIM Model.
        """
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
    """
    This module is used to extract features from the Vision Transformer Classifier in eval mode
    """
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
    to compute baseline metrics using a SotA animal re-id feature extractor.

    The model is not finetuned and only used to extract features.

    The model has been trained on external Animal Re-ID Data and has not been trained on chickens.
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
    A model that uses the same architecture as the MegaDescriptor-L-384 model
    i.e. the Swin Transformer, but is trained on the Chick4FreeID dataset

    The settings and hyperparameters mirror the settings and hyperparameters of the MegaDescriptorL384 training procedure.
    """
    # Disable logging during embedding training because the ArcFaceLoss takes an embedding isntead of class scores.
    # Without class scores available during training, the logging would fail.
    enable_logging: bool = False

    def __init__(
        self,
        batch_size_per_device,
        feature_dim,
        num_classes,
    ) -> None:
        super().__init__(
            model=timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", num_classes=0, pretrained=True),
            feature_dim=feature_dim,
            num_classes=num_classes,
            batch_size_per_device=batch_size_per_device,
            freeze_model=False,
            enable_logging=self.enable_logging
        )
        
    
    def build_critierion(self):
        """
        For rationale why the ArcFaceLoss is used, see the paper of the MegaDescriptorL384 model
        """
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




class SwinL384(LinearClassifier):
    """
    A model that uses the same architecture as the MegaDescriptor-L-384 model
    i.e. the Swin Transformer, but is trained on the Chick4FreeID dataset

    The settings and hyperparameters mirror the settings and hyperparameters of the MegaDescriptorL384 training procedure.
    """
    # Disable logging during embedding training because the ArcFaceLoss takes an embedding isntead of class scores.
    # Without class scores available during training, the logging would fail.
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
        """
        For rationale why the ArcFaceLoss is used, see the paper of the MegaDescriptorL384 model
        """
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


HF_DATASET_DICT: Dict[str, Dataset] = {}
TRAIN_DATASET_CACHE: List[Tuple[Image.Image, int]] = []
VALIDATION_DATASET_CACHE: List[Tuple[Image.Image, int]] = []
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
        validation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        resize: int = 384,
        test_run: bool = False,
        val_split: float = 0.1,  # Validation split ratio
        dataset_subset = "chicken-re-id-all-visibility",
    ) -> None:
        """
        Args:
            root: Passed to torchvision.datasets.VisionDataset
            train: If True, creates dataset from training set, otherwise creates from test set.
            validation: If True, creates dataset from validation set.
            transform: A function/transform that takes in an PIL image and returns a transformed version.
            target_transform: A function/transform that takes in the target (integer) and transforms it.
            resize: The size of the image after resizing (quadratic)
            test_run: If True, only returns and caches the first 50 images of the dataset
            val_split: The proportion of the training data to use for validation.
        """
        global HF_DATASET_DICT, TRAIN_DATASET_CACHE, VALIDATION_DATASET_CACHE, TEST_DATASET_CACHE
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.resize = resize
        self.transform = transform
        self.target_transform = target_transform
        self.test_run = test_run

        if not HF_DATASET_DICT:
            HF_DATASET_DICT = load_dataset(
                "dariakern/Chicks4FreeID", 
                dataset_subset, 
                download_mode="reuse_cache_if_exists"
            )
        
        if not TRAIN_DATASET_CACHE and not VALIDATION_DATASET_CACHE:
            print_rank_zero("Caching train images in memory...")
            full_train_cache = [
                self._load_row(data) for data in tqdm(self.check_test_run(HF_DATASET_DICT["train"]))
            ]  

            # Extract the targets to perform stratified split
            targets = [target for _, target in full_train_cache]

            # Stratified split
            stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=1)
            train_idx, val_idx = next(stratified_split.split(full_train_cache, targets))
            
            TRAIN_DATASET_CACHE = [full_train_cache[i] for i in train_idx]
            VALIDATION_DATASET_CACHE = [full_train_cache[i] for i in val_idx]
        
        if not TEST_DATASET_CACHE:
            print_rank_zero("Caching test images in memory...")
            TEST_DATASET_CACHE = [
                self._load_row(data) for data in tqdm(self.check_test_run(HF_DATASET_DICT["test"]))
            ]

        # Assign the appropriate split
        if train:
            self.split = TRAIN_DATASET_CACHE
        elif validation:
            self.split = VALIDATION_DATASET_CACHE
        else:
            self.split = TEST_DATASET_CACHE


    def __len__(self):
        return len(self.split)

    def check_test_run(self, gen):
        yield from gen# (gen if not self.test_run else islice(gen, 0, 50))
        
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
    
    def clear_cache():
        global HF_DATASET_DICT, TRAIN_DATASET_CACHE, VALIDATION_DATASET_CACHE, TEST_DATASET_CACHE
        HF_DATASET_DICT = {}
        TRAIN_DATASET_CACHE = []
        VALIDATION_DATASET_CACHE = []
        TEST_DATASET_CACHE = []



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
            # Disable resizing because it's already done in the dataset
            # self.resize_transform,
            T.RandomRotation(360),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            self.method_specific_augmentation
        ])

        # Transform for linear eval training and kkn training
        self.eval_train_transform = T.Compose([
            # Disable resizing because it's already done in the dataset
            # self.resize_transform,
            T.RandomRotation(360),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            self.normalize_transform,
        ])

        # Transform for all validation datasets
        self.val_transform = T.Compose([
            # Disable resizing because it's already done in the dataset
            # self.resize_transform,
            T.ToTensor(),
            self.normalize_transform
        ])


        self.embedding_train_dataset = ChicksVisionDataset(
            train=True,
            transform=self.embedding_train_transform,
            test_run=self.cfg.test_run,
            dataset_subset=self.cfg.dataset_subset
        )

        self.knn_train_dataset = self.linear_train_dataset = ChicksVisionDataset(
            train=True,
            transform=self.eval_train_transform,
            dataset_subset=self.cfg.dataset_subset
        )
   
        self.knn_val_dataset = self.linear_val_dataset  = self.embedding_val_dataset = ChicksVisionDataset(
            train=False,
            validation=True,
            transform=self.val_transform,
            dataset_subset=self.cfg.dataset_subset
        )

        self.knn_test_dataset = self.linear_test_dataset  = self.embedding_test_dataset = ChicksVisionDataset(
            train=False,
            validation=False,
            transform=self.val_transform,
            dataset_subset=self.cfg.dataset_subset
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
            test_dataset = self.knn_test_dataset,
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
            test_dataset = self.linear_test_dataset,
            log_name="linear_eval",
        )
    
    def embedding_training(self):
        print_rank_zero(f"### Training {self.name} embedding model... ")
        
        self.train(
            classifier = self.model,
            epochs = self.cfg.epochs,
            train_dataset = self.embedding_train_dataset,
            val_dataset = self.embedding_val_dataset,
            test_dataset = self.embedding_test_dataset,
            log_name="embedding_training",
        )


    @timing_decorator
    def train(self, classifier, epochs, train_dataset, val_dataset, test_dataset, log_name):
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

        test_dataloader = DataLoader(
            test_dataset,
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

        trainer.validate(
            model=classifier,
            dataloaders=test_dataloader,
            ckpt_path=None if epochs == 1 else "best",
        )

        # Print the current run results
        for metric in metric_callback.val_metrics.keys():
            max_value = (metric_callback.val_metrics[metric])[-1]
            print_rank_zero(f"{self.name} {log_name} {metric}: {max_value}")
        
        # Update the metric values in a markdown and csv file
        metrics = {
            metric: (value)[-1]
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
                
        

    
class MegaDescriptorL384Baseline(BaselineMethod):
    normalize_transform = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    skip_embedding_training = True
    feature_dim = 1536        
        
    def __init__(self, args):
        super().__init__(args)
        self.model = MegaDescriptorL384()


class MegaDescriptorL384FinetuneBaseline(BaselineMethod):
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


class SwinL384Baseline(BaselineMethod):
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
        self.model = SwinL384(
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




class Baseline:
    """
    The main class that runs the baseline methods.
    """
    methods: Dict[str, Type[BaselineMethod]] = {
        #"swav": SwAVBaseline,  # Removed
        #"aim": AIMBaseline,    # Removed
        #"resnet50": ResNet50Baseline, # Removed, Resnet worked around 90% top1, it is kinda old tho tbh so it's not further pursued
        "vit_b_16": ViT_B_16Baseline,
        "mega_descriptor_finetune": MegaDescriptorL384FinetuneBaseline,
        "mega_descriptor": MegaDescriptorL384Baseline,
        "swin_transformer": SwinL384Baseline,
    }

    all_subsets = ["chicken-re-id-all-visibility", "chicken-re-id-best-visibility"]

    @timing_decorator
    def run(self, args):
        """
        Run the class as specified in the config.
            
            args: argparse.Namespace - The CLI arguments, used as kwargs to instantiate a Config object.
        """
        cfg = Config(**vars(args))

        if cfg.aggregate_metrics:
            subsets = cfg.dataset_subsets or self.all_subsets
            for subset in subsets:
                cfg.log_dir = Path(subset)
                self.aggregate_metrics(cfg)
            return
        
        cfg.baseline_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        methods = cfg.methods or list(self.methods.keys())
        subsets = cfg.dataset_subsets or self.all_subsets
        
        for subset in subsets:
            ChicksVisionDataset.clear_cache()
            cfg.dataset_subset = subset
            cfg.log_dir = Path(subset)
            print_rank_zero(f"# Running: {methods} on subset {subset}...")   
            for method in methods:
                self.methods[method](cfg).run_baseline_method()
            
            print_rank_zero(f"# Results saved in {cfg.log_dir / cfg.baseline_id}") 
        
        print_rank_zero(f"# All baselines metrics computed!")
            

    
    def calculate_mean_std(self, df: pd.DataFrame, groupby: List[str]):
        """
        Calculate the mean and standard deviation for all numerical columns grouped by two identifiers,
        and return a DataFrame with the results in the ± notation.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            groupby (List[str]): List of which columns to group by.

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

    def aggregate_metrics(self, cfg: Config):

        # Agglomerate the all metric files
        metrics = list(cfg.log_dir.glob("**/metrics*.csv"))
        result_metrics = pd.concat([pd.read_csv(metric) for metric in metrics], ignore_index=True)
        result_metrics.dropna(inplace=True)
        result_metrics.to_csv(cfg.log_dir / "agglomerated_metrics.csv", index=False)
        result_metrics.to_markdown(cfg.log_dir / "agglomerated_metrics.md", index=False, floatfmt=".4f")

        # Aggregate the agglomerated metrics with error bars
        result_metrics = self.calculate_mean_std(result_metrics, ["Setting", "Evaluation"])
        result_metrics.to_csv(cfg.log_dir / "aggregated_metrics.csv", index=False)
        result_metrics.to_markdown(cfg.log_dir / "aggregated_metrics.md", index=False, floatfmt=".4f")
        print_rank_zero(f"Aggregated metrics saved in {cfg.log_dir}")


parser = argparse.ArgumentParser(description='Baseline metrics for the paper Chicks4FreeID')
#parser.add_argument("--log-dir", type=Path, default=str(Config.log_dir))
parser.add_argument("--batch-size-per-device", type=int, default=Config.batch_size_per_device) #default=32) #default=128)
parser.add_argument("--epochs", type=int, default=Config.epochs)
parser.add_argument("--num-workers", type=int, default=Config.num_workers)
parser.add_argument("--checkpoint-path", type=Path, default=Config.checkpoint_path)
parser.add_argument("--methods", type=str, nargs="+", default=Config.methods, choices=Baseline.methods.keys(), required=False)
#parser.add_argument("--num-classes", type=int, default=Config.num_classes) # Will be 50 for the Chicks4FreeID dataset
parser.add_argument("--skip-embedding-training", action="store_true", default=Config.skip_embedding_training)
parser.add_argument("--skip-knn-eval", action="store_true", default=Config.skip_knn_eval)
parser.add_argument("--skip-linear-eval", action="store_true", default=Config.skip_linear_eval)
#parser.add_argument("--test-run", action="store_true", default=Config.test_run) # For debugging only
parser.add_argument("--aggregate-metrics", action="store_true", default=Config.aggregate_metrics)

if __name__ == "__main__":
    args = parser.parse_args()
    Baseline().run(args)

