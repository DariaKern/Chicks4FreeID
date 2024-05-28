from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero


from lightly_benchmarks.imagenet.resnet50 import barlowtwins
from lightly_benchmarks.imagenet.resnet50 import byol
from lightly_benchmarks.imagenet.resnet50 import dcl
from lightly_benchmarks.imagenet.resnet50 import dclw
from lightly_benchmarks.imagenet.resnet50 import dino
from lightly_benchmarks.imagenet.resnet50 import finetune_eval
from lightly_benchmarks.imagenet.resnet50 import knn_eval
from lightly_benchmarks.imagenet.resnet50 import linear_eval
from lightly_benchmarks.imagenet.resnet50 import mocov2
from lightly_benchmarks.imagenet.resnet50 import simclr
from lightly_benchmarks.imagenet.resnet50 import swav
from lightly_benchmarks.imagenet.resnet50 import tico
from lightly_benchmarks.imagenet.resnet50 import vicreg
from lightly_benchmarks.imagenet.vitb16 import aim
from lightly_benchmarks.imagenet.resnet50 import mega_descriptor_L384
from lightly_benchmarks.imagenet.resnet50 import resnet50

from pathlib import Path
from typing import Callable, Optional, Union
from datasets import Dataset
from datasets import load_dataset

import torchvision

        




METHODS = {
    "swav": {"model": swav.SwAV, "transform": swav.transform},
    "resnet50": {"model": resnet50.ResNet50Classifier, "transform": resnet50.transform}, 
    "mega_descriptor_l384": {"model": mega_descriptor_L384.MegaDescriptorL384, "transform": mega_descriptor_L384.transform},
    #"barlowtwins": {"model": barlowtwins.BarlowTwins, "transform": barlowtwins.transform,},
    #"byol": {"model": byol.BYOL, "transform": byol.transform},
    #"dcl": {"model": dcl.DCL, "transform": dcl.transform},
    #"dclw": {"model": dclw.DCLW, "transform": dclw.transform},
    #"aim": {"model": aim.AIM, "transform": aim.transform},
    #"dino": {"model": dino.DINO, "transform": dino.transform},
    #"mocov2": {"model": mocov2.MoCoV2, "transform": mocov2.transform},
    #"simclr": {"model": simclr.SimCLR, "transform": simclr.transform},
    #"tico": {"model": tico.TiCo, "transform": tico.transform},
    #"vicreg": {"model": vicreg.VICReg, "transform": vicreg.transform},
}

parser = ArgumentParser("Chicks4FreeID ResNet50 Benchmarks")
parser.add_argument("--log-dir", type=Path, default="benchmark_logs")
parser.add_argument("--batch-size-per-device", type=int, default=128)#default=32) #default=128)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--accelerator", type=str, default="auto") # default="ddp" or "ddp2" or "gpu" or "cpu
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--precision", type=str, default="16-mixed") # "16-mixed")
parser.add_argument("--ckpt-path", type=Path, default=None)
parser.add_argument("--compile-model", action="store_true")
parser.add_argument("--methods", type=str, nargs="+")
parser.add_argument("--num-classes", type=int, default=50)
parser.add_argument("--skip-knn-eval", action="store_true")
parser.add_argument("--skip-linear-eval", action="store_true")
parser.add_argument("--skip-finetune-eval", action="store_true", default=True)



def main(
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    epochs: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    compile_model: bool,
    methods: Union[Sequence[str], None],
    num_classes: int,
    skip_knn_eval: bool,
    skip_linear_eval: bool,
    skip_finetune_eval: bool,
    ckpt_path: Union[Path, None],
) -> None:
    torch.set_float32_matmul_precision("high")

    method_names = methods or METHODS.keys()

    for method in method_names:
        print_rank_zero(f"Running benchmark for method {method}...")
        
        method_dir = (
            log_dir / method / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ).resolve()
        model = METHODS[method]["model"](
            batch_size_per_device=batch_size_per_device, num_classes=num_classes
        )

        if compile_model and hasattr(torch, "compile"):
            # Compile model if PyTorch supports it.
            print_rank_zero("Compiling model...")
            model = torch.compile(model)
        
        input_size = 384 
        if hasattr(model, "input_size"):
            input_size = model.input_size

        feature_dim = 2048
        if hasattr(model, "feature_dim"):
            feature_dim = model.feature_dim
        
        normalize = T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"])
        if hasattr(model, "normalize"):
            normalize = model.normalize

        # Transform for pretaining of the embedding
        pretrain_transform = T.Compose([
            T.Resize(input_size),
            T.RandomRotation(360),
            METHODS[method]["transform"]
        ])

        # Transform for linear eval training and fine-tuning
        eval_transform = T.Compose(
            [
                T.RandomResizedCrop(input_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        )

        # Transform for validation and knn evaluation
        val_transform = T.Compose(
            [
                T.Resize(input_size),
                T.ToTensor(),
                normalize
            ]
        )


        common_args = {
            "model": model,
            "train_dir": train_dir,
            "val_dir": val_dir,
            "log_dir": method_dir,
            "batch_size_per_device": batch_size_per_device,
            "num_workers": num_workers,
            "accelerator": accelerator,
            "devices": devices,
            "precision": precision,
        }

        

        if hasattr(model, "is_unsupervised") and model.is_unsupervised:
            print_rank_zero("Unsupervise setting uses a pretrained model - skipping pretraining")
        elif epochs <= 0:
            print_rank_zero("Epochs <= 0, skipping pretraining.")
            if ckpt_path is not None:
                model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        else:
            pretrain(
                method=method,
                epochs=epochs,
                ckpt_path=ckpt_path,
                train_transform=pretrain_transform,
                val_transform=val_transform,
                **common_args
            )

        if skip_knn_eval:
            print_rank_zero("Skipping KNN eval.")
        else:
            knn_eval.knn_eval(
                transform=val_transform,
                num_classes=num_classes,
                **common_args
            )
 

        if skip_linear_eval:
            print_rank_zero("Skipping linear eval.")
        else:
            linear_eval.linear_eval(
                num_classes=num_classes,
                train_transform=eval_transform,
                val_transform=val_transform,
                feature_dim=feature_dim,
                **common_args
            )



        if skip_finetune_eval:
            print_rank_zero("Skipping fine-tune eval.")
        else:
            raise NotImplementedError(
                "Fine-tune evaluation is currently not implemented for all methods"
            )
            finetune_eval.finetune_eval(
                train_transform=eval_transform,
                val_transform=val_transform,
                num_classes=num_classes,
                feature_dim=feature_dim,
                **common_args
            )


def pretrain(
    model: LightningModule,
    method: str,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    epochs: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    ckpt_path: Union[Path, None],
    train_transform: Callable,
    val_transform: Callable,

) -> None:
    print_rank_zero(f"Running pretraining for {method}...")


    train_dataset = LightlyDataset(input_dir=(train_dir), transform=train_transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=False,
    )

    val_dataset = LightlyDataset(input_dir=(val_dir), transform=val_transform)
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    # Train model.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        #devices=devices,
        callbacks=[
            LearningRateMonitor(),
            # Stop if training loss diverges.
            # EarlyStopping(monitor="train_loss", patience=int(1e12), check_finite=True),
            DeviceStatsMonitor(),
            metric_callback,
        ],
        logger=TensorBoardLogger(save_dir=(log_dir), name="pretrain"),
        precision=precision,
        #strategy="ddp_find_unused_parameters_true",
        #sync_batchnorm=accelerator != "cpu",  # Sync batchnorm is not supported on CPU.
        #gradient_clip_val=5.0,
        num_sanity_val_steps=0,
        log_every_n_steps=0,
        #check_val_every_n_epoch=5,
        #accumulate_grad_batches=4
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
    for metric in ["val_online_cls_top1", "val_online_cls_top5"]:
        print_rank_zero(f"max {metric}: {max(metric_callback.val_metrics[metric])}")





# Monkeypatch LightlyDataset such that __init__ initializes a Chick4FreeReIDBestTorchVisionDataset


class Chicks4FreeReIDBestTorchVisionDataset(torchvision.datasets.VisionDataset):
    """
    Provides the HuggingFace dataset as a Torchvision dataset.
    The dataset will return a tuple of the input and output based on the supervised_keys.
    """
    
    def __init__(
        self,
        root: Union[str, Path] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.hf_dataset: Dataset = load_dataset(
            "dariakern/Chicks4FreeID", 
            "chicken-re-id-best-visibility", 
            trust_remote_code=True, 
            split="train" if train else "test",
            keep_in_memory=True
        )
        
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Retrieve data at the specified index
        img, target = self.hf_dataset[idx].values()
            
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __str__(self):
        return self
    

old_init = LightlyDataset.__init__
def new_init(self, input_dir, transform):
    try:
        LightlyDataset.__init__ = old_init

        torch_ds = Chicks4FreeReIDBestTorchVisionDataset(train=input_dir=="train", download=True)        
        ds = LightlyDataset.from_torch_dataset(torch_ds, transform)

        self.__dict__.update(ds.__dict__)
    finally:
        LightlyDataset.__init__ = new_init



if __name__ == "__main__":
    args = parser.parse_args()
    args.train_dir = "train"
    args.val_dir = "test"
    LightlyDataset.__init__ = new_init
    main(**vars(args))
