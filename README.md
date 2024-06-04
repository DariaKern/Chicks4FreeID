# 🐔 Chicks4FreeID
The very first publicly available chicken re-identification dataset
is available on 🤗 **Hugging Face**: [huggingface.co/datasets/dariakern/Chicks4FreeID](https://huggingface.co/datasets/dariakern/Chicks4FreeID)

<img src="./wiki/chickenDataset.png">


## 🤗 Usage

> ```shell
> pip install datasets
> ```

Load the data:
```python
from datasets import load_dataset
train_ds = load_dataset("dariakern/Chicks4FreeID", split="train")
train_ds[0]
```

Output: 
> ```python
> {'crop': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=2630x2630 at 0x7AA95E7D1720>,
> 'identity': 43}
> ```

> [!TIP]
> Find more information on how to work with 🤗  [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets/v2.19.0/index)


## 📊 Baseline

To establish a baseline on the dataset, we explore 3 approaches

1. We evaluate the SotA model in animal re-identification: MegaDescriptor-L-384, a feature extractor, pre-trained on many species and identities. 
   
   `timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)`
2. We train MegaDescriptor-L-384's underlying architecture; a Swin-Transformer, in the same way it has been used to build the MegaDescriptor-L-384, but now on our own dataset. 
   
   `timm.create_model('swin_large_patch4_window12_384')`
3. We train a Vision Transformer (ViT-B/16) as a fully supervised baseline, and focus on embeddings by replacing the classifier head with a linear layer.
   
   `from torchvision.models import vit_b_16`

Evaluation settings are based on:

- Linear: [SimCLR](https://dl.acm.org/doi/abs/10.5555/3524938.3525087)
- k-NN: [InstDist](https://doi.org/10.1109/CVPR.2018.00393)

Metrics are from torchmetrics 

- mAP: `MulticlassAveragePrecision(average="macro")`
- top1: `MulticlassAccuracy(top_k=1)`
- top5: `MulticlassAccuracy(top_k=5)`

Below are the metrics for the test set. Standard deviations are based on 3 runs:

| Setting                            | Evaluation          | mAP           | top-1         | top-5         |
|:-----------------------------------|:--------------------|:--------------|:--------------|:--------------|
| MegaDescriptor-L-384 (frozen)      | k-NN                | 0.649 ± 0.044 | 0.709 ± 0.026 | 0.924 ± 0.027 |
| MegaDescriptor-L-384 (frozen)      | Linear              | 0.935 ± 0.005 | 0.883 ± 0.009 | 0.985 ± 0.003 |
| Swin-L-384                         | k-NN                | 0.837 ± 0.062 | 0.881 ± 0.041 | 0.983 ± 0.010 |
| Swin-L-384                         | Linear              | 0.963 ± 0.022 | 0.922 ± 0.042 | 0.987 ± 0.012 |
| ViT-B/16                           | k-NN                | 0.893 ± 0.010 | 0.923 ± 0.005 | 0.985 ± 0.019 |
| ViT-B/16                           | Linear              | 0.976 ± 0.007 | 0.928 ± 0.002 | 0.990 ± 0.012 |


The most interesting observation in this table is that, even though the MegaDescriptor-L-384 feature extractor has never seen our dataset, its embeddings are still relatively helpful in identifiying the chickens, even when compared to the fully supervised approaches. 

## 🧑‍💻 Replicate the baseline

```shell
git clone https://github.com/DariaKern/Chicks4FreeID
cd Chicks4FreeID
pip install requirements.txt
python run_baseline.py
```

You can pass different options, depending on your hardware configuration

```shell
python run_baseline.py --devices=4 --batch-size-per-device=128 
```

For a full list of arguments type

```shell
python run_baseline.py --help
```

In a sepearte shell, open tensorboard to view progress and results

```shell
tensorboard --logdir baseline_logs
```

> [!NOTE]
> Differnt low-level accelerator implementations (TPU, MPS, CUDA) yield different results. The original hardware config for the reported results is based on the MPS implementation accessible on a 64GB Apple M3 Max chip (2023) 💻 - it is recommened to run the baseline script with at least 64GB of VRAM / Shared RAM. On this device, one run takes around `9:30h`


## ⏳ Timeline
- [2024/05/30] DOI created: [https://doi.org/10.57967/hf/2345](https://doi.org/10.57967/hf/2345) 
- [2024/05/23] the first version of the dataset was uploaded to Hugging Face. [https://huggingface.co/datasets/dariakern/Chicks4FreeID](https://huggingface.co/datasets/dariakern/Chicks4FreeID)

## 📝 Papers and systems citing the Chicks4FreeID dataset
coming soon ...

## 🖋️ Citation 
```tex
@misc{kern2024Chicks4FreeID,
      title={Chicks4freeID: A Benchmark Dataset for Chicken Re-Identification}, 
      author={Daria Kern and Tobias Schiele and Ulrich Klauck and Winfred Ingabire},
      year={2024},
      doi={https://doi.org/10.57967/hf/2345},
      note={in preparation for NeurIPS 2024 Datasets and Benchmarks Track}
}
```
