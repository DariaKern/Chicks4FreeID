# 🐔 Chicks4FreeID
The very first publicly available chicken re-identification dataset
is available on 🤗 **Hugging Face**: [huggingface.co/datasets/dariakern/Chicks4FreeID](https://huggingface.co/datasets/dariakern/Chicks4FreeID)

<img src="./wiki/chickenDataset.png">


## 🤗 Usage

> ```shell
> pip install datasets
> ```

> ```python
> from datasets import load_dataset
> train_ds = load_dataset("dariakern/Chicks4FreeID", split="train")
> train_ds[0]
> ```

```python
{'crop': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=2630x2630 at 0x7AA95E7D1720>,
 'identity': 43}
```


> [!TIP]
> Find more information on how to work with 🤗  [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets/v2.19.0/index)


## 📊 Baseline

To establish a baseline on the dataset, we explore 3 approaches

1. We evaluate the SotA model in animal re-identification: MegaDescriptor-L384, a feature extractor, pre-trained on many species and identities.
2. We train a Swin-Transformer, in the same way it has been used to build the MegaDescriptorL384, but now on our own dataset.
3. We train a generic Vision Transformer (ViT-B/16) as a fully supervised baseline (not focused on usable embeddings)

We evaluate the embeddings using two methods:

1. A simple KNN as suggested in the [InstDist](https://www.semanticscholar.org/paper/Unsupervised-Feature-Learning-via-Non-parametric-Wu-Xiong/155b7782dbd713982a4133df3aee7adfd0b6b304) paper
2. A generic linear classifier, as suggested in the SimCLR paper

Below are the metrics for the test set:

| Setting                            | mAP knn                   | mAP linear                   | top1 knn                   | top1 linear                   | top5 knn                   | top5 linear                   |
|:-----------------------------------|:--------------------------|:-----------------------------|:---------------------------|:------------------------------|:---------------------------|:------------------------------|
| MegaDescriptor-L384 (Frozen)        | 0.664 ± 0.040             | 0.935 ± 0.006                | 0.717 ± 0.026              | 0.879 ± 0.006                 | 0.923 ± 0.033              | 0.984 ± 0.004                 |
| SwinL384                   | 0.850 ± 0.083             | 0.962 ± 0.031                | 0.890 ± 0.054              | **0.923** ± 0.059             | 0.978 ± 0.004              | **0.988** ± 0.016             |
| ViT-B/16                  | **0.893** ± 0.013         | **0.975** ± 0.008            | **0.923** ± 0.006          | 0.928 ± 0.003                 | **0.980** ± 0.020          | 0.987 ± 0.012                 |

The most interesting observation in this table is that, even though the MegaDescriptorL384 feature extractor has never seen our dataset, it's still doing okay in identifying the chickens, compared to the fully supervised approaches. 

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

> [!IMPORTANT]
> Differnt low-level accelerator implementations (TPU, MPS, CUDA) yield different results. The original hardware config for the reported results is based on the MPS implementation accessible on a 64GB Apple M3 Max chip (2023) 💻 - it is recommened to run the baseline script with at least 64GB of VRAM / Shared RAM.


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
