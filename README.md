# Chicks4FreeID
The very first publicly available chicken re-identification dataset
COMING SOON

<img src="./wiki/chickenDataset.png">

## Papers and systems citing the Chicks4FreeID dataset
coming soon ...

## Replicate the benchmmark

```shell
git clone https://github.com/DariaKern/Chicks4FreeID
cd Chicks4FreeID
pip install requirements.txt
python run_benchmark.py
```

You can pass different options, depending on your hardware configuration

```shell
python run_benchmark.py --devices=4 --batch-size-per-device=128 
```

For a full list of arguments type

```shell
python run_benchmark.py --help
```

In a sepearte shell, open tensorboard to view progress and results

```shell
tensorboard --logdir benchmark_logs
```

> [!IMPORTANT]
> Differnt low-level accelerator implementations (TPU, MPS, CUDA) yield different results. The original hardware config for the results reported in the paper is based on the MPS implementation accessible on a 64GB Apple M3 Max chip (2023).


## Citation
```tex
@misc{kern2024Chicks4FreeID,
      title={Chicks4freeID: A Benchmark Dataset for Chicken Re-Identification}, 
      author={Daria Kern and Tobias Schiele and Ulrich Klauck and Anjali DeSilva and Winfred Ingabire},
      year={2024},
      note={in preparation for NeurIPS 2024 Datasets and Benchmarks Track}
}
```
