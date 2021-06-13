# Code release for team DeeperBiggerBetter for MAG-240M in the OGB KDD Cup

Please refer to **[OGB-LSC paper](https://arxiv.org/abs/2103.09430)** for the detailed setting.

## Installation requirements
```
ogb>=1.3.0
torch>=1.7.0
pytorch-lightning>=1.2.0
torch-geometric==master (pip install git+https://github.com/rusty1s/pytorch_geometric.git)
```

## Baseline models

The `MAG240M-LSC` dataset will be automatically downloaded to the path denoted in `root.py`.
Please change its content accordingly if you want to download the dataset to a custom hard-drive or folder.
For each experiment, the test submission will be automatically saved in `./results/` after training is done.

Due to the file size of the `MAG240M-LSC` node feature matrix, some scripts may require up to 256GB RAM.


### Training

For training the model on k gpus, run:

```bash
python rgnn.py --device=k --model=rgat
```

For evaluating the `Relational-GAT` model on best validation checkpoint, run:

```bash
python rgnn.py --device=k --model=rgat --evaluate
```

## Performance

| R-GAT | 70.48 | 69.49 | 12.3M | GeForce RTX 2080 Ti (11GB GPU) |
| Ours | 70.48 | 69.49 | 12.3M | GeForce RTX 2080 Ti (11GB GPU) |

\* Test Accuracy is evaluated on the **hidden test set.**

## References

[6] Schlichtkrull *et al.*: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
