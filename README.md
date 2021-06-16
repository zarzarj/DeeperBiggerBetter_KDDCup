# Code release for team DeeperBiggerBetter for MAG-240M in the OGB KDD Cup

## Installation requirements
```
ogb>=1.3.0
torch>=1.7.0
pytorch-lightning>=1.2.0
torch-geometric==master (pip install git+https://github.com/rusty1s/pytorch_geometric.git)
jupyterlab (for post-processing)
```

## Dataset

The `MAG240M-LSC` dataset will be automatically downloaded to the path denoted in `root.py`.
Please change its content accordingly if you want to download the dataset to a custom hard-drive or folder.
For each experiment, the test submission will be automatically saved in `./results/` after training is done.

Due to the file size of the `MAG240M-LSC` node feature matrix, training requires at least 256GB RAM.


### Training

For training the model on k gpus, run:

```bash
python rgnn.py --device=k --accelerator='ddp' --model=rgat --hidden_channels=2048 --precision=16 --scheduler=cosine --optimizer=radam --extra_mlp --train_set=train_val --author_labels
```

For evaluating the model on the best validation checkpoint and save prediction logits, run:

```bash
python rgnn.py --device=k --accelerator='ddp' --evaluate --save_eval_probs
```

## Performance

| R-GAT [6] | 70.48 | 69.49 | 12.3M | GeForce RTX 2080 Ti (11GB GPU) |
| Ours (2-layer) | 71.08 | - | 81.2M | NVIDIA V100 (32GB GPU) |
| Ours (3-layer) | 71.87 | - | 99.5M | NVIDIA RTX 6000 (48GB GPU) |

\* Test Accuracy is evaluated on the **hidden test set.**

## References
This code is heavily based on [1].

[1] Hu *et al.*: [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687)

[6] Schlichtkrull *et al.*: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)


