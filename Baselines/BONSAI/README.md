# Bonsai: Gradient-free Graph Distillation for Node Classification

This is the PyTorch implementation for Bonsai: Gradient-free Graph Distillation for Node Classification

# Download Datasets

For Cora, Citeseer, and Pubmed, the code will directly download them from PyTorch Geometric. For Flickr, Ogbn-arxiv, and Reddit, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). They are available on [this Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) provided by the GraphSAINT team. Download the files and unzip them to `datasets` at the root directory.

# Instructions

1. Please install dependencies from requirements.txt in `python==3.9.19`
2. You can directly run `bash run.sh 0`. Here, `0` is the index of the GPU to run on it. If no index is passed, the code runs on CPU. 
3. The outputs will be saved to `saved_ours` directory. It also contains `train_bonsai.py` script along with `train_all.sh` to train a `GCN`, `GAT`, and `GIN` on the saved outputs to get the results.

# Some fine-details

## Sampling for scalable Rev-k-NN computation

The code doesn't perform sampling by default, but it can be enabled by passing `0 < frac_to_sample < 1` to `repr_to_dist` function in `main.py` in Ln. 610.

## Dataset train/val/test split

We over-ride the default train/val/test split in Ln. 545-550 of `main.py` for all datasets. Specifically,  
```python3
train, idx_test = train_test_split(range(nnodes), test_size=0.2, random_state=42)
rng = np.random.RandomState(seed=0)
idx_train = rng.choice(train, size=int(0.7 * len(train)), replace=False)
idx_val = list(set(range(nnodes)) - set(idx_train).union(set(idx_test)))
splits = {"train": idx_train, "val": idx_val, "test": idx_test}
```

# MAG240M

Please refer to the `mag240m` directory's `README` for the instructions.

# How to cite this work

```
@inproceedings{
gupta2025bonsai,
title={Bonsai: Gradient-free Graph Condensation for Node Classification},
author={Mridul Gupta and Samyak Jain and Vansh Ramani and Hariprasad Kodamana and Sayan Ranu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=5x88lQ2MsH}
}
```
