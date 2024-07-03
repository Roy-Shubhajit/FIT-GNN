# CoPart-GNN
<!-- ## Accuracies for `extra_node` varying `exp_setup`
![avg_acc](plots/cora_ave_acc.png)
![top_10_acc](plots/cora_top_10_acc.png)

## Important files added/modified
./results/cora.csv<br>
./save/ -->

## Datasets:
### Graph Classification:
1) [TUDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.TUDataset.html#torch_geometric.datasets.TUDataset) [Dataset Names for PyG](https://chrsmrrs.github.io/datasets/docs/datasets/): Usable datasets for our task are the ones which have `+` symbol under `Node Labels` and `Node Attr.` Ex: AIDS, ENZYMES, PROTEINS
2) [NeuroGraphDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.NeuroGraphDataset.html#torch_geometric.datasets.NeuroGraphDataset): `name = HCPGender`
3) [LRGBDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.LRGBDataset.html#torch_geometric.datasets.LRGBDataset): `name = Peptides-func`

### Graph Regression:
1) [QM9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html#torch_geometric.datasets.QM9)
2) [NeuroGraphDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.NeuroGraphDataset.html#torch_geometric.datasets.NeuroGraphDataset): `name = HCPWM | HCPFI`
3) [LRGBDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.LRGBDataset.html#torch_geometric.datasets.LRGBDataset): `name = Peptides-struct`
4) [BrcaTcga](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.BrcaTcga.html#torch_geometric.datasets.BrcaTcga)

### Node Regression:
1) [WikipediaNetwork](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WikipediaNetwork.html#torch_geometric.datasets.WikipediaNetwork): `name = chameleon | crocodile | squirrel`
2) [Airports](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Airports.html#torch_geometric.datasets.Airports): `name = USA | Brazil | Europe`
