# FIT-GNN: Faster Inference Time for GNNs that 'FIT' in Memory Using Coarsening
---

### Important File Locations:
- Training
    - Baseline: `main.py` with baseline set to true (alternatively `run_main.sh`)
    - Subgraph: `main.py` (alternatively `run_main.sh`)
- Inference: 
    - Baseline: `inference_baseline.py` (alternatively `run_inference_baseline.sh`)
    - Subgraph: `inference.py` (alternatively `run_inference.sh`)
- Saved Models: Models stored in the following directory: `./save/`

### Examples
Refer to the following `.sh` files for examples:
- Training (both subgraph and baseline): `run_main.sh`
- Inference for baselines: `run_inference_baseline.sh`
- Inference for subgraphs: `run_inference.sh`

### Dataset Info
Refer to the csv file: `dataset_info.csv`

### Requirements
```console
pip install -r requirements.txt
```

### Parameter Description:
- `dataset`:
    - Dataset name
        1) Node Classification: cora, citeseer, pubmed, dblp, Physics
        2) Node Regression: chameleon, squirrel, crocodile
        3) Graph Classification: ENZYMES, AIDS, PROTEINS
        4) Graph Regression: QM9, ZINC (subset)
- `experiment`: {fixed, random, few}
    - Parameter specific to Node Classification for splitting nodes into train, val and test sets.
        1) fixed: cora, citeseer, pubmed
        2) few: cora, citeseer, pubmed, dblp, Physics
        3) random: dblp, Physics
- `runs`: default = 20
    - Number of times to run node-level task
- `baseline`: default = True
    - To train the baseline model
- `train_fitgnn`: default = False
    - To train the FIT_GNN model
    - Note: If both `baseline` and `train_fitgnn` are set to be true, then `train_fitgnn` will be considered.
- `exp_setup`: {Gc_train_2_Gs_infer, Gs_tran_2_Gs_infer, Gc_train_2_Gs_train}
    - Type of experiment setup to run
        1) Gc_train_2_Gs_infer: Train and val on Gc >> Test on Gs
        2) Gs_train_2_Gs_infer: Train, val and test on Gs
        3) Gc_train_2_Gs_train: Train and val on Gc >> transfer learnt weights >> Train, val and test on Gs 
- `extra_node`: {True, False}
    - Boolean parameter to train model by incorporating extra nodes.
- `cluster_node`: {True, False}
    - Boolean parameter to train model by incorporating cluster nodes.
- `coarsening_ratio`: [0, 1]
    - Extent of coarsening, 0 implying fewer subgraphs created and more nodes in each subgraph while 1 indicating large number of subgraphs created and fewer number of nodes in each subgraph.
- `coarsening_method`: {variation_neighborhoods, algebraic_JC, affinity_GS, kron}
    - Method used to coarsen graphs into subgraphs.
- `output_dir`:
    - Directory to save best model.
- `task`: {node_cls, node_reg, graph_cls, graph_reg}
    - Type of node-level or graph-level task being performed.
- `multi_prop`: {True, False}
    - Boolean parameter specific to QM9 `dataset` for Node Regression task. Should be set to True while performing experiments using QM9, else False.
- `property`: {0, 1, ... , 18}
    - Parameter specific to QM9 `dataset` for Node Regression task. Should be given one of the 19 targets for prediction.
- `hidden`: default = 512
    - Number of nodes in hidden layers of GNN
- `epochs1`: default = 100
    - Parameter specific to Gc_train_2_Gs_infer `exp_setup`. Number of epochs to train on Gc.
- `epochs2`: default = 300
    - Parameter specific to Gs_train_2_Gs_infer `exp_setup`. Number of epochs to train on Gs.
- `num_layers1`: default = 2
    - Parameter specific to Gc_train_2_Gs_infer `exp_setup`. Number of layers in Gc training model.
- `num_layers2`: default = 2
    - Parameter specific to Gs_train_2_Gs_infer `exp_setup`. Number of layers in Gs training model.
- `train_ratio`: [0, 1], default = 0.3
    - Parameter specific to graph-level tasks. Ratio of graphs reserved for training to total number of graphs in dataset.
- `val_ratio`: [0, 1], default = 0.2
    - Parameter specific to graph-level tasks. Ratio of graphs reserved for validation to total number of graphs in dataset.
- `use_community_detection`: default = False
    - Leiden algorithm is used to detect the top k communities to construct a proxy graph of a large graph.
