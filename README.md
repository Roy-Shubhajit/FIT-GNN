# CoPart-GNN
<!-- ## Accuracies for `extra_node` varying `exp_setup`
![avg_acc](plots/cora_ave_acc.png)
![top_10_acc](plots/cora_top_10_acc.png)

## Important files added/modified
./results/cora.csv<br>
./save/ -->

## Resolved minor errors in `main.py`, `run.py` and `utils.py`.

## Errors:
### Node Classification
Terminal Input: python main.py --dataset cora --output_dir node_cls_final
Error: Weird loss_fn error. Not able to resolve this. 

### Graph Classification
Terminal Input: python main.py --dataset ENZYMES --output_dir graph_cls_final
Error: Shape mismatch in run.py (line 118): out = model(gc)

### Graph Regression
Terminal Input: python main.py --dataset QM9 --output_dir graph_reg_final
Error: Shape mismatch in run.py (line 118): out = model(gc)
