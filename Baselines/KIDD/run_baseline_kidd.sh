for dataset in PROTEINS AIDS;
do
  for layer_name in GCNConv GATConv SAGEConv GINConv;
  do
    for gpc in 1 10 50
    do
      python GKRRDistill.py --dataset $dataset --gpc $gpc --initialize RandomSplit --anm 4 --test_gnn_depth 2  --lrA 0 --lrX 0.001 --device 0 --hidden_dim 512 --layer_name $layer_name
    done
  done
done