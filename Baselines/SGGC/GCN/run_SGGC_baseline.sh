for dataset in cora citeseer pubmed dblp Physics
do
  for layer_name in GCNCov GATConv SAGEConv GINConv
  do
    for r in 0.1 0.3 0.5 0.7
    do
      python size.py --dataset $dataset --experiment random --coarsening_ratio $r
    done
  done
done
