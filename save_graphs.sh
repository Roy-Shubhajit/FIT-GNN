for coarsening_method in variation_edges variation_cliques heavy_edge variation_neighborhoods kron algebraic_JC 
do
    python save_graphs.py --dataset cora --normalize_features --coarsening_ratio 0.5 --coarsening_method $coarsening_method --cluster_node
done

