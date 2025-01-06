# python save_graphs.py --dataset cora --extra_node False --cluster_node False --normalize_features True --coarsening_ratio 0.1 --coarsening_method variation_neighborhoods
# python save_graphs.py --dataset citeseer --extra_node False --cluster_node True --normalize_features True --coarsening_ratio 0.3 --coarsening_method algebraic_JC
# python save_graphs.py --dataset cora --extra_node True --cluster_node False --normalize_features True --coarsening_ratio 0.5 --coarsening_method kron
# python save_graphs.py --dataset chameleon --extra_node False --cluster_node True --normalize_features True --coarsening_ratio 0.7 --coarsening_method variation_neighborhoods
# python save_graphs.py --dataset ENZYMES --extra_node False --cluster_node True --normalize_features True --coarsening_ratio 0.9 --coarsening_method algebraic_JC

# python save_graphs.py --dataset ogbn-products --extra_node True --cluster_node False --normalize_features True --coarsening_ratio 0.5 --coarsening_method variation_neighborhoods
#python save_graphs.py --dataset ogbn-products --extra_node False --cluster_node True --normalize_features True --coarsening_ratio 0.5 --coarsening_method variation_neighborhoods
# python save_graphs.py --dataset ogbn-products --extra_node False --cluster_node False --normalize_features True --coarsening_ratio 0.5 --coarsening_method variation_neighborhoods

#python save_graphs.py --dataset random --extra_node False --cluster_node False --normalize_features True --coarsening_ratio 0.5 --coarsening_method variation_neighborhoods --num_random_nodes 10000

# for coarsening_method in variation_edges variation_cliques heavy_edge
# do
#     python save_graphs.py --dataset cora --normalize_features --coarsening_ratio 0.5 --coarsening_method $coarsening_method
# done

for coarsening_method in variation_edges variation_cliques heavy_edge variation_neighborhoods kron algebraic_JC 
do
    python save_graphs.py --dataset cora --normalize_features --coarsening_ratio 0.5 --coarsening_method $coarsening_method --cluster_node
done

# for coarsening_method in variation_edges variation_cliques heavy_edge
# do
#     python save_graphs.py --dataset cora --normalize_features --coarsening_ratio 0.5 --coarsening_method $coarsening_method --extra_node
# done

