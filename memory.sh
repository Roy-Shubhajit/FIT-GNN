
method=variation_neighborhoods
for dataset in cora citeseer pubmed dblp Physics ogbn-products chameleon crocodile squirrel
    for r in 0.1 0.3 0.5 0.7
    do
        python3 memory.py --dataset $dataset --coarsening_method $method --coarsening_ratio $r --fitgnn --cluster_node
        python3 memory.py --dataset $dataset --coarsening_method $method --coarsening_ratio $r --fitgnn --extra_node
    done
done

for dataset in cora citeseer pubmed dblp Physics ogbn-products chameleon crocodile squirrel
do
    python3 memory.py --dataset $dataset
done