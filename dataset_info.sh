for dataset in ogbn-products\(community\) # WikiCS dblp Physics cora citeseer pubmed chameleon squirrel crocodile PROTEINS AIDS QM9 ZINC
do
    python dataset.py --dataset "$dataset"
done
