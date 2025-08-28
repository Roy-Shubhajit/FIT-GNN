for dataset in cora citeseer pubmed dblp Physics
do
    for r in 0.1 0.3 0.5 0.7
    do
        python train_gcond_transduct.py --dataset ${dataset} --nlayers=2 --lr_feat=1e-4 --gpu_id=0 --lr_adj=1e-4 --r=${r} --seed=1 --epoch=300 --save=1 --hidden=512
    done
done