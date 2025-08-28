for dataset in AIDS PROTEINS
do
    for layer_name in GCNConv GATConv SAGEConv GINConv
    do
        for ipc in 1 10 50
        do
            python main.py --dataset ${dataset} --init real --gpu_id=0 --nconvs=2 --dis=mse --lr_adj=0.01 --lr_feat=0.01 --epochs=1 --eval_init=1 --net_norm=none --pool=mean --seed=0 --ipc=${ipc} --save=0 --layer_name=${layer_name} --use_max_nodes_size=0
        done
    done
done