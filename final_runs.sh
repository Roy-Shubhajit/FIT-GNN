exp_setup="Gc_train_2_Gs_infer"
dataset="AIDS"
output_dir=graph_cls_final
task=graph_cls
hidden=512
num_layers1=2
num_layers2=2
epochs1=100
epochs2=300
early_stopping=10
weight_decay=0.0005
normalize_features=True
task=graph_cls
underscore="_"
cluster="cluster"
extra="extra"
for cluster_node in True
do
    for coarsening_method in variation_neighborhoods algebraic_JC affinity_GS kron
    do
        for coarsening_ratio in 0.1 0.5 0.9
        do
            for batch_size in 2 16 32 128
            do
                for lr in 0.001 0.01 0.1
                do
                output_dir=$dataset$underscore$exp_setup$underscore$coarsening_ratio$underscore$coarsening_method$underscore$cluster$underscore$batch_size$underscore$lr
                python main.py --dataset $dataset --output_dir $output_dir --exp_setup $exp_setup --cluster_node $cluster_node --coarsening_method $coarsening_method --coarsening_ratio $coarsening_ratio --task $task --hidden $hidden --num_layers1 $num_layers1 --num_layers2 $num_layers2 --epochs1 $epochs1 --epochs2 $epochs2 --early_stopping $early_stopping --weight_decay $weight_decay --normalize_features $normalize_features --batch_size $batch_size --lr $lr
                done
            done
        done
    done
done

for coarsening_method in variation_neighborhoods algebraic_JC affinity_GS kron
do
    for coarsening_ratio in 0.1 0.5 0.9
    do
        for batch_size in 2 16 32 128
        do
            for lr in 0.001 0.01 0.1
            do
            output_dir=$dataset$underscore$exp_setup$underscore$coarsening_ratio$underscore$coarsening_method$underscore$batch_size$underscore$lr
            python main.py --dataset $dataset --output_dir $output_dir --exp_setup $exp_setup --coarsening_method $coarsening_method --coarsening_ratio $coarsening_ratio --task $task --hidden $hidden --num_layers1 $num_layers1 --num_layers2 $num_layers2 --epochs1 $epochs1 --epochs2 $epochs2 --early_stopping $early_stopping --weight_decay $weight_decay --normalize_features $normalize_features --batch_size $batch_size --lr $lr
            done
        done
    done
done


for extra_node in True
do
    for coarsening_method in variation_neighborhoods algebraic_JC affinity_GS kron
    do
        for coarsening_ratio in 0.1 0.5 0.9
        do
            for batch_size in 2 16 32 128
            do
                for lr in 0.001 0.01 0.1
                do
                output_dir=$dataset$underscore$exp_setup$underscore$coarsening_ratio$underscore$coarsening_method$underscore$extra$underscore$batch_size$underscore$lr
                python main.py --dataset $dataset --output_dir $output_dir --exp_setup $exp_setup --extra_node $extra_node --coarsening_method $coarsening_method --coarsening_ratio $coarsening_ratio --task $task --hidden $hidden --num_layers1 $num_layers1 --num_layers2 $num_layers2 --epochs1 $epochs1 --epochs2 $epochs2 --early_stopping $early_stopping --weight_decay $weight_decay --normalize_features $normalize_features --batch_size $batch_size --lr $lr
                done
            done
        done
    done
done


# python main.py --dataset ENZYMES --output_dir graph_cls_final --exp_setup Gs_train_2_Gs_infer
# python main.py --dataset ENZYMES --output_dir graph_cls_final --extra_node True --exp_setup Gs_train_2_Gs_infer
# python main.py --dataset ENZYMES --output_dir graph_cls_final --cluster_node True --exp_setup Gs_train_2_Gs_infer