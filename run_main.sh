underscore="_"
cluster="cluster"
extra="extra"
gradient_method="GD"
baseline="baseline"
for dataset in cora chameleon PROTEINS ZINC_subset
do
    for coarsening_method in variation_neighborhoods # kron algebraic_JC heavy_edge variation_edges variation_cliques
    do
        for exp_setup in Gs_train_2_Gs_infer #Gc_train_2_Gs_infer Gs_train_2_Gs_infer Gc_train_2_Gs_train Gc_train_2_Gc_infer
        do
            for coarsening_ratio in 0.3
            do
                for lr in 0.01
                do
                    for loss_reduction in mean
                    do
                        for batch_size in 128
                        do
                            # Eg. 1: To run baseline model
                            output_dir=$dataset$underscore$exp_setup$underscore$coarsening_ratio$underscore$coarsening_method$underscore$cluster$underscore$lr$underscore$batch_size$underscore$gradient_method$underscore$baseline
                            python main.py --experiment random --dataset $dataset --output_dir $output_dir --normalize_features --lr $lr --runs 20 --loss_reduction $loss_reduction --batch_size $batch_size --gradient_method $gradient_method --epochs1 300 --epochs2 300 --baseline

                            # Eg. 2: To run FIT-GNN model
                            output_dir=$dataset$underscore$exp_setup$underscore$coarsening_ratio$underscore$coarsening_method$underscore$extra$underscore$lr$underscore$batch_size$underscore$gradient_method
                            python main.py --dataset $dataset --output_dir $output_dir --exp_setup $exp_setup --extra_node --coarsening_method $coarsening_method --coarsening_ratio $coarsening_ratio  --normalize_features --lr $lr --runs 20 --loss_reduction $loss_reduction --batch_size $batch_size --gradient_method $gradient_method --epochs1 300 --epochs2 300 --train_fitgnn
                        done
                    done
                done
            done
        done
    done
done