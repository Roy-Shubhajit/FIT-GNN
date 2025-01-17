#output_dir="cora_fixed_Gs_train_2_Gs_infer_0.1_variation_neighborhoods_cluster_1_0.01"
#python main.py --dataset cora --experiment fixed --output_dir $output_dir --exp_setup Gs_train_2_Gs_infer --cluster_node --coarsening_method variation_neighborhoods --coarsening_ratio 0.1 --runs 3 --lr 0.01 --batch_size 1 --epochs2 1

#output_dir="cora_fixed_Gs_train_2_Gs_infer_0.1_variation_neighborhoods_cluster_128_0.01"
#python main.py --dataset cora --experiment fixed --output_dir $output_dir --exp_setup Gs_train_2_Gs_infer --cluster_node --coarsening_method variation_neighborhoods --coarsening_ratio 0.1 --runs 3 --lr 0.01 --batch_size 128 --epochs2 1

# output_dir="chameleon_Gs_train_2_Gs_infer_0.3_kron_extra"
# python main.py --dataset chameleon --output_dir $output_dir --exp_setup Gs_train_2_Gs_infer --extra_node True --coarsening_method kron --coarsening_ratio 0.3 --runs 1

# output_dir="AIDS_Gc_train_2_Gs_train_0.7_algebraic_JC_cluster"
# python main.py --dataset AIDS --output_dir $output_dir --exp_setup Gc_train_2_Gs_train --cluster_node True --coarsening_method algebraic_JC --coarsening_ratio 0.7 --epochs2 1

# output_dir="QM9_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_extra_6"
# python main.py --dataset QM9 --output_dir $output_dir --exp_setup Gc_train_2_Gs_infer --extra_node True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --multi_prop True --property 6

# output_dir="cora_fixed_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_cluster"
# python main.py --dataset cora --experiment fixed --output_dir $output_dir --exp_setup Gc_train_2_Gs_infer --cluster_node True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --runs 1 --epochs1 1 --epochs2 1

# output_dir="chameleon_Gs_train_2_Gs_infer_0.3_kron_extra"
# python main.py --dataset chameleon --output_dir $output_dir --exp_setup Gs_train_2_Gs_infer --extra_node True --coarsening_method kron --coarsening_ratio 0.3 --runs 1 --epochs1 1 --epochs2 1

# output_dir="AIDS_Gc_train_2_Gc_infer_0.5_variation_neighborhoods_extra"
# python main.py --dataset AIDS --output_dir $output_dir --exp_setup Gc_train_2_Gc_infer --extra_node True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --epochs1 300 --epochs2 300 --lr 0.0001

# output_dir="ZINC_subset_Gc_train_2_Gc_infer_0.5_variation_neighborhoods_extra"
# python main.py --dataset ZINC_subset --output_dir $output_dir --exp_setup Gc_train_2_Gc_infer --extra_node True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --epochs1 300 --epochs2 300 --lr 0.0001

underscore="_"
cluster="cluster"
extra="extra"
MB="GD"
for dataset in ogbn-products #cora AIDS QM9 ZINC_subset chameleon 
do
    for coarsening_method in variation_neighborhoods 
    do
        for exp_setup in Gs_train_2_Gs_infer #Gc_train_2_Gs_infer Gs_train_2_Gs_infer Gc_train_2_Gs_train Gc_train_2_Gc_infer
        do
            for coarsening_ratio in 0.3 0.5
            do
                for lr in 0.01
                do
                    for loss_reduction in mean
                    do
                        for batch_size in 1
                        do
                            output_dir=$dataset$underscore$exp_setup$underscore$coarsening_ratio$underscore$coarsening_method$underscore$cluster$underscore$lr$underscore$batch_size$underscore$MB
                            python main.py --experiment random --dataset $dataset --output_dir $output_dir --exp_setup $exp_setup --cluster_node --coarsening_method $coarsening_method --coarsening_ratio $coarsening_ratio  --normalize_features --lr $lr --runs 1 --loss_reduction $loss_reduction --batch_size $batch_size --gradient_method $MB --epochs1 1 --epochs2 1
                            #output_dir=$dataset$underscore$exp_setup$underscore$coarsening_ratio$underscore$coarsening_method$underscore$extra$underscore$lr$underscore$batch_size$underscore$MB
                            #python main.py --dataset $dataset --output_dir $output_dir --exp_setup $exp_setup --extra_node --coarsening_method $coarsening_method --coarsening_ratio $coarsening_ratio  --normalize_features --lr $lr --runs 1 --loss_reduction $loss_reduction --batch_size $batch_size --gradient_method $MB --epochs1 2 --epochs2 2
                        done
                    done
                done
            done
        done
    done
done