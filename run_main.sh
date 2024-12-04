# output_dir="cora_fixed_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_cluster"
# python main.py --dataset cora --experiment fixed --output_dir $output_dir --exp_setup Gc_train_2_Gs_infer --cluster_node True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --runs 20

# output_dir="chameleon_Gs_train_2_Gs_infer_0.3_kron_extra"
# python main.py --dataset chameleon --output_dir $output_dir --exp_setup Gs_train_2_Gs_infer --extra_node True --coarsening_method kron --coarsening_ratio 0.3 --runs 1

# output_dir="AIDS_Gc_train_2_Gs_train_0.7_algebraic_JC_cluster"
# python main.py --dataset AIDS --output_dir $output_dir --exp_setup Gc_train_2_Gs_train --cluster_node True --coarsening_method algebraic_JC --coarsening_ratio 0.7 --epochs2 1

# output_dir="QM9_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_extra_6"
# python main.py --dataset QM9 --output_dir $output_dir --exp_setup Gc_train_2_Gs_infer --extra_node True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --multi_prop True --property 6

output_dir="cora_fixed_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_cluster"
python main.py --dataset cora --experiment fixed --output_dir $output_dir --exp_setup Gc_train_2_Gs_infer --cluster_node True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --runs 1 --epochs1 1 --epochs2 1

output_dir="chameleon_Gs_train_2_Gs_infer_0.3_kron_extra"
python main.py --dataset chameleon --output_dir $output_dir --exp_setup Gs_train_2_Gs_infer --extra_node True --coarsening_method kron --coarsening_ratio 0.3 --runs 1 --epochs1 1 --epochs2 1

output_dir="AIDS_Gc_train_2_Gs_train_0.7_algebraic_JC_cluster"
python main.py --dataset AIDS --output_dir $output_dir --exp_setup Gc_train_2_Gs_train --cluster_node True --coarsening_method algebraic_JC --coarsening_ratio 0.7 --epochs1 1 --epochs2 1

output_dir="ZINC_subset_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_extra"
python main.py --dataset ZINC_subset --output_dir $output_dir --exp_setup Gc_train_2_Gs_infer --extra_node True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --epochs1 1 --epochs2 1

