# python inference.py --dataset cora --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --experiment fixed --exp_setup Gc_train_2_Gs_infer --cluster_node True --path_b save/node_cls/baselines/ --model_name_b baseline_cora_fixed.pt --path_gs save/node_cls/cora_fixed_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_cluster/ --model_name_gs model.pt

# python inference.py --dataset chameleon --num_test_samples 20 --baseline True --coarsening_method kron --coarsening_ratio 0.3 --exp_setup Gs_train_2_Gs_infer --extra_node True --path_b save/node_reg/baselines/ --model_name_b baseline_chameleon.pt --path_gs save/node_reg/chameleon_Gs_train_2_Gs_infer_0.3_kron_extra/ --model_name_gs model.pt

# python inference.py --dataset AIDS --num_test_samples 20 --baseline True --coarsening_method algebraic_JC --coarsening_ratio 0.7 --exp_setup Gc_train_2_Gs_train --cluster_node True --path_b save/graph_cls/baselines/ --model_name_b baseline_AIDS.pt --path_gs save/graph_cls/AIDS_Gc_train_2_Gs_train_0.7_algebraic_JC_cluster/ --model_name_gs model.pt

# python inference.py --dataset QM9 --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --exp_setup Gc_train_2_Gs_infer --extra_node True --multi_prop True --property 6 --path_b save/graph_reg/baselines/ --model_name_b baseline_QM9_6.pt --path_gs save/graph_reg/QM9_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_extra_6/ --model_name_gs model.pt


# python inference.py --dataset cora --num_test_samples 5 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --experiment fixed --exp_setup Gc_train_2_Gs_infer --cluster_node True --path_b save/node_cls/baselines/ --model_name_b baseline_cora_fixed.pt --path_gs save/node_cls/cora_fixed_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_cluster/ --model_name_gs model.pt

# python inference.py --dataset chameleon --num_test_samples 5 --baseline True --coarsening_method kron --coarsening_ratio 0.3 --exp_setup Gs_train_2_Gs_infer --extra_node True --path_b save/node_reg/baselines/ --model_name_b baseline_chameleon.pt --path_gs save/node_reg/chameleon_Gs_train_2_Gs_infer_0.3_kron_extra/ --model_name_gs model.pt

# python inference.py --dataset AIDS --num_test_samples 5 --baseline True --coarsening_method algebraic_JC --coarsening_ratio 0.7 --exp_setup Gc_train_2_Gs_train --cluster_node True --path_b save/graph_cls/baselines/ --model_name_b baseline_AIDS.pt --path_gs save/graph_cls/AIDS_Gc_train_2_Gs_train_0.7_algebraic_JC_cluster/ --model_name_gs model.pt

python inference.py --dataset ZINC_subset --num_test_samples 5 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --exp_setup Gc_train_2_Gs_infer --extra_node True --path_b save/graph_reg/baselines/ --model_name_b baseline_ZINC_subset.pt --path_gs save/graph_reg/ZINC_subset_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_extra/ --model_name_gs model.pt