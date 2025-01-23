# Final

python inference_baseline.py --experiment random --dataset cora --num_test_samples 10 --path_b ./save/node_cls/baseline/cora_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt

python inference_baseline.py --experiment random --dataset chameleon --num_test_samples 10 --path_b ./save/node_reg/baseline/chameleon_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt

python inference_baseline.py --experiment random --dataset PROTEINS --num_test_samples 10 --path_b ./save/graph_cls/baseline/PROTEINS_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt

python inference_baseline.py --experiment random --dataset ZINC_subset --num_test_samples 10 --path_b ./save/graph_reg/baseline/ZINC_subset_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt

# python inference.py --experiment random --dataset PROTEINS --num_test_samples 10 --path_b ./save/graph_cls/baseline/PROTEINS_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt --baseline

# python inference.py --dataset PROTEINS --num_test_samples 1000 --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --experiment random --exp_setup Gc_train_2_Gc_infer --extra_node --path_gc ./save/graph_cls/PROTEINS_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_extra_0.01_128_GD/ --model_name_gc model.pt --path_b ./save/graph_cls/baseline/PROTEINS_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt --baseline

# python inference.py --dataset AIDS --num_test_samples 2000 --coarsening_method variation_neighborhoods --coarsening_ratio 0.3 --experiment random --exp_setup Gc_train_2_Gc_infer --extra_node --path_gc ./save/graph_cls/AIDS_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_extra_0.01_128_GD/ --model_name_gc model.pt --path_b ./save/graph_cls/baseline/AIDS_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt --baseline

# python inference.py --dataset ZINC_subset --num_test_samples 5000 --coarsening_method variation_neighborhoods --coarsening_ratio 0.1 --experiment random --exp_setup Gc_train_2_Gc_infer --extra_node --path_gc ./save/graph_reg/ZINC_subset_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_extra_0.01_128_GD/ --model_name_gc model.pt --path_b ./save/graph_reg/baseline/ZINC_subset_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt --baseline