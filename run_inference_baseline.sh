# Final

python inference_baseline.py --experiment random --dataset cora --num_test_samples 10 --path_b ./save/node_cls/baseline/cora_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt

python inference_baseline.py --experiment random --dataset chameleon --num_test_samples 10 --path_b ./save/node_reg/baseline/chameleon_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt

python inference_baseline.py --experiment random --dataset PROTEINS --num_test_samples 10 --path_b ./save/graph_cls/baseline/PROTEINS_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt

python inference_baseline.py --experiment random --dataset ZINC_subset --num_test_samples 10 --path_b ./save/graph_reg/baseline/ZINC_subset_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_0.01_128_GD_baseline/ --model_name_b model.pt