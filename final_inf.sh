# python inference.py --dataset citeseer --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.3 --experiment fixed --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b baseline_citeseer_fixed.pt --path_gs save/node_cls/final_testing/citeseer_fixed_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_cluster/ --model_name_gs model.pt
# python inference.py --dataset citeseer --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.1 --experiment fixed --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b baseline_citeseer_fixed.pt --path_gs save/node_cls/final_testing/citeseer_fixed_Gc_train_2_Gs_infer_0.1_variation_neighborhoods_cluster/ --model_name_gs model.pt

# python inference.py --dataset pubmed --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.3 --experiment fixed --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b baseline_pubmed_fixed.pt --path_gs save/node_cls/final_testing/pubmed_fixed_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_cluster/ --model_name_gs model.pt
# python inference.py --dataset pubmed --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.1 --experiment fixed --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b baseline_pubmed_fixed.pt --path_gs save/node_cls/final_testing/pubmed_fixed_Gc_train_2_Gs_infer_0.1_variation_neighborhoods_cluster/ --model_name_gs model.pt

# python inference.py --dataset dblp --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.3 --experiment random --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b baseline_dblp_random.pt --path_gs save/node_cls/final_testing/dblp_random_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_cluster/ --model_name_gs model.pt
# python inference.py --dataset dblp --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.1 --experiment random --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b baseline_dblp_random.pt --path_gs save/node_cls/final_testing/dblp_random_Gc_train_2_Gs_infer_0.1_variation_neighborhoods_cluster/ --model_name_gs model.pt

# python inference.py --dataset Physics --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.3 --experiment random --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b baseline_Physics_random.pt --path_gs save/node_cls/final_testing/Physics_random_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_cluster/ --model_name_gs model.pt
# python inference.py --dataset Physics --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.1 --experiment random --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b baseline_Physics_random.pt --path_gs save/node_cls/final_testing/Physics_random_Gc_train_2_Gs_infer_0.1_variation_neighborhoods_cluster/ --model_name_gs model.pt

# python inference.py --dataset PROTEINS --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.3 --exp_setup Gs_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.0001 --path_b final_models/ --model_name_b baseline_PROTEINS_batch_128_lr_0.0001.pt --path_gs final_models/model_folder/PROTEINS_Gs_train_2_Gs_infer_0.3_variation_neighborhoods_cluster_128_0.0001/ --model_name_gs model.pt
# python inference.py --dataset AIDS --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.3 --exp_setup Gc_train_2_Gc_infer --cluster_node True --batch_size 128 --lr 0.0001 --path_b final_models/NEWAIDS_Gc_train_2_Gc_infer_0.3_variation_neighborhoods_cluster_128_0.0001/ --model_name_b model.pt --path_gs final_models/NEWAIDS_Gc_train_2_Gc_infer_0.3_variation_neighborhoods_cluster_128_0.0001/ --model_name_gs model.pt
# python inference.py --dataset QM9 --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.3 --exp_setup Gc_train_2_Gs_infer --extra_node True --batch_size 128 --lr 0.001 --path_b final_models/model_folder/QM9_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_extra_128_0.001_6/ --model_name_b model.pt --path_gs final_models/model_folder/QM9_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_extra_128_0.001_6/ --model_name_gs model.pt --property 6
# python inference.py --dataset QM9 --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --exp_setup Gc_train_2_Gs_infer --extra_node True --batch_size 128 --lr 0.001 --path_b final_models/model_folder/QM9_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_extra_128_0.001_6/ --model_name_b model.pt --path_gs final_models/model_folder/QM9_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_extra_128_0.001_6/ --model_name_gs model.pt --property 6
python inference.py --dataset ZINC --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.3 --exp_setup Gs_train_2_Gs_infer --extra_node True --batch_size 128 --lr 0.0001 --path_b final_models/model_folder/ZINC_subset_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_extra_128_0.0001/ --model_name_b model.pt --path_gs final_models/model_folder/ZINC_subset_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_extra_128_0.0001/ --model_name_gs model.pt
python inference.py --dataset ZINC --num_test_samples 20 --baseline True --coarsening_method variation_neighborhoods --coarsening_ratio 0.5 --exp_setup Gs_train_2_Gs_infer --extra_node True --batch_size 128 --lr 0.0001 --path_b final_models/model_folder/ZINC_subset_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_extra_128_0.0001/ --model_name_b model.pt --path_gs final_models/model_folder/ZINC_subset_Gc_train_2_Gs_infer_0.3_variation_neighborhoods_extra_128_0.0001/ --model_name_gs model.pt


# # node_cls
# underscore="_"
# exp_setup="Gc_train_2_Gs_infer"
# cluster="cluster"
# baseline="baseline"
# coarsening_method="variation_neighborhoods"


# for dataset in citeseer pubmed dblp Physics
# do
#     for coarsening_ratio in 0.1 0.3
#     do
#         if [ $dataset == "Physics" -o $dataset == "dblp" ]
#         then
#             for experiment in random
#             do
#                 python inference.py --dataset $dataset --num_test_samples 20 --coarsening_method variation_neighborhoods --coarsening_ratio $coarsening_ratio --experiment $experiment --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b $baseline$underscore$dataset$underscore$experiment.pt --path_gs save/node_cls/final_testing/$dataset$underscore$experiment$underscore$exp_setup$underscore$coarsening_ratio$underscore$coarsening_method$underscore$cluster/ --model_name_gs model.pt
#             done
#         else
#             for experiment in fixed
#             do
#                 python inference.py --dataset $dataset --num_test_samples 20 --coarsening_method variation_neighborhoods --coarsening_ratio $coarsening_ratio --experiment $experiment --exp_setup Gc_train_2_Gs_infer --cluster_node True --batch_size 128 --lr 0.01 --path_b save/node_cls/baselines/ --model_name_b $baseline$underscore$dataset$underscore$experiment.pt --path_gs save/node_cls/final_testing/$dataset$underscore$experiment$underscore$exp_setup$underscore$coarsening_ratio$underscore$coarsening_method$underscore$cluster/ --model_name_gs model.pt
#             done
#         fi
#     done
# done