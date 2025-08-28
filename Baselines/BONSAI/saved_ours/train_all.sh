#!/bin/bash

# Check if CUDA device is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <CUDA_VISIBLE_DEVICES>"
  exit 1
fi

# Set CUDA device
CUDA_VISIBLE_DEVICES=$1

# Number of epochs
NUM_EPOCHS=1000  # or any number you want to set, modify here or pass as an argument

# Arrays for datasets, target fractions, and models
#datasets=("flickr" "ogbn-arxiv")
#target_fractions=(0.005 0.01 0.03)
target_fractions=(0.5)
datasets=("cora" "citeseer" "pubmed" "dblp" "Physics" "products") #"cora" "citeseer" "pubmed" "dblp" "Physics"
#target_fractions=(0.1 0.3 0.5 0.7)
models=("GCN")
#debug="-m pdb -c cont"
#debug="-m pdb"
debug=""

# Directory to save error logs
mkdir -p error_logs

# Loop through each combination of dataset, target fraction, and model
for ds in "${datasets[@]}"; do
  for tf in "${target_fractions[@]}"; do
    for model in "${models[@]}"; do
      echo "Running model $model on dataset $ds with target fraction $tf"
      
      for script in train_bonsai.py; do
        echo "Running $script with $model on $ds and tf=$tf"
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $debug $script -ds "$ds" -tf "$tf" -ne "$NUM_EPOCHS" -model "$model" 2>> error_logs/${script}_${ds}_${tf}_${model}_error.log
      done
    done
  done
done
