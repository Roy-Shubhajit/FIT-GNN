#!/bin/bash

# Check if CUDA device is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <CUDA_VISIBLE_DEVICES>"
  exit 1
fi

contains() {
  local var="$1"
  shift
  local list=("$@")
  for item in "${list[@]}"; do
    if [[ "$item" == "$var" ]]; then
      return 0
    fi
  done
  return 1
}

# Set CUDA device
CUDA_VISIBLE_DEVICES=$1

# Number of epochs
NUM_EPOCHS=1000  # or any number you want to set, modify here or pass as an argument

# Arrays for datasets, target fractions, and models
datasets=("cora" "citeseer" "pubmed" "flickr" "ogbn-arxiv" "reddit")
target_fractions=(0.005 0.01 0.03)
models=("GCN" "GAT" "GIN")
#debug="-m pdb -c cont"
#debug="-m pdb"
debug=""

# Directory to save error logs
mkdir -p error_logs

# Loop through each combination of dataset, target fraction, and model
for ds in "${datasets[@]}"; do
  for tf in "${target_fractions[@]}"; do
    for model in "${models[@]}"; do
      # echo "Running model $model on dataset $ds with target fraction $tf"
      
      # Run each Python script and save errors in the corresponding log file
      for script in train_gdem.py train_gcond.py; do
        echo "Running $script with $model on $ds and tf=$tf"
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $debug $script -ds "$ds" -tf "$tf" -ne "$NUM_EPOCHS" -model "$model" #2>> error_logs/${script}_${ds}_${tf}_${model}_error.log
      done

      # Run train_gcsr.py only for flickr-0.005
      script="train_gcsr.py"
      if contains "$ds" cora citeseer pubmed flickr; then
        if [[ "$ds" != "flickr" ]]; then
          echo "Running $script with $model on $ds and tf=$tf"
          CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $debug $script -ds "$ds" -tf "$tf" -ne "$NUM_EPOCHS" -model "$model" #2>> error_logs/${script}_${ds}_${tf}_${model}_error.log
        else
          if [[ "$ds" == "flickr" && "$tf" == "0.005" ]]; then
            echo "Running $script with $model on $ds and tf=$tf"
            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $debug $script -ds "$ds" -tf "$tf" -ne "$NUM_EPOCHS" -model "$model" #2>> error_logs/${script}_${ds}_${tf}_${model}_error.log
          else
            echo "Skipping ${script} for dataset=$ds and target_fraction=$tf"
          fi
        fi
      else
        echo "Skipping ${script} for dataset=$ds and target_fraction=$tf"
      fi
    done
  done
done
