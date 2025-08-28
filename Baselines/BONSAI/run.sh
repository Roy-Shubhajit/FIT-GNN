#!/bin/bash
# run on linux with `savelog` installed or change the logfile name manually
gpu=$0
rotate_logs(){
  local folder="$1"
  local file="$2"
  N=5
  if [ -z "$folder" ]; then
    echo "Error: No folder specified"
    return 1
  fi
  if [ ! -d "$folder" ]; then
    echo "Error: The specified folder doesn't exist"
    return 1
  fi
  savelog -n -c "${N}" "${folder}/${file}"
  # assumes that you have savelog
}
datasets=("cora" "citeseer" "pubmed" "dblp" "Physics" "products") #"cora" "citeseer" "pubmed" "dblp" "Physics"
sizes=(0.5)
models=("GCN")
for dataset in ${datasets[@]}
do
  dsdir="logs/${dataset}"
  mkdir -p "${dsdir}"
  for size in ${sizes[@]}
  do
    logdir="${dsdir}/frac-${size}"
    mkdir -p "${logdir}"
    log="log"
    rotate_logs "${logdir}" "${log}"
    logfile="${logdir}/${log}"
    echo "Writing logs to ${logfile}"
    printf "${dataset}-${size}\n" | tee "${logfile}"
    t1="$(date +'%s.%N')"
    CUDA_VISIBLE_DEVICES="${gpu}" python3 -u main.py --target_size_frac "${size}" --model "${models[0]}" --dataset "${dataset}" --nepochs 100 --save 2> >(while read line; do echo "err: $line"; done >&1) > >(while read line; do echo "$line"; done >&1) | tee -a "${logfile}"
    t2="$(date +'%s.%N')"
    dur=$(echo "${t2}-${t1}" | bc)
    printf "It took %ss\n" "${dur}" | tee -a "${logfile}"
  done
done
