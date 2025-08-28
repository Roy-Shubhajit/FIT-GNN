import warnings
warnings.simplefilter("ignore")
import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net
import numpy as np
from utils import load_data, coarsening
import os
from torch_geometric.data import Data
from torch_geometric.profile import get_data_size
from tqdm import tqdm
import csv
print("\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_features, args.num_classes, candidate, C_list, Gc_list = coarsening(args.dataset, 1-args.coarsening_ratio, args.coarsening_method)

    data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = load_data(
            args.dataset, candidate, C_list, Gc_list, args.experiment)
    data = data.to(device)
    coarsen_features = coarsen_features.to(device)
    coarsen_train_labels = coarsen_train_labels.to(device)
    coarsen_train_mask = coarsen_train_mask.to(device)
    coarsen_val_labels = coarsen_val_labels.to(device)
    coarsen_val_mask = coarsen_val_mask.to(device)
    coarsen_edge = coarsen_edge.to(device)

    coarsen_data = Data(x=coarsen_features, edge_index=coarsen_edge, train_labels=coarsen_train_labels, train_mask=coarsen_train_mask, val_labels=coarsen_val_labels, val_mask=coarsen_val_mask)
    gc_size = get_data_size(coarsen_data)
    g_size = get_data_size(data)

    print(f"Dataset: {args.dataset}")
    print(f"Coarsening Ratio:  {args.coarsening_ratio}")
    print(f"coarsen graph size: {gc_size}")
    print(f"full graph size: {g_size}")
    print(f"Target Fraction: {gc_size/g_size}")
    row = [args.dataset, args.coarsening_ratio, gc_size, g_size, gc_size/g_size]

    # Check if file exists
    file_exists = os.path.isfile("size.csv")

    # Write to CSV
    with open("size.csv", mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Dataset', 'Coarsening Ratio', 'Gc_Size', 'G_Size', 'Fraction'])
        writer.writerow(row)

    

