import json
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor


def load_saint_dataset(name, *, root="datasets"):
    with open(f"{root}/{name}/role.json", "r") as jsonfile:
        roles = json.load(jsonfile)
    train_nodes = roles["tr"]
    val_nodes = roles["va"]
    test_nodes = roles["te"]
    adj = sp.load_npz(f"{root}/{name}/adj_full.npz")
    if name == "ogbn-arxiv":
        adj = adj + adj.T
        adj[adj > 1] = 1
    adj = adj.tocoo()
    rows = adj.row
    cols = adj.col
    edge_index = np.stack((rows, cols), axis=0)
    feats = np.load(f"{root}/{name}/feats.npy")
    feats = feats.astype(np.float32)
    with open(f"{root}/{name}/class_map.json", "r") as jsonfile:
        class_map = json.load(jsonfile)
    num_nodes = feats.shape[0]
    ys = np.zeros((num_nodes,))
    for node, cls in class_map.items():
        ys[int(node)] = cls
    ys = ys.astype(np.int64)  # Long.
    nc = np.unique(ys).shape[0]
    train_mask = np.zeros((num_nodes,))
    train_mask[train_nodes] = 1
    train_mask = train_mask.astype(bool)
    val_mask = np.zeros((num_nodes,))
    val_mask[val_nodes] = 1
    val_mask = val_mask.astype(bool)
    test_mask = np.zeros((num_nodes,))
    test_mask[test_nodes] = 1
    test_mask = test_mask.astype(bool)
    #
    adj = adj.tolil()
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    adj = adj.dot(r_mat_inv)
    adj = adj.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(adj.row).unsqueeze(1)
    sparsecol = torch.LongTensor(adj.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(adj.data)
    adj = torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(adj.shape))
    adj = SparseTensor(
        row=adj._indices()[0],
        col=adj._indices()[1],
        value=adj._values(),
        sparse_sizes=adj.size(),
    )
    data = Data(
        x=torch.tensor(feats),
        edge_index=torch.tensor(edge_index).long(),
        y=torch.tensor(ys),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask),
        num_nodes=torch.tensor(num_nodes),
        num_classes=torch.tensor(nc),
        adj=adj,
    )
    return data
