# I have both sparse adj and edge_index available.
import numpy as np
import scipy.sparse as sp

import torch
import torch_sparse
from torch_sparse import SparseTensor


@torch.no_grad()
def one_wl_forward(x, adj):
    if isinstance(adj, torch_sparse.SparseTensor):
        output = torch_sparse.matmul(adj, x)
    else:
        output = torch.spmm(adj, x)
    return output


def compute_wl_representations(x, adj):
    adj = adj.tolil()
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    mask = rowsum == 0
    rowsum[mask] = 1
    r_inv = (1 / rowsum).flatten()
    r_mat_inv = sp.diags(r_inv)
    # r_inv = np.power(rowsum, -1 / 2).flatten()
    # r_inv[np.isinf(r_inv)] = 0.0
    # r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    adj = adj.dot(r_mat_inv)
    single_node_inds = np.nonzero(mask)[0]
    adj[single_node_inds, single_node_inds] = 1 # weighted_transition_matrix(G, q) in https://github.com/chens5/WL-distance/blob/main/utils/utils.py
    # Also possible to use a q like in https://github.com/chens5/WL-distance/blob/main/utils/utils.py
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
    return one_wl_forward(x, adj)
