r"""File containing GCN model classes.
"""

from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse

import torch_geometric.nn as gnn


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn"""

    def __init__(self, in_features, out_features, with_bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data.T)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, adj: Union[torch.Tensor, torch_sparse.SparseTensor]
    ) -> torch.Tensor:
        """Graph Convolutional Layer forward function"""
        if x.data.is_sparse:
            support = torch.spmm(x, self.weight)
        else:
            support = torch.mm(x, self.weight)

        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN_inductive(nn.Module):
    """A simple GCN"""

    def __init__(
        self,
        indim,
        outdim,
        hidim,
        nlayers=2,
        with_relu=True,
        with_bias=True,
        with_bn=False,
    ):
        super().__init__()
        self.with_relu = with_relu
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GraphConvolution(indim, outdim, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(hidim))
            self.layers.append(GraphConvolution(indim, hidim, with_bias=with_bias))
            for _ in range(nlayers - 2):
                self.layers.append(GraphConvolution(hidim, hidim, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(hidim))
            self.layers.append(GraphConvolution(hidim, outdim, with_bias=with_bias))

    def forward(
        self, x: torch.Tensor, adj: Union[torch.Tensor, torch_sparse.SparseTensor]
    ) -> torch.Tensor:
        r"""GCN forward function"""
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                if self.with_relu:
                    x = F.relu(x)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    """A simple 2 layered torch_geometric GCN"""

    def __init__(self, indim, outdim, *, hidim=128):
        super().__init__()
        self.l1 = gnn.GCNConv(indim, hidim)
        self.l2 = gnn.GCNConv(hidim, outdim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        r"""GCN forward function"""
        h = x
        h = F.relu(self.l1(h, edge_index))
        h = self.l2(h, edge_index)
        return h


class GAT(nn.Module):
    r"""A simple pytorch geometric GAT"""

    def __init__(self, indim, outdim, *, hidim=128):
        super().__init__()
        self.l1 = gnn.GATConv(indim, hidim)
        self.l2 = gnn.GATConv(hidim, outdim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        r"""GAT forward pass"""
        h = x
        h = F.relu(self.l1(h, edge_index))
        h = self.l2(h, edge_index)
        return h
    
class SAGE(nn.Module):
    r"""A simple pytorch geometric SAGE"""

    def __init__(self, indim, outdim, *, hidim=128):
        super().__init__()
        self.l1 = gnn.SAGEConv(indim, hidim)
        self.l2 = gnn.SAGEConv(hidim, outdim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        r"""SAGE forward pass"""
        h = x
        h = F.relu(self.l1(h, edge_index))
        h = self.l2(h, edge_index)
        return h


class GIN(nn.Module):
    r"""A simple pytorch geometric GIN"""

    def __init__(self, indim, outdim, *, hidim=128):
        super().__init__()
        self.l1 = gnn.GINConv(nn.Linear(indim, hidim))
        self.l2 = gnn.GINConv(nn.Linear(hidim, outdim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        h = F.relu(self.l1(h, edge_index))
        h = self.l2(h, edge_index)
        return h
