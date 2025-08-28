import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv
import torch_geometric.nn as pyg_nn


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        if args.layer_name != 'GINConv':
          LayerClass = getattr(pyg_nn, args.layer_name, None)
          self.conv1 = LayerClass(args.num_features, args.hidden)
          self.conv2 = LayerClass(args.hidden, args.num_classes)
        else:
          self.conv1 = GINConv(torch.nn.Sequential(torch.nn.Linear(args.num_features, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True)
          self.conv2 = GINConv(torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.num_classes)), train_eps=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)