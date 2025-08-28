import torch
import torch_scatter
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool

class Classify_node(torch.nn.Module):
    def __init__(self, args):
        super(Classify_node, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        LayerClass = getattr(pyg_nn, args.layer_name, None)
        if args.layer_name != 'GINConv':
            self.conv.append(LayerClass(args.num_features, args.hidden))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(args.hidden, args.hidden))
        else:
            self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.num_features, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = self.lt1(x)
        return F.log_softmax(x, dim=1)
    
class Regress_node(torch.nn.Module):
    def __init__(self, args):
        super(Regress_node, self).__init__()
        self.num_layers = args.num_layers1
        LayerClass = getattr(pyg_nn, args.layer_name, None)
        self.conv = torch.nn.ModuleList()
        if args.layer_name != 'GINConv':
            self.conv.append(LayerClass(args.num_features, args.hidden))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(args.hidden, args.hidden))
        else:
            self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.num_features, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
        self.lt1 = torch.nn.Linear(args.hidden, 1)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = self.lt1(x)
        return x
    
class Classify_graph_gc(torch.nn.Module):
    def __init__(self, args):
        super(Classify_graph_gc, self).__init__()
        self.num_layers = args.num_layers1
        LayerClass = getattr(pyg_nn, args.layer_name, None)
        self.conv = torch.nn.ModuleList()
        if args.layer_name != 'GINConv':
            self.conv.append(LayerClass(args.num_features, args.hidden))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(args.hidden, args.hidden))
        else:
            self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.num_features, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, gc):
        x, edge_index, batch = gc.x, gc.edge_index, gc.batch
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = global_max_pool(x, batch)
        x = self.lt1(x)
        return F.softmax(x, dim=1)

class Classify_graph_gs(torch.nn.Module):
    def __init__(self, args):
        super(Classify_graph_gs, self).__init__()
        self.num_layers = args.num_layers1
        LayerClass = getattr(pyg_nn, args.layer_name, None)
        self.conv = torch.nn.ModuleList()
        if args.layer_name != 'GINConv':
            self.conv.append(LayerClass(args.num_features, args.hidden))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(args.hidden, args.hidden))
        else:
            self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.num_features, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
        self.lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, set_gs, batch_tensor):
        X_main = torch.tensor([]).to(batch_tensor.device)
        for gs in set_gs:
            X = torch.tensor([]).to(batch_tensor.device)
            for g in gs:
                g = g.to(batch_tensor.device)
                x, edge_index, mask = g.x, g.edge_index, g.mask
                for i in range(self.num_layers):
                    x = self.conv[i](x, edge_index)
                    x = F.elu(x)
                    x = F.dropout(x, training=self.training)
                X = torch.cat((X, x[mask]), 0)
            X_main = torch.cat((X_main, X), 0)
        x = global_max_pool(X_main, batch_tensor.type(torch.int64))
        x = self.lt1(x)
        if len(x.shape) == 1:
            return F.softmax(x)
        return F.softmax(x, dim=1)
    
class Regress_graph_gc(torch.nn.Module):
    def __init__(self, args):
        super(Regress_graph_gc, self).__init__()
        self.num_layers = args.num_layers1
        LayerClass = getattr(pyg_nn, args.layer_name, None)
        self.conv = torch.nn.ModuleList()
        if args.layer_name != 'GINConv':
            self.conv.append(LayerClass(args.num_features, args.hidden))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(args.hidden, args.hidden))
        else:
            self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.num_features, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
        self.lt1 = torch.nn.Linear(args.hidden, 1)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, gc):
        x, edge_index, batch = gc.x, gc.edge_index, gc.batch
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.lt1(x)
        return x

class Regress_graph_gs(torch.nn.Module):
    def __init__(self, args):
        super(Regress_graph_gs, self).__init__()
        self.num_layers = args.num_layers1
        LayerClass = getattr(pyg_nn, args.layer_name, None)
        self.conv = torch.nn.ModuleList()
        if args.layer_name != 'GINConv':
            self.conv.append(LayerClass(args.num_features, args.hidden))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(args.hidden, args.hidden))
        else:
            self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.num_features, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
            for i in range(self.num_layers - 1):
                self.conv.append(LayerClass(torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU()), train_eps=True))
        self.lt1 = torch.nn.Linear(args.hidden, 1)

    def reset_parameters(self):
        for module in self.conv:
            module.reset_parameters()
        self.lt1.reset_parameters()

    def forward(self, set_gs, batch_tensor):
        X_main = torch.tensor([]).to(batch_tensor.device)
        for gs in set_gs:
            X = torch.tensor([]).to(batch_tensor.device)
            for g in gs:
                g = g.to(batch_tensor.device)
                x, edge_index, mask = g.x.float(), g.edge_index, g.mask
                for i in range(self.num_layers):
                    x = self.conv[i](x, edge_index)
                    x = F.elu(x)
                    x = F.dropout(x, training=self.training)
                X = torch.cat((X, x[mask]), 0)
            X_main = torch.cat((X_main, X), 0)
        x = global_mean_pool(X_main, batch_tensor.type(torch.int64))
        x = self.lt1(x)
        return x    
