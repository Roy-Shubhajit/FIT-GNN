import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool

class Classify_node(torch.nn.Module):
    def __init__(self, args):
        super(Classify_node, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
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
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
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
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
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
        return F.log_softmax(x, dim=1)

class Classify_graph_gs(torch.nn.Module):
    def __init__(self, args):
        super(Classify_graph_gs, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
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
            # temp = torch.max(X, 0)[0]  
            # if X_main.size()[0] == 0:
            #     X_main = temp          
            # else:
            #     X_main = torch.vstack((X_main, temp))
        x = global_max_pool(X_main, batch_tensor.type(torch.int64))
        x = self.lt1(x)
        if len(x.shape) == 1:
            return F.softmax(x)
        return F.softmax(x, dim=1)
    
class Regress_graph_gc(torch.nn.Module):
    def __init__(self, args):
        super(Regress_graph_gc, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
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
        x = global_max_pool(x, batch)
        x = self.lt1(x)
        return x

class Regress_graph_gs(torch.nn.Module):
    def __init__(self, args):
        super(Regress_graph_gs, self).__init__()
        self.num_layers = args.num_layers1
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(args.num_features, args.hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(args.hidden, args.hidden))
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
                x, edge_index, mask = g.x, g.edge_index, g.mask
                for i in range(self.num_layers):
                    x = self.conv[i](x, edge_index)
                    x = F.elu(x)
                    x = F.dropout(x, training=self.training)
                X = torch.cat((X, x[mask]), 0)
            X_main = torch.cat((X_main, X), 0)
        x = global_max_pool(X_main, batch_tensor.type(torch.int64))
        x = self.lt1(x)
        return x    
                  
class TransferNet(torch.nn.Module):
    def __init__(self, args, model1):
        super(TransferNet, self).__init__()
        self.num_layers = args.num_layers2
        self.conv = model1.conv
        self.new_lt1 = torch.nn.Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        self.new_lt1.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)    
            x = F.dropout(x, training=self.training)
        x = self.new_lt1(x)
        if len(x.shape) == 1:
            return F.log_softmax(x)
        return F.log_softmax(x, dim=1)