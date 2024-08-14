import torch
import argparse
from torch_geometric.datasets import WikipediaNetwork, TUDataset, Planetoid, Coauthor, CitationFull, QM9, ZINC
from utils import load_graph_data, coarsening_classification, coarsening_regression, coarsening_classification, coarsening_regression, load_data_classification, load_data_regression, colater 
from torch.utils.data import DataLoader as T_DataLoader
from network import Classify_graph_gs, Regress_graph_gs, Classify_node, Regress_node
from time import time
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool

class Classify_graph(torch.nn.Module):
    def __init__(self, num_layer, num_feature, num_hidden, num_classes):
        super(Classify_graph, self).__init__()
        self.num_layers = num_layer
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(num_feature, num_hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(num_hidden, num_hidden))
        self.lt1 = torch.nn.Linear(num_hidden, num_classes)

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

class Regress_graph(torch.nn.Module):
    def __init__(self, num_layer, num_feature, num_hidden):
        super(Regress_graph, self).__init__()
        self.num_layers = num_layer
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(num_feature, num_hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(num_hidden, num_hidden))
        self.lt1 = torch.nn.Linear(num_hidden, 1)

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
    
class Net1(torch.nn.Module):
    def __init__(self, num_features, hidden, num_layers, num_classes):
        super(Net1, self).__init__()
        self.num_layers = num_layers
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(num_features, hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(hidden, hidden))
        self.lt1 = torch.nn.Linear(hidden, num_classes)

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
        return F.log_softmax(x, dim = 1)

class Net2(torch.nn.Module):
    def __init__(self, num_features, hidden, num_layers):
        super(Net2, self).__init__()
        self.num_layers = num_layers
        self.conv = torch.nn.ModuleList()
        self.conv.append(GCNConv(num_features, hidden))
        for i in range(self.num_layers - 1):
            self.conv.append(GCNConv(hidden, hidden))
        self.lt1 = torch.nn.Linear(hidden, 1)

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

def arg_correction(args):
    if args.super_graph:    
        args.cluster_node = False
        args.extra_node = False
    elif args.cluster_node:
        args.extra_node = False
        args.super_graph = False
    elif args.extra_node:
        args.cluster_node = False
        args.super_graph = False
    return args

def process_dataset(args):
    # Node Classification
    if args.dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=args.dataset)
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    elif args.dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=args.dataset)
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    elif args.dataset == 'cora':
        dataset = Planetoid(root='./dataset', name=args.dataset)
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    elif args.dataset == 'citeseer':
        dataset = Planetoid(root='./dataset', name=args.dataset)
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    elif args.dataset == 'pubmed':
        dataset = Planetoid(root='./dataset', name=args.dataset)
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    #Node Regression
    elif args.dataset == 'chameleon':
        dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_reg'
    elif args.dataset == 'squirrel':
        dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_reg'
    elif args.dataset == 'crocodile':
        dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_reg'
    #Graph Classification
    elif args.dataset == 'ENZYMES':
        dataset = TUDataset(root='./dataset', name=args.dataset)
        for i in range(len(dataset)):
            if args.normalize_features:
                dataset[i].x = torch.nn.functional.normalize(dataset[i].x, p=1)
        args.task = 'graph_cls'
        args.num_classes = 6
        args.num_features = dataset[0].x.shape[1]
    elif args.dataset == 'PROTEINS':
        dataset = TUDataset(root='./dataset', name=args.dataset)
        for i in range(len(dataset)):
            if args.normalize_features:
                dataset[i].x = torch.nn.functional.normalize(dataset[i].x, p=1)
        args.task = 'graph_cls'
        args.num_features = dataset[0].x.shape[1]
        args.num_classes = 2
    elif args.dataset == 'AIDS':
        dataset = TUDataset(root='./dataset', name=args.dataset)
        for i in range(len(dataset)):
            if args.normalize_features:
                dataset[i].x = torch.nn.functional.normalize(dataset[i].x, p=1)
        args.task = 'graph_cls'
        args.num_classes = 2
        args.num_features = dataset[0].x.shape[1]
    #Graph Regression
    elif args.dataset == 'QM9':
        dataset = QM9(root='./dataset/QM9')
        for i in range(len(dataset)):
            if args.normalize_features:
                dataset[i].x = torch.nn.functional.normalize(dataset[i].x, p=1)
        args.task = 'graph_reg'
        args.num_features = dataset[0].x.shape[1]
    elif args.dataset == "ZINC":
        dataset = ZINC(root='./dataset', subset=True)
        args.task = 'graph_reg'
        args.num_features = dataset[0].x.shape[1]

    return dataset, args

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chameleon')
parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--exp_setup', type=str, default='Gc_train_2_Gs_infer') # 'Gc_train_2_Gs_infer', 'Gs_train_2_Gs_infer'
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--epochs1', type=int, default=100)
parser.add_argument('--epochs2', type=int, default=300)
parser.add_argument('--num_layers1', type=int, default=2)
parser.add_argument('--num_layers2', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_ratio', type=float, default=0.3)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--extra_node', type=bool, default=False)
parser.add_argument('--cluster_node', type=bool, default=False)
parser.add_argument('--super_graph', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--coarsening_ratio', type=float, default=0.5)
parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods') #'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'
parser.add_argument('--task', type = str, default = 'node_cls')         ### node_reg, graph_cls, graph_reg
parser.add_argument('--seed', type = int, default = None)               ### Seed for reproducibility
parser.add_argument('--property', type = int, default = 0)              ### Property for graph regression task
parser.add_argument('--num_test_samples', type = int, default = 20)     ### Number of test samples 
parser.add_argument('--path_b', type = str, default = "./final_models/") ### Path for baseline model
parser.add_argument('--model_name_b', type = str, default = "baseline_ENZYMES_batch_128_lr_0.001.pt") ### Baseline model name
parser.add_argument('--path_gs', type = str, default = "./save/graph_cls/ENZYMES_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_128_0.001/") ### Path for subgraph model
parser.add_argument('--model_name_gs', type = str, default = "model.pt") ### Subgraph model name
parser.add_argument('--baseline', type = bool, default = False)          ### If True, baseline model results will be saved
args = parser.parse_args()

args = arg_correction(args)
dataset, args = process_dataset(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

new_dataset = []

if args.task == "graph_cls":
    test_indices = np.random.choice(len(dataset), args.num_test_samples, replace=False)
    all_out_b = []
    all_label_b = []
    all_out_gs = []
    all_label_gs = []
    losses_gs = []
    losses_b = []
    times_b = []
    times_gs = []

    model_gs = Classify_graph_gs(args).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model_gs.load_state_dict(torch.load(args.path_gs + args.model_name_gs))
    model_gs.eval()

    model_b = Classify_graph(args.num_layers2, args.num_features, args.hidden, args.num_classes)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
    model_b.eval()
    num = 0
    for i in test_indices:
        try:
            args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, dataset[i], 1-args.coarsening_ratio, args.coarsening_method)
            Gc = load_graph_data(dataset[i], CLIST, GcLIST, candidate)
            Gs = subgraph_list
            new_dataset.append((dataset[i], Gc, Gs))
            num += 1
        except:
            continue

        colater_fn = colater()
        test_loader = T_DataLoader(new_dataset, batch_size=1, collate_fn=colater_fn)

        # Subgraph based model
        for batch in test_loader:
            set_gs = batch[1]
            y_ = batch[2].to(device).type(torch.long)
            batch_tensor = batch[3].to(device)
            t1 = time()
            out_gs = model_gs(set_gs, batch_tensor).to(device)
            t2 = time()
            times_gs.append(t2-t1)
            all_out_gs.append(out_gs.argmax().item())
            all_label_gs.append(y_.item())
            loss_gs = loss_fn(out_gs, y_)
            losses_gs.append(loss_gs.item())

        # Baseline model
        G = new_dataset[0][0]
        y = G.y.to(device).type(torch.long)
        t3 = time()
        out_b = model_b(G).to(device)
        t4 = time()
        times_b.append(t4-t3)
        all_out_b.append(out_b.argmax().item())
        all_label_b.append(y.item())
        loss_b = loss_fn(out_b, y)
        losses_b.append(loss_b.item())

        # print(f"\nSubgraph-Based Model:\nGround Truth: {y_.item()}\nPredicted: {out_gs.argmax().item()}\nOutput: {out_gs}\nLoss: {loss_gs.item()}\nTime: {t2-t1}s")
        # print(f"\nBaseline Model:\nGround Truth: {y.item()}\nPredicted: {out_b.argmax().item()}\nOutput: {out_b}\nLoss: {loss_b.item()}\nTime: {t4-t3}s")
    print(len(times_b), len(times_gs))

elif args.task == "graph_reg":
    test_indices = np.random.choice(len(dataset), args.num_test_samples, replace=False)
    losses_gs = []
    losses_b = []
    times_gs = []
    times_b = []
    
    model_gs = Regress_graph_gs(args).to(device)
    loss_fn = torch.nn.L1Loss().to(device)
    model_gs.load_state_dict(torch.load(args.path_gs + args.model_name_gs))
    model_gs.eval()

    model_b = Regress_graph(args.num_layers2, args.num_features, args.hidden)
    loss_fn = torch.nn.L1Loss().to(device)
    model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
    model_b.eval()
    num = 0
    for i in test_indices:    
        args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, dataset[i], 1-args.coarsening_ratio, args.coarsening_method)
        Gc = load_graph_data(dataset[i], CLIST, GcLIST, candidate)
        Gs = subgraph_list
        new_dataset.append((dataset[i], Gc, Gs))
        num += 1

        colater_fn = colater()
        test_loader = T_DataLoader(new_dataset, batch_size=1, collate_fn=colater_fn)

        # Subgraph based model
        for batch in test_loader:
            set_gs = batch[1]
            y_ = batch[2].to(device).type(torch.long)
            batch_tensor = batch[3].to(device)
            t1 = time()
            out_gs = model_gs(set_gs, batch_tensor).to(device)
            t2 = time()
            times_gs.append(t2-t1)
            loss_gs = loss_fn(out_gs, y_[:, args.property].view(-1, 1))
            losses_gs.append(loss_gs.item())
        
        # Baseline model
        G = new_dataset[0][0]
        y = G.y.to(device).type(torch.long)
        t3 = time()
        out_b = model_b(G).to(device)
        t4 = time()
        times_b.append(t4-t3)
        loss_b = loss_fn(out_b, y[:, args.property].view(-1, 1))
        losses_b.append(loss_b.item())
        
        print(f"Subgraph-Based Model:\nGround Truth: {y_[:, args.property].item()}\nPredicted: {out_gs.item()}\nOutput: {out_gs}\nLoss: {loss_gs.item()}\nTime: {t2-t1}s")
        print(f"\nBaseline Model:\nGround Truth: {y[:, args.property].item()}\nPredicted: {out_b.item()}\nOutput: {out_b}\nLoss: {loss_b.item()}\nTime: {t4-t3}s")

if args.task == "node_cls":
    args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
    args.num_classes, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, graphs = load_data_classification(args, dataset[0], candidate, C_list, Gc_list, args.experiment, subgraph_list)
    if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
    indices = []
    num = 0
    for i in np.random.permutation(len(graphs)):
        if num != args.num_test_samples:
            if args.cluster_node:
                index = np.random.choice(subgraph_list[i].orig_idx, 1)[0]
            elif args.extra_node:
                index = np.random.choice(subgraph_list[i].orig_idx[~np.isin(subgraph_list[i].orig_idx, subgraph_list[i].actual_ext)], 1)[0]
            j = subgraph_list[i].map_dict[index]
            indices.append((index, j, i))
            num += 1
        else:
            break
    losses_b = []
    losses_gs = []
    times_b = []
    times_gs = []
    all_label_b = []
    all_out_b = []
    all_label_gs = []
    all_out_gs = []

    # Subgraph based model
    model_gs = Net1(args.num_features, args.hidden, args.num_layers2, args.num_classes).to(device)
    loss_fn = torch.nn.NLLLoss().to(device)
    model_gs.load_state_dict(torch.load(args.path_gs + args.model_name_gs))
    model_gs.eval()

    for index, j, i in indices:
        x = graphs[i].x.to(device)
        y = graphs[i].y.to(device)
        edge_index = graphs[i].edge_index.to(device)
        print(f"Subgraph Model: {x.shape}, {y.shape}, {edge_index.shape}")
        t1 = time()
        out_gs = model_gs(x, edge_index).to(device)
        t2 = time()
        loss_gs = loss_fn(out_gs[j], y[j])
        all_label_gs.append(y[j].item())
        all_out_gs.append(out_gs[j].argmax().item())
        losses_gs.append(loss_gs.item())
        times_gs.append(t2 - t1)
        print(f"\nSubgraph-Based Model:\nGround Truth: {y[j]}\nPredicted: {out_gs[j].argmax().item()}\nOutput: {out_gs[j]}\nLoss: {loss_gs.item()}\nTime: {t2-t1}s\n")
    print(f"\nAverage time (subgraph): {np.mean(times_gs[1:])}\nAccuracy (subgraph): {np.sum(np.array(all_label_gs) == np.array(all_out_gs))}/{num}")
    
    # Baseline model
    model_b = Net1(dataset[0].x.shape[1], args.hidden, args.num_layers2, args.num_classes).to(device)
    loss_fn = torch.nn.NLLLoss().to(device)
    model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
    model_b.eval()
    for i in range(num):
        x_ = dataset[0].x.to(device)
        y_ = dataset[0].y.to(device)
        edge_index_ = dataset[0].edge_index.to(device)
        print(f"Baseline Model: {x_.shape}, {y_.shape}, {edge_index_.shape}")
        t3 = time()
        out_b = model_b(x_, edge_index_).to(device)
        t4 = time()
        loss_b = loss_fn(out_b[indices[i][0]], y_[indices[i][0]])
        all_label_b.append(y_[indices[i][0]].item())
        all_out_b.append(out_b[indices[i][0]].argmax().item())
        losses_b.append(loss_b.item())
        times_b.append(t4 - t3)
        print(f"\nBaseline Model:\nGround Truth: {y_[indices[i][0]]}\nPredicted: {out_b[indices[i][0]].argmax().item()}\nOutput: {out_b[indices[i][0]]}\nLoss: {loss_b.item()}\nTime: {t4 - t3}s\n")
    print(f"\nAverage time (baseline): {np.mean(times_b[1:])}\nAccuracy (baseline): {np.sum(np.array(all_label_b) == np.array(all_out_b))}/{num}")

    
    plt.figure()
    plt.title("Prediction Error")
    plt.plot(losses_b, label = "Baseline")
    plt.plot(losses_gs, label = "Subgraph")
    plt.legend()
    plt.savefig("prediction_error.png")

    plt.figure()
    plt.title("Time")
    plt.plot(times_b, label = "Baseline")
    plt.plot(times_gs, label = "Subgraph")
    plt.legend()
    plt.savefig("time.png")

elif args.task == "node_reg":
    args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
    graphs = load_data_regression(args, dataset[0], subgraph_list)

    indices = []
    num = 0
    for i in np.random.permutation(len(graphs)):
        if num != args.num_test_samples:
            if args.cluster_node:
                index = np.random.choice(subgraph_list[i].orig_idx, 1)[0]
            elif args.extra_node:
                index = np.random.choice(subgraph_list[i].orig_idx[~np.isin(subgraph_list[i].orig_idx, subgraph_list[i].actual_ext)], 1)[0]
            j = subgraph_list[i].map_dict[index]
            indices.append((index, j, i))
            num += 1
        else:
            break
    
    # Baseline model
    model_b = Regress_node(args).to(device)
    loss_fn = torch.nn.L1Loss().to(device)
    model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
    model_b.eval()
    losses_b = []
    times_b = []
    for i in range(num):
        x_ = dataset[0].x.to(device)
        y_ = dataset[0].y.to(device)
        edge_index_ = dataset[0].edge_index.to(device)
        t3 = time()
        out_b = model_b(x_, edge_index_).to(device)
        t4 = time()
        loss_b = loss_fn(out_b[indices[i][0]][0], y_[indices[i][0]])
        losses_b.append(loss_b.item())
        times_b.append(t4 - t3)
        print(f"\nBaseline Model:\nGround Truth: {y_[indices[i][0]]}\nPredicted: {out_b[indices[i][0]]}\nLoss: {loss_b.item()}\nTime: {t4 - t3}s\n")
    print(f"Average time (baseline): {np.mean(times_b[1:])}")

    # Subgraph based model
    model_gs = Regress_node(args).to(device)
    loss_fn = torch.nn.L1Loss().to(device)
    model_gs.load_state_dict(torch.load(args.path_gs + args.model_name_gs))
    model_gs.eval()
    losses_gs = []
    times_gs = []
    for index, j, i in indices:
        x = graphs[i].x.to(device)
        y = graphs[i].y.to(device)
        edge_index = graphs[i].edge_index.to(device)
        t1 = time()
        out_gs = model_gs(x, edge_index).to(device)
        t2 = time()
        loss_gs = loss_fn(out_gs[j][0], y[j])
        losses_gs.append(loss_gs.item())
        times_gs.append(t2 - t1)
        print(f"Subgraph-Based Model:\nGround Truth: {y[j]}\nPredicted: {out_gs[j]}\nLoss: {loss_gs.item()}\nTime: {t2-t1}s\n")
    print(f"Average time (subgraph): {np.mean(times_gs[1:])}")

    plt.figure()
    plt.title("Prediction Error")
    plt.plot(losses_b, label = "Baseline")
    plt.plot(losses_gs, label = "Subgraph")
    plt.legend()
    plt.savefig("prediction_error.png")

    plt.figure()
    plt.title("Time")
    plt.plot(times_b, label = "Baseline")
    plt.plot(times_gs, label = "Subgraph")
    plt.legend()
    plt.savefig("time.png")

file_path = f"final_results/{args.task}.csv"
with open(file_path, 'a') as f:
    if args.task == "node_cls":
        if args.baseline:
            f.write(f"{args.dataset},True,{args.experiment},None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{np.std(losses_b)},{np.sum(np.array(all_label_b) == np.array(all_out_b))/num}\n")
        f.write(f"{args.dataset},False,{args.experiment},{args.exp_setup},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gs[1:])},{np.mean(losses_gs)},{np.std(losses_gs)},{np.sum(np.array(all_label_gs) == np.array(all_out_gs))/num}\n")
    elif args.task == "node_reg":
        if args.baseline:
            f.write(f"{args.dataset},True,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{np.std(losses_b)}\n")
        f.write(f"{args.dataset},False,{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gs[1:])},{np.mean(losses_gs)},{np.std(losses_gs)}\n")
    elif args.task == 'graph_cls':
        if args.baseline:
            f.write(f"{args.dataset},True,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{np.std(losses_b)},{np.sum(np.array(all_label_b) == np.array(all_out_b))/num}\n")
        f.write(f"{args.dataset},False,{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gs[1:])},{np.mean(losses_gs)},{np.std(losses_gs)},{np.sum(np.array(all_label_gs) == np.array(all_out_gs))/num}\n")
    elif args.task == 'graph_reg':
        if args.baseline:
            f.write(f"{args.dataset},True,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{np.std(losses_b)}\n")
        f.write(f"{args.dataset},False,{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gs[1:])},{np.mean(losses_gs)},{np.std(losses_gs)}\n")
f.close()
