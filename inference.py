import warnings
warnings.simplefilter("ignore")
import os
import torch
import pickle
import argparse
from torch_geometric.datasets import WikipediaNetwork, TUDataset, Planetoid, Coauthor, CitationFull, QM9, ZINC, Flickr
from ogb.nodeproppred import PygNodePropPredDataset
from utils import load_graph_data, coarsening_classification, coarsening_regression, coarsening_classification, coarsening_regression, load_data_classification, load_data_regression, colater 
from torch.utils.data import DataLoader as T_DataLoader
from torch_geometric.data import DataLoader as G_DataLoader
from network import Classify_graph_gs, Regress_graph_gs, Classify_node, Regress_node, Classify_graph_gc, Regress_graph_gc
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        x, edge_index, batch = gc.x.float(), gc.edge_index, gc.batch
        for i in range(self.num_layers):
            x = self.conv[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, batch)
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
    if args.cluster_node:
        args.extra_node = False
    elif args.extra_node:
        args.cluster_node = False
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
    elif args.dataset == "Flickr":
        dataset = Flickr(root='./dataset')
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    elif args.dataset == "ogbn-products":
        dataset = PygNodePropPredDataset(name="ogbn-products", root='./dataset')
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
        dataset = ZINC(root='./dataset/ZINC', subset=False)
        args.task = 'graph_reg'
        args.num_features = dataset[0].x.shape[1]
    elif args.dataset == "ZINC_subset":
        dataset = ZINC(root='./dataset/ZINC', subset=True)
        args.task = 'graph_reg'
        args.num_features = dataset[0].x.shape[1]
    return dataset, args

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--experiment', type=str, default='fixed')
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--exp_setup', type=str, default='Gc_train_2_Gs_infer')
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--epochs1', type=int, default=100)
parser.add_argument('--epochs2', type=int, default=300)
parser.add_argument('--num_layers1', type=int, default=2)
parser.add_argument('--num_layers2', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_ratio', type=float, default=0.3)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--extra_node', action='store_true')
parser.add_argument('--cluster_node', action='store_true')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--use_community_detection', action='store_true')
parser.add_argument('--normalize_features', action='store_true')
parser.add_argument('--coarsening_ratio', type=float, default=0.5)
parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
parser.add_argument('--task', type = str, default = 'node_cls')
parser.add_argument('--seed', type = int, default = None)
parser.add_argument('--multi_prop', action='store_true')
parser.add_argument('--property', type = int, default = 0)
parser.add_argument('--num_test_samples', type = int, default = 20)                                                                             ### Number of test samples 
parser.add_argument('--path_b', type = str, default = "./save/node_cls/baseline/")                                                             ### Path for baseline model
parser.add_argument('--model_name_b', type = str, default = "baseline_cora_fixed.pt")                                                           ### Baseline model name
parser.add_argument('--path_gs', type = str, default = "./save/node_cls/cora_fixed_Gc_train_2_Gs_infer_0.5_variation_neighborhoods_cluster/")   ### Path for FIT-GNN model
parser.add_argument('--model_name_gs', type = str, default = "model.pt")                                                                        ### FIT-GNN model name
parser.add_argument('--path_gc', type = str, default = "./save/node_cls/cora_Gc_train_2_Gc_infer_0.5_variation_neighborhoods_extra/")          ### Path for coarsened graph model
parser.add_argument('--model_name_gc', type = str, default = "model.pt")                                                                        ### Coarsened graph model name
parser.add_argument('--baseline', action='store_true')                                                                                 ### If True, baseline model results will be saved
args = parser.parse_args()

if(args.dataset == 'ogbn-products' and args.baseline == True):
    raise Exception("For the ogbn-products dataset, please run inference_baseline.py file for the baseline.")

args = arg_correction(args)
dataset, args = process_dataset(args)


if (args.task == 'node_cls' or args.task == 'node_reg') and dataset[0].num_nodes > 170000:
    args.use_community_detection = True

node_type = "d"
if args.extra_node:
    node_type = "e"
elif args.cluster_node:
    node_type = "c"
if args.use_community_detection:
    graph_type = "community"
else:
    graph_type = "full"

print("###############################################")
print("\nDataset: ", args.dataset)
print("Task: ", args.task)
print("Coarsening Method: ", args.coarsening_method)
print("Coarsening Ratio: ", args.coarsening_ratio)
print("Extra Node: ", args.extra_node)
print("Cluster Node: ", args.cluster_node)
print("Community:", args.use_community_detection)
print("\n###############################################")


if args.task == "graph_cls":
    all_out_b = []
    all_label_b = []
    all_out_gs = []
    all_label_gs = []
    losses_gs = []
    losses_b = []
    times_b = []
    times_gs = []
    all_out_gc = []
    all_label_gc = []
    losses_gc = []
    times_gc = []

    # FIT-GNN model
    if args.exp_setup == "Gc_train_2_Gc_infer":
        model_gc = Classify_graph_gc(args).to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        model_gc.load_state_dict(torch.load(args.path_gc + args.model_name_gc))
        model_gc.eval()
    else:
        model_gs = Classify_graph_gs(args).to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        model_gs.load_state_dict(torch.load(args.path_gs + args.model_name_gs))
        model_gs.eval()
    
    if args.baseline:
        model_b = Classify_graph_gc(args).to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
        model_b.eval()

    num = 0
    new_datasets = []
    if os.path.exists(f"./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt"):
        print("Loading saved graphs...")
        Gs_ = torch.load(f"./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt")
        Gc_ = pickle.load(open(f"./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl", "rb"))
        saved_graph_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_saved_graph_list.pkl', 'rb'))
        test_indices = np.random.choice(len(saved_graph_list), args.num_test_samples, replace=False)
        for i in test_indices:
            try:
                new_datasets.append([[dataset[saved_graph_list[i]], Gc_[i], Gs_[i]]])
                num += 1
            except:
                continue
        args.num_features = dataset[0].x.shape[1]
    else:
        for i in np.random.permutation(len(dataset)):
            if num != args.num_test_samples:
                graph = dataset[i].to(device)
                try:
                    args.num_features, candidate, subgraph_list, CLIST, GcLIST = coarsening_classification(args, graph, 1-args.coarsening_ratio, args.coarsening_method)
                    Gc = load_graph_data(graph, CLIST, GcLIST, candidate)
                    Gs = subgraph_list
                    new_datasets.append([[graph, Gc, Gs]])
                    num += 1
                except:
                    continue
            else:
                break

    for j in tqdm(range(num)):
        colater_fn = colater()
        test_loader = T_DataLoader(new_datasets[j], batch_size=1, collate_fn=colater_fn)
        test_loader_baseline = G_DataLoader([new_datasets[j][0][0]], batch_size=1)

        # FIT-GNN model
        if args.exp_setup == 'Gc_train_2_Gc_infer':
            # FIT-GNN Coaarsened Graph Based model
            set_gc = new_datasets[j][0][1].to(device)
            y_ = set_gc.y.to(device).type(torch.long)
            t1 = time()
            out_gc = model_gc(set_gc).to(device)
            t2 = time()
            times_gc.append(t2-t1)
            all_out_gc.append(out_gc.argmax().item())
            all_label_gc.append(y_.item())
            loss_gc = loss_fn(out_gc, y_)
            losses_gc.append(loss_gc.item())
        else:
            # FIT-GNN Subgraph Based model
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

        if args.baseline:
            # Baseline model
            for batch in test_loader_baseline:
                G = batch.to(device)
                y = G.y.to(device).type(torch.long)
                t3 = time()
                out_b = model_b(G)
                t4 = time()
                times_b.append(t4-t3)
                all_out_b.append(out_b.argmax().item())
                all_label_b.append(y.item())
                loss_b = loss_fn(out_b, y)
                losses_b.append(loss_b.item())

        # Remove set_gc, set_gs and y_ from device memory
        if args.exp_setup == "Gc_train_2_Gc_infer":
            set_gc = set_gc.cpu()
            y_ = y_.cpu()
            del set_gc, y_
        else:
            y_ = y_.cpu()
            del set_gs, y_
    
    # FIT-GNN
    if args.exp_setup == "Gc_train_2_Gc_infer":
        print(f"\nAverage time (FIT-GNN - coarsened graph): {np.mean(times_gc[1:])}\nAccuracy (FIT-GNN - coarsened graph): {np.sum(np.array(all_label_gc) == np.array(all_out_gc))}/{num}")
    else:
        print(f"\nAverage time (FIT-GNN - subgraph): {np.mean(times_gs[1:])}\nAccuracy (FIT-GNN - subgraph): {np.sum(np.array(all_label_gs) == np.array(all_out_gs))}/{num}")
    
    if args.baseline:
        print(f"\nAverage time (baseline): {np.mean(times_b[1:])}\nAccuracy (baseline): {np.sum(np.array(all_label_b) == np.array(all_out_b))}/{num}")

elif args.task == "graph_reg":
    
    losses_gs = []
    losses_b = []
    times_gs = []
    times_b = []
    losses_gc = []
    times_gc = []
    
    # FIT-GNN
    if args.exp_setup == "Gc_train_2_Gc_infer":
        model_gc = Regress_graph_gc(args).to(device)
        loss_fn = torch.nn.L1Loss().to(device)
        model_gc.load_state_dict(torch.load(args.path_gc + args.model_name_gc))
        model_gc.eval()
    else:
        model_gs = Regress_graph_gs(args).to(device)
        loss_fn = torch.nn.L1Loss().to(device)
        model_gs.load_state_dict(torch.load(args.path_gs + args.model_name_gs))
        model_gs.eval()

    if args.baseline:
        model_b = Regress_graph(args.num_layers2, args.num_features, args.hidden).to(device)
        loss_fn = torch.nn.L1Loss().to(device)
        model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
        model_b.eval()
    
    num = 0
    new_datasets = []

    if os.path.exists(f"./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt"):
        print("Loading saved graphs...")
        Gs_ = torch.load(f"./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt")
        Gc_ = pickle.load(open(f"./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl", "rb"))
        saved_graph_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_saved_graph_list.pkl', 'rb'))
        test_indices = np.random.choice(len(saved_graph_list), args.num_test_samples, replace=False)
        for i in test_indices:
            try:
                new_datasets.append([[dataset[saved_graph_list[i]], Gc_[i], Gs_[i]]])
                num += 1
            except:
                continue
        args.num_features = dataset[0].x.shape[1]
    else:
        for i in np.random.permutation(len(dataset)):
            if num != args.num_test_samples:
                graph = dataset[i].to(device)
                try:
                    args.num_features, candidate, subgraph_list, CLIST, GcLIST = coarsening_regression(args, graph, 1-args.coarsening_ratio, args.coarsening_method)
                    Gc = load_graph_data(graph, CLIST, GcLIST, candidate)
                    Gs = subgraph_list
                    new_datasets.append([[graph, Gc, Gs]])
                    num += 1
                except:
                    continue
            else:
                break
                
    for j in tqdm(range(num)):
        colater_fn = colater()
        test_loader = T_DataLoader(new_datasets[j], batch_size=1, collate_fn=colater_fn)
        test_loader_baseline = G_DataLoader([new_datasets[j][0][0]], batch_size=1)
        # FIT-GNN
        if args.exp_setup == "Gc_train_2_Gc_infer":
            # FIT-GNN - Coarsened Graph based model
            set_gc = new_datasets[j][0][1].to(device)
            y_ = new_datasets[j][0][0].y.to(device).type(torch.float)
            t1 = time()
            out_gc = model_gc(set_gc).to(device)
            t2 = time()
            times_gc.append(t2-t1)
            if args.dataset == 'QM9':
                loss_gc = loss_fn(out_gc, y_[:, args.property].view(-1, 1))
            else:
                loss_gc = loss_fn(out_gc, y_)
            losses_gc.append(loss_gc.item())
        else:
            
            # FIT-GNN - Subgraph based model
            for batch in test_loader:
                set_gs = batch[1]
                y_ = batch[2].to(device).type(torch.float)
                batch_tensor = batch[3].to(device)
                t1 = time()
                out_gs = model_gs(set_gs, batch_tensor).to(device)
                t2 = time()
                times_gs.append(t2-t1)
                if args.dataset == 'QM9':
                    loss_gs = loss_fn(out_gs, y_[:, args.property].view(-1, 1))
                else:
                    loss_gs = loss_fn(out_gs, y_)
                losses_gs.append(loss_gs.item())
        
        if args.baseline:
            # Baseline model
            for batch in test_loader_baseline:
                G = batch.to(device)
                y = G.y.to(device).type(torch.float)
                t3 = time()
                out_b = model_b(G)
                t4 = time()
                times_b.append(t4-t3)
                if args.dataset == 'QM9':
                    loss_b = loss_fn(out_b, y[:, args.property].view(-1, 1))
                else:
                    loss_b = loss_fn(out_b, y)
                losses_b.append(loss_b.item())
                
        if args.exp_setup == "Gc_train_2_Gc_infer":
            set_gc = set_gc.cpu()
            y_ = y_.cpu()
            del set_gc, y_
        else:
            y_ = y_.cpu()
            del set_gs, y_

    # FIT-GNN
    if args.exp_setup == "Gc_train_2_Gc_infer":
        print(f"\nAverage time (FIT-GNN - coarsened graph): {np.mean(times_gc[1:])}\nAverage Loss (FIT-GNN - coarsened graph): {np.mean(losses_gc)}")
    else:
        print(f"\nAverage time (FIT-GNN - subgraph): {np.mean(times_gs[1:])}\nAverage Loss (FIT-GNN - subgraph): {np.mean(losses_gs)}")

    if args.baseline:
        print(f"\nAverage time (baseline): {np.mean(times_b[1:])}\nAverage Loss (baseline): {np.mean(losses_b)}")

elif args.task == "node_cls":
    dataset = dataset.to(device)
    args.num_classes = torch.unique(dataset.y).shape[0]
    if os.path.exists(f"./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt"):
        print("Loading saved graphs...")
        subgraph_list = torch.load(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
        candidate = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_candidate.pkl', 'rb'))
        C_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_C_list.pkl', 'rb'))
        Gc_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl', 'rb'))
        args.num_features = dataset[0].x.shape[1]     
        if args.use_community_detection:
            del dataset
            torch.cuda.empty_cache()
            data = torch.load(f'./dataset/{args.dataset}/saved/{graph_type}_data.pt')
            dataset = [data]
    else:
        print("Coarsening graphs...")
        args.num_features, candidate, C_list, Gc_list, subgraph_list = coarsening_classification(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
    args.num_classes, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, graphs = load_data_classification(args, dataset[0], candidate, C_list, Gc_list, args.experiment, subgraph_list)
    if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
    indices = []
    true_indices = []
    false_indices = []
    num = 0
    test_num = args.num_test_samples
    no_graphs = len(graphs)
    rat = test_num // no_graphs

    if test_num > dataset[0].x.shape[0]:
        print("Number of test nodes exceeds the maximum number of nodes in the graph. Hence, all nodes will be used as test nodes.")

    permu = np.random.permutation(no_graphs)
    maskie = [False]*no_graphs

    if test_num <= no_graphs:
        for i in np.random.permutation(len(graphs)):
            if num != args.num_test_samples:
                if args.extra_node:
                    index = np.random.choice(subgraph_list[i].orig_idx.cpu()[~np.isin(subgraph_list[i].orig_idx.cpu(), subgraph_list[i].actual_ext.cpu())], 1)[0]
                else:
                    index = np.random.choice(subgraph_list[i].orig_idx.cpu(), 1)[0]
                j = subgraph_list[i].map_dict[index]
                indices.append((index, j, i))
                num += 1
            else:
                break
    else:
        for i in permu:
            if test_num > 0:
                if args.extra_node:
                    if len(subgraph_list[i].orig_idx.cpu()[~np.isin(subgraph_list[i].orig_idx.cpu(), subgraph_list[i].actual_ext.cpu())]) >= rat:
                        maskie[i] = True
                        test_num = test_num - rat
                else:
                    if len(subgraph_list[i].orig_idx.cpu()) >= rat:
                        maskie[i] = True
                        test_num = test_num - rat
            else:
                break

        for i in permu:
            if maskie[i] == True:
                if args.extra_node:
                    index = np.random.choice(subgraph_list[i].orig_idx.cpu()[~np.isin(subgraph_list[i].orig_idx.cpu(), subgraph_list[i].actual_ext.cpu())], rat)
                else:
                    index = np.random.choice(subgraph_list[i].orig_idx.cpu(), rat)
                true_indices.append((index, i)) # one test sample: index = original node index in community graph, j = renamed index of node in that subgraph, i = index of subgraph in subgraph list
            else:
                if test_num > 0:
                    if args.extra_node:
                        index = subgraph_list[i].orig_idx.cpu()[~np.isin(subgraph_list[i].orig_idx.cpu(), subgraph_list[i].actual_ext.cpu())]
                    else:
                        index = subgraph_list[i].orig_idx.cpu()
                    test_num = test_num - len(index)
                    for ind in index:
                        ind = ind.item()
                        indices.append((ind, subgraph_list[i].map_dict[ind], i)) # one test sample: index = original node index in community graph, j = renamed index of node in that subgraph, i = index of subgraph in subgraph list   
        
        for itr in true_indices:
            if test_num > 0:
                if args.extra_node:
                    index = subgraph_list[itr[1]].orig_idx.cpu()[~np.isin(subgraph_list[itr[1]].orig_idx.cpu(), subgraph_list[itr[1]].actual_ext.cpu()) & ~np.isin(subgraph_list[itr[1]].orig_idx.cpu(), itr[0])]
                else:
                    index = subgraph_list[itr[1]].orig_idx.cpu()[~np.isin(subgraph_list[itr[1]].orig_idx.cpu(), itr[0])]
                test_num = test_num - len(index)
                for ind in index:
                    ind = ind.item()
                    indices.append((ind, subgraph_list[itr[1]].map_dict[ind], itr[1]))
            
            for ind in itr[0]:
                ind = ind.item()
                indices.append((ind, subgraph_list[itr[1]].map_dict[ind], itr[1]))

    num = len(indices)

    losses_b = []
    losses_gs = []
    times_b = []
    times_gs = []
    all_label_b = []
    all_out_b = []
    all_label_gs = []
    all_out_gs = []
    
    if args.baseline:
        # Baseline model
        model_b = Net1(dataset[0].x.shape[1], args.hidden, args.num_layers2, args.num_classes).to(device)
        loss_fn = torch.nn.NLLLoss().to(device)
        model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
        model_b.eval()
        graph_data = G_DataLoader(dataset, batch_size=1)
        fig = plt.figure()
        for i in range(num):
            for graph in graph_data:
                graph = graph.to(device)
                t3 = time()
                out_b = model_b(graph.x, graph.edge_index).to(device)
                t4 = time()
                loss_b = loss_fn(out_b[indices[i][0]].reshape(1,-1), graph.y[indices[i][0]].reshape(1))
                all_label_b.append(graph.y[indices[i][0]].item())
                all_out_b.append(out_b[indices[i][0]].argmax().item())
                losses_b.append(loss_b.item())
                times_b.append(t4 - t3)
            
        print(f"Average time (baseline): {np.mean(times_b[1:])}\nAccuracy (baseline): {np.sum(np.array(all_label_b) == np.array(all_out_b))}/{num}")

    # FIT-GNN model
    model_gs = Net1(args.num_features, args.hidden, args.num_layers2, args.num_classes).to(device)
    loss_fn = torch.nn.NLLLoss().to(device)
    model_gs.load_state_dict(torch.load(args.path_gs + args.model_name_gs))
    model_gs.eval()
    for index, j, i in tqdm(indices, colour='blue'):
        x = graphs[i].x.to(device)
        y = graphs[i].y.to(device)
        edge_index = graphs[i].edge_index.to(device)
        t1 = time()
        out_gs = model_gs(x, edge_index).to(device)
        t2 = time()       
        loss_gs = loss_fn(out_gs[j].reshape(1,-1), y[j].reshape(1))
        all_label_gs.append(y[j].item())        
        all_out_gs.append(out_gs[j].argmax().item())
        losses_gs.append(loss_gs.item())
        times_gs.append(t2 - t1)
        x = x.cpu()
        y = y.cpu()
        edge_index = edge_index.cpu()
        del x, y, edge_index
    print(f"\nAverage time (FIT-GNN): {np.mean(times_gs[1:])}\nAccuracy (FIT-GNN): {np.sum(np.array(all_label_gs) == np.array(all_out_gs))}/{num}")
        

elif args.task == "node_reg":
    dataset = dataset.to(device)
    if os.path.exists(f"./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt"):
        print("Loading saved graphs...")
        subgraph_list = torch.load(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
        args.num_features = dataset[0].x.shape[1]
    else:
        args.num_features, subgraph_list = coarsening_regression(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
    graphs = load_data_regression(args, dataset[0], subgraph_list)

    indices = []
    true_indices = []
    false_indices = []
    num = 0
    test_num = args.num_test_samples
    no_graphs = len(graphs)
    rat = test_num // no_graphs
    
    if test_num > dataset[0].x.shape[0]:
        print("Number of test nodes exceeds the maximum number of nodes in the graph. Hence, all nodes will be used as test nodes.")

    permu = np.random.permutation(no_graphs)
    maskie = [False]*no_graphs

    if test_num <= no_graphs:
        for i in np.random.permutation(len(graphs)):
            if num != args.num_test_samples:
                if args.extra_node:
                    index = np.random.choice(subgraph_list[i].orig_idx.cpu()[~np.isin(subgraph_list[i].orig_idx.cpu(), subgraph_list[i].actual_ext.cpu())], 1)[0]
                else:
                    index = np.random.choice(subgraph_list[i].orig_idx.cpu(), 1)[0]
                j = subgraph_list[i].map_dict[index]
                indices.append((index, j, i))
                num += 1
            else:
                break
    else:
        for i in permu:
            if test_num > 0:
                if args.extra_node:
                    if len(subgraph_list[i].orig_idx.cpu()[~np.isin(subgraph_list[i].orig_idx.cpu(), subgraph_list[i].actual_ext.cpu())]) >= rat:
                        maskie[i] = True
                        test_num = test_num - rat
                else:
                    if len(subgraph_list[i].orig_idx.cpu()) >= rat:
                        maskie[i] = True
                        test_num = test_num - rat
            else:
                break

        for i in permu:
            if maskie[i] == True:
                if args.extra_node:
                    index = np.random.choice(subgraph_list[i].orig_idx.cpu()[~np.isin(subgraph_list[i].orig_idx.cpu(), subgraph_list[i].actual_ext.cpu())], rat)
                else:
                    index = np.random.choice(subgraph_list[i].orig_idx.cpu(), rat)
                true_indices.append((index, i)) # one test sample: index = original node index in community graph, j = renamed index of node in that subgraph, i = index of subgraph in subgraph list
            else:
                if test_num > 0:
                    if args.extra_node:
                        index = subgraph_list[i].orig_idx.cpu()[~np.isin(subgraph_list[i].orig_idx.cpu(), subgraph_list[i].actual_ext.cpu())]
                    else:
                        index = subgraph_list[i].orig_idx.cpu()
                    test_num = test_num - len(index)
                    for ind in index:
                        ind = ind.item()
                        indices.append((ind, subgraph_list[i].map_dict[ind], i)) # one test sample: index = original node index in community graph, j = renamed index of node in that subgraph, i = index of subgraph in subgraph list   
        
        for itr in true_indices:
            if test_num > 0:
                if args.extra_node:
                    index = subgraph_list[itr[1]].orig_idx.cpu()[~np.isin(subgraph_list[itr[1]].orig_idx.cpu(), subgraph_list[itr[1]].actual_ext.cpu()) & ~np.isin(subgraph_list[itr[1]].orig_idx.cpu(), itr[0])]
                else:
                    index = subgraph_list[itr[1]].orig_idx.cpu()[~np.isin(subgraph_list[itr[1]].orig_idx.cpu(), itr[0])]
                test_num = test_num - len(index)
                for ind in index:
                    ind = ind.item()
                    indices.append((ind, subgraph_list[itr[1]].map_dict[ind], itr[1]))
            
            for ind in itr[0]:
                ind = ind.item()
                indices.append((ind, subgraph_list[itr[1]].map_dict[ind], itr[1]))

    num = len(indices)
    del true_indices, false_indices, maskie, permu
    if args.baseline:
        # Baseline model
        model_b = Regress_node(args).to(device)
        loss_fn = torch.nn.L1Loss().to(device)
        model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
        model_b.eval()
        losses_b = []
        times_b = []
        y_b = []
        for i in range(num):
            x_ = dataset[0].x.to(device)
            y_ = dataset[0].y.to(device)
            edge_index_ = dataset[0].edge_index.to(device)
            t3 = time()
            out_b = model_b(x_, edge_index_).to(device)
            t4 = time()
            loss_b = loss_fn(out_b[indices[i][0]][0], y_[indices[i][0]])
            losses_b.append(loss_b.item())
            y_b.append(y_[indices[i][0]].item())
            times_b.append(t4 - t3)

    # FIT-GNN model
    model_gs = Regress_node(args).to(device)
    loss_fn = torch.nn.L1Loss().to(device)
    model_gs.load_state_dict(torch.load(args.path_gs + args.model_name_gs))
    model_gs.eval()
    losses_gs = []
    times_gs = []
    y_gs = []
    for index, j, i in tqdm(indices, colour='blue'):
        x = graphs[i].x.to(device)
        y = graphs[i].y.to(device)
        edge_index = graphs[i].edge_index.to(device)
        t1 = time()
        out_gs = model_gs(x, edge_index).to(device)
        t2 = time()
        loss_gs = loss_fn(out_gs[j][0], y[j])
        y_gs.append(y[j].item())
        losses_gs.append(loss_gs.item())
        times_gs.append(t2 - t1)
        x = x.cpu()
        y = y.cpu()
        edge_index = edge_index.cpu()
        del x, y, edge_index

    print(f"\nAverage time (FIT-GNN): {np.mean(times_gs[1:])}\nAverage Loss (FIT-GNN): {np.mean(losses_gs/np.std(y_gs))}")

    if args.baseline:
        print(f"Average time (baseline): {np.mean(times_b[1:])}\nAverage Loss (baseline): {np.mean(losses_b/np.std(y_b))}")

if not os.path.exists("inference_results"):
    os.makedirs("inference_results")

if not os.path.exists(f"inference_results/{args.task}.csv"):
    with open(f"inference_results/{args.task}.csv", 'w') as f:
        if args.task == "node_cls":
            f.write("dataset,baseline,experiment,exp_setup,coarsening_method,coarsening_ratio,extra_node,cluster_node,hidden,num_test_samples,num_layers,batch_size,lr,avg_inf_time,avg_loss,acc\n")
        elif args.task == "node_reg":
            f.write("dataset,baseline,exp_setup,coarsening_method,coarsening_ratio,extra_node,cluster_node,hidden,num_test_samples,num_layers,batch_size,lr,avg_inf_time,avg_loss\n")
        elif args.task == "graph_cls":
            f.write("dataset,baseline,exp_setup,coarsening_method,coarsening_ratio,extra_node,cluster_node,hidden,num_test_samples,num_layers,batch_size,lr,avg_inf_time,avg_loss,acc\n")
        elif args.task == "graph_reg":
            f.write("dataset,baseline,exp_setup,coarsening_method,coarsening_ratio,extra_node,cluster_node,hidden,num_test_samples,num_layers,batch_size,lr,avg_inf_time,avg_loss,property\n")
    f.close()         
file_path = f"inference_results/{args.task}.csv"

with open(file_path, 'a') as f:
    if args.task == "node_cls":
        if args.baseline:
            f.write(f"{args.dataset},True,{args.experiment},None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{np.sum(np.array(all_label_b) == np.array(all_out_b))/num}\n")
        f.write(f"{args.dataset},False,{args.experiment},{args.exp_setup},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gs[1:])},{np.mean(losses_gs)},{np.sum(np.array(all_label_gs) == np.array(all_out_gs))/num}\n")
    elif args.task == "node_reg":
        if args.baseline:
            f.write(f"{args.dataset},True,None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b/np.std(y_b))}\n")
        f.write(f"{args.dataset},False,{args.exp_setup},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gs[1:])},{np.mean(losses_gs)/np.std(y_gs)}\n")
    elif args.task == 'graph_cls':
        if args.baseline:
            f.write(f"{args.dataset},True,None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{np.sum(np.array(all_label_b) == np.array(all_out_b))/num}\n")
        if args.exp_setup == "Gc_train_2_Gc_infer":
            f.write(f"{args.dataset},False,{args.exp_setup},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gc[1:])},{np.mean(losses_gc)},{np.sum(np.array(all_label_gc) == np.array(all_out_gc))/num}\n")
        else:
            f.write(f"{args.dataset},False,{args.exp_setup},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gs[1:])},{np.mean(losses_gs)},{np.sum(np.array(all_label_gs) == np.array(all_out_gs))/num}\n")
    elif args.task == 'graph_reg':
        if args.baseline:
            if args.dataset == 'QM9':
                f.write(f"{args.dataset},True,None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{args.property}\n")
            else:
                f.write(f"{args.dataset},True,None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},None\n")
        if args.dataset == 'QM9':
            if args.exp_setup == "Gc_train_2_Gc_infer":
                f.write(f"{args.dataset},False,{args.exp_setup},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gc[1:])},{np.mean(losses_gc)},{args.property}\n")
            else:
                f.write(f"{args.dataset},False,{args.exp_setup},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gs[1:])},{np.mean(losses_gs)},{args.property}\n")
        else:
            if args.exp_setup == "Gc_train_2_Gc_infer":
                f.write(f"{args.dataset},False,{args.exp_setup},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gc[1:])},{np.mean(losses_gc)},None\n")
            else:
                f.write(f"{args.dataset},False,{args.exp_setup},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},512,{num},{args.num_layers2},{args.batch_size},{args.lr},{np.mean(times_gs[1:])},{np.mean(losses_gs)},None\n")
f.close()
