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
        dataset = PygNodePropPredDataset(name="ogbn-products", root='./dataset/')
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
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--epochs1', type=int, default=100)
parser.add_argument('--epochs2', type=int, default=300)
parser.add_argument('--num_layers1', type=int, default=2)
parser.add_argument('--num_layers2', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_ratio', type=float, default=0.3)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--use_community_detection', action='store_true')
parser.add_argument('--normalize_features', action='store_true')
parser.add_argument('--task', type = str, default = 'node_cls')
parser.add_argument('--seed', type = int, default = None)
parser.add_argument('--multi_prop', action='store_true')
parser.add_argument('--property', type = int, default = 0)
parser.add_argument('--num_test_samples', type = int, default = 20)                                                                             ### Number of test samples 
parser.add_argument('--path_b', type = str, default = "./save/node_cls/baseline/")                                                             ### Path for baseline model
parser.add_argument('--model_name_b', type = str, default = "baseline_cora_fixed.pt")                                                           ### Baseline model name                                                                    ### Coarsened graph model name                                                                                ### If True, baseline model results will be saved
args = parser.parse_args()

dataset, args = process_dataset(args)

if (args.task == 'node_cls' or args.task == 'node_reg') and dataset[0].num_nodes > 170000:
    args.use_community_detection = True

node_type = "d"
if args.use_community_detection:
    graph_type = "community"
else:
    graph_type = "full"

print("###############################################")
print("\nDataset: ", args.dataset)
print("Task: ", args.task)
print("Community:", args.use_community_detection)
print("\n###############################################")


if args.task == "graph_cls":
    all_out_b = []
    all_label_b = []
    losses_b = []
    times_b = []

    model_b = Classify_graph(args.num_layers2, args.num_features, args.hidden, args.num_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
    model_b.eval()

    num = 0
    perm_graphs = np.random.permutation(len(dataset))
    if args.num_test_samples > len(dataset):
        args.num_test_samples = len(dataset)
    new_datasets = []
    for i in perm_graphs[:args.num_test_samples]:
        new_datasets.append(dataset[i])
    num = len(new_datasets)
    for j in range(num):
        test_loader = G_DataLoader([new_datasets[j]], batch_size=1)
        for graph in test_loader:
            graph = graph.to(device)
            t3 = time()
            out_b = model_b(graph)
            t4 = time()
            times_b.append(t4-t3)
            all_out_b.append(out_b.argmax().item())
            all_label_b.append(graph.y.item())
            loss_b = loss_fn(out_b, graph.y)
            losses_b.append(loss_b.item())

    print(f"\nAverage time (baseline): {np.mean(times_b[1:])}\nAccuracy (baseline): {np.sum(np.array(all_label_b) == np.array(all_out_b))}/{num}")

elif args.task == "graph_reg":

    losses_b = []
    times_b = []

    model_b = Regress_graph(args.num_layers2, args.num_features, args.hidden).to(device)
    loss_fn = torch.nn.L1Loss().to(device)
    model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
    model_b.eval()
    
    perm_graphs = np.random.permutation(len(dataset))
    if args.num_test_samples > len(dataset):
        args.num_test_samples = len(dataset)
    new_datasets = []
    for i in perm_graphs[:args.num_test_samples]:
        new_datasets.append(dataset[i])
    num = len(new_datasets)
    
    for j in range(num):
        test_loader = G_DataLoader([new_datasets[j]], batch_size=1)
        for graph in test_loader:
            graph = graph.to(device)
            graph.y = graph.y.type(torch.float)
            t3 = time()
            out_b = model_b(graph)
            t4 = time()
            times_b.append(t4-t3)
            if args.dataset == 'QM9':
                loss_b = loss_fn(out_b, graph.y[:, args.property].view(-1, 1))
            else:
                loss_b = loss_fn(out_b, graph.y)
            losses_b.append(loss_b.item())
        
    print(f"\nAverage time (baseline): {np.mean(times_b[1:])}\nAverage Loss (baseline): {np.mean(losses_b)}")

elif args.task == "node_cls":
    dataset = dataset.to(device)
    data = dataset[0]
    args.num_classes = torch.unique(dataset.y).shape[0]
    args.num_features = dataset[0].x.shape[1]
    if args.use_community_detection and os.path.exists(f"./dataset/{args.dataset}/saved/{graph_type}_data.pt"):
        del dataset
        torch.cuda.empty_cache()
        print("Loading saved graph...")
        data = torch.load(f'./dataset/{args.dataset}/saved/{graph_type}_data.pt')
                               
    perm_nodes = np.random.permutation(data.num_nodes)
    if args.num_test_samples > data.num_nodes:
        args.num_test_samples = data.num_nodes
    indices = perm_nodes[:args.num_test_samples]
    num = len(indices)
    losses_b = []
    times_b = []
    all_label_b = []
    all_out_b = []
    graph_data = G_DataLoader([data], batch_size=1)
    model_b = Net1(data.x.shape[1], args.hidden, args.num_layers2, args.num_classes).to(device)
    loss_fn = torch.nn.NLLLoss().to(device)
    model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
    model_b.eval()
    fig = plt.figure()
    for i in tqdm(range(num)):
        for graph in graph_data:
            graph = graph.to(device)
            t3 = time()
            out_b = model_b(graph.x, graph.edge_index).to(device)
            t4 = time()
            loss_b = loss_fn(out_b[i].view(1,-1), graph.y[i].flatten())
            losses_b.append(loss_b.item())
            all_label_b.append(graph.y[i].item())
            all_out_b.append(out_b[i].argmax().item())
            times_b.append(t4 - t3)

    print(f"Average time (baseline): {np.mean(times_b[1:])}\nAccuracy (baseline): {np.sum(np.array(all_label_b) == np.array(all_out_b))}/{num}")
        
elif args.task == "node_reg":
    dataset = dataset.to(device)
    args.num_features = dataset[0].x.shape[1]
    perm_nodes = np.random.permutation(dataset[0].num_nodes)
    if args.num_test_samples > dataset[0].num_nodes:
        args.num_test_samples = dataset[0].num_nodes
    indices = perm_nodes[:args.num_test_samples]
    num = len(indices)
    graph_data = G_DataLoader(dataset, batch_size=1)
    model_b = Regress_node(args).to(device)
    loss_fn = torch.nn.L1Loss().to(device)
    model_b.load_state_dict(torch.load(args.path_b + args.model_name_b))
    model_b.eval()
    losses_b = []
    times_b = []
    y_b = []
    for i in range(num):
        for graph in graph_data:
            graph = graph.to(device)
            t3 = time()
            out_b = model_b(graph.x, graph.edge_index).to(device)
            t4 = time()
            loss_b = loss_fn(out_b[indices[i]][0], graph.y[indices[i]])
            losses_b.append(loss_b.item())
            y_b.append(graph.y[indices[i]].item())
            times_b.append(t4 - t3)

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
        f.write(f"{args.dataset},True,{args.experiment},None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{np.sum(np.array(all_label_b) == np.array(all_out_b))/num}\n")
    elif args.task == "node_reg":
        f.write(f"{args.dataset},True,None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b/np.std(y_b))}\n")
    elif args.task == 'graph_cls':
        f.write(f"{args.dataset},True,None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{np.sum(np.array(all_label_b) == np.array(all_out_b))/num}\n")
    elif args.task == 'graph_reg':
        if args.dataset == 'QM9':
            f.write(f"{args.dataset},True,None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},{args.property}\n")
        else:
            f.write(f"{args.dataset},True,None,None,None,None,None,512,{num},{args.num_layers2},None,0.01,{np.mean(times_b[1:])},{np.mean(losses_b)},None\n")
f.close()