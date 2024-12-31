from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, WikipediaNetwork, TUDataset, ZINC, QM9
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
import torch
from utils import coarsening_classification, coarsening_regression, load_graph_data
import argparse
from tqdm import tqdm
import pickle
import os
import warnings
warnings.simplefilter('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    elif args.dataset == "ogbn-products":
        dataset = PygNodePropPredDataset(name="ogbn-products", root='/hdfs1/Data/weather/CoarseGNN_Hrriday/OGB/dataset/')
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
        if args.normalize_features:
            for i in range(len(dataset)):
                dataset[i].x = torch.nn.functional.normalize(dataset[i].x, p=1)
        args.task = 'graph_cls'
    elif args.dataset == 'PROTEINS':
        dataset = TUDataset(root='./dataset', name=args.dataset)
        if args.normalize_features:
            for i in range(len(dataset)):
                dataset[i].x = torch.nn.functional.normalize(dataset[i].x, p=1)
        args.task = 'graph_cls'
    elif args.dataset == 'AIDS':
        dataset = TUDataset(root='./dataset', name=args.dataset)
        args.task = 'graph_cls'
    #Graph Regression
    elif args.dataset == 'QM9':
        dataset = QM9(root='./dataset/QM9')
        args.task = 'graph_reg'
        args.multi_prop = True
    elif args.dataset == 'ZINC_full':
        dataset = ZINC(root='./dataset/ZINC', subset=False)
        args.task = 'graph_reg'
    elif args.dataset == 'ZINC_subset':
        dataset = ZINC(root='./dataset/ZINC', subset=True)
        args.task = 'graph_reg'
    elif args.dataset == 'random':
        print("Here\nNum Random Nodes: ", args.num_random_nodes)
        # create a cyclic graph with args.num_random_nodes nodes
        x = torch.randint(0, 10, (args.num_random_nodes, 1), dtype=torch.float64)
        edge_index = torch.zeros(2, 4*args.num_random_nodes, dtype=torch.long)
        for i in tqdm(range(args.num_random_nodes)):
            edge_index[0][2*i] = i
            edge_index[1][2*i] = (i+1)%args.num_random_nodes
            edge_index[0][2*i+1] = (i+1)%args.num_random_nodes
            edge_index[1][2*i+1] = i
            edge_index[0][2*args.num_random_nodes + 2*i] = i
            edge_index[1][2*args.num_random_nodes + 2*i] = (i+2)%args.num_random_nodes
            edge_index[0][2*args.num_random_nodes + 2*i+1] = (i+2)%args.num_random_nodes
            edge_index[1][2*args.num_random_nodes + 2*i+1] = i
        dataset = [Data(x=x, edge_index=edge_index, y = torch.zeros(args.num_random_nodes, dtype=torch.long))]
        args.task = 'graph_cls'
        print("Graph formed...")
    return dataset, args

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

def save(args, path, candidate = None, C_list = None, Gc_list = None, subgraph_list = None, saved_graph_list = None):
    if not os.path.exists(path):
        os.makedirs(path)
    node_type = "d"
    if args.extra_node:
        node_type = "e"
    elif args.cluster_node:
        node_type = "c"
    if args.task == 'node_cls':
        with open(path + f'{args.coarsening_ratio}{node_type}_candidate.pkl', 'wb') as f:
            pickle.dump(candidate, f)
        f.close()
        with open(path + f'{args.coarsening_ratio}{node_type}_C_list.pkl', 'wb') as f:
            pickle.dump(C_list, f)
        f.close()
        with open(path + f'{args.coarsening_ratio}{node_type}_Gc_list.pkl', 'wb') as f:
            pickle.dump(Gc_list, f)
        f.close()
        torch.save(subgraph_list, path + f'{args.coarsening_ratio}{node_type}_subgraph_list.pt')
    elif args.task == 'node_reg':
        torch.save(subgraph_list, path + f'{args.coarsening_ratio}{node_type}_subgraph_list.pt')
    elif args.task == 'graph_cls':
        with open(path + f'{args.coarsening_ratio}{node_type}_Gc_list.pkl', 'wb') as f:
            pickle.dump(Gc_list, f)
        f.close()
        with open(path + f'{args.coarsening_ratio}{node_type}_saved_graph_list.pkl', 'wb') as f:
            pickle.dump(saved_graph_list, f)
        f.close()
        torch.save(subgraph_list, path + f'{args.coarsening_ratio}{node_type}_subgraph_list.pt')
    else:
        with open(path + f'{args.coarsening_ratio}{node_type}_Gc_list.pkl', 'wb') as f:
            pickle.dump(Gc_list, f)
        f.close()
        with open(path + f'{args.coarsening_ratio}{node_type}_saved_graph_list.pkl', 'wb') as f:
            pickle.dump(saved_graph_list, f)
        f.close()
        torch.save(subgraph_list, path + f'{args.coarsening_ratio}{node_type}_subgraph_list.pt')
    print("Saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--extra_node', type=bool, default=False)
    parser.add_argument('--cluster_node', type=bool, default=False)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--task', type = str, default = 'node_cls')
    parser.add_argument('--multi_prop', type =bool, default=False)
    parser.add_argument('--property', type = int, default = 0)
    parser.add_argument('--super_graph', type=bool, default=False)
    parser.add_argument('--num_random_nodes', type=int, default=100)
    parser.add_argument('--use_community_detection', type=bool, default=False)
    # parser.add_argument('--experiment', type=str, default='fixed')
    # parser.add_argument('--runs', type=int, default=20)
    # parser.add_argument('--exp_setup', type=str, default='Gc_train_2_Gs_infer')
    # parser.add_argument('--hidden', type=int, default=512)
    # parser.add_argument('--epochs1', type=int, default=100)   
    # parser.add_argument('--epochs2', type=int, default=300)
    # parser.add_argument('--num_layers1', type=int, default=2)
    # parser.add_argument('--num_layers2', type=int, default=2)
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--train_ratio', type=float, default=0.3)
    # parser.add_argument('--val_ratio', type=float, default=0.2)
    # parser.add_argument('--early_stopping', type=int, default=10)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--weight_decay', type=float, default=0.0005)
    # parser.add_argument('--output_dir', type=str, required=True)
    # parser.add_argument('--seed', type = int, default = None)
    args = parser.parse_args()

args = arg_correction(args)
dataset, args = process_dataset(args)

if args.task == 'node_cls':
    dataset = dataset.to(device)
    args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
    save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', candidate=candidate, C_list=C_list, Gc_list=Gc_list, subgraph_list=subgraph_list)
    
elif args.task == 'node_reg':
    dataset = dataset.to(device)
    args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
    save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', subgraph_list=subgraph_list)
    
elif args.task == 'graph_cls':
    Gc_ = []
    Gs_ = []
    saved_graph_list = []
    classes = set()
    dataset = dataset.to(device)
    for i in tqdm(range(len(dataset))):
        try:
            args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, dataset[i], 1-args.coarsening_ratio, args.coarsening_method)
            Gc = load_graph_data(dataset[i], CLIST, GcLIST, candidate)
            saved_graph_list.append(i)
            # Gs = subgraph_list
            # new_dataset.append((dataset[i], Gc, Gs))
            Gc_.append(Gc)
            Gs_.append(subgraph_list)
            classes.add(dataset[i].y.item())
        except:
            pass
    args.num_classes = len(classes)
    save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', Gc_list=Gc_, subgraph_list=Gs_, saved_graph_list=saved_graph_list)
    
else:
    Gc_ = []
    Gs_ = []
    saved_graph_list = []
    dataset = dataset.to(device)
    for i in tqdm(range(len(dataset))):
        try:
            args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, dataset[i], 1-args.coarsening_ratio, args.coarsening_method)
            Gc = load_graph_data(dataset[i], CLIST, GcLIST, candidate)
            saved_graph_list.append(i)
            # Gs = subgraph_list
            # new_dataset.append((dataset[i], Gc, Gs))
            Gc_.append(Gc)
            Gs_.append(subgraph_list)
        except:
            pass
    save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', Gc_list=Gc_, subgraph_list=Gs_, saved_graph_list=saved_graph_list)
