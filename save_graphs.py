import warnings
warnings.simplefilter("ignore")
from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, WikipediaNetwork, TUDataset, ZINC, QM9
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
import torch
import igraph as ig
import leidenalg
from utils import coarsening_classification, coarsening_regression, load_graph_data, merge_communities
import argparse
from tqdm import tqdm
import pickle
import os
import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)
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
        print("Num Random Nodes: ", args.num_random_nodes)
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
    if args.cluster_node:
        args.extra_node = False
    elif args.extra_node:
        args.cluster_node = False
    return args

def save(args, path, candidate = None, C_list = None, Gc_list = None, subgraph_list = None, saved_graph_list = None):
    if not os.path.exists(path):
        os.makedirs(path)
    node_type = "d"
    if args.extra_node:
        node_type = "e"
    elif args.cluster_node:
        node_type = "c"
    if args.use_community_detection ==  True:
        graph_type = "community"
    else:
        graph_type = "full"
    if args.task == 'node_cls':
        with open(path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_candidate.pkl', 'wb') as f:
            pickle.dump(candidate, f)
        f.close()
        with open(path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_C_list.pkl', 'wb') as f:
            pickle.dump(C_list, f)
        f.close()
        with open(path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl', 'wb') as f:
            pickle.dump(Gc_list, f)
        f.close()
        torch.save(subgraph_list, path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
    elif args.task == 'node_reg':
        torch.save(subgraph_list, path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
    elif args.task == 'graph_cls':
        with open(path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl', 'wb') as f:
            pickle.dump(Gc_list, f)
        f.close()
        with open(path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_saved_graph_list.pkl', 'wb') as f:
            pickle.dump(saved_graph_list, f)
        f.close()
        torch.save(subgraph_list, path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
    else:
        with open(path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl', 'wb') as f:
            pickle.dump(Gc_list, f)
        f.close()
        with open(path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_saved_graph_list.pkl', 'wb') as f:
            pickle.dump(saved_graph_list, f)
        f.close()
        torch.save(subgraph_list, path + f'{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
    print("Saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--extra_node', action='store_true')
    parser.add_argument('--cluster_node', action='store_true')
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--task', type = str, default = 'node_cls')
    parser.add_argument('--multi_prop', action='store_true')
    parser.add_argument('--property', type = int, default = 0)
    parser.add_argument('--num_random_nodes', type=int, default=100)
    parser.add_argument('--use_community_detection', action='store_true')
    
    args = parser.parse_args()
    args = arg_correction(args)
    dataset, args = process_dataset(args)

if args.task == 'node_cls':
    dataset = dataset.to(device)
    data = dataset[0]
    if args.use_community_detection:
        graph_type = "community"
        print("Using community detection")
        g_ig = ig.Graph(n=data.num_nodes, edges=data.edge_index.t().tolist())
        part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
        mapping = {}
        for i, c in enumerate(part.membership):
            if int(c) not in mapping.keys():
                mapping[int(c)] = []
            mapping[int(c)].append(i)
        data = merge_communities(data, mapping, 165000)
        del dataset
        torch.save(data, f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{graph_type}_data.pt')
    args.num_features, candidate, C_list, Gc_list, subgraph_list = coarsening_classification(args, data, 1-args.coarsening_ratio, args.coarsening_method)
    save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', candidate=candidate, C_list=C_list, Gc_list=Gc_list, subgraph_list=subgraph_list)
    
elif args.task == 'node_reg':
    dataset = dataset.to(device)
    data = dataset[0]
    if args.use_community_detection:
        graph_type = "community"
        print("Using community detection")
        g_ig = ig.Graph(n=data.num_nodes, edges=data.edge_index.t().tolist())
        part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
        mapping = {}
        for i, c in enumerate(part.membership):
            if int(c) not in mapping.keys():
                mapping[int(c)] = []
            mapping[int(c)].append(i)
        data = merge_communities(data, mapping, 165000)
        del dataset
        torch.save(data, f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{graph_type}_data.pt')
    args.num_features, subgraph_list = coarsening_regression(args, data, 1-args.coarsening_ratio, args.coarsening_method)
    save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', subgraph_list=subgraph_list)
    
elif args.task == 'graph_cls':
    Gc_ = []
    Gs_ = []
    saved_graph_list = []
    classes = set()
    for i in tqdm(range(len(dataset)), colour='blue'):
            graph = dataset[i]
            graph = graph.to(device)
            args.num_features, candidate, subgraph_list, CLIST, GcLIST = coarsening_classification(args, graph, 1-args.coarsening_ratio, args.coarsening_method)
            Gc = load_graph_data(graph, CLIST, GcLIST, candidate)
            saved_graph_list.append(i)
            Gc_.append(Gc)
            Gs_.append(subgraph_list)
            classes.add(graph.y.item())
    args.num_classes = len(classes)
    save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', Gc_list=Gc_, subgraph_list=Gs_, saved_graph_list=saved_graph_list)
    
else:
    Gc_ = []
    Gs_ = []
    saved_graph_list = []
    for i in tqdm(range(len(dataset)), colour='blue'):
        try:
            graph = dataset[i]
            graph = graph.to(device)
            args.num_features, candidate, subgraph_list, CLIST, GcLIST = coarsening_regression(args, graph, 1-args.coarsening_ratio, args.coarsening_method)
            Gc = load_graph_data(graph, CLIST, GcLIST, candidate)
            saved_graph_list.append(i)
            Gc_.append(Gc)
            Gs_.append(subgraph_list)
        except:
            pass
    save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', Gc_list=Gc_, subgraph_list=Gs_, saved_graph_list=saved_graph_list)