import csv
import warnings
warnings.simplefilter("ignore")
import os
import torch
import pickle
import argparse
import igraph as ig
import leidenalg
from torch_geometric.profile import get_data_size
from run import *
from utils import node_mem_save
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
import torch_scatter
from torch.utils.tensorboard import SummaryWriter
from utils import coarsening_classification, coarsening_regression, load_graph_data, merge_communities
from torch_geometric.datasets import WikipediaNetwork, TUDataset, Planetoid, Coauthor, CitationFull, ZINC, QM9, WikiCS
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
        '''dataset = [torch.load(
            "/hdfs1/Data/Shubhajit/Project/CoPart-GNN/dataset/ogbn-products/saved/community_data.pt",
            map_location="cpu",
            weights_only=False,
        )]'''
        dataset = PygNodePropPredDataset(name="ogbn-products", root='./dataset/')
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    elif args.dataset == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root='./dataset/')
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    elif args.dataset == 'WikiCS':
        dataset = WikiCS(root='./dataset/WikiCS')
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    elif args.dataset == "ogbn-proteins":
        dataset = PygNodePropPredDataset(name="ogbn-proteins", root='./dataset/')
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)

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
    return dataset, args

def arg_correction(args):
    if args.cluster_node:
        args.extra_node = False
    elif args.extra_node:
        args.cluster_node = False
    if args.experiment == 'fixed':
        if args.dataset == 'ogbn-products' or args.dataset == 'dblp' or args.dataset == 'Physics'or args.dataset == 'WikiCS' or args.dataset == 'Flickr':
            args.experiment = 'random'
    return args

def save(args, path, candidate = None, C_list = None, Gc_list = None, subgraph_list = None, saved_graph_list = None):
    if not os.path.exists(path):
        os.makedirs(path)
    node_type = "d"
    if args.extra_node:
        node_type = "e"
    elif args.cluster_node:
        node_type = "c"
    if args.use_community_detection == True:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--experiment', type=str, default='random')
    parser.add_argument('--train_ratio', type=float, default=0.3)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--extra_node', action='store_true')
    parser.add_argument('--cluster_node', action='store_true')
    parser.add_argument('--use_community_detection', action='store_true')
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--task', type = str, default = 'node_cls')
    parser.add_argument('--seed', type = int, default = None)
    parser.add_argument('--fitgnn', action='store_true')
    parser.add_argument('--multi_prop', action='store_true')
    args = parser.parse_args()

    args = arg_correction(args)
    dataset, args = process_dataset(args)

    node_type = "d"
    if args.extra_node:
        node_type = "e"
    elif args.cluster_node:
        node_type = "c"

    if args.use_community_detection:
        graph_type = "community"
    else:
        graph_type = "full"
    
    if args.task == 'node_cls':
        
        dataset = dataset.to(device)
        data = dataset[0]
        if args.dataset == 'ogbn-products':
            args.num_classes = 47
        else:
            args.num_classes = torch.unique(data.y).shape[0]
        
        if args.use_community_detection:
            if os.path.exists(f'./dataset/{args.dataset}/saved/{graph_type}_data.pt'):
                del dataset
                torch.cuda.empty_cache()
                data = torch.load(f'./dataset/{args.dataset}/saved/{graph_type}_data.pt', weights_only=False)
            else:
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
                torch.cuda.empty_cache()
                torch.save(data, f'./dataset/{args.dataset}/saved/{graph_type}_data.pt')
        args.num_features = data.x.shape[1]
        if args.fitgnn:
            if os.path.exists(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt'):
                print("Loading saved graphs...")
                subgraph_list = torch.load(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt', weights_only=False)
                candidate = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_candidate.pkl', 'rb'))
                C_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_C_list.pkl', 'rb'))
                Gc_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl', 'rb'))
            else:
                print("Coarsening graphs...")
                args.num_features, candidate, C_list, Gc_list, subgraph_list = coarsening_classification(args, data, 1-args.coarsening_ratio, args.coarsening_method)
                save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', candidate=candidate, C_list=C_list, Gc_list=Gc_list, subgraph_list=subgraph_list)
            node_mem_save(args=args, data=data, candidate=candidate, C_list=C_list, Gc_list=Gc_list, subgraph_list=subgraph_list)
        else:
            max_mem = get_data_size(data)
            max_mem = max_mem / (1024 * 1024)
            if not os.path.exists("memory_usage.csv"):
                with open("memory_usage.csv", mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Dataset", "Coarsening_Method", "Coarsening_Ratio", "Task", "Type", "Max_Memory_Usage_Graphs(MB)"])
                    writer.writerow([args.dataset, "None", 1.0, args.task, "None", max_mem])
            else:
                with open("memory_usage.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([args.dataset, "None", 1.0, args.task, "None", max_mem])
    elif args.task == 'node_reg':
        dataset = dataset.to(device)
        data = dataset[0]
        if args.fitgnn:
            if args.use_community_detection:
                if os.path.exists(f'./dataset/{args.dataset}/saved/{graph_type}_data.pt'):
                    del dataset
                    torch.cuda.empty_cache()
                    data = torch.load(f'./dataset/{args.dataset}/saved/{graph_type}_data.pt', weights_only=False)
                else:
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
                    torch.cuda.empty_cache()
                    torch.save(data, f'./dataset/{args.dataset}/saved/{graph_type}_data.pt')
            args.num_features = data.x.shape[1]
            if args.fitgnn:
                if os.path.exists(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt'):
                    print("Loading saved graphs...")
                    subgraph_list = torch.load(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt', weights_only=False)
                else:
                    print("Coarsening graphs...")
                    args.num_features, subgraph_list = coarsening_regression(args, data, 1-args.coarsening_ratio, args.coarsening_method)
                    save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', subgraph_list=subgraph_list)
                node_mem_save(args=args, data=data, subgraph_list=subgraph_list)
        else:
            max_mem = get_data_size(data)
            max_mem = max_mem / (1024 * 1024)
            if not os.path.exists("memory_usage.csv"):
                with open("memory_usage.csv", mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Dataset", "Coarsening_Method", "Coarsening_Ratio", "Task", "Type", "Max_Memory_Usage_Graphs(MB)"])
                    writer.writerow([args.dataset, "None", 1.0, args.task, "None", max_mem])
            else:
                with open("memory_usage.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([args.dataset, "None", 1.0, args.task, "None", max_mem])
