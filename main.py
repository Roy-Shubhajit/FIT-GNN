import warnings
warnings.simplefilter("ignore")
import os
import torch
import pickle
import argparse
from run import *
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
import torch_scatter
from torch.utils.tensorboard import SummaryWriter
from utils import coarsening_classification, coarsening_regression, load_graph_data
from torch_geometric.datasets import WikipediaNetwork, TUDataset, Planetoid, Coauthor, CitationFull, ZINC, QM9
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
    parser.add_argument('--experiment', type=str, default='fixed')
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--exp_setup', type=str, default='Gc_train_2_Gs_infer') # Gc_train_2_Gc_infer for graph level tasks
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
    parser.add_argument('--super_graph', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--use_community_detection', action='store_true')
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--task', type = str, default = 'node_cls')
    parser.add_argument('--seed', type = int, default = None)
    parser.add_argument('--multi_prop', action='store_true')
    parser.add_argument('--property', type = int, default = 0)
    args = parser.parse_args()

    args = arg_correction(args)
    dataset, args = process_dataset(args)
    if args.task == 'node_cls' and dataset[0].num_nodes > 165000:
        args.use_community_detection = True
    elif args.task == 'node_reg' and dataset[0].num_nodes > 165000:
        args.use_community_detection = True

    path = f"save/{args.task}/"+args.output_dir+"/"
    if not os.path.exists('save'):
        os.makedirs('save')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(path)

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
        if os.path.exists(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt'):
            print("Loading saved graphs...")
            subgraph_list = torch.load(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
            candidate = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_candidate.pkl', 'rb'))
            C_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_C_list.pkl', 'rb'))
            Gc_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl', 'rb'))
            args.num_features = dataset[0].x.shape[1]
        else:
            print("Coarsening graphs...")
            args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
            save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', candidate=candidate, C_list=C_list, Gc_list=Gc_list, subgraph_list=subgraph_list)
        node_classification(args, path, dataset, writer, candidate, C_list, Gc_list, subgraph_list)
    elif args.task == 'node_reg':
        dataset = dataset.to(device)
        if os.path.exists(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt'):
            print("Loading saved graphs...")
            subgraph_list = torch.load(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
            args.num_features = dataset[0].x.shape[1]
        else:
            print("Coarsening graphs...")
            args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
            save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', subgraph_list=subgraph_list)
        node_regression(args, path, dataset, writer, subgraph_list)     
    elif args.task == 'graph_cls':
        new_dataset = []
        Gc_ = []
        Gs_ = []
        saved_graph_list = []
        classes = set()
        if os.path.exists(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt'):
            print("Loading saved graphs...")
            Gc_ = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl', 'rb'))
            Gs_ = torch.load(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
            saved_graph_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_saved_graph_list.pkl', 'rb'))
            for i in tqdm(range(len(saved_graph_list))):
                classes.add(dataset[saved_graph_list[i]].y.item())
                new_dataset.append((dataset[saved_graph_list[i]], Gc_[i], Gs_[i]))
            args.num_features = dataset[0].x.shape[1]
        else:
            print("Coarsening graphs...")
            for i in tqdm(range(len(dataset))):
                try:
                    graph = dataset[i]
                    graph = graph.to(device)
                    args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, graph, 1-args.coarsening_ratio, args.coarsening_method)
                    Gc = load_graph_data(graph, CLIST, GcLIST, candidate)
                    Gs = subgraph_list
                    new_dataset.append((graph, Gc, Gs))
                    Gs_.append(Gs)
                    Gc_.append(Gc)
                    classes.add(graph.y.item())
                    saved_graph_list.append(i)
                except:
                    pass
            save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', Gc_list=Gc_, subgraph_list=Gs_, saved_graph_list=saved_graph_list)
        args.num_classes = len(classes)
        graph_classification(args, path, writer, new_dataset)    
    else:
        new_dataset = []
        Gc_ = []
        Gs_ = []
        saved_graph_list = []
        if os.path.exists(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt'):
            print("Loading saved graphs...")
            Gc_ = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_Gc_list.pkl', 'rb'))
            Gs_ = torch.load(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_subgraph_list.pt')
            saved_graph_list = pickle.load(open(f'./dataset/{args.dataset}/saved/{args.coarsening_method}/{args.coarsening_ratio}_{node_type}_{graph_type}_saved_graph_list.pkl', 'rb'))
            for i in tqdm(range(len(saved_graph_list))):
                new_dataset.append((dataset[saved_graph_list[i]], Gc_[i], Gs_[i]))
            args.num_features = dataset[0].x.shape[1]
        else:
            print("Coarsening graphs...")   
            for i in tqdm(range(len(dataset))):
                try:
                    graph = dataset[i]
                    graph = graph.to(device)
                    args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, graph, 1-args.coarsening_ratio, args.coarsening_method)
                    Gc = load_graph_data(graph, CLIST, GcLIST, candidate)
                    Gs = subgraph_list
                    new_dataset.append((graph, Gc, Gs))
                    Gs_.append(Gs)
                    Gc_.append(Gc)
                    saved_graph_list.append(i)
                except:
                    pass
            save(args, path = f'./dataset/{args.dataset}/saved/{args.coarsening_method}/', Gc_list=Gc_, subgraph_list=Gs_, saved_graph_list=saved_graph_list)
        graph_regression(args, path, writer, new_dataset)
