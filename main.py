import os
import torch
import argparse
from run import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import coarsening_classification, coarsening_regression, load_graph_data
from torch_geometric.datasets import WikipediaNetwork, TUDataset, Planetoid, Coauthor, CitationFull, QM7b, QM9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_dataset(args):
    # Node Classification
    if args.dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=args.dataset)
        args.task = 'node_cls'
    elif args.dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=args.dataset)
        args.task = 'node_cls'
    elif args.dataset == 'cora':
        dataset = Planetoid(root='./dataset', name=args.dataset)
        args.task = 'node_cls'
    elif args.dataset == 'citeseer':
        dataset = Planetoid(root='./dataset', name=args.dataset)
        args.task = 'node_cls'
    elif args.dataset == 'pubmed':
        dataset = Planetoid(root='./dataset', name=args.dataset)
        args.task = 'node_cls'
    #Node Regression
    elif args.dataset == 'chameleon':
        dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
        args.task = 'node_reg'
    elif args.dataset == 'squirrel':
        dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
        args.task = 'node_reg'
    elif args.dataset == 'chrocodile':
        dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
        args.task = 'node_reg'
    #Graph Classification
    elif args.dataset == 'ENZYMES':
        dataset = TUDataset(root='./dataset', name=args.dataset)
        args.task = 'graph_cls'
    elif args.dataset == 'PROTEINS':
        dataset = TUDataset(root='./dataset', name=args.dataset)
        args.task = 'graph_cls'
    elif args.dataset == 'AIDS':
        dataset = TUDataset(root='./dataset', name=args.dataset)
        args.task = 'graph_cls'
    #Graph Regression
    elif args.dataset == 'QM7b':
        dataset = QM7b(root='./dataset', name=args.dataset)
        args.task = 'graph_reg'
    elif args.dataset == 'QM9':
        dataset = QM9(root='./dataset', name=args.dataset)
        args.task = 'graph_reg'

    
    '''if args.task == 'node_cls':
        if args.dataset == 'dblp':
            dataset = CitationFull(root='./dataset', name=args.dataset)
        elif args.dataset == 'Physics':
            dataset = Coauthor(root='./dataset/Physics', name=args.dataset)
        else:
            dataset = Planetoid(root='./dataset', name=args.dataset)
        data = dataset[0]
        # num_classes = len(set(np.array(data.y)))                          ### Might want to Uncomment this to return the number of classes as well 
    elif args.task == 'node_reg':
        dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
        data = dataset[0]
    elif args.task == 'graph_cls':
        if args.dataset == 'ENZYMES':
            dataset = TUDataset(root='./dataset', name=args.dataset)
        else:
            dataset = Planetoid(root='./dataset', name=args.dataset)
        # dataset = TUDataset(root='./dataset', name=args.dataset)
        data = dataset
    else:                                                                   
        dataset = TUDataset(root='./dataset', name = args.dataset)          ### Need to add datasets for graph_reg but have temporarily added dataset for graph_cls
        data = dataset
    # if args.normalize_features:
    #     data.x = torch.nn.functional.normalize(data.x, p=1)     '''          ### Attention needed: Normalizing the features
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

if __name__ == "__main__":
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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods') #'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron'
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--task', type = str, default = 'node_cls')         ### node_reg, graph_cls, graph_reg
    parser.add_argument('--seed', type = int, default = None)               ### Seed for reproducibility
    args = parser.parse_args()

    args = arg_correction(args)

    path = f"save/{args.task}/"+args.output_dir+"/"
    if not os.path.exists('save'):
        os.makedirs('save')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(path)

    dataset, args = process_dataset(args)

    if args.task == 'node_cls':
        args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
        node_classification(args, path, dataset, writer, candidate, C_list, Gc_list, subgraph_list)
        
    elif args.task == 'node_reg':
        args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, dataset[0], 1-args.coarsening_ratio, args.coarsening_method)
        node_regression(args, path, writer, subgraph_list)
        
    elif args.task == 'graph_cls':
        new_dataset = []
        for i in tqdm(range(len(dataset))):
            args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_classification(args, dataset[i], 1-args.coarsening_ratio, args.coarsening_method)
            Gc = load_graph_data(dataset[i], CLIST, GcLIST, candidate)
            Gs = list(component_2_subgraphs.values())
            new_dataset.append((dataset[i], Gc, Gs))
        graph_classification(args, path, writer, new_dataset)
        
    else:
        new_dataset = []
        for i in range(len(dataset)):
            args.num_features, candidate, C_list, Gc_list, subgraph_list, component_2_subgraphs, CLIST, GcLIST = coarsening_regression(args, dataset[i], 1-args.coarsening_ratio, args.coarsening_method)
            Gc = load_graph_data(dataset[i], CLIST, GcLIST, candidate)
            Gs = list(component_2_subgraphs.values())
            new_dataset.append((dataset[i], Gc, Gs))
        graph_regression(args, path, writer, new_dataset)