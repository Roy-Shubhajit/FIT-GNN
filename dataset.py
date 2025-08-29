import warnings
warnings.simplefilter("ignore")
import os
from tqdm import tqdm
import argparse
import torch
import igraph as ig
import leidenalg
from torch_geometric.utils import homophily
from ogb.nodeproppred import PygNodePropPredDataset
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch_geometric.datasets import WikipediaNetwork, TUDataset, Planetoid, Coauthor, CitationFull, ZINC, QM9, WikiCS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def merge_communities(data, mapping, k):
    sorted_mapping = sorted(mapping.items(), key=lambda x: len(x[1]), reverse=True)
    new_nodes = torch.tensor([], dtype=torch.long)
    for i in range(len(sorted_mapping)):
        if len(new_nodes) + len(sorted_mapping[i][1]) <= k:
            new_nodes = torch.cat((new_nodes, torch.tensor(sorted_mapping[i][1], dtype=torch.long)))
            if len(new_nodes) == k:
                break
    new_data = data.subgraph(new_nodes)
    return new_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
args = parser.parse_args()
edge_homo, node_home, ins_homo, node_reg_std = None, None, None, None
if args.dataset== 'dblp':
    dataset = CitationFull(root='./dataset', name=args.dataset)
    task = 'node_cls'
elif args.dataset== 'Physics':
    dataset = Coauthor(root='./dataset/Physics', name=args.dataset)
    task = 'node_cls'
elif args.dataset== 'cora':
    dataset = Planetoid(root='./dataset', name=args.dataset)
    task = 'node_cls'
elif args.dataset== 'citeseer':
    dataset = Planetoid(root='./dataset', name=args.dataset)
    task = 'node_cls'
elif args.dataset== 'pubmed':
    dataset = Planetoid(root='./dataset', name=args.dataset)
    task = 'node_cls'
elif args.dataset == 'WikiCS':
    dataset = WikiCS(root='./dataset/WikiCS')
    task = 'node_cls'
elif args.dataset== 'ogbn-products(community)':
    if os.path.exists('./dataset/ogbn_products/saved/community_data.pt'):
        data = torch.load('./dataset/ogbn_products/saved/community_data.pt')
    else:
        data = PygNodePropPredDataset(name="ogbn-products", root='/hdfs1/Data/weather/CoarseGNN_Hrriday/OGB/dataset/')
        print("Using community detection")
        g_ig = ig.Graph(n=data.num_nodes, edges=data.edge_index.t().tolist())
        part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
        mapping = {}
        for i, c in enumerate(part.membership):
            if int(c) not in mapping.keys():
                mapping[int(c)] = []
            mapping[int(c)].append(i)
        print(data)
        data = merge_communities(data, mapping, 165000)
        torch.save(data, f'./dataset/ogbn_products/saved/community_data.pt')
    dataset = [data]
    task = 'node_cls'
elif args.dataset == 'OGBN-Products':
    dataset = PygNodePropPredDataset(name="ogbn-products", root='./dataset/')
    task = 'node_cls'
elif args.dataset== 'chameleon':
    dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
    task = 'node_reg'
elif args.dataset == 'squirrel':
    dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
    task = 'node_reg'
elif args.dataset == 'crocodile':
    dataset = WikipediaNetwork(root='./dataset', name=args.dataset, geom_gcn_preprocess=False)
    task = 'node_reg'
elif args.dataset== 'PROTEINS':
    dataset = TUDataset(root='./dataset', name=args.dataset)
    task = 'graph_cls'
elif args.dataset== 'AIDS':
    dataset = TUDataset(root='./dataset', name=args.dataset)
    task = 'graph_cls'
elif args.dataset== 'QM9':
    dataset = QM9(root='./dataset/QM9')
    task = 'graph_reg'
elif args.dataset== 'ZINC':
    dataset = ZINC(root='./dataset/ZINC', subset=True)
    task = 'graph_reg'

if task == 'node_cls' or task == 'node_reg':
    #check whether dataset has Data object in [0]:
    if isinstance(dataset, list):
        data = dataset[0]
    else:
        data = dataset
    num_graphs = 1
    num_nodes = data.x.size(0)
    num_features = data.x.size(1)
    if task == 'node_cls':
        if args.dataset== 'ogbn-products(community)':
            num_classes = 47
        else:
            num_classes = dataset.num_classes
        num_targets = None
        edge_homo = homophily(data.edge_index, data.y, method="edge")
        node_home = homophily(data.edge_index, data.y, method="node")
        ins_homo = homophily(data.edge_index, data.y, method="edge_insensitive")
    else:
        if len(data.y.size()) == 1:
            num_targets = 1
        else:
            num_targets = data.y.size(1)    
        num_classes = None
        node_reg_std = torch.std(data.y)
    num_edges = data.edge_index.size(1)//2
    del data
else:
    num_graphs = len(dataset)
    num_nodes = torch.mean(torch.tensor([data.num_nodes//2 for data in dataset], dtype=torch.float))
    num_edges = torch.mean(torch.tensor([data.num_edges//2 for data in dataset], dtype=torch.float))
    num_features = dataset.num_features
    if task == 'graph_cls':
        num_classes = dataset.num_classes
        num_targets = None
    else:
        if len(dataset[0].y.size()) == 1:
            num_targets = 1
        else:
            num_targets = dataset[0].y.size(1)
        num_classes = None

if not os.path.exists("dataset_info.csv"):
    with open("dataset_info.csv", "w") as f:
        f.write("dataset,task,num_graphs,num_nodes/avg_num_nodes,num_edges/num_edges,num_features,num_classes,num_targets,std,homophily(edge),homophily(node),homophily(edge_insensitive)\n")
    f.close()

with open("dataset_info.csv", "a") as f:
    f.write(f"{args.dataset},{task},{num_graphs},{num_nodes},{num_edges},{num_features},{num_classes},{num_targets},{node_reg_std},{edge_homo},{node_home},{ins_homo}\n")
f.close()