import os
import torch
import igraph as ig
import leidenalg
from utils import merge_communities
from ogb.nodeproppred import PygNodePropPredDataset
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch_geometric.datasets import WikipediaNetwork, TUDataset, Planetoid, Coauthor, CitationFull, ZINC, QM9

dataset_names = ['dblp', 'Physics', 'cora', 'citeseer', 'pubmed', 'ogbn-products(community)','chameleon', 'squirrel', 'crocodile', 'PROTEINS',"AIDS","QM9","ZINC"]

for _ in dataset_names:
    if _ == 'dblp':
        dataset = CitationFull(root='./dataset', name=_)
        task = 'node_cls'
    elif _ == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=_)
        task = 'node_cls'
    elif _ == 'cora':
        dataset = Planetoid(root='./dataset', name=_)
        task = 'node_cls'
    elif _ == 'citeseer':
        dataset = Planetoid(root='./dataset', name=_)
        task = 'node_cls'
    elif _ == 'pubmed':
        dataset = Planetoid(root='./dataset', name=_)
        task = 'node_cls'
    elif _ == 'ogbn-products(community)':
        if os.path.exists('./dataset/ogbn-products/saved/community_data.pt'):
            data = torch.load('./dataset/ogbn-products/saved/community_data.pt')
        else:
            dataset = PygNodePropPredDataset(name="ogbn-products", root='./dataset/')
            print("Using community detection")
            g_ig = ig.Graph(n=data.num_nodes, edges=data.edge_index.t().tolist())
            part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
            mapping = {}
            for i, c in enumerate(part.membership):
                if int(c) not in mapping.keys():
                    mapping[int(c)] = []
                mapping[int(c)].append(i)
            data = merge_communities(data, mapping, 165000)
            torch.save(data, f'./dataset/ogbn-products/saved/community_data.pt')
        dataset = [data]
        task = 'node_cls'
    elif _ == 'chameleon':
        dataset = WikipediaNetwork(root='./dataset', name=_, geom_gcn_preprocess=False)
        task = 'node_reg'
    elif _ == 'squirrel':
        dataset = WikipediaNetwork(root='./dataset', name=_, geom_gcn_preprocess=False)
        task = 'node_reg'
    elif _ == 'crocodile':
        dataset = WikipediaNetwork(root='./dataset', name=_, geom_gcn_preprocess=False)
        task = 'node_reg'
    elif _ == 'PROTEINS':
        dataset = TUDataset(root='./dataset', name=_)
        task = 'graph_cls'
    elif _ == 'AIDS':
        dataset = TUDataset(root='./dataset', name=_)
        task = 'graph_cls'
    elif _ == 'QM9':
        dataset = QM9(root='./dataset/QM9')
        task = 'graph_reg'
    elif _ == 'ZINC':
        dataset = ZINC(root='./dataset/ZINC', subset=True)
        task = 'graph_reg'

    if task == 'node_cls' or task == 'node_reg':
        data = dataset[0]
        num_graphs = 1
        num_nodes = data.x.size(0)
        num_features = data.x.size(1)
        if task == 'node_cls':
            if _ == 'ogbn-products(community)':
                num_classes = 47
            else:
                num_classes = dataset.num_classes
            num_targets = None
        else:
            if len(data.y.size()) == 1:
                num_targets = 1
            else:
                num_targets = data.y.size(1)    
            num_classes = None
        num_edges = data.edge_index.size(1)//2
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
            f.write("dataset,task,num_graphs,num_nodes/avg_num_nodes,num_edges/num_edges,num_features,num_classes,num_targets\n")
        f.close()

    with open("dataset_info.csv", "a") as f:
        f.write(f"{_},{task},{num_graphs},{num_nodes},{num_edges},{num_features},{num_classes},{num_targets}\n")
    f.close()