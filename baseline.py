import warnings
warnings.simplefilter("ignore")
import os
import torch
import time
import argparse
import igraph as ig
import leidenalg
from tqdm import tqdm
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
import torch_scatter
from torch.utils.tensorboard import SummaryWriter
from utils import merge_communities
from torch_geometric.datasets import WikipediaNetwork, TUDataset, Planetoid, Coauthor, CitationFull, ZINC, QM9
import logging
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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
    elif args.dataset == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root='/hdfs1/Data/weather/CoarseGNN_Shubhajit/CoPart-GNN/dataset/')
        if args.normalize_features:
            dataset.x = torch.nn.functional.normalize(dataset.x, p=1)
        args.task = 'node_cls'
    elif args.dataset == "ogbn-proteins":
        dataset = PygNodePropPredDataset(name="ogbn-proteins", root='/hdfs1/Data/weather/CoarseGNN_Shubhajit/CoPart-GNN/dataset/')
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
        dataset = WikipediaNetwork(root='./datas    args = arg_correction(args)et', name=args.dataset, geom_gcn_preprocess=False)
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

class Net2(torch.nn.Module):
    def __init__(self, num_features, hidden, num_layers, num_classes):
        super(Net2, self).__init__()
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
    
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def splits_classification(data, num_classes, exp):
    if exp!='fixed':
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        else:
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--experiment', type=str, default='fixed')
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.3)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--gradient_method', type=str, default='GD') #GD: Gradient_Descent, MB: Mini_Batch
    parser.add_argument('--use_community_detection', action='store_true')
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--task', type = str, default = 'node_cls')
    parser.add_argument('--seed', type = int, default = None)
    parser.add_argument('--multi_prop', action='store_true')
    parser.add_argument('--loss_reduction', type = str, default = 'mean')
    parser.add_argument('--property', type = int, default = 0)
    args = parser.parse_args()

    dataset, args = process_dataset(args)
    if (args.task == 'node_cls' or args.task == 'node_reg') and dataset[0].num_nodes > 170000:
        args.use_community_detection = True

    path = f"save/baseline/{args.task}/"+args.output_dir+"/"
    if not os.path.exists('save'):
        os.makedirs('save')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = SummaryWriter(path)

    if args.use_community_detection:
        graph_type = "community"
    else:
        graph_type = "full"
    
    if args.task == "node_cls":
        dataset = dataset.to(device)
        data = dataset[0]
        args.num_classes = torch.unique(data.y).shape[0]

        if args.use_community_detection:
            print("Using community detection")
            g_ig = ig.Graph(n=dataset.num_nodes, edges=dataset.edge_index.t().tolist())
            part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
            mapping = {}
            for i, c in enumerate(part.membership):
                if int(c) not in mapping.keys():
                    mapping[int(c)] = []
                mapping[int(c)].append(i)
            data = merge_communities(dataset, mapping, 100000)
            del dataset
            torch.cuda.empty_cache()
            torch.save(data, f'./dataset/{args.dataset}/saved/{graph_type}_data.pt')

        data = splits_classification(data, args.num_classes, exp = args.experiment)
        model = Net2(data.x.shape[1], args.hidden, args.num_layers, args.num_classes).to(device)
        loss_fn = torch.nn.NLLLoss()
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        avg_time = 0
        all_acc = []
        for run in range(args.runs):
            best_val_loss = 100000
            model.reset_parameters()

            data = data.to(device)
            for epoch in tqdm(range(args.epochs)):
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    val_loss = loss_fn(out[data.val_mask], data.y[data.val_mask])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        #save model
                        torch.save(model.state_dict(), f'./save/node_cls/baselines/baseline_{args.dataset}_{args.experiment}.pt')
            
            start = time.time()
            out = model(data.x, data.edge_index)
            avg_time += time.time() - start
            test_loss = loss_fn(out[data.test_mask], data.y[data.test_mask])
            acc = int(torch.sum(torch.argmax(out, dim=1) == data.y).item()) / len(data.y)
            all_acc.append(acc)
        top_acc= sorted(all_acc, reverse = True)[:10]
        print(f'{args.dataset} {args.experiment}: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f} time: {avg_time/args.runs:.4f}')
    elif args.task == 'node_reg':
        pass
    elif args.task == 'graph_cls':
        pass
    else:
        pass


