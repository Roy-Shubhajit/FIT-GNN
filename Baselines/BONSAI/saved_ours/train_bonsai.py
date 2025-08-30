import copy
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from model import GCN, GCN_inductive, GAT, GIN, SAGE


saint_datasets = ["flickr", "ogbn-arxiv", "reddit"]

def nop_scaler(x):
    return x


def load_dataset(dataset_name, root):
    global saint_datasets
    if dataset_name in saint_datasets:
        from sklearn.preprocessing import StandardScaler
        from load_saint_dataset import load_saint_dataset
        data = load_saint_dataset(dataset_name, root=root)
        feat_full = data.x.cpu().numpy()
        scaler = StandardScaler()
        scaler.fit(feat_full)
        feat_full = torch.tensor(scaler.transform(feat_full))
        data.x = feat_full
        return {"data": data, "scaler": lambda feat: scaler.transform(feat)}
    else:
        from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, WikiCS
        if dataset_name == "dblp":
            data = CitationFull(root=root+"/dblp", name="dblp")._data
        elif dataset_name == "Physics" or dataset_name == "physics":
            data = Coauthor(root=root+"/Physics", name="Physics")._data
        elif dataset_name == "WikiCS" or dataset_name == "wikics":
            data = WikiCS(root=root+"/WikiCS")._data
        elif dataset_name == "products":
            data = PygNodePropPredDataset(
            name="ogbn-products", root=root+"/ogbn-products"
        )._data
        #change the dimension of labels to 1D
            if isinstance(data.y, torch.Tensor) and data.y.dim() > 1:
                data.y = data.y.squeeze(1)
        else:
            data = Planetoid(root=root, name=dataset_name)._data
        return {"data": data, "scaler": nop_scaler}


def load_syn_dataset(dataset_name, target_frac, synthetic_root, scaler):
    global saint_datasets
    from pathlib import Path
    path = Path(synthetic_root) / f"{dataset_name}-{target_frac}/data_m_0.9.pt"
    data_syn = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data_syn, dict):
        data = data_syn["data"]
        feat_syn = data.x.cpu().numpy()
        feat_syn = torch.tensor(scaler(feat_syn))
        data.x = feat_syn
        data_syn["data"] = data
    else:
        feat_syn = data_syn.x.cpu().numpy()
        feat_syn = torch.tensor(scaler(feat_syn))
        data_syn.x = feat_syn
        features_used = np.arange(data_syn.x.shape[1])
        data_syn = {"data": data_syn, "features_used": features_used}
    if dataset_name in saint_datasets:
        import scipy.sparse as sp
        from torch_sparse import SparseTensor
        data = data_syn["data"]
        d = np.ones(data.edge_index.shape[1])
        r = data.edge_index[0].cpu().numpy()
        c = data.edge_index[1].cpu().numpy()
        n = data.x.shape[0]
        adj = sp.csr_matrix((d, (r, c)), shape=(n, n))
        adj = adj.tolil()
        adj = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1 / 2).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        adj = r_mat_inv.dot(adj)
        adj = adj.dot(r_mat_inv)
        adj = adj.tocoo().astype(np.float32)
        sparserow = torch.LongTensor(adj.row).unsqueeze(1)
        sparsecol = torch.LongTensor(adj.col).unsqueeze(1)
        sparseconcat = torch.cat((sparserow, sparsecol), 1)
        sparsedata = torch.FloatTensor(adj.data)
        adj = torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(adj.shape))
        adj = SparseTensor(
            row=adj._indices()[0],
            col=adj._indices()[1],
            value=adj._values(),
            sparse_sizes=adj.size(),
        )
        data_syn["data"].adj = adj
    return data_syn


def get_splits(nnodes, nclasses):
    num_of_train_nodes, num_of_val_nodes = 20 * nclasses, 30 * nclasses
    total_train_val_nodes = num_of_train_nodes + num_of_val_nodes
    num_of_test_nodes = nnodes - num_of_train_nodes - num_of_val_nodes
    train_node_fraction = num_of_train_nodes / total_train_val_nodes
    val_node_fraction = num_of_val_nodes / total_train_val_nodes
    test_node_fraction = num_of_test_nodes / nnodes
    
    train_, test = train_test_split(range(nnodes), test_size=test_node_fraction, random_state=42)
    rng = np.random.RandomState(seed=0)
    sample_size = int(len(train_) * train_node_fraction)
    train = rng.choice(train_, (sample_size,), replace=False)
    val = list(set(range(nnodes))-set(train).union(set(test)))
    return {"train": train, "test": test, "val": val}


def train_backend_inductive(model, nepochs, data, data_syn, splits, writer):
    opt = optim.Adam(model.parameters())
    loss_fn = lambda out, y: F.nll_loss(out, y)
    loop = tqdm(range(nepochs), ascii=True, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    d = data["data"]
    d = d.to(device, "x", "adj", "y")
    d_syn = data_syn["data"]
    d_syn = d_syn.to(device, "x", "adj", "y", "target")
    test = splits["test"]
    val = splits["val"]
    for epoch in loop:
        model.train()
        out = model(d_syn.x, d_syn.adj)
        loss = loss_fn(out[d_syn.target], d_syn.y[d_syn.target])
        opt.zero_grad()
        loss.backward()
        opt.step()
        writer.add_scalar('loss/train', loss.item(), epoch)
        model.eval()
        with torch.no_grad():
            out = model(d.x, d.adj)
            loss = loss_fn(out[val], d.y[val])
            writer.add_scalar('loss/val', loss.item(), epoch)
            preds = out[val].max(1)[1].cpu().numpy()
            acc = accuracy_score(d.y[val].cpu().numpy(), preds)
            writer.add_scalar('acc/val', acc, epoch)
            if acc > best_acc_val:
                best_acc_val = acc
                weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    with torch.no_grad():
        out = model(d.x, d.adj)
        preds = out[test].max(1)[1].cpu().numpy()
        acc = accuracy_score(d.y[test].cpu().numpy(), preds)
    writer.add_scalar('test_acc/test', acc)
    return acc


def train_backend_pyg(args, model, nepochs, data, data_syn, splits, writer):
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loop = tqdm(range(nepochs), ascii=True, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    d = data["data"]
    d = d.to(device, "x", "edge_index", "y")
    d_syn = data_syn["data"]
    d_syn = d_syn.to(device, "x", "edge_index", "y", "target")
    test = splits["test"]
    val = splits["val"]
    # print(d_syn, d_syn.x.type(), d_syn.edge_index.type())
    from my_profiling import profile
    with profile(False):
        for epoch in loop:
            model.train()
            out = model(d_syn.x, d_syn.edge_index)
            loss = loss_fn(out[d_syn.target], d_syn.y[d_syn.target])
            opt.zero_grad()
            loss.backward()
            opt.step()
            writer.add_scalar('loss/train', loss.item(), epoch)
            model.eval()
            with torch.no_grad():
                out = model(d.x, d.edge_index)
                loss = loss_fn(out[val], d.y[val])
                writer.add_scalar('loss/val', loss.item(), epoch)
                preds = out[val].max(1)[1].cpu().numpy()
                acc = accuracy_score(d.y[val].cpu().numpy(), preds)
                writer.add_scalar('acc/val', acc, epoch)
                if acc > best_acc_val:
                    best_acc_val = acc
                    weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    with torch.no_grad():
        out = model(d.x, d.edge_index)
        preds = out[test].max(1)[1].cpu().numpy()
        acc = accuracy_score(d.y[test].cpu().numpy(), preds)
    print(f"Dataset: {args["ds"].lower()}, Target Fraction: {args["tf"]}, Model: {args["model"]}, Test Accuracy: {acc}")
    file_exists = os.path.isfile("results.csv")
    with open("results.csv", "a") as f:
        if not file_exists:
            f.write("dataset,target_fraction,model,test_accuracy\n")
        f.write(f"{args['ds'].lower()},{args['tf']},{args['model']},{acc}\n")
    writer.add_scalar('test_acc/test', acc)
    return acc


def train_model(args, model_type, model, nepochs, data, data_syn, splits, writer):
    if model_type == "GCN_inductive":
        train_backend_inductive(model, nepochs, data, data_syn, splits, writer)
    else:
        train_backend_pyg(args, model, nepochs, data, data_syn, splits, writer)


def main(args):
    global saint_datasets
    data = load_dataset(args["ds"].lower(), "/datasets")
    data_syn = load_syn_dataset(args["ds"], args["tf"], args["sr"], data["scaler"])
    data["data"].x = data["data"].x[:,data_syn["features_used"]]
    nnodes = data["data"].x.shape[0]
    splits = get_splits(nnodes, torch.unique(data["data"].y).shape[0])
    model_type = "GCN_inductive" if args["model"] == "GCN" and args["ds"] in saint_datasets else args["model"]
    model_type = args["model"]
    model_class = globals()[model_type]
    nclasses = torch.unique(data["data"].y).shape[0]
    input_dim = data["data"].x.shape[1]
    output_dim = nclasses
    hidden_dim = 1024 if args["ds"] in saint_datasets else 128
    model = model_class(indim=input_dim, outdim=output_dim, hidim=hidden_dim)
    writer = SummaryWriter(f'logs/bonsai_{args["ds"]}_{args["model"]}_{args["tf"]}')
    from timing import Timer
    with Timer(f"Training Bonsai for {args['ne']} epochs") as timer:
        train_model(args, model_type, model, args["ne"], data, data_syn, splits, writer)
    writer.add_scalar("experiment/time", timer.dur)


if __name__=="__main__":
    root = (Path('..') / 'datasets').absolute()
    synthetic_root = "/saved_ours"
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", required=True) # dataset
    parser.add_argument("-tf", required=True) # target_frac
    parser.add_argument("-ne", type=int, required=True) # nepochs
    parser.add_argument("-dr", default=root) # data root
    parser.add_argument("-sr", default=synthetic_root) # synthetic root
    parser.add_argument("-model", required=True) # model: GCN, GAT, SAGE, GIN?
    args = parser.parse_args()
    main(vars(args))
