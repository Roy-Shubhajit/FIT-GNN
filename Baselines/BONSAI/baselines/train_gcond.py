import copy
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from models import GCNE, GCN_GDEM, GATE, GINE, GCN


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
        edge_attr = torch.ones(data.edge_index.shape[1], device=data.edge_index.device)[:, None]
        data.edge_attr = edge_attr
        return {"data": data, "scaler": lambda feat: scaler.transform(feat)}
    else:
        from torch_geometric.datasets import Planetoid
        data = Planetoid(root=root, name=dataset_name)._data
        edge_attr = torch.ones(data.edge_index.shape[1], device=data.edge_index.device)[:, None]
        data.edge_attr = edge_attr
        return {"data": data, "scaler": nop_scaler}


def load_syn_dataset(dataset_name, target_frac, synthetic_root, scaler):
    global saint_datasets
    from pathlib import Path
    path = Path(synthetic_root) / f"{dataset_name}-{target_frac}"
    adj_file = path / "adj_1.pt"
    feat_file = path / "feat_1.pt"
    label_file = path / "labels_1.pt"
    adj = torch.load(adj_file, map_location="cpu")
    feat = torch.load(feat_file, map_location="cpu")
    label = torch.load(label_file, map_location="cpu")
    import scipy.sparse as sp
    from torch_sparse import SparseTensor
    adj = sp.csr_matrix(adj.numpy())
    adj = adj.tocoo()
    row, col = adj.row, adj.col
    edge_weights = torch.tensor(adj.tocsr()[row,col]).squeeze()[:, None]
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)]).long()
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
    data_syn = Data(x=feat, edge_index=edge_index, edge_attr=edge_weights, y=label, adj=adj)
    return {"data":data_syn}


def get_splits(nnodes):
    train_, test = train_test_split(range(nnodes), test_size=0.2, random_state=42)
    rng = np.random.RandomState(seed=0)
    sample_size = int(len(train_) * 0.7)
    train = rng.choice(train_, (sample_size,), replace=False)
    val = list(set(range(nnodes))-set(train).union(set(test)))
    return {"train": train, "test": test, "val": val}


def train_backend_GDEM(model, nepochs, data, data_syn, splits, writer):
    opt = optim.Adam(model.parameters())
    loss_fn = lambda out, y: F.nll_loss(out, y)
    loop = tqdm(range(nepochs), ascii=True, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    d = data["data"]
    d = d.to(device, "x", "adj", "y")
    d_syn = data_syn["data"]
    d_syn = d_syn.to(device, "x", "adj", "y")
    test = splits["test"]
    val = splits["val"]
    for epoch in loop:
        model.train()
        out = model(d_syn.x, d_syn.adj)
        loss = loss_fn(out, d_syn.y)
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


def train_backend_pyg(model, nepochs, data, data_syn, splits, writer):
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loop = tqdm(range(nepochs), ascii=True, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    d = data["data"]
    d_syn = data_syn["data"]
    test = splits["test"]
    val = splits["val"]
    # print(d_syn, d_syn.x.type(), d_syn.edge_index.type())
    from my_profiling import profile
    with profile(False):
        for epoch in loop:
            model.train()
            d_syn = d_syn.to(device, "x", "edge_index", "y", "edge_attr")
            out = model(d_syn.x, d_syn.edge_index, d_syn.edge_attr)
            loss = loss_fn(out, d_syn.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            writer.add_scalar('loss/train', loss.item(), epoch)
            model.eval()
            with torch.no_grad():
                d = d.to(device, "x", "edge_index", "y", "edge_attr")
                out = model(d.x, d.edge_index, d.edge_attr)
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
        out = model(d.x, d.edge_index, d.edge_attr)
        preds = out[test].max(1)[1].cpu().numpy()
        acc = accuracy_score(d.y[test].cpu().numpy(), preds)
    writer.add_scalar('test_acc/test', acc)
    return acc


def train_model(model_type, model, nepochs, data, data_syn, splits, writer):
    if model_type in ["GATE"]:
        data_syn["data"].edge_attr = data_syn["data"].edge_attr.squeeze()
        data["data"].edge_attr = data["data"].edge_attr.squeeze()
    if model_type == "GCN_GDEM":
        train_backend_GDEM(model, nepochs, data, data_syn, splits, writer)
    else:
        train_backend_pyg(model, nepochs, data, data_syn, splits, writer)


def main(args):
    global saint_datasets
    data = load_dataset(args["ds"].lower(), args["dr"])
    data_syn = load_syn_dataset(args["ds"].lower(), args["tf"], args["sr"], data["scaler"])
    nnodes = data["data"].x.shape[0]
    splits = get_splits(nnodes)
    model_type = "GCN_GDEM" if args["model"] == "GCN" and args["ds"] in saint_datasets else args["model"]
    model_type = args["model"]
    model_type = model_type + "E" #if not model_type.startswith("GCN") else model_type
    model_class = globals()[model_type]
    nclasses = torch.unique(data["data"].y).shape[0]
    input_dim = data["data"].x.shape[1]
    output_dim = nclasses
    hidden_dim = 1024 if args["ds"] in saint_datasets else 128
    model = model_class(indim=input_dim, outdim=output_dim, hidim=hidden_dim)
    writer = SummaryWriter(f'logs/GCond_{args["ds"]}_{args["model"]}_{args["tf"]}')
    from timing import Timer
    with Timer(f"Training GCond for {args['ne']} epochs") as timer:
        train_model(model_type, model, args["ne"], data, data_syn, splits, writer)
    writer.add_scalar("experiment/time", timer.dur)


if __name__=="__main__":
    from pathlib import Path
    root = str(Path("..").absolute() / "datasets")
    synthetic_root = "saved_gcond"
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", required=True) # dataset
    parser.add_argument("-tf", required=True) # target_frac
    parser.add_argument("-ne", type=int, required=True) # nepochs
    parser.add_argument("-dr", default=root) # data root
    parser.add_argument("-sr", default=synthetic_root) # synthetic root
    parser.add_argument("-model", required=True) # model: GCN, GAT, GIN?
    args = parser.parse_args()
    main(vars(args))
