import warnings
warnings.simplefilter("ignore")
import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch_scatter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as T_DataLoader
from torch_geometric.loader import DataLoader as G_DataLoader
from utils import load_data_classification, load_data_regression, train_test_val_split, colater, NLLLoss_numpy, L1Loss_numpy
from network import Classify_node, Regress_node, Classify_graph_gc, Classify_graph_gs, Regress_graph_gc, Regress_graph_gs
import logging
from torch_geometric.profile import get_data_size
logging.disable(logging.INFO)
logging.disable(logging.WARNING)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not os.path.exists("results"):
    os.mkdir("results")

def node_train_Gc(model, x, edge_index, mask, y, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    mask = mask.to(device)
    out = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def node_val_Gc(model, x, edge_index, mask, y, loss_fn):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    mask = mask.to(device)
    out = model(x, edge_index)
    loss = loss_fn(out[mask], y[mask])
    return loss.item()

def node_infer_Gs_GD(args, model, graph_data, loss_fn, infer_type):
    total_loss = 0
    total_time = 0
    n = 0
    all_out = np.array([])
    all_label = np.array([])
    max_mem = 0
    if args.task == 'node_cls':
        all_out = all_out.astype(np.int32)
        all_out = all_out.reshape(0, args.num_classes)
        all_label = all_label.astype(np.int32)
    for graph in graph_data:
        max_mem = max(max_mem, get_data_size(graph))
    print(max_mem)
    
        
def node_infer_Gs_MB(args, model, graph_data, loss_fn, infer_type):
    total_loss = 0
    total_time = 0
    all_out = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)
    max_mem = 0
    for graph in graph_data:
        max_mem = max(max_mem, get_data_size(graph))
    print(max_mem)

def node_train_Gs_GD(model, graph_data, loss_fn, optimizer, args):
    total_loss = 0
    all_out = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)
    model.train()
    optimizer.zero_grad()
    max_mem = 0
    for graph in graph_data:
        max_mem = max(max_mem, get_data_size(graph))
    with open('memory.csv', 'a') as f:
        f.write(f"{args.dataset}, False, {args.coarsening_ratio}, {args.task}, extra, {max_mem/(10**6)}\n")
    f.close()
    quit()
        
def node_train_Gs_MB(model, graph_data, loss_fn, optimizer, args):
    total_loss = 0
    all_out = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)
    model.train()
    optimizer.zero_grad()
    max_mem = 0
    for graph in graph_data:
        max_mem = max(max_mem, get_data_size(graph))
    with open('memory.csv', 'a') as f:
        f.write(f"{args.dataset}, False, {args.coarsening_ratio}, {args.task}, {max_mem/(10**6)}\n")
    f.close()
    quit()

def graph_train_Gc(args, model, loader, optimizer, loss_fn):
    total_loss = 0
    model.train()
    optimizer.zero_grad()
    for batch in loader:
        gc = batch[0].to(device)
        y = batch[2].to(device).type(torch.long)
        out = model(gc)
        if args.multi_prop:
            loss = loss_fn(out, y[:, args.property].view(-1, 1))
        else:
            loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def graph_val_Gc(args, model, loader, loss_fn):
    total_loss = 0
    model.eval()
    for batch in loader:
        gc = batch[0].to(device)
        y = batch[2].to(device).type(torch.long)
        out = model(gc)
        if args.multi_prop:
            loss = loss_fn(out, y[:, args.property].view(-1, 1))
        else:
            loss = loss_fn(out, y)
        total_loss += loss.item()
    if args.task == "graph_cls":
        return total_loss / len(loader), int(torch.sum(torch.argmax(out, dim=1) == y).item()) / len(y)
    else:
        return total_loss / len(loader), 0

def graph_train_Gs(args, model, loader, optimizer, loss_fn):
    total_loss = 0
    model.train()
    optimizer.zero_grad()
    for batch in loader:
        set_gs = batch[1]
        y = batch[2].to(device).type(torch.long)
        batch_tensor = batch[3].to(device)
        out = model(set_gs, batch_tensor)
        if args.multi_prop:
            loss = loss_fn(out, y[:, args.property].view(-1, 1))
        else:
            loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def graph_infer_Gs(args, model, loader, loss_fn):
    total_loss = 0
    model.eval()
    all_labels = torch.tensor([], dtype=torch.float32).to(device)
    for batch in loader:
        set_gs = batch[1]
        y = batch[2].to(device).type(torch.long)
        batch_tensor = batch[3].to(device)
        out = model(set_gs, batch_tensor)
        if args.multi_prop:
            loss = loss_fn(out, y[:, args.property].view(-1, 1))
            all_labels = torch.cat((all_labels, y[:, args.property]))
        else:
            loss = loss_fn(out, y)
            all_labels = torch.cat((all_labels, y))
        total_loss += loss.item()
    if args.task == 'graph_cls':
        acc = int(torch.sum(torch.argmax(out, dim=1) == y).item()) / len(y)
    else:
        total_loss = total_loss / torch.std(all_labels).item()
        acc = 0
    return total_loss / len(loader), acc

def node_classification(args, path, dataset, writer, candidate, C_list, Gc_list, subgraph_list):
    all_loss = []
    all_acc = []
    all_time = []
    args.num_classes, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, graphs = load_data_classification(args, dataset, candidate, C_list, Gc_list, args.experiment, subgraph_list)
    if args.normalize_features:
        coarsen_features = F.normalize(coarsen_features, p=1)
    graph_data = G_DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    
    for run in range(args.runs):
        run_writer = SummaryWriter(path + "/run_"+str(run+1), )
        model = Classify_node(args).to(device)
        loss_fn = torch.nn.NLLLoss(reduction=args.loss_reduction).to(device)
        loss_fn_np = NLLLoss_numpy(reduction=args.loss_reduction)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.exp_setup == 'Gc_train_2_Gs_train':
            best_val_loss_Gc =  float('inf')
            best_val_loss_Gs =  float('inf')
            best_test_loss = float('inf')
            best_test_acc = 0
            best_test_time = 0
            #Train and Val on Gc
            for epoch in tqdm(range(args.epochs1), desc=f"Run {run+1}", colour='red'):
                train_loss = node_train_Gc(model, coarsen_features, coarsen_edge, coarsen_train_mask, coarsen_train_labels, loss_fn, optimizer)
                run_writer.add_scalar('Gc_train_loss', train_loss, epoch)
                val_loss = node_val_Gc(model, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, loss_fn)
                run_writer.add_scalar('Gc_val_loss', val_loss, epoch)

                if val_loss < best_val_loss_Gc or epoch == 0:
                    best_val_loss_Gc = val_loss
                    torch.save(model.state_dict(), path+'/model.pt') 

            #Train and val on Gs
            model.load_state_dict(torch.load(path+'/model.pt'))
            for epoch in tqdm(range(args.epochs2), desc=f"Run {run+1}", colour='green'):
                if args.gradient_method == "GD":
                    train_loss, train_max_mem = node_train_Gs_GD(model, graph_data, loss_fn, optimizer, args)
                    val_loss, val_acc, val_time, val_max_mem = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'val')
                    test_loss, test_acc, test_time, test_max_mem = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
                else:
                    train_loss, train_max_mem = node_train_Gs_MB(model, graph_data, loss_fn, optimizer, args)
                    val_loss, val_acc, val_time, val_max_mem = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'val')
                    test_loss, test_acc, test_time, test_max_mem = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')
                    
                run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
                run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
                run_writer.add_scalar('Gs_val_acc', val_acc, epoch)
                run_writer.add_scalar('Gs_test_loss', test_loss, epoch)
                run_writer.add_scalar('Gs_test_acc', test_acc, epoch)
                run_writer.add_scalar('Gs_train_max_mem', train_max_mem, epoch)
                run_writer.add_scalar('Gs_val_max_mem', val_max_mem, epoch)
                run_writer.add_scalar('Gs_test_max_mem', test_max_mem, epoch)

                if val_loss < best_val_loss_Gs or epoch == 0:
                    best_val_loss_Gs = val_loss
                    best_test_loss = test_loss
                    best_test_acc = test_acc
                    best_test_time = test_time
                    torch.save(model.state_dict(), path+'/model.pt')
                    
            all_loss.append(best_test_loss)
            all_acc.append(best_test_acc)
            all_time.append(best_test_time)
        
        elif args.exp_setup == 'Gc_train_2_Gs_infer':
            best_val_loss_Gc =  float('inf')
            best_val_loss_Gs =  float('inf')
            best_test_loss = float('inf')
            best_test_acc = 0
            best_test_time = 0
            #Train and Val on Gc
            for epoch in tqdm(range(args.epochs1), desc=f"Run {run+1}", colour='red'):
                train_loss_gc = node_train_Gc(model, coarsen_features, coarsen_edge, coarsen_train_mask, coarsen_train_labels, loss_fn, optimizer)
                val_loss_gc = node_val_Gc(model, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, loss_fn)
                
                if args.gradient_method == "GD":
                    val_loss_gs, val_acc, val_time, val_max_mem = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'val')
                    test_loss, test_acc, test_time, test_max_mem = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
                else:
                    val_loss_gs, val_acc, val_time, val_max_mem = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'val')
                    test_loss, test_acc, test_time, test_max_mem = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')
                
                run_writer.add_scalar('Gc_val_loss', val_loss_gc, epoch)
                run_writer.add_scalar('Gc_train_loss', train_loss_gc, epoch)
                run_writer.add_scalar('Gs_val_loss', val_loss_gs, epoch)
                run_writer.add_scalar('Gs_val_acc', val_acc, epoch)
                run_writer.add_scalar('Gs_test_loss', test_loss, epoch)
                run_writer.add_scalar('Gs_test_acc', test_acc, epoch)
                run_writer.add_scalar('Gs_val_max_mem', val_max_mem, epoch)
                run_writer.add_scalar('Gs_test_max_mem', test_max_mem, epoch)

                if val_loss_gc < best_val_loss_Gc or epoch == 0:
                    best_val_loss_Gc = val_loss_gc
                    best_test_loss = test_loss
                    best_test_acc = test_acc
                    best_test_time = test_time
                    torch.save(model.state_dict(), path+'/model.pt') 

            all_loss.append(best_test_loss)
            all_acc.append(best_test_acc)
            all_time.append(best_test_time)

        elif args.exp_setup == 'Gs_train_2_Gs_infer':
            if run == 0:
                del coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge
                torch.cuda.empty_cache()
            best_val_loss_Gs =  float('inf')
            best_test_loss = float('inf')
            best_test_acc = 0
            best_test_time = 0
            #Train on Gs
            for epoch in tqdm(range(args.epochs2), desc=f"Run {run+1}", colour='green'):
                if args.gradient_method == "GD":
                    train_loss, train_max_mem = node_train_Gs_GD(model, graph_data, loss_fn, optimizer,args)
                    val_loss, val_acc, val_time, val_max_mem = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'val')
                    test_loss, test_acc, test_time, test_max_mem = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
                else:
                    train_loss, train_max_mem = node_train_Gs_MB(model, graph_data, loss_fn, optimizer,args)
                    val_loss, val_acc, val_time, val_max_mem = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'val')
                    test_loss, test_acc, test_time, test_max_mem = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')

                run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
                run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
                run_writer.add_scalar('Gs_val_acc', val_acc, epoch)
                run_writer.add_scalar('Gs_test_loss', test_loss, epoch)
                run_writer.add_scalar('Gs_test_acc', test_acc, epoch)
                run_writer.add_scalar('Gs_train_max_mem', train_max_mem, epoch)
                run_writer.add_scalar('Gs_val_max_mem', val_max_mem, epoch)
                run_writer.add_scalar('Gs_test_max_mem', test_max_mem, epoch)

                if val_loss < best_val_loss_Gs or epoch == 0:
                    best_val_loss_Gs = val_loss
                    best_test_loss = test_loss
                    best_test_acc = test_acc
                    best_test_time = test_time
                    torch.save(model.state_dict(), path+'/model.pt')
                
            all_loss.append(best_test_loss)
            all_acc.append(best_test_acc)
            all_time.append(best_test_time)

    top_acc = sorted(all_acc, reverse=True)[:10]
    top_loss = sorted(all_loss)[:10]

    if not os.path.exists(f"results/{args.dataset}.csv"):
        with open(f"results/{args.dataset}.csv", 'w') as f:
            f.write('dataset,coarsening_method,coarsening_ratio,experiment,exp_setup,extra_nodes,cluster_node,community_used,hidden,runs,num_layers,batch_size,lr,ave_acc,ave_time,top_10_acc,best_acc,top_10_loss,best_loss\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.experiment},{args.exp_setup},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_acc)} +/- {np.std(all_acc)},{np.mean(all_time)},{np.mean(top_acc)} +/- {np.std(top_acc)}, {top_acc[0]}, {np.mean(top_loss)} +/- {np.std(top_loss)}, {top_loss[0]}\n")
    print("#####################################################################")
    print(f"dataset: {args.dataset}")
    print(f"experiment: {args.experiment}")
    print(f"exp_setup: {args.exp_setup}")
    print(f"extra_nodes: {args.extra_node}")
    print(f"cluster_node: {args.cluster_node}")
    print(f"hidden: {args.hidden}")
    print(f"runs: {args.runs}")
    print(f"num_layers: {args.num_layers1}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"coarsening_ratio: {args.coarsening_ratio}")
    print(f"coarsening_method: {args.coarsening_method}")
    print(f"ave_acc: {np.mean(all_acc)} +/- {np.std(all_acc)}")
    print(f"ave_time: {np.mean(all_time)}")
    print(f"top_10_acc: {np.mean(top_acc)} +/- {np.std(top_acc)}")
    print(f"best_acc: {top_acc[0]}")
    print(f"top_10_loss: {np.mean(top_loss)} +/- {np.std(top_loss)}")
    print(f"best_loss: {top_loss[0]}")
    print(f"maximum memory: {test_max_mem}")
    print("#####################################################################")

def node_regression(args, path, dataset, writer, subgraph_list):
    all_loss = []
    all_acc = []
    all_time = []
    graphs = load_data_regression(args, dataset, subgraph_list)
    graph_data = G_DataLoader(graphs, batch_size=args.batch_size, shuffle=False, pin_memory=False)

    for run in range(args.runs):
        run_writer = SummaryWriter(path + "/run_"+str(run+1))
        model = Regress_node(args).to(device)
        loss_fn = torch.nn.L1Loss(reduction=args.loss_reduction).to(device)
        loss_fn_np = L1Loss_numpy(reduction=args.loss_reduction)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_val_loss =  float('inf')
        best_test_loss = float('inf')
        best_test_acc = 0
        best_test_time = 0
        #Train on Gs
        for epoch in tqdm(range(args.epochs2), desc=f"Run {run+1}", colour='green'):
            if args.gradient_method == "GD":
                train_loss, train_max_mem = node_train_Gs_GD(model, graph_data, loss_fn, optimizer,args)
                val_loss, val_acc, val_time, val_max_mem = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'val')
                test_loss, test_acc, test_time, test_max_mem = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
            else:
                train_loss, train_max_mem = node_train_Gs_MB(model, graph_data, loss_fn, optimizer,args)
                val_loss, val_acc, val_time, val_max_mem = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'val')
                test_loss, test_acc, test_time, test_max_mem = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')
                
            run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
            run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
            run_writer.add_scalar('Gs_test_loss', test_loss, epoch)
            run_writer.add_scalar('Gs_train_max_mem', train_max_mem, epoch)
            run_writer.add_scalar('Gs_val_max_mem', val_max_mem, epoch)
            run_writer.add_scalar('Gs_test_max_mem', test_max_mem, epoch)

            if val_loss < best_val_loss or epoch == 0:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_test_time = test_time
                torch.save(model.state_dict(), path+'/model.pt')
        
        all_loss.append(best_test_loss)
        all_time.append(best_test_time)

    top_loss = sorted(all_loss)[:10]

    if not os.path.exists(f"results/{args.dataset}.csv"):
        with open(f"results/{args.dataset}.csv", 'w') as f:
            f.write('dataset,coarsening_method,coarsening_ratio,extra_nodes,cluster_node,community_used,hidden,runs,num_layers,batch_size,lr,ave_time,top_10_loss,best_loss\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_time)},{np.mean(top_loss)} +/- {np.std(top_loss)},{top_loss[0]}\n")
    print("#####################################################################")
    print(f"dataset: {args.dataset}")
    print(f"extra_nodes: {args.extra_node}")
    print(f"cluster_node: {args.cluster_node}")
    print(f"hidden: {args.hidden}")
    print(f"runs: {args.runs}")
    print(f"num_layers: {args.num_layers1}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"coarsening_ratio: {args.coarsening_ratio}")
    print(f"coarsening_method: {args.coarsening_method}")
    print(f"ave_time: {np.mean(all_time)}")
    print(f"top_10_loss: {np.mean(top_loss)} +/- {np.std(top_loss)}")
    print(f"best_loss: {top_loss[0]}")
    print(f"maximum memory: {test_max_mem}")
    print("#####################################################################")

def graph_classification(args, path, writer, dataset):
    train_split, test_split, val_split = train_test_val_split(dataset, shuffle=True)
    colater_fn = colater()
    train_loader = T_DataLoader(train_split, batch_size=args.batch_size, shuffle=True, collate_fn=colater_fn)
    test_loader = T_DataLoader(test_split, batch_size=args.batch_size, shuffle=True, collate_fn=colater_fn)
    val_loader = T_DataLoader(val_split, batch_size=args.batch_size, shuffle=True, collate_fn=colater_fn)

    model_gc = Classify_graph_gc(args).to(device)
    model_gs = Classify_graph_gs(args).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer_gc = torch.optim.Adam(model_gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_gs = torch.optim.Adam(model_gs.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.exp_setup == 'Gc_train_2_Gs_train':
        best_val_loss_Gc =  float('inf')
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        best_test_acc = 0
        for epoch in tqdm(range(args.epochs1), desc=f"Gc Train", colour='red'):
            train_loss_gc = graph_train_Gc(args, model_gc, train_loader, optimizer_gc, loss_fn)
            val_loss_gc, val_acc_gc = graph_val_Gc(args, model_gc, val_loader, loss_fn)
            writer.add_scalar('Gc_train_loss', train_loss_gc, epoch)
            writer.add_scalar('Gc_val_loss', val_loss_gc, epoch)
            if val_loss_gc < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss_gc
                torch.save(model_gc.state_dict(), path+'model.pt')
        
        model_gs.load_state_dict(torch.load(path+'/model.pt'))
        for epoch in tqdm(range(args.epochs2), desc=f"Gs Train", colour='green'):
            train_loss = graph_train_Gs(args, model_gs, train_loader, optimizer_gs, loss_fn)
            val_loss, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            writer.add_scalar('Gs_train_loss', train_loss, epoch)
            writer.add_scalar('Gs_val_loss', val_loss, epoch)
            writer.add_scalar('Gs_test_loss', test_loss, epoch)
            writer.add_scalar('Gs_val_acc', val_acc, epoch)
            writer.add_scalar('Gs_test_acc', test_acc, epoch)

            if val_loss < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss
                best_test_loss = test_loss
                best_test_acc = test_acc
                torch.save(model_gs.state_dict(), path+'/model.pt')
    
    elif args.exp_setup == "Gc_train_2_Gc_infer":
        best_val_loss_Gc =  float('inf')
        best_test_loss = float('inf')
        best_test_acc = 0
        for epoch in tqdm(range(args.epochs1), desc=f"Gc Train", colour='red'):
            train_loss = graph_train_Gc(args, model_gc, train_loader, optimizer_gc, loss_fn)
            val_loss, val_acc = graph_val_Gc(args, model_gc, val_loader, loss_fn)
            test_loss, test_acc = graph_val_Gc(args, model_gc, test_loader, loss_fn)
            writer.add_scalar('Gc_train_loss', train_loss, epoch)
            writer.add_scalar('Gc_val_loss', val_loss, epoch)
            writer.add_scalar('Gc_test_loss', test_loss, epoch)
            writer.add_scalar('Gc_val_acc', val_acc, epoch)
            writer.add_scalar('Gc_test_acc', test_acc, epoch)

            if val_loss < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss
                best_test_loss = test_loss
                best_test_acc = test_acc
                torch.save(model_gc.state_dict(), path+'/model.pt')

    elif args.exp_setup == 'Gc_train_2_Gs_infer':
        best_val_loss_Gc =  float('inf')
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        best_test_acc = 0
        best_val_acc = 0
        for epoch in tqdm(range(args.epochs1), desc=f"Gc Train", colour='red'):
            train_loss = graph_train_Gc(args, model_gc, train_loader, optimizer_gc, loss_fn)
            val_loss_gc, val_acc_gc = graph_val_Gc(args, model_gc, val_loader, loss_fn)
            
            val_loss_gs, val_acc_gs = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            
            writer.add_scalar('Gc_val_loss', val_loss_gc, epoch)
            writer.add_scalar('Gc_train_loss', train_loss, epoch)
            writer.add_scalar('Gs_val_loss', val_loss_gs, epoch)
            writer.add_scalar('Gs_test_loss', test_loss, epoch)
            writer.add_scalar('Gs_val_acc', val_acc_gs, epoch)
            writer.add_scalar('Gs_test_acc', test_acc, epoch)
            
            if val_loss_gc < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss_gc
                best_test_loss = test_loss
                best_test_acc = test_acc
                torch.save(model_gc.state_dict(), path+'model.pt')

    elif args.exp_setup == "Gs_train_2_Gs_infer":
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        best_test_acc = 0
        for epoch in tqdm(range(args.epochs2), desc=f"Gs Train", colour='green'):
            train_loss = graph_train_Gs(args, model_gs, train_loader, optimizer_gs, loss_fn)
            val_loss, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            writer.add_scalar('Gs_train_loss', train_loss, epoch)
            writer.add_scalar('Gs_val_loss', val_loss, epoch)
            writer.add_scalar('Gs_test_loss', test_loss, epoch)
            writer.add_scalar('Gs_val_acc', val_acc, epoch)
            writer.add_scalar('Gs_test_acc', test_acc, epoch)

            if val_loss < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss
                best_test_loss = test_loss
                best_test_acc = test_acc
                torch.save(model_gs.state_dict(), path+'/model.pt')

    if not os.path.exists(f"results/{args.dataset}.csv"):
        with open(f"results/{args.dataset}.csv", 'w') as f:
            f.write('dataset,coarsening_method,coarsening_ratio,exp_setup,extra_nodes,cluster_node,community_used,hidden,num_layers1,num_layers2,epochs1,epochs2,batch_size,lr,best_test_loss,best_test_acc\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.exp_setup},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.num_layers1},{args.num_layers2},{args.epochs1},{args.epochs2},{args.batch_size},{args.lr},{best_test_loss},{best_test_acc}\n")
    print("#####################################################################")
    print(f"dataset: {args.dataset}")
    print(f"exp_setup: {args.exp_setup}")
    print(f"extra_nodes: {args.extra_node}")
    print(f"cluster_node: {args.cluster_node}")
    print(f"hidden: {args.hidden}")
    print(f"num_layers: {args.num_layers1}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"coarsening_ratio: {args.coarsening_ratio}")
    print(f"coarsening_method: {args.coarsening_method}")
    print(f"best_test_loss: {best_test_loss}")
    print(f"best_test_acc: {best_test_acc}")
    print("#####################################################################")

def graph_regression(args, path, writer, dataset):
    train_split, test_split, val_split = train_test_val_split(dataset, shuffle=True)
    colater_fn = colater()
    train_loader = T_DataLoader(train_split, batch_size=args.batch_size, shuffle=True, collate_fn=colater_fn)
    test_loader = T_DataLoader(test_split, batch_size=args.batch_size, shuffle=True, collate_fn=colater_fn)
    val_loader = T_DataLoader(val_split, batch_size=args.batch_size, shuffle=True, collate_fn=colater_fn)

    model_gc = Regress_graph_gc(args).to(device)
    model_gs = Regress_graph_gs(args).to(device)
    loss_fn = torch.nn.L1Loss().to(device)
    optimizer_gc = torch.optim.Adam(model_gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_gs = torch.optim.Adam(model_gs.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.exp_setup == 'Gc_train_2_Gs_train':
        best_val_loss_Gc =  float('inf')
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        for epoch in tqdm(range(args.epochs1), desc=f"Gc Train", colour='red'):
            train_loss_gc = graph_train_Gc(args, model_gc, train_loader, optimizer_gc, loss_fn)
            val_loss_gc, val_acc = graph_val_Gc(args, model_gc, val_loader, loss_fn)
            writer.add_scalar('Gc_val_loss', val_loss_gc, epoch)
            writer.add_scalar('Gc_train_loss', train_loss_gc, epoch)
            if val_loss_gc < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss_gc
                torch.save(model_gc.state_dict(), path+'model.pt')
        
        model_gs.load_state_dict(torch.load(path+'/model.pt'))
        for epoch in tqdm(range(args.epochs2), desc=f"Gs Train", colour='green'):
            train_loss_gs = graph_train_Gs(args, model_gs, train_loader, optimizer_gs, loss_fn)
            val_loss_gs, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            writer.add_scalar('Gs_train_loss', train_loss_gs, epoch)
            writer.add_scalar('Gs_val_loss', val_loss_gs, epoch)
            writer.add_scalar('Gs_test_loss', test_loss, epoch)

            if val_loss_gs < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss_gs
                best_test_loss = test_loss
                torch.save(model_gs.state_dict(), path+'/model.pt')
    
    elif args.exp_setup == "Gc_train_2_Gc_infer":
        best_val_loss_Gc =  float('inf')
        best_test_loss = float('inf')
        for epoch in tqdm(range(args.epochs1), desc=f"Gc Train", colour='red'):
            train_loss = graph_train_Gc(args, model_gc, train_loader, optimizer_gc, loss_fn)
            val_loss, val_acc = graph_val_Gc(args, model_gc, val_loader, loss_fn)
            test_loss, test_acc = graph_val_Gc(args, model_gc, test_loader, loss_fn)
            writer.add_scalar('Gc_train_loss', train_loss, epoch)
            writer.add_scalar('Gc_val_loss', val_loss, epoch)
            writer.add_scalar('Gc_test_loss', test_loss, epoch)
            writer.add_scalar('Gc_val_acc', val_acc, epoch)
            writer.add_scalar('Gc_test_acc', test_acc, epoch)

            if val_loss < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss
                best_test_loss = test_loss
                torch.save(model_gc.state_dict(), path+'/model.pt')

    elif args.exp_setup == 'Gc_train_2_Gs_infer':
        best_val_loss_Gc =  float('inf')
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        for epoch in tqdm(range(args.epochs1), desc=f"Gc train", colour='red'):
            train_loss = graph_train_Gc(args, model_gc, train_loader, optimizer_gc, loss_fn)
            val_loss_gc, val_acc_gc = graph_val_Gc(args, model_gc, val_loader, loss_fn)
            val_loss_gs, val_acc_gs = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            
            writer.add_scalar('Gc_train_loss', train_loss, epoch)
            writer.add_scalar('Gc_val_loss', val_loss_gc, epoch)
            writer.add_scalar('Gs_val_loss', val_loss_gs, epoch)
            writer.add_scalar('Gs_test_loss', test_loss, epoch)
            if val_loss_gc < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss_gc
                best_val_loss_Gs = val_loss_gs
                best_test_loss = test_loss
                torch.save(model_gc.state_dict(), path+'model.pt')
            
    elif args.exp_setup == "Gs_train_2_Gs_infer":
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        for epoch in tqdm(range(args.epochs2), desc=f"Gs Train", colour='green'):
            train_loss = graph_train_Gs(args, model_gs, train_loader, optimizer_gs, loss_fn)
            val_loss, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            writer.add_scalar('Gs_train_loss', train_loss, epoch)
            writer.add_scalar('Gs_val_loss', val_loss, epoch)
            writer.add_scalar('Gs_test_loss', test_loss, epoch)

            if val_loss < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss
                best_test_loss = test_loss
                torch.save(model_gs.state_dict(), path+'/model.pt')

    if not os.path.exists(f"results/{args.dataset}.csv"):
        with open(f"results/{args.dataset}.csv", 'w') as f:
            if args.multi_prop:
                f.write('dataset,coarsening_method,coarsening_ratio,exp_setup,extra_nodes,cluster_node,community_used,hidden,num_layers1,num_layers2,epochs1,epochs2,batch_size,lr,best_test_loss,property_idx}\n')
            else:
                f.write('dataset,coarsening_method,coarsening_ratio,exp_setup,extra_nodes,cluster_node,community_used,hidden,num_layers1,num_layers2,epochs1,epochs2,batch_size,lr,best_test_loss}\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        if args.multi_prop:
            f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.exp_setup},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.num_layers1},{args.num_layers2},{args.epochs1},{args.epochs2},{args.batch_size},{args.lr},{best_test_loss},{args.property}\n")
        else:
            f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.exp_setup},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.num_layers1},{args.num_layers2},{args.epochs1},{args.epochs2},{args.batch_size},{args.lr},{best_test_loss}\n")
    print("#####################################################################")
    print(f"dataset: {args.dataset}")
    print(f"exp_setup: {args.exp_setup}")
    print(f"extra_nodes: {args.extra_node}")
    print(f"cluster_node: {args.cluster_node}")
    print(f"hidden: {args.hidden}")
    print(f"num_layers: {args.num_layers1}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"coarsening_ratio: {args.coarsening_ratio}")
    print(f"coarsening_method: {args.coarsening_method}")
    print(f"best_test_loss: {best_test_loss}")
    if args.multi_prop:
        print(f"property_idx: {args.property}")
    print("#####################################################################")
