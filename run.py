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
from utils import load_data_classification, load_data_regression, splits_classification, splits_regression, train_test_val_split, colater, NLLLoss_numpy, L1Loss_numpy
from network import Classify_node, Regress_node, Classify_graph_gc, Classify_graph_gs, Regress_graph_gc, Regress_graph_gs
from torch_geometric.profile import get_data_size
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not os.path.exists("results"):
    os.mkdir("results")

### FIT-GNN Functions

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
    if args.task == 'node_cls':
        all_out = all_out.astype(np.int32)
        all_out = all_out.reshape(0, args.num_classes)
        all_label = all_label.astype(np.int32)
    for graph in graph_data:
        model.eval()
        if infer_type == 'test':
            if True in graph.test_mask:
                x = graph.x.to(device)
                y = graph.y.to(device)
                edge_index = graph.edge_index.to(device)
                start_time = time.time()
                out = model(x, edge_index)
                total_time += time.time() - start_time
                test_mask = graph.test_mask.to(device)
                if args.task == 'node_reg':
                    all_out = np.concatenate((all_out,out[test_mask].flatten().cpu().detach().numpy()), axis=0)
                else:
                    all_out = np.concatenate((all_out,out[test_mask].cpu().detach().numpy()), axis=0)
                all_label = np.concatenate((all_label, y[test_mask].flatten().cpu().numpy()))
                del x, y, edge_index, test_mask
                torch.cuda.empty_cache()
                n += 1
            else:
                continue
        else:
            if True in graph.val_mask:
                x = graph.x.to(device)
                y = graph.y.to(device)
                edge_index = graph.edge_index.to(device)
                start_time = time.time()
                out = model(x, edge_index)
                total_time += time.time() - start_time
                val_mask = graph.val_mask.to(device)
                if args.task == 'node_reg':
                    all_out = np.concatenate((all_out,out[val_mask].flatten().cpu().detach().numpy()), axis=0)
                else:
                    all_out = np.concatenate((all_out,out[val_mask].cpu().detach().numpy()), axis=0)
                all_label = np.concatenate((all_label, y[val_mask].flatten().cpu().numpy()))
                del x, y, edge_index, val_mask
                torch.cuda.empty_cache()
                n += 1
            else:
                continue
    if args.task == 'node_cls':
        loss = loss_fn.compute_loss(all_out, all_label)
        total_loss += loss.item()
        acc = np.sum(np.argmax(all_out,axis=1) == all_label) / len(all_label)
        if args.loss_reduction == 'mean':
            return total_loss, acc, total_time
        else:
            return total_loss/len(all_out), acc, total_time
    else:
        loss = loss_fn.compute_loss(all_out, all_label)
        total_loss += loss.item()
        total_loss = total_loss / np.std((all_label)).item()
        acc = 0
        if args.loss_reduction == 'mean':
            return total_loss, acc, total_time
        else:
            return total_loss/len(all_out), acc, total_time
        
def node_infer_Gs_MB(args, model, graph_data, loss_fn, infer_type):
    total_loss = 0
    total_time = 0
    all_out = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)
    for graph in graph_data:
        model.eval()
        x = graph.x.to(device)
        y = graph.y.to(device)
        edge_index = graph.edge_index.to(device)
        if infer_type == 'test':
            if True in graph.test_mask:
                start_time = time.time()
                out = model(x, edge_index)
                total_time += time.time() - start_time
                test_mask = graph.test_mask.to(device)
                if args.task == 'node_cls':
                    loss = loss_fn(out[test_mask], y[test_mask].type(torch.long).flatten())
                else:
                    loss = loss_fn(out[test_mask].view(-1, 1), y[test_mask].view(-1, 1))
                all_out = torch.cat((all_out, out[test_mask]), dim=0)
                all_label = torch.cat((all_label, y[test_mask]), dim=0)
                total_loss += loss.item()
                del x, y, edge_index, test_mask
                torch.cuda.empty_cache()
            else:
                continue
        else:
            if True in graph.val_mask:
                start_time = time.time()
                out = model(x, edge_index)
                total_time += time.time() - start_time
                val_mask = graph.val_mask.to(device)
                if args.task == 'node_cls':
                    loss = loss_fn(out[val_mask], y[val_mask].type(torch.long).flatten())
                else:
                    loss = loss_fn(out[val_mask].view(-1, 1), y[val_mask].view(-1, 1))
                all_out = torch.cat((all_out, out[val_mask]), dim=0)
                all_label = torch.cat((all_label, y[val_mask]), dim=0)
                total_loss += loss.item()
                del x, y, edge_index, val_mask
                torch.cuda.empty_cache()
            else:
                continue
        
    if args.loss_reduction == 'mean':
        if args.task == 'node_cls':
            acc = int(torch.sum(torch.argmax(all_out, dim=1) == all_label).item()) / len(all_label)
            return total_loss/len(graph_data), acc, total_time
        else:
            acc  = 0
            return total_loss/(torch.std(all_label).item()*len(graph_data)), acc, total_time
    else:
        if args.task == 'node_cls':
            acc = int(torch.sum(torch.argmax(all_out, dim=1) == all_label).item()) / len(all_label)
            return total_loss/len(all_out), acc, total_time
        else:
            acc = 0
            return total_loss/(len(all_out)*torch.std(all_label).item()), acc, total_time

def node_train_Gs_GD(model, graph_data, loss_fn, optimizer, args):
    total_loss = 0
    all_out = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)
    model.train()
    optimizer.zero_grad()
    max_mem = 0
    for graph in graph_data:
        max_mem = max(max_mem, get_data_size(graph))
        train_mask = graph.train_mask.to(device)
        if True in train_mask:
            x = graph.x.to(device)
            y = graph.y.to(device)
            edge_index = graph.edge_index.to(device)
            out = model(x, edge_index)
            all_out = torch.cat((all_out, out[train_mask]), dim=0)
            all_label = torch.cat((all_label, y[train_mask]), dim=0)
        else:
            continue
        del x, y, edge_index, train_mask
        torch.cuda.empty_cache()
            
    if args.task == 'node_cls':
        loss = loss_fn(all_out, all_label.type(torch.long).flatten())
    else:
        loss = loss_fn(all_out.view(-1, 1), all_label.view(-1, 1))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    if args.loss_reduction == 'mean':
        if args.task == 'node_cls':
            return total_loss
        else:
            return total_loss/torch.std(all_label).item()
    else:
        if args.task == 'node_cls':
            return total_loss/len(all_out)
        else:
            return total_loss/(len(all_out)*torch.std(all_label).item())
        
def node_train_Gs_MB(model, graph_data, loss_fn, optimizer, args):
    total_loss = 0
    all_out = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)
    model.train()
    optimizer.zero_grad()
    for graph in graph_data:
        train_mask = graph.train_mask.to(device)
        if True in train_mask:
            x = graph.x.to(device)
            y = graph.y.to(device)
            edge_index = graph.edge_index.to(device)
            out = model(x, edge_index)
            if args.task == 'node_cls':
                loss = loss_fn(out[train_mask], y[train_mask].type(torch.long).flatten())
            else:
                loss = loss_fn(out[train_mask].view(-1, 1), y[train_mask].view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_out = torch.cat((all_out, out[train_mask]), dim=0)
            all_label = torch.cat((all_label, y[train_mask]), dim=0)
            del x, y, edge_index, train_mask
            torch.cuda.empty_cache()
        else:
            continue
    if args.loss_reduction == 'mean':
        if args.task == 'node_cls':
            return total_loss/len(graph_data)
        else:
            return total_loss/(torch.std(all_label).item()*len(graph_data))
    else:
        if args.task == 'node_cls':
            return total_loss/len(all_out)
        else:
            return total_loss/(len(all_out)*torch.std(all_label).item())

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
                    train_loss = node_train_Gs_GD(model, graph_data, loss_fn, optimizer, args)
                    val_loss, val_acc, val_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'val')
                else:
                    train_loss = node_train_Gs_MB(model, graph_data, loss_fn, optimizer, args)
                    val_loss, val_acc, val_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'val')
                    
                run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
                run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
                run_writer.add_scalar('Gs_val_acc', val_acc, epoch)
                if val_loss < best_val_loss_Gs or epoch == 0:
                    best_val_loss_Gs = val_loss
                    torch.save(model.state_dict(), path+'/model.pt')
                    
                if args.run_intermediate_inference and epoch % args.intermediate_inference_freq == 0:
                    model.load_state_dict(torch.load(path+'/model.pt'))
                    if args.gradient_method == "GD":
                        test_loss, test_acc, test_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
                    else:
                        test_loss, test_acc, test_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')
                    run_writer.add_scalar('Gs_test_loss_intermediate', test_loss, epoch)
                    if not os.path.exists(f"results/{args.dataset}_intermediate_inference.csv"):
                        with open(f"results/{args.dataset}_intermediate_inference.csv", 'w') as f:
                            f.write('epoch,test_loss,test_acc,test_time\n')
                    with open(f"results/{args.dataset}_intermediate_inference.csv", 'a') as f:
                        f.write(f"{epoch},{test_loss},{test_acc},{test_time}\n")

            model.load_state_dict(torch.load(path+'/model.pt'))
            if args.gradient_method == "GD":
                test_loss, test_acc, test_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
            else:
                test_loss, test_acc, test_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')
            
            writer.add_scalar('Gs_test_loss', test_loss, run)
            writer.add_scalar('Gs_test_acc', test_acc, run)
                    
            all_loss.append(test_loss)
            all_acc.append(test_acc)
            all_time.append(test_time)
        
        elif args.exp_setup == 'Gc_train_2_Gs_infer':
            best_val_loss_Gc =  float('inf')
            best_val_loss_Gs =  float('inf')
            #Train and Val on Gc
            for epoch in tqdm(range(args.epochs1), desc=f"Run {run+1}", colour='red'):
                train_loss_gc = node_train_Gc(model, coarsen_features, coarsen_edge, coarsen_train_mask, coarsen_train_labels, loss_fn, optimizer)
                val_loss_gc = node_val_Gc(model, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, loss_fn)
                
                run_writer.add_scalar('Gc_val_loss', val_loss_gc, epoch)
                run_writer.add_scalar('Gc_train_loss', train_loss_gc, epoch)
                if val_loss_gc < best_val_loss_Gc or epoch == 0:
                    best_val_loss_Gc = val_loss_gc
                    torch.save(model.state_dict(), path+'/model.pt') 
                    
            model.load_state_dict(torch.load(path+'/model.pt'))
            if args.gradient_method == "GD":
                val_loss_gs, val_acc, val_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'val')
                test_loss, test_acc, test_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
            else:
                val_loss_gs, val_acc, val_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'val')
                test_loss, test_acc, test_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')
                
            writer.add_scalar('Gs_val_loss', val_loss_gs, run)
            writer.add_scalar('Gs_val_acc', val_acc, run)
            writer.add_scalar('Gs_test_loss', test_loss, run)
            writer.add_scalar('Gs_test_acc', test_acc, run)
            all_loss.append(test_loss)
            all_acc.append(test_acc)
            all_time.append(test_time)

        elif args.exp_setup == 'Gs_train_2_Gs_infer':
            if run == 0:
                del coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge
                torch.cuda.empty_cache()
            best_val_loss_Gs =  float('inf')
            #Train on Gs
            for epoch in tqdm(range(args.epochs2), desc=f"Run {run+1}", colour='green'):
                if args.gradient_method == "GD":
                    train_loss = node_train_Gs_GD(model, graph_data, loss_fn, optimizer,args)
                    val_loss, val_acc, val_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'val')
                else:
                    train_loss = node_train_Gs_MB(model, graph_data, loss_fn, optimizer,args)
                    val_loss, val_acc, val_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'val')
                run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
                run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
                run_writer.add_scalar('Gs_val_acc', val_acc, epoch)

                if val_loss < best_val_loss_Gs or epoch == 0:
                    best_val_loss_Gs = val_loss
                    torch.save(model.state_dict(), path+'/model.pt')
                    
                if args.run_intermediate_inference and epoch % args.intermediate_inference_freq == 0:
                    model.load_state_dict(torch.load(path+'/model.pt'))
                    if args.gradient_method == "GD":
                        test_loss, test_acc, test_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
                    else:
                        test_loss, test_acc, test_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')
                    run_writer.add_scalar('Gs_test_loss_intermediate', test_loss, epoch)
                    if not os.path.exists(f"results/{args.dataset}_intermediate_inference.csv"):
                        with open(f"results/{args.dataset}_intermediate_inference.csv", 'w') as f:
                            f.write('epoch,test_loss,test_acc,test_time\n')
                    with open(f"results/{args.dataset}_intermediate_inference.csv", 'a') as f:
                        f.write(f"{epoch},{test_loss},{test_acc},{test_time}\n")
                    
            model.load_state_dict(torch.load(path+'/model.pt'))
            if args.gradient_method == "GD":
                test_loss, test_acc, test_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
            else:
                test_loss, test_acc, test_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')
            writer.add_scalar('Gs_test_loss', test_loss, run)
            writer.add_scalar('Gs_test_acc', test_acc, run)
            all_loss.append(test_loss)
            all_acc.append(test_acc)
            all_time.append(test_time)

    top_acc = sorted(all_acc, reverse=True)[:10]
    top_loss = sorted(all_loss)[:10]

    if not os.path.exists(f"results/{args.dataset}.csv"):
        with open(f"results/{args.dataset}.csv", 'w') as f:
            f.write('dataset,coarsening_method,coarsening_ratio,experiment,exp_setup,layer_name,extra_nodes,cluster_node,community_used,hidden,runs,num_layers,batch_size,lr,ave_acc,ave_time,top_10_acc,best_acc,top_10_loss,best_loss\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.experiment},{args.exp_setup},{args.layer_name},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_acc)} +/- {np.std(all_acc)},{np.mean(all_time)},{np.mean(top_acc)} +/- {np.std(top_acc)}, {top_acc[0]}, {np.mean(top_loss)} +/- {np.std(top_loss)}, {top_loss[0]}\n")
    print("############################### FIT-GNN MODEL ###############################")
    print(f"dataset: {args.dataset}")
    print(f"experiment: {args.experiment}")
    print(f"exp_setup: {args.exp_setup}")
    print(f"layer_name: {args.layer_name}")
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
    print("#############################################################################")

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
        best_val_loss_Gs =  float('inf')
        #Train on Gs
        for epoch in tqdm(range(args.epochs2), desc=f"Run {run+1}", colour='green'):
            if args.gradient_method == "GD":
                train_loss = node_train_Gs_GD(model, graph_data, loss_fn, optimizer,args)
                val_loss, val_acc, val_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'val')
            else:
                train_loss = node_train_Gs_MB(model, graph_data, loss_fn, optimizer,args)
                val_loss, val_acc, val_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'val')
            run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
            run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
            run_writer.add_scalar('Gs_val_acc', val_acc, epoch)

            if val_loss < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss
                torch.save(model.state_dict(), path+'/model.pt')
                
        model.load_state_dict(torch.load(path+'/model.pt'))
        if args.gradient_method == "GD":
            test_loss, test_acc, test_time = node_infer_Gs_GD(args, model, graph_data, loss_fn_np, 'test')
        else:
            test_loss, test_acc, test_time = node_infer_Gs_MB(args, model, graph_data, loss_fn, 'test')
        writer.add_scalar('Gs_test_loss', test_loss, run)
        writer.add_scalar('Gs_test_acc', test_acc, run)
        all_loss.append(test_loss)
        all_acc.append(test_acc)
        all_time.append(test_time)

    top_loss = sorted(all_loss)[:10]

    if not os.path.exists(f"results/{args.dataset}.csv"):
        with open(f"results/{args.dataset}.csv", 'w') as f:
            f.write('dataset,coarsening_method,coarsening_ratio,layer_name,extra_nodes,cluster_node,community_used,hidden,runs,num_layers,batch_size,lr,ave_time,top_10_loss,best_loss\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.layer_name},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_time)},{np.mean(top_loss)} +/- {np.std(top_loss)},{top_loss[0]}\n")
    print("############################### FIT-GNN MODEL ###############################")
    print(f"dataset: {args.dataset}")
    print(f"Layer_name: {args.layer_name}")
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
    print("#############################################################################")

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
            f.write('dataset,coarsening_method,coarsening_ratio,exp_setup,layer_name,extra_nodes,cluster_node,community_used,hidden,num_layers1,num_layers2,epochs1,epochs2,batch_size,lr,best_test_loss,best_test_acc\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.exp_setup},{args.layer_name},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.num_layers1},{args.num_layers2},{args.epochs1},{args.epochs2},{args.batch_size},{args.lr},{best_test_loss},{best_test_acc}\n")
    print("############################### FIT-GNN MODEL ###############################")
    print("FIT-GNN MODEL")
    print(f"dataset: {args.dataset}")
    print(f"exp_setup: {args.exp_setup}")
    print(f"layer_name: {args.layer_name}")
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
    print("#############################################################################")

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
                f.write('dataset,coarsening_method,coarsening_ratio,exp_setup,layer_name,extra_nodes,cluster_node,community_used,hidden,num_layers1,num_layers2,epochs1,epochs2,batch_size,lr,best_test_loss,property_idx}\n')
            else:
                f.write('dataset,coarsening_method,coarsening_ratio,exp_setup,layer_name,extra_nodes,cluster_node,community_used,hidden,num_layers1,num_layers2,epochs1,epochs2,batch_size,lr,best_test_loss\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        if args.multi_prop:
            f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.exp_setup},{args.layer_name},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.num_layers1},{args.num_layers2},{args.epochs1},{args.epochs2},{args.batch_size},{args.lr},{best_test_loss},{args.property}\n")
        else:
            f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.exp_setup},{args.layer_name},{args.extra_node},{args.cluster_node},{args.use_community_detection},{args.hidden},{args.num_layers1},{args.num_layers2},{args.epochs1},{args.epochs2},{args.batch_size},{args.lr},{best_test_loss}\n")
    print("############################### FIT-GNN MODEL ###############################")
    print(f"dataset: {args.dataset}")
    print(f"exp_setup: {args.exp_setup}")
    print(f"layer_name: {args.layer_name}")
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
    print("#############################################################################")

### Baseline Functions

def node_classification_baseline(args, path, data, writer):
    all_loss = []
    all_acc = []
    all_time = []
    data = splits_classification(data, args.num_classes, args.experiment)
    graph_data = G_DataLoader([data], batch_size=args.batch_size, shuffle=False)
    model = Classify_node(args).to(device)
    loss_fn = torch.nn.NLLLoss(reduction=args.loss_reduction).to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for run in range(args.runs):
        run_writer = SummaryWriter(path + "/run_"+str(run+1))
        model.reset_parameters()
        best_val_loss = float('inf')
        avg_time = 0
        for graph in graph_data:
            for epoch in tqdm(range(args.epochs1), desc=f"Run {run+1}", colour='orange'):
                model.train()
                optimizer.zero_grad()
                graph = graph.to(device)
                out = model(graph.x, graph.edge_index)
                loss = loss_fn(out[graph.train_mask], graph.y[graph.train_mask].flatten())
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    out = model(graph.x, graph.edge_index)
                    val_loss = loss_fn(out[graph.val_mask], graph.y[graph.val_mask].flatten())
                    if val_loss < best_val_loss or epoch == 0:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), path+'/model.pt')
                run_writer.add_scalar('val_loss', val_loss, epoch)
                run_writer.add_scalar('train_loss', loss, epoch)
                    
            model.load_state_dict(torch.load(path+'/model.pt'))
            model.eval()
            with torch.no_grad():
                start = time.time()
                out = model(graph.x, graph.edge_index)
                avg_time += time.time() - start
                test_loss = loss_fn(out[graph.test_mask], graph.y[graph.test_mask].flatten())
            test_acc = np.sum(np.argmax(out[graph.test_mask].cpu().numpy(), axis=1) == graph.y[graph.test_mask].flatten().cpu().numpy()) / len(graph.y[graph.test_mask])
            writer.add_scalar('test_loss', test_loss, run)
            writer.add_scalar('test_acc', test_acc, run)
            all_loss.append(test_loss.item())
            all_acc.append(test_acc)
            all_time.append(avg_time)
    top_acc = sorted(all_acc, reverse=True)[:10]
    top_loss = sorted(all_loss)[:10]
    
    if not os.path.exists(f"results/baseline/{args.dataset}.csv"):
        with open(f"results/baseline/{args.dataset}.csv", 'w') as f:
            f.write('dataset,eperiment,layer_name,runs,num_layers,batch_size,lr,ave_acc,ave_time,top_10_acc,best_acc,top_10_loss,best_loss\n')
    with open(f"results/baseline/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.experiment},{args.layer_name},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_acc)} +/- {np.std(all_acc)},{np.mean(all_time)},{np.mean(top_acc)} +/- {np.std(top_acc)}, {top_acc[0]}, {np.mean(top_loss)} +/- {np.std(top_loss)}, {top_loss[0]}\n")
    print("############################## BASELINE MODEL ##############################")
    print(f"dataset: {args.dataset}")
    print(f"experiment: {args.experiment}")
    print(f"layer_name: {args.layer_name}")
    print(f"hidden: {args.hidden}")
    print(f"runs: {args.runs}")
    print(f"num_layers: {args.num_layers1}")
    print(f"lr: {args.lr}")
    print(f"ave_acc: {np.mean(all_acc)} +/- {np.std(all_acc)}")
    print(f"ave_time: {np.mean(all_time)}")
    print(f"top_10_acc: {np.mean(top_acc)} +/- {np.std(top_acc)}")
    print(f"best_acc: {top_acc[0]}")
    print(f"top_10_loss: {np.mean(top_loss)} +/- {np.std(top_loss)}")
    print(f"best_loss: {top_loss[0]}")
    print("############################################################################")
    
def node_regression_baseline(args, path, data, writer):
    all_loss = []
    all_time = []
    data = splits_regression(data, args.train_ratio, args.val_ratio)
    graph_data = G_DataLoader([data], batch_size=args.batch_size, shuffle=False)
    for run in range(args.runs):
        run_writer = SummaryWriter(path + "/run_"+str(run+1))
        model = Regress_node(args).to(device)
        loss_fn = torch.nn.L1Loss(reduction=args.loss_reduction).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val_loss = float('inf')
        avg_time = 0
        for graph in graph_data:
            for epoch in tqdm(range(args.epochs1), desc=f"Run {run+1}", colour='orange'):
                model.train()
                optimizer.zero_grad()
                graph = graph.to(device)
                out = model(graph.x, graph.edge_index)
                loss = loss_fn(out[graph.train_mask], graph.y[graph.train_mask].flatten())
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    out = model(graph.x, graph.edge_index)
                    val_loss = loss_fn(out[graph.val_mask], graph.y[graph.val_mask].flatten())
                    if val_loss < best_val_loss or epoch == 0:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), path+'/model.pt')
                run_writer.add_scalar('val_loss', val_loss/torch.std(graph.y[graph.val_mask]), epoch)
                run_writer.add_scalar('train_loss', loss/torch.std(graph.y[graph.train_mask]), epoch)
        
            model.load_state_dict(torch.load(path+'/model.pt'))
            model.eval()
            with torch.no_grad():
                start = time.time()
                out = model(graph.x, graph.edge_index)
                avg_time += time.time() - start
                test_loss = loss_fn(out[graph.test_mask], graph.y[graph.test_mask].flatten())
            writer.add_scalar('test_loss', test_loss, run)
            all_loss.append((test_loss/torch.std(graph.y[graph.test_mask])).item())
            all_time.append(avg_time)
    top_loss = sorted(all_loss)[:10]
    
    if not os.path.exists(f"results/baseline/{args.dataset}.csv"):
        with open(f"results/baseline/{args.dataset}.csv", 'w') as f:
            f.write('dataset,experiment,layer_name,runs,num_layers,batch_size,lr,ave_time,top_10_loss,best_loss\n')
    with open(f"results/baseline/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.experiment},{args.layer_name},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_time)},{np.mean(top_loss)} +/- {np.std(top_loss)},{top_loss[0]}\n")
    print("############################## BASELINE MODEL ##############################")
    print(f"dataset: {args.dataset}")
    print(f"experiment: {args.experiment}")
    print(f"layer_name: {args.layer_name}")
    print(f"hidden: {args.hidden}")
    print(f"runs: {args.runs}")
    print(f"num_layers: {args.num_layers1}")
    print(f"lr: {args.lr}")
    print(f"ave_time: {np.mean(all_time)}")
    print(f"top_10_loss: {np.mean(top_loss)} +/- {np.std(top_loss)}")
    print(f"best_loss: {top_loss[0]}")
    print("############################################################################")
    
def graph_classification_baseline(args, path, data, writer):
    train_split, test_split, val_split = train_test_val_split(data, shuffle=True)
    train_loader = G_DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
    test_loader = G_DataLoader(test_split, batch_size=args.batch_size, shuffle=False)
    val_loader = G_DataLoader(val_split, batch_size=args.batch_size, shuffle=False)
    model = Classify_graph_gc(args).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_test_acc = 0
    for epoch in tqdm(range(args.epochs1), colour='orange'):
        train_loss = 0
        test_loss = 0
        val_loss = 0
        val_pred = np.array([])
        val_true = np.array([])
        test_pred = np.array([])
        test_true = np.array([])
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch.y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = loss_fn(out, batch.y)
                val_loss += loss.item()
                val_pred = np.concatenate((val_pred, torch.argmax(out, dim=1).cpu().numpy()), axis=0)
                val_true = np.concatenate((val_true, batch.y.cpu().numpy()), axis=0)
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = loss_fn(out, batch.y)
                test_loss += loss.item()
                test_pred = np.concatenate((test_pred, torch.argmax(out, dim=1).cpu().numpy()), axis=0)
                test_true = np.concatenate((test_true, batch.y.cpu().numpy()), axis=0)
        
        val_acc = np.sum(val_pred == val_true) / len(val_true)
        test_acc = np.sum(test_pred == test_true) / len(test_true)
        writer.add_scalar('val_loss', val_loss/len(val_loader), epoch)
        writer.add_scalar('train_loss', train_loss/len(train_loader), epoch)
        writer.add_scalar('test_loss', test_loss/len(test_loader), epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        if val_loss < best_val_loss or epoch == 0:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_test_acc = test_acc
            torch.save(model.state_dict(), path+'/model.pt')
    if not os.path.exists(f"results/baseline/{args.dataset}.csv"):
        with open(f"results/baseline/{args.dataset}.csv", 'w') as f:
            f.write('dataset,experiment,layer_name,runs,num_layers,batch_size,lr,best_test_loss,best_test_acc\n')
    with open(f"results/baseline/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.experiment},{args.layer_name},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{best_test_loss},{best_test_acc}\n")
    print("############################## BASELINE MODEL ##############################")
    print(f"dataset: {args.dataset}")
    print(f"experiment: {args.experiment}")
    print(f"layer_name: {args.layer_name}")
    print(f"hidden: {args.hidden}")
    print(f"runs: {args.runs}")
    print(f"num_layers: {args.num_layers1}")
    print(f"lr: {args.lr}")
    print(f"best_test_loss: {best_test_loss}")
    print(f"best_test_acc: {best_test_acc}")
    print("############################################################################")
    
def graph_regression_baseline(args, path, data, writer):
    train_split, test_split, val_split = train_test_val_split(data, shuffle=True)
    train_loader = G_DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
    test_loader = G_DataLoader(test_split, batch_size=args.batch_size, shuffle=False)
    val_loader = G_DataLoader(val_split, batch_size=args.batch_size, shuffle=False)

    model = Regress_graph_gc(args).to(device)
    loss_fn = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    for epoch in tqdm(range(args.epochs1), colour='orange'):
        train_loss = 0
        test_loss = 0
        val_loss = 0
        test_true = np.array([])
        val_true = np.array([])
        train_true = np.array([])
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            batch.x = batch.x.float()
            out = model(batch)
            if args.multi_prop:
                loss = loss_fn(out, batch.y[:, args.property].view(-1, 1))
                train_true = np.concatenate((train_true, batch.y[:, args.property].cpu().numpy()), axis=0)
            else:
                loss = loss_fn(out, batch.y)
                train_true = np.concatenate((train_true, batch.y.cpu().numpy()), axis=0)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch.x = batch.x.float()
                out = model(batch)
                if args.multi_prop:
                    loss = loss_fn(out, batch.y[:, args.property].view(-1, 1))
                    val_true = np.concatenate((val_true, batch.y[:, args.property].cpu().numpy()), axis=0)
                else:    
                    loss = loss_fn(out, batch.y)
                    val_true = np.concatenate((val_true, batch.y.cpu().numpy()), axis=0)
                val_loss += loss.item()
            for batch in test_loader:
                batch = batch.to(device)
                batch.x = batch.x.float()
                out = model(batch)
                if args.multi_prop:
                    loss = loss_fn(out, batch.y[:, args.property])
                    test_true = np.concatenate((test_true, batch.y[:, args.property].cpu().numpy()), axis=0)
                else:
                    loss = loss_fn(out, batch.y)
                    test_true = np.concatenate((test_true, batch.y.cpu().numpy()), axis=0)
                test_loss += loss.item()

        val_std = np.std(val_true)
        test_std = np.std(test_true)
        train_std = np.std(train_true)
        writer.add_scalar('val_loss', val_loss/(len(val_loader)*val_std), epoch)
        writer.add_scalar('train_loss', train_loss/(len(train_loader)*train_std), epoch)
        writer.add_scalar('test_loss', test_loss/(len(test_loader)*test_std), epoch)
        if val_loss/(val_std*len(val_loader)) < best_val_loss or epoch == 0:
            best_val_loss = val_loss/(val_std*len(val_loader))
            best_test_loss = test_loss/(test_std*len(test_loader))
            torch.save(model.state_dict(), path+'/model.pt')
    if not os.path.exists(f"results/baseline/{args.dataset}.csv"):
        with open(f"results/baseline/{args.dataset}.csv", 'w') as f:
            if args.multi_prop:
                f.write('dataset,experiment,layer_name,runs,num_layers,batch_size,lr,best_test_loss,property_idx\n')
            else:
                f.write('dataset,experiment,layer_name,runs,num_layers,batch_size,lr,best_test_loss\n')
    with open(f"results/baseline/{args.dataset}.csv", 'a') as f:
        if args.multi_prop:
            f.write(f"{args.dataset},{args.experiment},{args.layer_name},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{best_test_loss},{args.property}\n")
        else:
            f.write(f"{args.dataset},{args.experiment},{args.layer_name},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{best_test_loss}\n")
    print("############################## BASELINE MODEL ##############################")
    print(f"dataset: {args.dataset}")
    print(f"experiment: {args.experiment}")
    print(f"layer_name: {args.layer_name}")
    print(f"hidden: {args.hidden}")
    print(f"runs: {args.runs}")
    print(f"num_layers: {args.num_layers1}")
    print(f"lr: {args.lr}")
    if args.multi_prop:
        print(f"property_idx: {args.property}")
    print(f"best_test_loss: {best_test_loss}")
    print("############################################################################")
    
    
    
    
            
                
            
                
    
