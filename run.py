import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as T_DataLoader
from torch_geometric.loader import DataLoader as G_DataLoader
from utils import load_data_classification, load_data_regression, train_test_val_split, colater
from network import Classify_node, Regress_node, Classify_graph_gc, Classify_graph_gs, Regress_graph_gc, Regress_graph_gs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def node_infer_Gs(args, model, graph_data, loss_fn, infer_type):
    total_loss = 0
    total_time = 0
    n = 0
    all_out = torch.tensor([], dtype=torch.float32).to(device)
    all_label = torch.tensor([], dtype=torch.float32).to(device)
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
                # loss = loss_fn(out[test_mask], y[test_mask])
                # total_loss += loss.item()
                all_out = torch.cat((all_out, out[test_mask]), dim=0)
                all_label = torch.cat((all_label, y[test_mask]), dim=0)
            else:
                continue
        else:
            if True in graph.val_mask:
                start_time = time.time()
                out = model(x, edge_index)
                total_time += time.time() - start_time
                val_mask = graph.val_mask.to(device)
                # loss = loss_fn(out[val_mask], y[val_mask])
                # total_loss += loss.item()
                all_out = torch.cat((all_out, out[val_mask]), dim=0)
                all_label = torch.cat((all_label, y[val_mask]), dim=0)
            else:
                continue
        n = n + 1
    if args.task == 'node_cls':
        loss = loss_fn(all_out, all_label)
        total_loss += loss.item()
        acc = int(torch.sum(torch.argmax(all_out, dim=1) == all_label).item()) / len(all_label)
    else:
        loss = loss_fn(all_out.view(-1, 1), all_label.view(-1, 1))
        total_loss += loss.item()/torch.std(all_label).item()
        acc = 0
    
    return total_loss / n, acc, total_time

def node_train_Gs(model, graph_data, loss_fn, optimizer):
    total_loss = 0
    n = 0
    all_out = torch.tensor([], dtype=torch.float32).to(device)
    all_label = torch.tensor([], dtype=torch.float32).to(device)
    model.train()
    optimizer.zero_grad()
    for graph in graph_data:
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
        n = n + 1
    loss = loss_fn(all_out.view(-1, 1), all_label.view(-1, 1))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    return total_loss / n

def graph_train_Gc(model, loader, optimizer, loss_fn):
    total_loss = 0
    model.train()
    optimizer.zero_grad()
    for batch in loader:
        gc = batch[0].to(device)
        y = batch[2].to(device)
        out = model(gc)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def graph_val_Gc(model, loader, loss_fn):
    total_loss = 0
    model.eval()
    for batch in loader:
        gc = batch[0].to(device)
        y = batch[2].to(device)
        out = model(gc)
        loss = loss_fn(out, y)
        total_loss += loss.item()
    return total_loss / len(loader)

def graph_train_Gs(model, loader, optimizer, loss_fn):
    total_loss = 0
    model.train()
    optimizer.zero_grad()
    for batch in loader:
        set_gs = batch[1]
        y = batch[2].to(device)
        batch_tensor = batch[3].to(device)
        out = model(set_gs, batch_tensor)
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
        y = batch[2].to(device)
        batch_tensor = batch[3].to(device)
        out = model(set_gs, batch_tensor)
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

    for run in range(args.run):
        run_writer = SummaryWriter(path + "/run_"+str(run+1))
        coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, graphs = load_data_classification(args, dataset[0], candidate, C_list, Gc_list, args.experiment, subgraph_list)
        if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
        graph_data = G_DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

        model = Classify_node(args).to(device)
        loss_fn = torch.nn.NLLLoss().to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.exp_setup == 'Gc_train_2_Gs_train':
            best_val_loss_Gc =  float('inf')
            best_val_loss_Gs =  float('inf')
            #Train and Val on Gc
            for epoch in tqdm(range(args.epochs1)):
                train_loss = node_train_Gc(model, coarsen_features, coarsen_edge, coarsen_train_mask, coarsen_train_labels, loss_fn, optimizer)
                run_writer.add_scalar('Gc_train_loss', train_loss, epoch)
                val_loss = node_val_Gc(model, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, loss_fn)
                run_writer.add_scalar('Gc_val_loss', val_loss, epoch)

                if val_loss < best_val_loss_Gc or epoch == 0:
                    best_val_loss_Gc = val_loss
                    torch.save(model.state_dict(), path+'/model.pt') 

            #Train and val on Gs
            model.load_state_dict(torch.load(path+'/model.pt'))# changed from model = model.load_state_dict(torch.load(path+'/model.pt')) -> model.load_state_dict(torch.load(path+'/model.pt'))
            for epoch in tqdm(range(args.epochs2)):
                train_loss = node_train_Gs(model, graph_data, loss_fn, optimizer)
                run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
                val_loss, val_acc, val_time = node_infer_Gs(args, model, graph_data, loss_fn, 'val')
                run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
                run_writer.add_scalar('Gs_val_acc', val_acc, epoch)

                if val_loss < best_val_loss_Gs or epoch == 0:
                    best_val_loss_Gs = val_loss
                    torch.save(model.state_dict(), path+'/model.pt')
            
            #Test on Gs
            model.load_state_dict(torch.load(path+'/model.pt'))# changed from model = model.load_state_dict(torch.load(path+'/model.pt')) -> model.load_state_dict(torch.load(path+'/model.pt'))
            test_loss, test_acc, test_time = node_infer_Gs(args, model, graph_data, loss_fn, 'test')
            
            writer.add_scalar('Gs_test_loss', test_loss, run)
            writer.add_scalar('Gs_test_acc', test_acc, run)
            all_loss.append(test_loss)
            all_acc.append(test_acc)
            all_time.append(test_time)
        
        elif args.exp_setup == 'Gc_train_2_Gs_infer':
            best_val_loss_Gc =  float('inf')
            best_val_loss_Gs =  float('inf')
            #Train and Val on Gc
            for epoch in tqdm(range(args.epochs1)):
                train_loss = node_train_Gc(model, coarsen_features, coarsen_edge, coarsen_train_mask, coarsen_train_labels, loss_fn, optimizer)
                run_writer.add_scalar('Gc_train_loss', train_loss, epoch)
                val_loss = node_val_Gc(model, coarsen_features, coarsen_edge, coarsen_val_mask, coarsen_val_labels, loss_fn)
                run_writer.add_scalar('Gc_val_loss', val_loss, epoch)

                if val_loss < best_val_loss_Gc or epoch == 0:
                    best_val_loss_Gc = val_loss
                    torch.save(model.state_dict(), path+'/model.pt') 

            #Infer on Gs
            model.load_state_dict(torch.load(path+'/model.pt')) # changed from model = model.load_state_dict(torch.load(path+'/model.pt')) -> model.load_state_dict(torch.load(path+'/model.pt'))
            val_loss, val_acc, val_time = node_infer_Gs(args, model, graph_data, loss_fn, 'val')
            run_writer.add_scalar('Gs_val_loss', val_loss, 0)
            run_writer.add_scalar('Gs_val_acc', val_acc, 0)
            test_loss, test_acc, test_time = node_infer_Gs(args, model, graph_data, loss_fn, 'test')
            writer.add_scalar('Gs_test_acc', test_acc, run)
            writer.add_scalar('Gs_test_loss', test_loss, run)
            all_loss.append(test_loss)
            all_acc.append(test_acc)
            all_time.append(test_time)

        elif args.exp_setup == 'Gs_train_2_Gs_infer':
            best_val_loss_Gs =  float('inf')
            #Train on Gs
            for epoch in tqdm(range(args.epochs2)):
                train_loss = node_train_Gs(model, graph_data, loss_fn, optimizer)
                run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
                val_loss, val_acc, val_time = node_infer_Gs(args, model, graph_data, loss_fn, 'val')
                run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
                run_writer.add_scalar('Gs_val_acc', val_acc, epoch)

                if val_loss < best_val_loss_Gs or epoch == 0:
                    best_val_loss_Gs = val_loss
                    torch.save(model.state_dict(), path+'/model.pt')
            
            #Test on Gs
            model.load_state_dict(torch.load(path+'/model.pt')) # changed from model = model.load_state_dict(torch.load(path+'/model.pt')) -> model.load_state_dict(torch.load(path+'/model.pt'))
            test_loss, test_acc, test_time = node_infer_Gs(args, model, graph_data, loss_fn, 'test')
            writer.add_scalar('Gs_test_loss', test_loss, run)
            writer.add_scalar('Gs_test_acc', test_acc, run)
            all_loss.append(test_loss)
            all_acc.append(test_acc)
            all_time.append(test_time)

    top_acc = sorted(all_acc, reverse=True)[:10]
    top_loss = sorted(all_loss)[:10]

    if not os.path.exists(f"results/{args.dataset}.csv"):
        with open(f"results/{args.dataset}.csv", 'w') as f:
            f.write('dataset,coarsening_method,coarsening_ratio,experiment,exp_setup,extra_nodes,cluster_node,hidden,runs,num_layers,batch_size,lr,ave_acc,ave_time,top_10_acc,best_acc,top_10_loss,best_loss\n')

    with open(f"results/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.experiment},{args.exp_setup},{args.extra_node},{args.cluster_node},{args.hidden},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_acc)} +/- {np.std(all_acc)},{np.mean(all_time)},{np.mean(top_acc)} +/- {np.std(top_acc)}, {top_acc[0]}, {np.mean(top_loss)} +/- {np.std(top_loss)}, {top_loss[0]}\n")
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
    print("#####################################################################")

def node_regression(args, path, writer, subgraph_list):
    all_loss = []
    all_acc = []
    all_time = []

    for run in range(args.runs):
        run_writer = SummaryWriter(path + "/run_"+str(run+1))
        graphs = load_data_regression(args, args.dataset, subgraph_list)
        graph_data = G_DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

        model = Regress_node(args).to(device)
        loss_fn = torch.nn.L1Loss().to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_val_loss_Gs =  float('inf')
        #Train on Gs
        for epoch in tqdm(range(args.epochs2)):
            train_loss = node_train_Gs(model, graph_data, loss_fn, optimizer)
            run_writer.add_scalar('Gs_train_loss', train_loss, epoch)
            val_loss, val_acc, val_time = node_infer_Gs(args, model, graph_data, loss_fn, 'val')
            run_writer.add_scalar('Gs_val_loss', val_loss, epoch)
            run_writer.add_scalar('Gs_val_acc', val_acc, epoch)

            if val_loss < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss
                torch.save(model.state_dict(), path+'/model.pt')
        
        #Test on Gs
        model.load_state_dict(torch.load(path+'/model.pt'))
        test_loss, test_acc, test_time = node_infer_Gs(args, model, graph_data, loss_fn, 'test')
        writer.add_scalar('Gs_test_loss', test_loss, run)
        writer.add_scalar('Gs_test_acc', test_acc, run)
        all_loss.append(test_loss)
        all_acc.append(test_acc)
        all_time.append(test_time)

    top_acc = sorted(all_acc, reverse=True)[:10]
    top_loss = sorted(all_loss)[:10]

    if not os.path.exists(f"results_3/{args.dataset}.csv"):
        with open(f"results_3/{args.dataset}.csv", 'w') as f:
            f.write('dataset,coarsening_method,coarsening_ratio,experiment,exp_setup,extra_nodes,cluster_node,hidden,runs,num_layers,batch_size,lr,ave_time,top_10_loss,best_loss\n')

    with open(f"results_3/{args.dataset}.csv", 'a') as f:
        f.write(f"{args.dataset},{args.coarsening_method},{args.coarsening_ratio},{args.experiment},{args.exp_setup},{args.extra_node},{args.cluster_node},{args.hidden},{args.runs},{args.num_layers1},{args.batch_size},{args.lr},{np.mean(all_time)},{np.mean(top_loss)} +/- {np.std(top_loss)},{top_loss[0]}\n")
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
    print(f"ave_time: {np.mean(all_time)}")
    print(f"top_10_loss: {np.mean(top_loss)} +/- {np.std(top_loss)}")
    print(f"best_loss: {top_loss[0]}")
    print("#####################################################################")

def graph_classification(args, path, writer, dataset):
    train_split, test_split, val_split = train_test_val_split(dataset, shuffle=True)
    colater_fn = colater()
    train_loader = T_DataLoader(train_split, batch_size=args.batch_size, shuffle=True, collate_fn=colater_fn)
    test_loader = T_DataLoader(test_split, batch_size=args.batch_size, shuffle=True, collate_fn=colater_fn)
    val_loader = T_DataLoader(val_split, batch_size=args.batch_size, shuffle=True, collate_fn=colater_fn)

    model_gc = Classify_graph_gc(args).to(device)
    model_gs = Classify_graph_gs(args).to(device)
    loss_fn = torch.nn.NLLLoss().to(device)
    optimizer_gc = torch.optim.Adam(model_gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_gs = torch.optim.Adam(model_gs.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.exp_setup == 'Gc_train_2_Gs_train':
        best_val_loss_Gc =  float('inf')
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        best_test_acc = 0
        best_val_acc = 0
        for epoch in tqdm(range(args.epochs1)):
            train_loss = graph_train_Gc(model_gc, train_loader, optimizer_gc, loss_fn)
            writer.add_scaler('Gc_train_loss', train_loss, epoch)
            val_loss = graph_val_Gc(model_gc, val_loader, loss_fn)
            writer.add_scaler('Gc_val_loss', val_loss, epoch)
            if val_loss < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss
                torch.save(model_gc.state_dict(), path+'model.pt')
        
        model_gs.load_state_dict(torch.load(path+'/model.pt'))
        for epoch in tqdm(range(args.epochs2)):
            train_loss = graph_train_Gs(model_gs, train_loader, optimizer_gs, loss_fn)
            val_loss, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            writer.add_scaler('Gs_train_loss', train_loss, epoch)
            writer.add_scaler('Gs_val_loss', val_loss, epoch)
            writer.add_scaler('Gs_test_loss', test_loss, epoch)
            writer.add_scaler('Gs_val_acc', val_acc, epoch)
            writer.add_scaler('Gs_test_acc', test_acc, epoch)

            if val_loss < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_val_acc = val_acc
                torch.save(model_gs.state_dict(), path+'/model.pt')

    elif args.exp_setup == 'Gc_train_2_Gs_infer':
        best_val_loss_Gc =  float('inf')
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        best_test_acc = 0
        best_val_acc = 0
        for epoch in tqdm(range(args.epochs1)):
            train_loss = graph_train_Gc(model_gc, train_loader, optimizer_gc, loss_fn)
            writer.add_scaler('Gc_train_loss', train_loss, epoch)
            val_loss = graph_val_Gc(model_gc, val_loader, loss_fn)
            writer.add_scaler('Gc_val_loss', val_loss, epoch)
            if val_loss < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss
                torch.save(model_gc.state_dict(), path+'model.pt')
            
        model_gs.load_state_dict(torch.load(path+'/model.pt'))
        val_loss, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
        test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
        best_val_loss_Gs = val_loss
        best_test_loss = test_loss
        best_test_acc = test_acc
        writer.add_scaler('Gs_val_loss', val_loss, epoch)
        writer.add_scaler('Gs_test_loss', test_loss, epoch)
        writer.add_scaler('Gs_val_acc', val_acc, epoch)
        writer.add_scaler('Gs_test_acc', test_acc, epoch)

    elif args.exp_setup == "Gs_train_2_Gs_infer":
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        best_test_acc = 0
        best_val_acc = 0
        for epoch in tqdm(range(args.epochs2)):
            train_loss = graph_train_Gs(model_gs, train_loader, optimizer_gs, loss_fn)
            val_loss, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            writer.add_scaler('Gs_train_loss', train_loss, epoch)
            writer.add_scaler('Gs_val_loss', val_loss, epoch)
            writer.add_scaler('Gs_test_loss', test_loss, epoch)
            writer.add_scaler('Gs_val_acc', val_acc, epoch)
            writer.add_scaler('Gs_test_acc', test_acc, epoch)

            if val_loss < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_val_acc = val_acc
                torch.save(model_gs.state_dict(), path+'/model.pt')

    #here we need to print the results and save it in a csv file

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
        for epoch in tqdm(range(args.epochs1)):
            train_loss = graph_train_Gc(model_gc, train_loader, optimizer_gc, loss_fn)
            writer.add_scaler('Gc_train_loss', train_loss, epoch)
            val_loss = graph_val_Gc(model_gc, val_loader, loss_fn)
            writer.add_scaler('Gc_val_loss', val_loss, epoch)
            if val_loss < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss
                torch.save(model_gc.state_dict(), path+'model.pt')
        
        model_gs.load_state_dict(torch.load(path+'/model.pt'))
        for epoch in tqdm(range(args.epochs2)):
            train_loss = graph_train_Gs(model_gs, train_loader, optimizer_gs, loss_fn)
            val_loss, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            writer.add_scaler('Gs_train_loss', train_loss, epoch)
            writer.add_scaler('Gs_val_loss', val_loss, epoch)
            writer.add_scaler('Gs_test_loss', test_loss, epoch)

            if val_loss < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_val_acc = val_acc
                torch.save(model_gs.state_dict(), path+'/model.pt')

    elif args.exp_setup == 'Gc_train_2_Gs_infer':
        best_val_loss_Gc =  float('inf')
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        for epoch in tqdm(range(args.epochs1)):
            train_loss = graph_train_Gc(model_gc, train_loader, optimizer_gc, loss_fn)
            writer.add_scaler('Gc_train_loss', train_loss, epoch)
            val_loss = graph_val_Gc(model_gc, val_loader, loss_fn)
            writer.add_scaler('Gc_val_loss', val_loss, epoch)
            if val_loss < best_val_loss_Gc or epoch == 0:
                best_val_loss_Gc = val_loss
                torch.save(model_gc.state_dict(), path+'model.pt')
            
        model_gs.load_state_dict(torch.load(path+'/model.pt'))
        val_loss, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
        test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
        best_val_loss_Gs = val_loss
        writer.add_scaler('Gs_val_loss', val_loss, epoch)
        writer.add_scaler('Gs_test_loss', test_loss, epoch)

    elif args.exp_setup == "Gs_train_2_Gs_infer":
        best_val_loss_Gs =  float('inf')
        best_test_loss = float('inf')
        for epoch in tqdm(range(args.epochs2)):
            train_loss = graph_train_Gs(model_gs, train_loader, optimizer_gs, loss_fn)
            val_loss, val_acc = graph_infer_Gs(args, model_gs, val_loader, loss_fn)
            test_loss, test_acc = graph_infer_Gs(args, model_gs, test_loader, loss_fn)
            writer.add_scaler('Gs_train_loss', train_loss, epoch)
            writer.add_scaler('Gs_val_loss', val_loss, epoch)
            writer.add_scaler('Gs_test_loss', test_loss, epoch)

            if val_loss < best_val_loss_Gs or epoch == 0:
                best_val_loss_Gs = val_loss
                best_test_loss = test_loss
                torch.save(model_gs.state_dict(), path+'/model.pt')

    #here we need to print the results and save it in a csv file






    
