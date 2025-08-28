import warnings
warnings.simplefilter("ignore")
import torch
from torch_geometric.utils import to_dense_adj
from graph_coarsening.coarsening_utils import *
from torch_geometric.data import Data, Batch
import torch_scatter
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix, degree
from tqdm import tqdm
import igraph as ig
import leidenalg
import logging
logging.disable(logging.INFO)
logging.disable(logging.WARNING)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_test_val_split(dataset, shuffle=True):
    N = len(dataset)
    if shuffle:
        idx = torch.randperm(N)
    else:
        idx = torch.arange(N)
    train = []
    val = []
    test = []
    for i in range(N):
        if i < N//2:
            train.append(dataset[idx[i]])
        elif i < 3*N//4 and i >= N//2:
            val.append(dataset[idx[i]])
        else:
            test.append(dataset[idx[i]])
    return train, test, val

def create_distribution_tensor(input_tensor, class_count):
    if input_tensor.dtype != torch.int64:
        input_tensor = input_tensor.long()
    distribution_tensor = torch.zeros(class_count, dtype=torch.int64).to(device=input_tensor.device)
    unique_classes, counts = input_tensor.unique(return_counts=True)
    distribution_tensor[unique_classes-1] = counts
    return distribution_tensor

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def neighbour(G, node):
    edge_index = G.edge_index.cpu().numpy()
    edges_connected_to_k = np.nonzero(edge_index[0] == node)[0]
    neighbors_k = edge_index[1][edges_connected_to_k].flatten()
    return neighbors_k

def nodes_2_neighbours(G, nodes):
    edge_index = G.edge_index.cpu().numpy()
    mask = np.isin(edge_index[0], nodes)
    connected_targets = np.unique(edge_index[1, mask])
    return torch.tensor(connected_targets).to(device)

def neighbor_2_cluster(Nt_node, node_2_comp_node, comp_node_2_meta_node):
    connected_clusters = np.array([], dtype=int)
    for node in Nt_node:
        comp_node = node_2_comp_node[node]
        meta_node = comp_node_2_meta_node[comp_node]
        connected_clusters = np.append(connected_clusters, meta_node)
    connected_clusters = np.unique(connected_clusters)
    return connected_clusters

def extract_components(H):
        if H.A.shape[0] != H.A.shape[1]:
            H.logger.error('Inconsistent shape to extract components. '
                           'Square matrix required.')
            return None

        if H.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        graphs = []

        visited = np.zeros(H.A.shape[0], dtype=bool)

        while not visited.all():
            stack = set([np.nonzero(~visited)[0][0]])
            comp = []

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    comp.append(v)
                    visited[v] = True

                    stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                      if not visited[idx]]))

            comp = sorted(comp)
            G = H.subgraph(comp)
            G.info = {'orig_idx': comp}
            graphs.append(G)

        return graphs

def orig_to_new_map(idxs):
    comp_node_2_node, node_2_comp_node = {}, {}
    for i, idx in enumerate(idxs):
        comp_node_2_node[i] = idx
        node_2_comp_node[idx] = i
    return comp_node_2_node, node_2_comp_node

def subgraph_mapping(map_dict):
    subgraph_mapping = {}
    for i in map_dict[0].keys():
        new_map = map_dict[0][i]
        if len(map_dict) > 1:
            for j in range(1, len(map_dict)):
                new_map = map_dict[j][new_map]
        subgraph_mapping[i] = new_map
    return subgraph_mapping

def metanode_to_node_mapping_new(comp_node_2_meta_node, comp_node_2_node):
    metanode_2_node = {}
    for comp_node, metanode in comp_node_2_meta_node.items():
        if metanode not in metanode_2_node.keys():
            metanode_2_node[metanode] = np.array([comp_node_2_node[comp_node]], dtype=np.int32)
        else:
            metanode_2_node[metanode] = np.append(metanode_2_node[metanode], comp_node_2_node[comp_node])
    return metanode_2_node

def merge_communities(data, mapping, k):
    sorted_mapping = sorted(mapping.items(), key=lambda x: len(x[1]), reverse=True)
    new_nodes = torch.tensor([], dtype=torch.long).to(device)
    for i in range(len(sorted_mapping)):
        if len(new_nodes) + len(sorted_mapping[i][1]) <= k:
            new_nodes = torch.cat((new_nodes, torch.tensor(sorted_mapping[i][1], dtype=torch.long).to(device)))
            if len(new_nodes) == k:
                break
    new_data = data.subgraph(new_nodes)
    return new_data

def coarsening_classification(args, data, coarsening_ratio, coarsening_method):
    G = gsp.graphs.Graph(W=to_scipy_sparse_matrix(edge_index=data.edge_index, num_nodes=data.num_nodes).tocsr()) #W=to_scipy_sparse_matrix(edge_index=data.edge_index, num_nodes=data.num_nodes).tocsr()
    components = G.extract_components()
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]
    Gc_list=[]
    CLIST = []
    GcLIST = []
    comp_node_2_meta_node_list = []
    subgraph_list=[]
    while number < len(candidate):
        H = candidate[number]
        H_feature = data.x[H.info['orig_idx']].cpu()
        comp_node_2_node, node_2_comp_node = orig_to_new_map(H.info['orig_idx'])
        if len(H.info['orig_idx']) > 1:
            C, Gc, mapping_dict_list = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            adj = Gc.A
            C_dot_H_feature = C.dot(H_feature)
            CLIST.append(C)
            GcLIST.append(Gc)
            if len(H.info['orig_idx']) > 10:
                C_list.append(C)
                Gc_list.append(Gc)
            if coarsening_method == "kron" or coarsening_method == "algebraic_JC" or coarsening_method == "heavy_edge" or coarsening_method == "variation_edges":
                rows, cols = C.nonzero()
                mapping_dict = {}
                for i, j in zip(rows, cols):
                    mapping_dict[j] = i
                col_sum = C.sum(axis=1)
                no_node = np.where(col_sum == 0)[0]
                max_node_in_mapping = np.argwhere(col_sum == col_sum.max())[0][0]
                other_nodes = set(range(len(H.info['orig_idx']))) - set(mapping_dict.keys())
                for node in other_nodes:
                    mapping_dict[node] = max_node_in_mapping
                for node in no_node:
                    mapping_dict[node] = max_node_in_mapping
                comp_node_2_meta_node = mapping_dict
            else:
                comp_node_2_meta_node = subgraph_mapping(mapping_dict_list) 
            comp_node_2_meta_node_list.append(comp_node_2_meta_node)
            meta_node_2_node = metanode_to_node_mapping_new(comp_node_2_meta_node, comp_node_2_node)
            if args.task == "node_cls":
                for key, value in tqdm(meta_node_2_node.items(), colour='blue'):
                    value = np.sort(value)
                    actual_ext = np.array([], dtype=np.int32)
                    num_nodes = len(value)
                    if args.cluster_node:
                        node_2_subgraph_node = {v.item(): i for i, v in enumerate(value)}
                        new_edges = np.array([], dtype=np.int32)
                        new_features = np.array([])
                        meta_node_2_new_node = {}
                        for node in value:
                            N_node = neighbour(data, node)
                            Nt_node = N_node[~np.isin(N_node, value)]
                            connected_clusters = neighbor_2_cluster(Nt_node, node_2_comp_node, comp_node_2_meta_node)
                            for cluster in connected_clusters:
                                if cluster not in meta_node_2_new_node.keys():
                                    meta_node_2_new_node[cluster] = np.array([num_nodes])
                                    if len(actual_ext.shape) <= 1:
                                        actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster]), axis=0)
                                        actual_ext = actual_ext.reshape(1, 1)
                                    else:
                                        actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster].reshape(1,1)), axis=0)
                                    new_feature = C_dot_H_feature[cluster]
                                    if len(new_features.shape) <= 1:
                                        new_features = new_feature
                                        new_features = new_features.reshape(1, len(new_feature))
                                    else:
                                        new_features = np.concatenate((new_features, new_feature.reshape(1,-1)), axis=0)
                                    num_nodes += 1
                                e1 = np.array([node_2_subgraph_node[node], meta_node_2_new_node[cluster][0]], dtype=np.int32)
                                e2 = np.array([meta_node_2_new_node[cluster][0], node_2_subgraph_node[node]], dtype=np.int32)
                                if len(new_edges.shape) <= 1:
                                    new_edges = np.concatenate((new_edges, e1), axis=0)
                                    new_edges = new_edges.reshape(1, 2)
                                    new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                                else:
                                    new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                    new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                            
                        if len(meta_node_2_new_node.keys()) > 1:
                            cluster_keys = list(meta_node_2_new_node.keys())
                            for i in range(len(cluster_keys)-1):
                                for j in range(i+1, len(cluster_keys)):
                                    if adj[cluster_keys[i], cluster_keys[j]] or adj[cluster_keys[j], cluster_keys[i]]:
                                        e1 = np.array([meta_node_2_new_node[cluster_keys[i]][0], meta_node_2_new_node[cluster_keys[j]][0]], dtype=np.int32)
                                        e2 = np.array([meta_node_2_new_node[cluster_keys[j]][0], meta_node_2_new_node[cluster_keys[i]][0]], dtype=np.int32)
                                        new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                        new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                        value = np.unique(np.sort(value))
                        value = torch.tensor(value).to(device)
                    elif args.extra_node:
                        extra_node = nodes_2_neighbours(data, value)
                        value = torch.tensor(value).to(device)
                        actual_ext = extra_node[~torch.isin(extra_node, value)]
                        value = torch.cat((value, actual_ext), dim=0)

                    if not torch.is_tensor(value):
                        value = torch.tensor(value).to(device)
                    value, _ = torch.sort(value)

                    mappiing = {}
                    for i in range(len(value)):
                        mappiing[value[i].item()] = i
                    M = data.subgraph(value)
                    M.actual_ext = actual_ext
                    M.orig_idx = value
                    if args.cluster_node:
                        M.x = torch.cat((M.x, torch.tensor(new_features).to(device).float()), dim=0)
                        M.edge_index = torch.cat((M.edge_index.T, torch.tensor(new_edges, dtype=torch.long).to(device)), dim=0).T
                        if len(M.y.size()) > 1:
                            M.y = torch.cat((M.y, torch.zeros((new_features.shape[0], M.y.shape[1])).to(device).long()))
                        else:
                            M.y = torch.cat((M.y, torch.zeros(len(new_features)).to(device).long()))
                        for new_node in actual_ext:
                            mappiing[new_node.item()] = new_node.item()
                    if args.extra_node:
                        M.mask = torch.tensor(((len(value) - len(actual_ext))*[True] + [False]*len(actual_ext)), dtype=torch.bool).to(device)
                    elif args.cluster_node:
                        M.mask = torch.tensor([True]*len(value) + [False]*len(actual_ext), dtype=torch.bool).to(device)
                    else:
                        M.mask = torch.tensor([True]*len(value), dtype=torch.bool).to(device)
                    M.map_dict = mappiing
                    subgraph_list.append(M)           
            else:
                for key, value in meta_node_2_node.items():
                    value = np.sort(value)
                    actual_ext = np.array([], dtype=np.int32)
                    num_nodes = len(value)
                    if args.cluster_node:
                        node_2_subgraph_node = {v.item(): i for i, v in enumerate(value)}
                        new_edges = np.array([], dtype=np.int32)
                        new_features = np.array([])
                        meta_node_2_new_node = {}
                        for node in value:
                            N_node = neighbour(data, node)
                            Nt_node = N_node[~np.isin(N_node, value)]
                            connected_clusters = neighbor_2_cluster(Nt_node, node_2_comp_node, comp_node_2_meta_node)
                            for cluster in connected_clusters:
                                if cluster not in meta_node_2_new_node.keys():
                                    meta_node_2_new_node[cluster] = np.array([num_nodes])
                                    if len(actual_ext.shape) <= 1:
                                        actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster]), axis=0)
                                        actual_ext = actual_ext.reshape(1, 1)
                                    else:
                                        actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster].reshape(1,1)), axis=0)
                                    new_feature = C_dot_H_feature[cluster]
                                    if len(new_features.shape) <= 1:
                                        new_features = new_feature
                                        new_features = new_features.reshape(1, len(new_feature))
                                    else:
                                        new_features = np.concatenate((new_features, new_feature.reshape(1,-1)), axis=0)
                                    num_nodes += 1
                                e1 = np.array([node_2_subgraph_node[node], meta_node_2_new_node[cluster][0]], dtype=np.int32)
                                e2 = np.array([meta_node_2_new_node[cluster][0], node_2_subgraph_node[node]], dtype=np.int32)
                                if len(new_edges.shape) <= 1:
                                    new_edges = np.concatenate((new_edges, e1), axis=0)
                                    new_edges = new_edges.reshape(1, 2)
                                    new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                                else:
                                    new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                    new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                            
                        if len(meta_node_2_new_node.keys()) > 1:
                            cluster_keys = list(meta_node_2_new_node.keys())
                            for i in range(len(cluster_keys)-1):
                                for j in range(i+1, len(cluster_keys)):
                                    if adj[cluster_keys[i], cluster_keys[j]] or adj[cluster_keys[j], cluster_keys[i]]:
                                        e1 = np.array([meta_node_2_new_node[cluster_keys[i]][0], meta_node_2_new_node[cluster_keys[j]][0]], dtype=np.int32)
                                        e2 = np.array([meta_node_2_new_node[cluster_keys[j]][0], meta_node_2_new_node[cluster_keys[i]][0]], dtype=np.int32)
                                        new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                        new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                        value = np.unique(np.sort(value))
                        value = torch.tensor(value).to(device)
                    elif args.extra_node:
                        extra_node = nodes_2_neighbours(data, value)
                        value = torch.tensor(value).to(device)
                        actual_ext = extra_node[~torch.isin(extra_node, value)]
                        value = torch.cat((value, actual_ext), dim=0)
                        
                    if not torch.is_tensor(value):
                        value = torch.tensor(value).to(device)
                    value, _ = torch.sort(value)
                    
                    mappiing = {}
                    for i in range(len(value)):
                        mappiing[value[i].item()] = i
                    M = data.subgraph(value)
                    M.actual_ext = actual_ext
                    M.orig_idx = value
                    if args.cluster_node:
                        M.x = torch.cat((M.x, torch.tensor(new_features).to(device).float()), dim=0)
                        M.edge_index = torch.cat((M.edge_index.T, torch.tensor(new_edges, dtype=torch.long).to(device)), dim=0).T
                        if len(M.y.size()) > 1:
                            M.y = torch.cat((M.y, torch.zeros((new_features.shape[0], M.y.shape[1])).to(device).long()))
                        else:
                            M.y = torch.cat((M.y, torch.zeros(len(new_features)).to(device).long()))
                        for new_node in actual_ext:
                            mappiing[new_node.item()] = new_node.item()
                    if args.extra_node:
                        M.mask = torch.tensor(((len(value) - len(actual_ext))*[True] + [False]*len(actual_ext)), dtype=torch.bool).to(device)
                    elif args.cluster_node:
                        M.mask = torch.tensor([True]*len(value) + [False]*len(actual_ext), dtype=torch.bool).to(device)
                    else:
                        M.mask = torch.tensor([True]*len(value), dtype=torch.bool).to(device)
                    M.map_dict = mappiing
                    subgraph_list.append(M)

        else:
            comp_node_2_meta_node = {0: 0}
            comp_node_2_meta_node_list.append(comp_node_2_meta_node)
            meta_node_2_node = metanode_to_node_mapping_new(comp_node_2_meta_node, comp_node_2_node)
            for key, value in meta_node_2_node.items():
                value = torch.LongTensor(value).to(device)
                value, _ = torch.sort(value)
                actual_ext = torch.LongTensor([]).to(device)
                M = data.subgraph(value)
                M.actual_ext = actual_ext
                M.orig_idx = value
                mappiing = {}
                for i in range(len(value)):
                    mappiing[value[i].item()] = i
                M.map_dict = mappiing
                M.mask = torch.tensor([True], dtype=torch.bool).to(device)
                subgraph_list.append(M)

        number += 1
    if args.task == "node_cls":
        return data.x.shape[1], candidate, C_list, Gc_list, subgraph_list
    else:
        return data.x.shape[1], candidate, subgraph_list, CLIST, GcLIST

def coarsening_regression(args, data, coarsening_ratio, coarsening_method):
    G = gsp.graphs.Graph(W=to_scipy_sparse_matrix(edge_index=data.edge_index, num_nodes=data.num_nodes).tocsr())
    components = G.extract_components()
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]
    Gc_list=[]
    CLIST = []
    GcLIST = [] 
    subgraph_list=[]
    while number < len(candidate):
        H = candidate[number]
        H_feature = data.x[H.info['orig_idx']].cpu()
        comp_node_2_node, node_2_comp_node = orig_to_new_map(H.info['orig_idx'])
        if len(H.info['orig_idx']) > 1:
            C, Gc, mapping_dict_list = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            adj = Gc.A
            C_dot_H_feature = C.dot(H_feature)
            CLIST.append(C)
            GcLIST.append(Gc)
            if len(H.info['orig_idx']) > 10:
                C_list.append(C)
                Gc_list.append(Gc)
            if coarsening_method == "kron" or coarsening_method == "algebraic_JC" or coarsening_method == "heavy_edge" or coarsening_method == "variation_edges":
                rows, cols = C.nonzero()
                mapping_dict = {}
                for i, j in zip(rows, cols):
                    mapping_dict[j] = i
                col_sum = C.sum(axis=1)
                no_node = np.where(col_sum == 0)[0]
                max_node_in_mapping = np.argwhere(col_sum == col_sum.max())[0][0]
                other_nodes = set(range(len(H.info['orig_idx']))) - set(mapping_dict.keys())
                for node in other_nodes:
                    mapping_dict[node] = max_node_in_mapping
                for node in no_node:
                    mapping_dict[node] = max_node_in_mapping
                comp_node_2_meta_node = mapping_dict
            else:
                comp_node_2_meta_node = subgraph_mapping(mapping_dict_list)
            meta_node_2_node = metanode_to_node_mapping_new(comp_node_2_meta_node, comp_node_2_node)
            if args.task == "graph_reg":
                for key, value in meta_node_2_node.items():
                    value = np.sort(value)
                    node_2_subgraph_node = {v.item(): i for i, v in enumerate(value)}
                    actual_ext = np.array([], dtype=np.int32)
                    num_nodes = len(value)
                    if args.cluster_node:
                        new_edges = np.array([], dtype=np.int32)
                        new_features = np.array([])
                        meta_node_2_new_node = {}
                        for node in value:
                            N_node = neighbour(data, node)
                            Nt_node = N_node[~np.isin(N_node, value)]
                            connected_clusters = neighbor_2_cluster(Nt_node, node_2_comp_node, comp_node_2_meta_node)
                            for cluster in connected_clusters:
                                if cluster not in meta_node_2_new_node.keys():
                                    meta_node_2_new_node[cluster] = np.array([num_nodes])
                                    if len(actual_ext.shape) <= 1:
                                        actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster]), axis=0)
                                        actual_ext = actual_ext.reshape(1, 1)
                                    else:
                                        actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster].reshape(1,1)), axis=0)
                                    new_feature = C_dot_H_feature[cluster]
                                    if len(new_features.shape) <= 1:
                                        new_features = new_feature
                                        new_features = new_features.reshape(1, len(new_feature))
                                    else:
                                        new_features = np.concatenate((new_features, new_feature.reshape(1,-1)), axis=0)
                                    num_nodes += 1
                                e1 = np.array([node_2_subgraph_node[node], meta_node_2_new_node[cluster][0]], dtype=np.int32)
                                e2 = np.array([meta_node_2_new_node[cluster][0], node_2_subgraph_node[node]], dtype=np.int32)
                                if len(new_edges.shape) <= 1:
                                    new_edges = np.concatenate((new_edges, e1), axis=0)
                                    new_edges = new_edges.reshape(1, 2)
                                    new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                                else:
                                    new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                    new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                            
                        if len(meta_node_2_new_node.keys()) > 1:
                            cluster_keys = list(meta_node_2_new_node.keys())
                            for i in range(len(cluster_keys)-1):
                                for j in range(i+1, len(cluster_keys)):
                                    if adj[cluster_keys[i], cluster_keys[j]] or adj[cluster_keys[j], cluster_keys[i]]:
                                        e1 = np.array([meta_node_2_new_node[cluster_keys[i]][0], meta_node_2_new_node[cluster_keys[j]][0]], dtype=np.int32)
                                        e2 = np.array([meta_node_2_new_node[cluster_keys[j]][0], meta_node_2_new_node[cluster_keys[i]][0]], dtype=np.int32)
                                        new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                        new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)

                        value = np.unique(np.sort(value))
                        value = torch.tensor(value).to(device)
                    elif args.extra_node:
                        extra_node = nodes_2_neighbours(data, value)
                        value = torch.tensor(value).to(device)
                        actual_ext = extra_node[~torch.isin(extra_node, value)]
                        value = torch.cat((value, actual_ext), dim=0)
                        
                    if not torch.is_tensor(value):
                        value = torch.tensor(value).to(device)
                    value, _ = torch.sort(value)
                    mappiing = {}
                    for i in range(len(value)):
                        mappiing[value[i].item()] = i
                    M = data.subgraph(value)
                    M.actual_ext = actual_ext
                    M.orig_idx = value
                    if args.cluster_node:
                        M.x = torch.cat((M.x, torch.tensor(new_features).to(device).float()), dim=0)
                        M.edge_index = torch.cat((M.edge_index.T, torch.tensor(new_edges, dtype=torch.long).to(device)), dim=0).T
                        if args.task == "graph_reg":
                            if len(M.y.size()) > 1:   
                                M.y = torch.cat((M.y, torch.zeros((new_features.shape[0], M.y.shape[1])).to(device)))
                            else:
                                M.y = torch.cat((M.y, torch.zeros(new_features.shape[0]).to(device)))
                        else:
                            M.y = torch.cat((M.y, torch.zeros(len(new_features)).to(device)))
                        for new_node in actual_ext:
                            mappiing[new_node.item()] = new_node.item()
                    M.map_dict = mappiing
                    if args.extra_node:
                        M.mask = torch.tensor(((len(value) - len(actual_ext))*[True] + [False]*len(actual_ext)), dtype=torch.bool).to(device)
                    elif args.cluster_node:
                        M.mask = torch.tensor([True]*len(value) + [False]*len(actual_ext), dtype=torch.bool).to(device)
                    else:
                        M.mask = torch.tensor([True]*len(value), dtype=torch.bool).to(device)
                    subgraph_list.append(M)
            else:
                for key, value in tqdm(meta_node_2_node.items(), colour='blue'):
                    value = np.sort(value)
                    node_2_subgraph_node = {v.item(): i for i, v in enumerate(value)}
                    actual_ext = np.array([], dtype=np.int32)
                    num_nodes = len(value)
                    if args.cluster_node:
                        new_edges = np.array([], dtype=np.int32)
                        new_features = np.array([])
                        meta_node_2_new_node = {}
                        for node in value:
                            N_node = neighbour(data, node)
                            Nt_node = N_node[~np.isin(N_node, value)]
                            connected_clusters = neighbor_2_cluster(Nt_node, node_2_comp_node, comp_node_2_meta_node)
                            for cluster in connected_clusters:
                                if cluster not in meta_node_2_new_node.keys():
                                    meta_node_2_new_node[cluster] = np.array([num_nodes])
                                    if len(actual_ext.shape) <= 1:
                                        actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster]), axis=0)
                                        actual_ext = actual_ext.reshape(1, 1)
                                    else:
                                        actual_ext = np.concatenate((actual_ext, meta_node_2_new_node[cluster].reshape(1,1)), axis=0)
                                    new_feature = C_dot_H_feature[cluster]
                                    if len(new_features.shape) <= 1:
                                        new_features = new_feature
                                        new_features = new_features.reshape(1, len(new_feature))
                                    else:
                                        new_features = np.concatenate((new_features, new_feature.reshape(1,-1)), axis=0)
                                    num_nodes += 1
                                e1 = np.array([node_2_subgraph_node[node], meta_node_2_new_node[cluster][0]], dtype=np.int32)
                                e2 = np.array([meta_node_2_new_node[cluster][0], node_2_subgraph_node[node]], dtype=np.int32)
                                if len(new_edges.shape) <= 1:
                                    new_edges = np.concatenate((new_edges, e1), axis=0)
                                    new_edges = new_edges.reshape(1, 2)
                                    new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                                else:
                                    new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                    new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)
                            
                        if len(meta_node_2_new_node.keys()) > 1:
                            cluster_keys = list(meta_node_2_new_node.keys())
                            for i in range(len(cluster_keys)-1):
                                for j in range(i+1, len(cluster_keys)):
                                    if adj[cluster_keys[i], cluster_keys[j]] or adj[cluster_keys[j], cluster_keys[i]]:
                                        e1 = np.array([meta_node_2_new_node[cluster_keys[i]][0], meta_node_2_new_node[cluster_keys[j]][0]], dtype=np.int32)
                                        e2 = np.array([meta_node_2_new_node[cluster_keys[j]][0], meta_node_2_new_node[cluster_keys[i]][0]], dtype=np.int32)
                                        new_edges = np.concatenate((new_edges, e1.reshape(1,-1)), axis=0)
                                        new_edges = np.concatenate((new_edges, e2.reshape(1,-1)), axis=0)

                        value = np.unique(np.sort(value))
                        value = torch.tensor(value).to(device)
                    elif args.extra_node:
                        extra_node = nodes_2_neighbours(data, value)
                        value = torch.tensor(value).to(device)
                        actual_ext = extra_node[~torch.isin(extra_node, value)]
                        value = torch.cat((value, actual_ext), dim=0)
                        
                    if not torch.is_tensor(value):
                        value = torch.tensor(value).to(device)
                    value, _ = torch.sort(value)
                    mappiing = {}
                    for i in range(len(value)):
                        mappiing[value[i].item()] = i
                    M = data.subgraph(value)
                    M.actual_ext = actual_ext
                    M.orig_idx = value
                    if args.cluster_node:
                        M.x = torch.cat((M.x, torch.tensor(new_features).to(device).float()), dim=0)
                        M.edge_index = torch.cat((M.edge_index.T, torch.tensor(new_edges, dtype=torch.long).to(device)), dim=0).T
                        if args.task == "graph_reg":
                            M.y = torch.cat((M.y, torch.zeros((new_features.shape[0], M.y.shape[1])).to(device)))
                        else:
                            M.y = torch.cat((M.y, torch.zeros(len(new_features)).to(device)))
                        for new_node in actual_ext:
                            mappiing[new_node.item()] = new_node.item()
                    M.map_dict = mappiing
                    if args.extra_node:
                        M.mask = torch.tensor(((len(value) - len(actual_ext))*[True] + [False]*len(actual_ext)), dtype=torch.bool).to(device)
                    elif args.cluster_node:
                        M.mask = torch.tensor([True]*len(value) + [False]*len(actual_ext), dtype=torch.bool).to(device)
                    else:
                        M.mask = torch.tensor([True]*len(value), dtype=torch.bool).to(device)
                    subgraph_list.append(M)
        else:
            comp_node_2_meta_node = {0: 0}
            meta_node_2_node = metanode_to_node_mapping_new(comp_node_2_meta_node, comp_node_2_node)
            for key, value in meta_node_2_node.items():
                value = torch.LongTensor(value).to(device)
                value, _ = torch.sort(value)
                actual_ext = torch.LongTensor([]).to(device)
                M = data.subgraph(value)
                M.actual_ext = actual_ext
                M.orig_idx = value
                mappiing = {}
                for i in range(len(value)):
                    mappiing[value[i].item()] = i
                M.map_dict = mappiing
                M.mask = torch.tensor([True], dtype=torch.bool).to(device)
                subgraph_list.append(M)
        number += 1
    if args.task == "node_reg":
        return data.x.shape[1], subgraph_list
    else:
        return data.x.shape[1], candidate, subgraph_list, CLIST, GcLIST

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def splits_classification(data, num_classes, exp):
    if exp!='fixed':
        indices = []
        for i in range(num_classes):
            if len(data.y.shape) > 1:
                labels = data.y.flatten()
                index = (labels == i).nonzero().view(-1)
            else:
                index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        elif exp == 'few':
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)
        elif exp == 'ogbn_split':
            num_nodes = np.random.permutation(data.num_nodes)
            train_index = torch.tensor(num_nodes[:int(0.08*data.num_nodes)], dtype=torch.long)
            val_index = torch.tensor(num_nodes[int(0.08*data.num_nodes):int(0.1*data.num_nodes)], dtype=torch.long)
            test_index = torch.tensor(num_nodes[int(0.1*data.num_nodes):], dtype=torch.long)
            

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data

def splits_regression(data, train_ratio, val_ratio):
    if train_ratio + val_ratio >= 1:
        raise ValueError('train_ratio + val_ratio should be less than 1')
    num_nodes = data.x.shape[0]
    num_train = int(train_ratio*num_nodes)
    num_val = int(val_ratio*num_nodes)
    perm_nodes = torch.randperm(num_nodes)
    train_nodes = perm_nodes[:num_train]
    val_nodes = perm_nodes[num_train:num_train+num_val]
    test_nodes = perm_nodes[num_train+num_val:]
    data.train_mask = index_to_mask(train_nodes, size=num_nodes)
    data.val_mask = index_to_mask(val_nodes, size=num_nodes)
    data.test_mask = index_to_mask(test_nodes, size=num_nodes)

    return data

def load_data_classification(args, dataset, candidate, C_list, Gc_list, exp, subgraph_list):
    n_classes = args.num_classes
    data = splits_classification(dataset, n_classes, exp)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    labels = data.y.cpu()
    features = data.x.cpu()

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()

    new_graphs = []

    for graph in subgraph_list:
        F = Data(x=graph.x, edge_index=graph.edge_index, y=graph.y, num_classes=n_classes)
        F.train_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
        F.val_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
        F.test_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
        for node, new_node in graph.map_dict.items():
            if train_mask[node]:
                F.train_mask[new_node] = True
            if val_mask[node]:
                F.val_mask[new_node] = True
            if test_mask[node]:
                F.test_mask[new_node] = True
            if args.extra_node and node in graph.actual_ext:
                F.train_mask[new_node] = False
                F.val_mask[new_node] = False
                F.test_mask[new_node] = False
            if args.cluster_node and new_node in graph.actual_ext:
                F.train_mask[new_node] = False
                F.val_mask[new_node] = False
                F.test_mask[new_node] = False
        new_graphs.append(F)

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]

        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            if train_labels.shape[1] == 1 and train_labels.shape[-1] == n_classes:
                train_labels = train_labels.squeeze(1)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            if val_labels.shape[1] == 1 and val_labels.shape[-1] == n_classes:
                val_labels = val_labels.squeeze(1)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])
            
            C = C_list[number]
            Gc = Gc_list[number]
            
            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    coarsen_edge = np.array([coarsen_row, coarsen_col])
    coarsen_edge = torch.LongTensor(coarsen_edge)
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()
    del dataset, candidate, C_list, Gc_list, subgraph_list
    torch.cuda.empty_cache()
    return n_classes, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, new_graphs
    
def load_data_regression(args, dataset, subgraph_list):
    data = splits_regression(dataset, args.train_ratio, args.val_ratio)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    new_graphs = []

    for graph in subgraph_list:
        F = Data(x=graph.x, edge_index=graph.edge_index, y=graph.y)
        F.train_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
        F.val_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
        F.test_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
        for node, new_node in graph.map_dict.items():
            if train_mask[node]:
                F.train_mask[new_node] = True
            if val_mask[node]:
                F.val_mask[new_node] = True
            if test_mask[node]:
                F.test_mask[new_node] = True
            if args.extra_node and node in graph.actual_ext:
                F.train_mask[new_node] = False
                F.val_mask[new_node] = False
                F.test_mask[new_node] = False
            if args.cluster_node and new_node in graph.actual_ext:
                F.train_mask[new_node] = False
                F.val_mask[new_node] = False
                F.test_mask[new_node] = False
        new_graphs.append(F)
    return new_graphs

def load_graph_data(data, C_LIST, GC_LIST, candidate):
    features = data.x.cpu()
    y = data.y.cpu()
    number = 0
    coarsen_node = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        if len(keep)>1:
            C = C_LIST[number]
            GC = GC_LIST[number]
            H_features = features[keep]

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)

            if coarsen_row is None:
                coarsen_row = GC.W.tocoo().row
                coarsen_col = GC.W.tocoo().col
            else:
                current_row = GC.W.tocoo().row + coarsen_node
                current_col = GC.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += GC.W.shape[0]
        else:
            coarsen_features = torch.cat([coarsen_features, features[keep]], dim=0)
            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1
    
    coarsen_edge = torch.LongTensor(np.array([coarsen_row, coarsen_col]))
    Gc = Data(x=coarsen_features, edge_index=coarsen_edge, y=y)
    return Gc

def adj_matrix_2_edge_index(adj_matrix):
    edge_index = []
    for i in range(adj_matrix.shape[0] - 1):
        for j in range(i+1, adj_matrix.shape[1]):
            if adj_matrix[i][j]:
                edge_index.append([i, j])
                edge_index.append([j, i])
    return torch.LongTensor(edge_index).T

def extract_masks(data):
    masks = []
    if type(data[0]) == list:
        for subgraph_list in data:
            temp = []
            for subgraph in subgraph_list:
                temp.append(subgraph.mask)
            masks.append(temp)
    else:
        for subgraph in data:
            masks.append(subgraph.mask)
    return masks

def load_data(Gc_dict, Gs_dict, y, args):
    if args.seed != None:
        np.random.seed(args.seed)
    l = len(Gc_dict)
    perm = np.random.permutation(len(Gc_dict))
    splits = [int(l*args.train_ratio), int(l*args.val_ratio), l - (int(l*args.train_ratio) + int(l*args.val_ratio))]
    train, val, test = [], [], []
    for iter, ele in enumerate(perm):
        if iter < splits[0]:
            train.append([Gc_dict[ele], Gs_dict[ele], extract_masks(Gs_dict[ele]), y[ele]])
        elif iter < splits[0] + splits[1]:
            val.append([Gc_dict[ele], Gs_dict[ele], extract_masks(Gs_dict[ele]), y[ele]])
        else:
            test.append([Gc_dict[ele], Gs_dict[ele], extract_masks(Gs_dict[ele]), y[ele]])

    return train, val, test

class colater:
    def __init__(self):
        pass
    def __call__(self, data_list):
        batch_tensor = torch.tensor([])
        GS = []
        GC = []
        Y = torch.tensor([])
        for i, graph_data in enumerate(data_list):
            graph = graph_data[0]
            batch_tensor = torch.cat((batch_tensor, torch.full((graph.x.shape[0],), i, dtype=torch.long)))
            Y = torch.cat((Y, graph.y.cpu()))
            GC.append(graph_data[1])
            GS.append(graph_data[2])
        GC_batch = Batch.from_data_list(GC)
        return GC_batch, GS, Y, batch_tensor
    
class NLLLoss_numpy:
    """
    A class to compute the Negative Log-Likelihood (NLL) Loss.
    """
    def __init__(self, reduction='mean'):
        """
        Initialize the NLLLoss class.

        Parameters:
        reduction (str): Specifies the reduction to apply to the output:
                         - 'mean': return the mean of the loss.
                         - 'sum': return the sum of the loss.
        """
        if reduction not in ['mean', 'sum']:
            raise ValueError("Reduction must be 'mean' or 'sum'.")
        self.reduction = reduction

    def compute_loss(self, log_probs, targets):
        """
        Compute the NLL Loss.

        Parameters:
        log_probs (numpy.ndarray): Log probabilities of the predictions with shape (N, C),
                                   where N is the number of samples and C is the number of classes.
        targets (numpy.ndarray): True target indices with shape (N,).

        Returns:
        float: The NLL loss.
        """
        if len(log_probs.shape) != 2:
            raise ValueError("log_probs must be a 2D array with shape (N, C).")
        if len(targets.shape) != 1 or targets.shape[0] != log_probs.shape[0]:
            raise ValueError("targets must be a 1D array with the same number of elements as rows in log_probs.")
        if not np.all((targets >= 0) & (targets < log_probs.shape[1])):
            raise ValueError("Targets must contain valid class indices.")

        # Extract the log probabilities corresponding to the target classes
        nll_values = -log_probs[np.arange(log_probs.shape[0]), targets]

        # Apply the reduction
        if self.reduction == 'mean':
            return np.mean(nll_values)
        elif self.reduction == 'sum':
            return np.sum(nll_values)
        
class L1Loss_numpy:
    """
    A class to compute the L1 Loss.
    """
    def __init__(self, reduction='mean'):
        """
        Initialize the L1Loss class.

        Parameters:
        reduction (str): Specifies the reduction to apply to the output:
                         - 'mean': return the mean of the loss.
                         - 'sum': return the sum of the loss.
        """
        if reduction not in ['mean', 'sum']:
            raise ValueError("Reduction must be 'mean' or 'sum'.")
        self.reduction = reduction

    def compute_loss(self, predictions, targets):
        """
        Compute the L1 Loss.

        Parameters:
        predictions (numpy.ndarray): Predicted values with shape (N,).
        targets (numpy.ndarray): True target values with shape (N,).

        Returns:
        float: The L1 loss.
        """
        if len(predictions.shape) != 1 or predictions.shape[0] != targets.shape[0]:
            raise ValueError("Predictions and targets must be 1D arrays with the same number of elements.")
        if self.reduction == 'mean':
            return np.mean(np.abs(predictions - targets))
        elif self.reduction == 'sum':
            return np.sum(np.abs(predictions - targets))