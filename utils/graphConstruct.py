"""
Graph construction utilities for building social graph and hypergraphs.
Creates:
1. Friendship graph for social connections
2. Two hypergraphs for capturing high-order semantic memory:
   - Source-user hypergraph: connects users who share the same diffusion source
   - Item-user hypergraph: connects users who appear in the same cascade
"""
import torch
import pickle
import scipy.sparse as ss
import numpy as np
from torch_geometric.data import Data
from dataLoader import Options

def _convert_sp_mat_to_sp_tensor(X):
    """
    Convert scipy sparse matrix to PyTorch sparse tensor.
    
    Args:
        X (scipy.sparse.csr_matrix): Input sparse matrix.
    
    Returns:
        torch.sparse.FloatTensor: Converted PyTorch sparse tensor.
    """
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def Read_all_cascade(data_name):
    """
    Read all cascades from the original data file.
    
    Args:
        data_name (str): Name of the dataset.
    
    Returns:
        tuple: (all_cascade, source_users) where:
            - all_cascade: List of all cascades (each cascade is list of user IDs)
            - source_users: List of source (root) users for each cascade.
    """
    options = Options(data_name)
    with open(options.u2idx_dict, 'rb') as handle:
        u2idx = pickle.load(handle)

    all_cascade = []
    source_users = []
    for line in open(options.data):
        if len(line.strip()) == 0:
            continue
        chunks = line.strip().strip(',').split(',')
        cascade = []
        for chunk in chunks:
            try:
                if len(chunk.split()) == 2:
                    user, timestamp = chunk.split()
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()
                    source_users.append(u2idx[root])
                    cascade.append(u2idx[root])
            except:
                print(chunk)
            cascade.append(u2idx[user])
        all_cascade.append(cascade)
    return all_cascade, source_users

def ConRelationGraph(data):
    """
    Construct the friendship (social connection) graph.
    
    Args:
        data (str): Name of the dataset.
    
    Returns:
        torch_geometric.data.Data: PyTorch Geometric Data object containing
            the social graph with bidirectional edges.
    """
    options = Options(data)

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)

    edges_list = []
    with open(options.net_data, 'r') as handle:
        relation_list = handle.read().strip().split("\n")
        relation_list = [edge.split(',') for edge in relation_list]

        relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if
                         edge[0] in _u2idx and edge[1] in _u2idx]
        relation_list_reverse = [edge[::-1] for edge in relation_list]
        edges_list += relation_list
        edges_list += relation_list_reverse

    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()
    graph = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)

    return graph

def ConHypergraph(data_name, user_size):
    """
    Construct two hypergraphs for semantic memory:
    1. Source-user hypergraph: connects users that share the same diffusion source
    2. Item-user (cascade-item) hypergraph: connects users that co-occur in the same cascade
    
    Uses spectral normalization for hypergraph convolution.
    
    Args:
        data_name (str): Name of the dataset.
        user_size (int): Total number of users.
    
    Returns:
        tuple: (HG_Source, HG_Item) where each is a PyTorch sparse tensor
            containing the normalized incidence matrix for the hypergraph.
    """
    all_cascade, source_users = Read_all_cascade(data_name)

    # Construct item-user hypergraph
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_cascade)):
        items = np.unique(all_cascade[j])
        length = len(items)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(items[i])
            data.append(1)

    H_T = ss.csr_matrix((data, indices, indptr), shape=(len(all_cascade), user_size))
    H_T_sum = 1.0 / H_T.sum(axis=1).reshape(1, -1)
    H_T_sum[H_T_sum == float("inf")] = 0

    BH_T = H_T.T.multiply(H_T_sum)
    BH_T = BH_T.T
    H = H_T.T

    H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)
    DH = DH.T
    HG_Item = np.dot(DH, BH_T).tocoo()

    # Construct source-user hypergraph
    source_dict = {}
    for k in source_users:
        source_dict[k] = []

    for j in range(len(all_cascade)):
        source_u = all_cascade[j][0]
        u_list = list(set(source_dict[source_u] + all_cascade[j]))
        source_dict[source_u] = u_list

    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(source_users)):
        idx = source_users[j]
        source = np.unique(source_dict[idx])
        length = len(source)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(source[i])
            data.append(1)

    H_S = ss.csr_matrix((data, indices, indptr), shape=(len(source_users), user_size))
    H_S_sum = 1.0 / H_S.sum(axis=1).reshape(1, -1)
    H_S_sum[H_S_sum == float("inf")] = 0

    BH_T = H_S.T.multiply(H_S_sum)
    BH_T = BH_T.T
    H = H_S.T

    H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)
    DH = DH.T
    HG_Source = np.dot(DH, BH_T).tocoo()

    HG_Source = _convert_sp_mat_to_sp_tensor(HG_Source)
    HG_Item = _convert_sp_mat_to_sp_tensor(HG_Item)

    return HG_Source, HG_Item
