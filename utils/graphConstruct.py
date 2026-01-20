import torch
import pickle
import scipy.sparse as ss
import numpy as np
from torch_geometric.data import Data
from dataLoader import Options, Read_all_cascade


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


'''Friendship network'''
def ConRelationGraph(data):

    options = Options(data)

    _u2idx = {}
    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)

    n_node = len(_u2idx)

    edges_list = []
    # 建反边
    with open(options.net_data, 'r') as handle:
        relation_list = handle.read().strip().split("\n")
        relation_list = [edge.split(',') for edge in relation_list]

        relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if
                         edge[0] in _u2idx and edge[1] in _u2idx]
        relation_list_reverse = [edge[::-1] for edge in relation_list]
        edges_list += relation_list_reverse

    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()
    graph = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)

    return graph

'''Hypergraph'''
def ConHypergraph(data_name, user_size):

    all_cascade, source_users = Read_all_cascade(data_name)
    # source_users 每一个cascade的头集合
    '''U-I hypergraph'''
    indptr, indices, data = [], [], []
    indptr.append(0)
    # indptr 数组的作用就是告诉我们每行的非零元素在 data 和 indices 数组中的起始位置
    for j in range(len(all_cascade)):
        items = np.unique(all_cascade[j]) # 去重 排序

        length = len(items)

        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(items[i])
            data.append(1)

    H_T = ss.csr_matrix((data, indices, indptr), shape=(len(all_cascade), user_size)) # [cas_num, user_size]
    # 每一行代表一个级联（超边），超边里哪些用户有出现（1）
    H_T_sum = 1.0 / H_T.sum(axis=1).reshape(1, -1)
    H_T_sum[H_T_sum == float("inf")] = 0 # [1, cas_num]

    # BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
    BH_T = H_T.T.multiply(H_T_sum) # 逐元素乘积  [user_size, cas_num]
    BH_T = BH_T.T # [cas_num, user_size]
    H = H_T.T # [cas_num, user_size]

    H_sum = 1.0 / H.sum(axis=1).reshape(1, -1) # [1, user_size]
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum) # [cas_num, user_size]
    # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    DH = DH.T # [user_size, cas_num]
    HG_Item = np.dot(DH, BH_T).tocoo() # [user_size, user_size]


    ''' U-S hypergraph'''
    source_dict = {}
    for k in source_users:
        source_dict[k] = []

    for j in range(len(all_cascade)):
        source_u = all_cascade[j][0]
        u_list = list(set(source_dict[source_u]+ all_cascade[j]))
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

    H_S = ss.csr_matrix((data, indices, indptr), shape=(len(source_users), user_size)) # [source_num, user_size]

    H_S_sum = 1.0 / H_S.sum(axis=1).reshape(1, -1)
    H_S_sum[H_S_sum == float("inf")] = 0

    # BH_T = H_S.T.multiply(1.0 / H_S.sum(axis=1).reshape(1, -1))
    BH_T = H_S.T.multiply(H_S_sum)
    BH_T = BH_T.T
    H = H_S.T

    H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)
    # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    DH = DH.T
    HG_Source = np.dot(DH, BH_T).tocoo() # [user_size, user_size]

    HG_Source = _convert_sp_mat_to_sp_tensor(HG_Source)
    HG_Item = _convert_sp_mat_to_sp_tensor(HG_Item)
    # 得到每个用户与其他用户之间的cas关系、源头cas矩阵
    return HG_Source, HG_Item
