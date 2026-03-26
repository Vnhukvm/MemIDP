"""
Data loading and preprocessing module for information diffusion prediction.
This module handles data reading, splitting, and dataset creation for training.
"""
import os.path
import random
import numpy as np
import torch
import Constants
import pickle
from torch.utils.data import Dataset

class Options(object):
    """Configuration class storing file paths for different datasets."""
    def __init__(self, data_name='android'):
        """
        Initialize options with file paths based on dataset name.
        
        Args:
            data_name (str): Name of the dataset. Defaults to 'android'.
        """
        self.data = 'data/' + data_name + '/cascades.txt'
        self.u2idx_dict = 'data/' + data_name + '/u2idx.pickle'
        self.idx2u_dict = 'data/' + data_name + '/idx2u.pickle'
        self.net_data = 'data/' + data_name + '/edges.txt'
        self.related_cas = 'data/' + data_name + '/related_cas.pickle'
        self.train_data = 'data/' + data_name + '/cascadetrain.txt'
        self.valid_data = 'data/' + data_name + '/cascadevalid.txt'
        self.test_data = 'data/' + data_name + '/cascadetest.txt'

def _readFromFile(filename, with_EOS):
    """
    Read cascades and timestamps from a text file.
    
    Args:
        filename (str): Path to the input file.
        with_EOS (bool): Whether to add EOS token at the end of sequence.
    
    Returns:
        tuple: (t_cascades, timestamps) where each is a list of lists.
    """
    t_cascades = []
    timestamps = []

    for line in open(filename):
        if len(line.strip()) == 0:
            continue

        timestamplist = []
        userlist = []
        chunks = line.strip().split()
        for i, chunk in enumerate(chunks):
            try:
                user, timestamp = chunk.split(',')
            except:
                print(chunk)

            userlist.append(int(user))
            timestamplist.append(float(timestamp))

        if with_EOS:
            userlist.append(Constants.EOS)
            timestamplist.append(timestamplist[-1])

        t_cascades.append(userlist)
        timestamps.append(timestamplist)

    return t_cascades, timestamps

def Read_data(data_name, with_EOS=True):
    """
    Read processed train/validation/test splits from disk.
    
    Args:
        data_name (str): Name of the dataset.
        with_EOS (bool): Whether to add EOS token. Defaults to True.
    
    Returns:
        tuple: (train, valid, test, user_size) where each split contains
            list of cascades and list of timestamps.
    """
    options = Options(data_name)

    with open(options.u2idx_dict, 'rb') as handle:
        u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        idx2u = pickle.load(handle)
    user_size = len(u2idx)

    train, train_t  = _readFromFile(options.train_data, with_EOS)
    valid, valid_t = _readFromFile(options.valid_data, with_EOS)
    test, test_t = _readFromFile(options.test_data, with_EOS)

    train = [train, train_t]
    valid = [valid, valid_t]
    test = [test, test_t]

    return train, valid, test, user_size

def buildIndex(data):
    """
    Build user-to-index mapping from raw data.
    
    Args:
        data (str): Path to raw cascade data.
    
    Returns:
        tuple: (user_size, u2idx, idx2u) containing vocabulary info.
    """
    user_set = set()
    u2idx = {}
    idx2u = []

    lineid = 0
    for line in open(data):
        lineid += 1
        if len(line.strip()) == 0:
            continue
        chunks = line.strip().split(',')
        for chunk in chunks:
            try:
                if len(chunk.split()) == 2:
                    user, timestamp = chunk.split()
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()
                    user_set.add(root)
            except:
                print(line)
                print(chunk)
                print(lineid)
            user_set.add(user)
    pos = 0
    u2idx['<blank>'] = pos
    idx2u.append('<blank>')
    pos += 1
    u2idx['</s>'] = pos
    idx2u.append('</s>')
    pos += 1

    for user in user_set:
        u2idx[user] = pos
        idx2u.append(user)
        pos += 1
    user_size = len(user_set) + 2
    print("user_size : %d" % (user_size))
    return user_size, u2idx, idx2u

def Split_data(data_name, train_rate=0.8, valid_rate=0.1, load_dict=False):
    """
    Split raw data into train/validation/test sets and save to files.
    
    Args:
        data_name (str): Name of the dataset.
        train_rate (float): Fraction of data for training. Defaults to 0.8.
        valid_rate (float): Fraction of data for validation. Defaults to 0.1.
        load_dict (bool): Whether to load existing user index. Defaults to False.
    """
    options = Options(data_name)

    if not load_dict:
        user_size, u2idx, idx2u = buildIndex(options.data)
        with open(options.u2idx_dict, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.idx2u_dict, 'wb') as handle:
            pickle.dump(idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(options.u2idx_dict, 'rb') as handle:
            u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)

    user_size = len(u2idx)

    t_cascades = []
    timestamps = []
    for line in open(options.data):
        if len(line.strip()) == 0:
            continue
        timestamplist = []
        userlist = []

        chunks = line.strip().strip(',').split(',')
        for chunk in chunks:
            try:
                if len(chunk.split()) == 2:
                    user, timestamp = chunk.split()
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()
                    userlist.append((u2idx[root]))
                    timestamplist.append(float(timestamp))
            except:
                print(chunk)

            userlist.append((u2idx[user]))
            timestamplist.append(float(timestamp))

        t_cascades.append(userlist)
        timestamps.append(timestamplist)

    # Sort cascades by timestamp
    order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x: x[1])]
    timestamps = sorted(timestamps)
    t_cascades[:] = [t_cascades[i] for i in order]
    cas_idx = [i for i in range(len(t_cascades))]

    # Split into train/validation/test
    train_idx_ = int(train_rate * len(t_cascades))
    train = t_cascades[0:train_idx_]
    train_t = timestamps[0:train_idx_]

    valid_idx_ = int((train_rate + valid_rate) * len(t_cascades))
    valid = t_cascades[train_idx_:valid_idx_]
    valid_t = timestamps[train_idx_:valid_idx_]

    test = t_cascades[valid_idx_:]
    test_t = timestamps[valid_idx_:]

    # Write splits to files
    with open(options.train_data, 'w') as file:
        file.truncate(0)

    with open(options.train_data, 'w') as f:
        for i in range(len(train)):
            data = list(zip(train[i], train_t[i]))
            data_lst0 = [','.join(map(str, x)) for x in data]
            data = ' '.join(data_lst0)
            f.writelines(data + "\n")

    with open(options.valid_data, 'w') as file:
        file.truncate(0)

    with open(options.valid_data, 'w') as f:
        for i in range(len(valid)):
            data = list(zip(valid[i], valid_t[i]))
            data_lst0 = [','.join(map(str, x)) for x in data]
            data = ' '.join(data_lst0)
            f.writelines(data + "\n")

    with open(options.test_data, 'w') as file:
        file.truncate(0)

    with open(options.test_data, 'w') as f:
        for i in range(len(test)):
            data = list(zip(test[i], test_t[i]))
            data_lst0 = [','.join(map(str, x)) for x in data]
            data = ' '.join(data_lst0)
            f.writelines(data + "\n")

    # Print dataset statistics
    total_len = sum(len(i) - 1 for i in t_cascades)
    train_size = len(train_t)
    valid_size = len(valid_t)
    test_size = len(test_t)
    print("training size:%d\n   valid size:%d\n  testing size:%d" % (train_size, valid_size, test_size))
    print("total size:%d " % (len(t_cascades)))
    print("average length:%f" % (total_len / len(t_cascades)))
    print('maximum length:%f' % (max(len(cas) for cas in t_cascades)))
    print('minimum length:%f' % (min(len(cas) for cas in t_cascades)))
    print("user size:%d" % (user_size - 2))


def Read_all_cascade(data_name, with_EOS=False):

    options = Options(data_name)

    '''user size'''
    with open(options.u2idx_dict, 'rb') as handle:
        u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        idx2u = pickle.load(handle)
    user_size = len(u2idx)

    '''load train data, validation data and test data'''
    train, train_t  = _readFromFile(options.train_data, with_EOS)
    valid, valid_t = _readFromFile(options.valid_data, with_EOS)
    test, test_t = _readFromFile(options.test_data, with_EOS)

    all_cascade = train + valid + test

    source_user = []
    for line in all_cascade:
        source_user.append(line[0])

    source_user = list(set(source_user))

    return all_cascade, source_user


def related_cas_(data_name, mini_seq_len=5, with_EOS=True, opt=None):
    options = Options(data_name)
    options.related_cas = 'data/' + opt.data_name + '/related_cas_w{}_top{}.pickle'.format(opt.window_size, opt.top_k)
    # print(options.related_cas)
    if os.path.exists(options.related_cas): 
        return pickle.load(open(options.related_cas, 'rb'))




class diffusionDataset(Dataset):
    """
    PyTorch Dataset for diffusion cascade data.
    Handles padding and prepares sequences with retrieved related cascades for memory.
    """
    def __init__(self, cascade, time, related_cas, max_len, top_k, window_size):
        """
        Initialize the diffusion dataset.
        
        Args:
            cascade (list): List of diffusion cascades (each cascade is list of user IDs).
            time (list): List of timestamps corresponding to each cascade.
            related_cas (list): Pre-retrieved related cascades for memory retrieval.
            max_len (int): Maximum sequence length for padding/truncation.
            top_k (int): Number of top related cascades to retrieve.
            window_size (int): Window size for slicing each related cascade.
        """
        self.cascade  = cascade
        self.time = time
        self.related_cas = related_cas
        self.max_len = max_len
        self.top_k = top_k
        self.window_size = window_size

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.cascade)

    def __getitem__(self, index):
        """
        Get one sample from dataset.
        For a cascade [u1, u2, u3, ..., un, EOS], creates prediction tasks for each step:
        - Input: [u1], Target: [u2]
        - Input: [u1, u2], Target: [u3]
        - ...
        - Input: [u1, ..., u_{n-1}], Target: [EOS]
        
        Args:
            index (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (seq, target, time_seq, target_time, related, related_mask) where:
                - seq: Input sequence [max_len-1]
                - target: Target sequence [max_len-1]
                - time_seq: Input timestamps [max_len-1]
                - target_time: Target timestamps [max_len-1]
                - related: Retrieved related cascades [(max_len-1)*top_k*window_size]
                - related_mask: Mask for related cascade padding [(max_len-1)*top_k*window_size]
        """
        cascade = self.cascade[index]
        time    = self.time[index]
        related_cas = self.related_cas[index]

        if len(cascade) < self.max_len:
            pad_len = self.max_len - len(cascade)
            cascade += [Constants.PAD] * pad_len
            time += [0] * pad_len

        seq = cascade[:-1]
        time_seq = time[:-1]
        related = []
        related_mask = []
        for related_item in related_cas:
            pad_related_item = related_item + [Constants.PAD] * (self.top_k * self.window_size - len(related_item))
            pad_mask = [float(1)] * len(related_item) + [float(0)] * (self.top_k * self.window_size - len(related_item))
            related += pad_related_item
            related_mask += pad_mask
        target = cascade[1:]
        target_time = time[1:]

        return torch.LongTensor(seq), torch.LongTensor(target), torch.FloatTensor(time_seq), torch.FloatTensor(target_time), torch.LongTensor(related), torch.FloatTensor(related_mask)