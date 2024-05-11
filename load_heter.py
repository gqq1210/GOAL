import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import pickle as pkl
import networkx as nx
import random
import os
import scipy
from torch_geometric.utils import to_undirected

import warnings
warnings.filterwarnings('ignore')


class NCDataset(object):
    def __init__(self, name, root=''):

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None
    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_geom_gcn_dataset(name, device):
    fulldata = scipy.io.loadmat('./data/' + name + '.mat')
    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index.to(device),
                     'node_feat': node_feat.to(device),
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize(mx):
    """Row-normalize sparse matrix""" 
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def load_data(dataset_str, device=None):
    if dataset_str in ['citeseer', 'cora', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]


    elif dataset_str in ['chameleon', 'film', 'squirrel']:
        dataset = load_geom_gcn_dataset(dataset_str, device)
        features = dataset.graph['node_feat']
        labels = dataset.label
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    
    train_rate = 0.4
    val_rate = 0.4
    idx_train = range(0, int(len(labels) * train_rate))
    idx_val = range(int(len(labels) * train_rate), int(len(labels) * (train_rate + val_rate)))
    idx_test = range(int(len(labels) * (train_rate + val_rate)), int(len(labels)))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if dataset_str in ['citeseer', 'cora', 'pubmed']:
        labels = torch.LongTensor(labels.argmax(1))

        adj_dense = normalize(adj + sp.eye(adj.shape[0])).todense() 
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))

        return adj_dense, adj, features, labels, idx_train, idx_val, idx_test
    elif dataset_str in ['chameleon', 'film', 'squirrel']:
        return dataset, features, labels, idx_train, idx_val, idx_test
    
    

def create_clean(labels, idx_val, clean_label_num):

    data_list_clean = {}
    for j in range(int(labels.max()) + 1):
        data_list_clean[j] = [i + int(len(labels) * 0.4) for i, label in enumerate(labels[idx_val]) if label == j]
    list_clean = []
    num = int(clean_label_num / (int(labels.max()) + 1))
    for i, ind in data_list_clean.items():
        np.random.shuffle(ind)
        list_clean.append(ind[0:num])
    idx_clean = np.array(list_clean)
    idx_clean = idx_clean.flatten()
    idx_clean = torch.LongTensor(idx_clean)
    return idx_clean