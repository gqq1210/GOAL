import torch
import numpy as np
import scipy.sparse as sparse
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from scipy import sparse
from torch_geometric.nn import GATConv, JumpingKnowledge
from torch.nn import Parameter, Linear, LeakyReLU, Dropout
from utils import kl_categorical
from torch import nn
import time

import scipy.sparse
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """

    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    """ our implementation """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes, nclass, noise_labels, idx_train, idx_clean, clean_list, model_type,
                 num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                 use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.noise_labels = noise_labels
        self.idx_train = idx_train
        self.idx_clean = idx_clean
        self.clean_list = clean_list
        self.model_type = model_type
        self.nclass = nclass

        self.feature_embed = MLP(in_channels, hidden_channels,
                                 hidden_channels, num_layers=num_mlp_layers, dropout=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(
                    hidden_channels*2*len(self.convs)))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout 

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(
                row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, data, infer_z=None):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']

        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)

        x = F.normalize(x, dim=1)

        
        if self.model_type == 'Q' or 'Re':
            prototype_embed = []
            for i in range(self.nclass):
                prototype_embed.append(x[self.clean_list[i]].mean(axis=0)) 
            prototype_embed = torch.stack(prototype_embed)
            clean_labels = self.noise_labels[self.idx_clean]

        
        if self.model_type == 'P':
            H = x
        elif self.model_type == 'Q':
            clean_x = prototype_embed[clean_labels]
            x_sim_prototype = torch.mm(x, prototype_embed.t())
            fake_label = torch.argmax(x_sim_prototype, dim=1)
            label_only_L = prototype_embed[fake_label]
            label_only_L[self.idx_clean] = clean_x
            H = (x + label_only_L)/2

        elif self.model_type == 'Re':
            infer_z = F.softmax(infer_z, dim=1)
            embed = torch.mm(infer_z, prototype_embed)
            H = (x + embed)/2
        
        return H



        
class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, noise_labels, idx_train, idx_clean, clean_list, model_type):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.noise_labels = noise_labels
        self.idx_train = idx_train
        self.idx_clean = idx_clean
        self.clean_list = clean_list
        self.model_type = model_type
        self.nhid = nhid
        self.nclass = nclass

    def forward(self, x, adj, infer_z=None):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.normalize(x, dim=1)

        
        if self.model_type == 'Q' or 'Re':
            prototype_embed = []
            for i in range(self.nclass):
                prototype_embed.append(x[self.clean_list[i]].mean(axis=0)) 
            prototype_embed = torch.stack(prototype_embed)
            clean_labels = self.noise_labels[self.idx_clean]

        
        if self.model_type == 'P':
            H = x
        elif self.model_type == 'Q':
            clean_x = prototype_embed[clean_labels]
            x_sim_prototype = torch.mm(x, prototype_embed.t())
            fake_label = torch.argmax(x_sim_prototype, dim=1)
            label_only_L = prototype_embed[fake_label]
            label_only_L[self.idx_clean] = clean_x
            H = (x + label_only_L)/2
        elif self.model_type == 'Re':
            infer_z = F.softmax(infer_z, dim=1)
            embed = torch.mm(infer_z, prototype_embed)
            H = (x + embed)/2
   
        return H
    
    
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)



    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support) 
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
