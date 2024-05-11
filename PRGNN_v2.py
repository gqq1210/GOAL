import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import random
import itertools
import scipy.io as scio
import sys
import copy
from model.Model_v2 import GCN, H2GCN
from utils import kl_categorical, nor
from noise_utils import uniform_mix_C, flip_labels_C
from sklearn.metrics.pairwise import cosine_similarity
from load_heter import load_data, create_clean
gpu = 1
device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) 
    random.seed(args.seed)
    
    if args.dataset in ['citeseer', 'cora', 'pubmed']:
        adj_normal, adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
        features = features.to(device)
        adj_normal = torch.FloatTensor(adj_normal).to(device)
    elif args.dataset in ['chameleon', 'film', 'squirrel']:
        dataset, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, device)
    else:
        print("Dataset Error!")
        exit()
        

    num_classes = int(labels.max()) + 1

    noise_init_list = [[] for _ in range(num_classes)]
    for i in range(len(idx_train)):
        noise_init_list[labels[idx_train[i]]].append(int(idx_train[i]))

    idx_clean = create_clean(labels, idx_val, args.clean_num)
    clean_init_list = [[] for _ in range(num_classes)]
    for i in range(len(idx_clean)):
        clean_init_list[labels[idx_clean[i]]].append(int(idx_clean[i]))

    label_true = torch.tensor(labels)

    idx_noise = idx_train
    if args.noise_type == 'uniform':
        C = uniform_mix_C(args.ptb, num_classes)
    elif args.noise_type == 'flip':
        C = flip_labels_C(args.ptb, num_classes)
    noise_labels = labels.numpy().copy()
    for i in idx_noise:
        noise_labels[i] = np.random.choice(num_classes, p=C[labels[i]])
    noise_labels = torch.tensor(noise_labels)
    _onehot_z = torch.FloatTensor(np.eye(num_classes)[noise_labels]).to(device)

    if args.dataset in ['citeseer', 'cora', 'pubmed']:
        GCN_Pi = GCN(nfeat=features.shape[1],
                    nhid=args.hidden_num,
                    nclass=num_classes,
                    dropout=args.dropout,
                    noise_labels=noise_labels,
                    idx_train=idx_train,
                    idx_clean=idx_clean,
                    clean_list=clean_init_list,
                    model_type='P').to(device)

        GCN_Qi = GCN(nfeat=features.shape[1],
                    nhid=args.hidden_num,
                    nclass=num_classes,
                    dropout=args.dropout,
                    noise_labels=noise_labels,
                    idx_train=idx_train,
                    idx_clean=idx_clean,
                    clean_list=clean_init_list,
                    model_type='Q').to(device)
                    
        GCN_Re = GCN(nfeat=features.shape[1],
                    nhid=args.hidden_num,
                    nclass=num_classes,
                    dropout=args.dropout,
                    noise_labels=noise_labels,
                    idx_train=idx_train,
                    idx_clean=idx_clean,
                    clean_list=clean_init_list,
                    model_type='Re').to(device)

    elif args.dataset in ['chameleon', 'film', 'squirrel']:
        num_layers = 2
        num_mlp_layers = 1
        GCN_Pi = H2GCN(features.shape[1], args.hidden_num, num_classes, dataset.graph['edge_index'],
                     dataset.graph['num_nodes'], 
                     num_classes, noise_labels, idx_train, idx_clean, clean_init_list, model_type='P',
                     num_layers=num_layers, dropout=args.dropout,
                     num_mlp_layers=num_mlp_layers).to(device)

        GCN_Qi = H2GCN(features.shape[1], args.hidden_num, num_classes, dataset.graph['edge_index'],
                        dataset.graph['num_nodes'], 
                        num_classes, noise_labels, idx_train, idx_clean, clean_init_list, model_type='Q',
                        num_layers=num_layers, dropout=args.dropout,
                        num_mlp_layers=num_mlp_layers).to(device)

        GCN_Re = H2GCN(features.shape[1], args.hidden_num, num_classes, dataset.graph['edge_index'],
                        dataset.graph['num_nodes'], 
                        num_classes, noise_labels, idx_train, idx_clean, clean_init_list, model_type='Re',
                        num_layers=num_layers, dropout=args.dropout,
                        num_mlp_layers=num_mlp_layers).to(device)
        GCN_Pi.reset_parameters()
        GCN_Qi.reset_parameters()
        GCN_Re.reset_parameters()

    
    idx_val_test = np.concatenate((idx_val, idx_test), axis=0) 
    optimizer = torch.optim.Adam(itertools.chain(GCN_Pi.parameters(), GCN_Qi.parameters(), GCN_Re.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    max_test = 0
    max_val = 0
    best_epoch = 0

    for epoch in range(1, args.epochs+1):
        GCN_Pi.train()
        GCN_Qi.train()
        GCN_Re.train()
        optimizer.zero_grad()

        if args.dataset in ['citeseer', 'cora', 'pubmed']:
            y1 = GCN_Pi(features, adj_normal, None, noise_init_list) 
            y2 = GCN_Qi(features, adj_normal)
            Ly = GCN_Re(features, adj_normal, y2, None, device)
        elif args.dataset in ['chameleon', 'film', 'squirrel']:
            y1 = GCN_Pi(dataset, None, noise_init_list) 
            y2 = GCN_Qi(dataset)
            Ly = GCN_Re(dataset, y2, None, device)

        _y = F.softmax(Ly)
        y_z = torch.cosine_similarity(_y[idx_train], _onehot_z[idx_train])
        y_z = nor(y_z)
        sample_weight = y_z.uniform_(0, 1).to(device)

         
        clean_list = copy.deepcopy(clean_init_list)
        for i in range(len(sample_weight)):
            if float(sample_weight[i]) > args.threshold:
                clean_list[noise_labels[i]].append(i)

        loss = criterion(Ly[idx_train,:], noise_labels[idx_train].to(device).long())
        loss = args.eta * (loss * sample_weight).mean()
        loss = loss + args.beta * F.nll_loss(F.log_softmax(y1[idx_clean], dim=1), noise_labels[idx_clean].to(device).long())
        loss = loss + F.nll_loss(F.log_softmax(Ly[idx_clean], dim=1), noise_labels[idx_clean].to(device).long()) 
        loss = loss + kl_categorical(y2[idx_val_test], y1[idx_val_test])

        loss.backward()
        optimizer.step()
        
        GCN_Qi.eval() 
        GCN_Re.eval() 
        if args.dataset in ['citeseer', 'cora', 'pubmed']:
            y2 = GCN_Qi(features, adj_normal)
            pred_y_Re = torch.argmax(F.softmax(GCN_Re(features, adj_normal, y2, None, device), dim=1), dim=1)
        elif args.dataset in ['chameleon', 'film', 'squirrel']:
            y2 = GCN_Qi(dataset)
            pred_y_Re = torch.argmax(F.softmax(GCN_Re(dataset, y2, None, device), dim=1), dim=1)
        acc_test = torch.sum(pred_y_Re[idx_test] == label_true.to(device)[idx_test]).float() / label_true[idx_test].shape[0]
        acc_val = torch.sum(pred_y_Re[idx_val] == label_true.to(device)[idx_val]).float() / label_true[idx_val].shape[0]

        GCN_Qi.clean_list = clean_list
        GCN_Re.clean_list = clean_list
        idx_add = []
        for i in range(num_classes):
            idx_add = np.concatenate((idx_add, clean_list[i]),  axis=0)
        GCN_Qi.idx_clean = idx_add
        GCN_Re.idx_clean = idx_add

        if acc_val.detach().clone().cpu().numpy() > max_val:
            max_val = acc_val.detach().clone().cpu().numpy()
            max_test = acc_test.detach().clone().cpu().numpy()
            best_epoch = epoch

    return max_test


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--seed', default=0)
    args.add_argument('--dataset', default='chameleon',help='The dataset:cora,citeseer,pubmed,chameleon,film,squirrel')
    args.add_argument('--learning_rate', type=float, default=1e-4)
    args.add_argument('--epochs', type=int, default=500)
    args.add_argument('--hidden_num', type=int, default=256,help='The number of neurons in hidden layer of GNNs')
    args.add_argument('--dropout', type=float, default=0.4)
    args.add_argument('--weight_decay', type=float, default=5e-5,help='The weight decay of GNNs')
    args.add_argument('--ptb',type=float,default=0.8,help='The ratio of label noise')
    args.add_argument('--noise_type',type=str,default='uniform',help='The type of label noise:uniform,flip')
    args.add_argument('--clean_num',type=int,default=25,help='The number of nodes in initial clean sets')
    args.add_argument('--threshold',type=float,default=0.7)
    args.add_argument('--eta',type=float,default=0.1)
    args.add_argument('--beta',type=float,default=5)
    args = args.parse_args()

    print(args)
    result = []
    for i in range(5):
        args.seed = int(i)
        _result = main(args)
        result.append(_result)
        print(_result)
    print(np.mean(result), np.std(result))
