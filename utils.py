import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import pickle as pkl
import networkx as nx
import random


def nor(p):
    d_min = p.min()
    if p.min() < 0:
        p += torch.abs(d_min)
        d_min = p.min()
    d_max = p.max()
    dst = d_max - d_min
    norm_data = (p - d_min).true_divide(dst)
    return norm_data

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, tau):
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))    

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
         mean: bool = True):
    h1 = z1
    h2 = z2
    l1 = semi_loss(h1, h2, 0.5)
    l2 = semi_loss(h2, h1, 0.5)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret    