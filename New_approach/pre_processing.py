import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
import numpy as np
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
import random
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import GATConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import global_add_pool
import math
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected

EPS = 1e-8

class ShowerData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'long_edge_index':
            return self.long_x.size(0)
        if key == 'trans_edge_index':
            return self.trans_x.size(0)
        if key == 'trans_batch':
            return int(self.n_active)
        #if key in ['trans_layer_ids', 'trans_E_l', 'trans_nhits']:
        #    return 0
        return 0
        #return super().__inc__(key, value, *args, **kwargs)
 
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['long_edge_index', 'trans_edge_index']:
            return 1
        return 0   

class ShowerDataset(Dataset):
    def __init__(self, root_dirs, stats=None):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
 
        self.files = []
        for root in root_dirs:
            for f in sorted(os.listdir(root)):
                if f.endswith('.pt') and os.path.isfile(os.path.join(root, f)):
                    self.files.append(os.path.join(root, f))
 
        self.stats = stats
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        for _ in range(len(self.files)):
            path = self.files[idx]
            try:
                data = torch.load(path, weights_only=False)
                shower = ShowerData()
                for key, value in data:
                    setattr(shower, key, value)
                data = shower
                break
            except Exception:
                print(f"[WARNING] Skipping bad file: {path}")
                idx = (idx + 1) % len(self.files)
        else:
            raise RuntimeError("All files are corrupted!")
 
        if self.stats is not None:
            data = self._transform(data)
 
        return data
 
    def _transform(self, data):
        s = self.stats
 
        data.E_inc   = (torch.log(data.E_inc   + EPS) - s['E_inc_mean'])   / s['E_inc_std']
        data.E_total = (torch.log(data.E_total + EPS) - s['E_total_mean']) / s['E_total_std']

        lx = data.long_x.clone()
        lx[:, 0] = (torch.log(lx[:, 0] + EPS) - s['long_E_mean']) / s['long_E_std']
        lx[:, 1] = (torch.log(lx[:, 1] + 1)   - s['long_n_mean']) / s['long_n_std']
        data.long_x = lx

        data.trans_E_l   = (torch.log(data.trans_E_l            + EPS) - s['tE_mean']) / s['tE_std']
        data.trans_nhits = (torch.log(data.trans_nhits.float()  + 1)   - s['tn_mean']) / s['tn_std']

        tx = data.trans_x.clone()
        tx[:, 0] = (torch.log(tx[:, 0] + EPS) - s['te_mean']) / s['te_std']
        tx[:, 1] = (tx[:, 1]                   - s['tx_mean']) / s['tx_std']
        tx[:, 2] = (tx[:, 2]                   - s['ty_mean']) / s['ty_std']
        tx[:, 3] = (torch.log(tx[:, 3] + EPS) - s['tr_mean']) / s['tr_std']
        tx[:, 4] =  tx[:, 4] / math.pi
        data.trans_x = tx
 
        return data

def compute_stats(train_dataset):
    log_E_inc, log_E_total         = [], []
    log_E_l_active, log_n_all      = [], []
    log_tE_l, log_tn               = [], []
    log_te, tx_all, ty_all, log_tr = [], [], [], []
 
    for i, data in enumerate(train_dataset):
        if i % 5000 == 0:
            print(f"  {i} / {len(train_dataset)}")
 
        log_E_inc.append(torch.log(data.E_inc   + EPS))
        log_E_total.append(torch.log(data.E_total + EPS))
        E_l_vals  = data.long_x[:, 0]
        nhit_vals = data.long_x[:, 1]
        active    = data.active
 
        log_E_l_active.append(torch.log(E_l_vals[active] + EPS))
        log_n_all.append(      torch.log(nhit_vals        + 1))

        log_tE_l.append(torch.log(data.trans_E_l           + EPS))
        log_tn.append(  torch.log(data.trans_nhits.float() + 1))

        tx = data.trans_x
        log_te.append(torch.log(tx[:, 0] + EPS))
        tx_all.append(tx[:, 1])
        ty_all.append(tx[:, 2])
        log_tr.append(torch.log(tx[:, 3] + EPS))
 
    def _ms(lst):
        t = torch.stack(lst) if lst[0].dim() == 0 else torch.cat(lst)
        return float(t.mean()), float(t.std().clamp_min(1e-6))

    stats = {}
    stats['E_inc_mean'],   stats['E_inc_std']   = _ms(log_E_inc)
    stats['E_total_mean'], stats['E_total_std'] = _ms(log_E_total)
    stats['long_E_mean'],  stats['long_E_std']  = _ms(log_E_l_active)
    stats['long_n_mean'],  stats['long_n_std']  = _ms(log_n_all)
    stats['tE_mean'],      stats['tE_std']      = _ms(log_tE_l)
    stats['tn_mean'],      stats['tn_std']      = _ms(log_tn)
    stats['te_mean'],      stats['te_std']      = _ms(log_te)
    stats['tx_mean'],      stats['tx_std']      = _ms(tx_all)
    stats['ty_mean'],      stats['ty_std']      = _ms(ty_all)
    stats['tr_mean'],      stats['tr_std']      = _ms(log_tr)
 
    return stats

def inverse_transform(stats, key, value):
    if key == 'phi':
        return value * math.pi
 
    mean = stats[f'{key}_mean']
    std  = stats[f'{key}_std']
    log_val = value * std + mean          # some are even not log, but just naming the variable
 
    if key in ['E_inc', 'E_total', 'long_E', 'tE', 'te', 'tr']:
        return torch.exp(torch.as_tensor(log_val)) - EPS
    elif key in ['long_n', 'tn']:
        return torch.exp(torch.as_tensor(log_val)) - 1.0
    else:
        return torch.as_tensor(log_val)       #tx, ty
    
train_paths = [
    "/eos/user/a/abkumar/a_freshlook_at_HGCALO/train_graphs_0",
    "/eos/user/a/abkumar/a_freshlook_at_HGCALO/train_graphs_1",
    "/eos/user/a/abkumar/a_freshlook_at_HGCALO/train_graphs_2",
    
]
val_paths  = ["/eos/user/a/abkumar/a_freshlook_at_HGCALO/val_graphs_0"]
test_paths = ["/eos/user/a/abkumar/a_freshlook_at_HGCALO/test_graphs_0"]

raw_train = ShowerDataset(train_paths, stats=None)
stats = compute_stats(raw_train)

train_dataset = ShowerDataset(train_paths, stats=stats)
val_dataset   = ShowerDataset(val_paths,   stats=stats)
test_dataset  = ShowerDataset(test_paths,  stats=stats)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=4, pin_memory=True, follow_batch = ['long_x', 'trans_x'])
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True, follow_batch = ['long_x', 'trans_x'])
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=4, pin_memory=True, follow_batch = ['long_x', 'trans_x'])
