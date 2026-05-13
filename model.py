import os
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LAYERS = 47

class ShowerData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'layer_edge_index':
            return self.layer_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GraphFolderDataset(Dataset):
    def __init__(self, root_dir, stats=None):
        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pt") and os.path.isfile(os.path.join(root_dir, f))
        ])
        self.stats = stats

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # -------------------------------
        # SAFE LOADING
        # -------------------------------
        for _ in range(len(self.files)):
            path = self.files[idx]
            try:
                g_old = torch.load(path, weights_only=False)
                break
            except Exception:
                print(f"[WARNING] Skipping bad file: {path}")
                idx = (idx + 1) % len(self.files)
        else:
            raise RuntimeError("All files are corrupted!")

        g = ShowerData(x=g_old.x.clone(),
                       edge_index=g_old.edge_index,
                       y=g_old.y.clone())

        for key in g_old.keys():
            if key not in ['x', 'edge_index', 'y']:
                setattr(g, key, getattr(g_old, key))

        # -------------------------------
        # PREPROCESSING
        # -------------------------------
        g.x[:, 3] = torch.log(g.x[:, 3] + 1e-6)
        g.y[0] = torch.log(g.y[0] + 1e-6)

        # 🔥 prevent overflow
        node_energy = torch.exp(torch.clamp(g.x[:, 3], max=10)) - 1e-6

        E_sum = node_energy.sum()
        gen_energy = torch.exp(torch.clamp(g.y[0], max=10)) - 1e-6

        g.E_sum = torch.log(E_sum + 1e-6)
        g.E_ratio = E_sum / (gen_energy + 1e-6)

        z = g.x[:, 2]
        layer_idx = torch.round(z).long().clamp(0, MAX_LAYERS - 1)

        x0 = g.x[:, 0]
        y0 = g.x[:, 1]

        sum_xe = scatter_add(x0 * node_energy, layer_idx, dim=0, dim_size=MAX_LAYERS)
        sum_ye = scatter_add(y0 * node_energy, layer_idx, dim=0, dim_size=MAX_LAYERS)
        sum_e = scatter_add(node_energy, layer_idx, dim=0, dim_size=MAX_LAYERS)

        # -------------------------------
        # LAYER FEATURES (SAFE)
        # -------------------------------
        layer_loge = torch.full_like(sum_e, torch.log(torch.tensor(1e-6)))
        nonzero = sum_e > 0
        layer_loge[nonzero] = torch.log(sum_e[nonzero] + 1e-6)

        x_center = torch.zeros(MAX_LAYERS, device=x0.device)
        y_center = torch.zeros(MAX_LAYERS, device=y0.device)

        x_center[nonzero] = sum_xe[nonzero] / (sum_e[nonzero] + 1e-6)
        y_center[nonzero] = sum_ye[nonzero] / (sum_e[nonzero] + 1e-6)

        x_width = torch.zeros(MAX_LAYERS, device=x0.device)
        y_width = torch.zeros(MAX_LAYERS, device=y0.device)

        for l in range(MAX_LAYERS):
            mask = layer_idx == l
            if mask.sum() == 0:
                continue

            e = node_energy[mask]
            denom = e.sum() + 1e-6

            xc = x_center[l]
            yc = y_center[l]

            x_width[l] = torch.sqrt(((x0[mask] - xc)**2 * e).sum() / denom)
            y_width[l] = torch.sqrt(((y0[mask] - yc)**2 * e).sum() / denom)

        hits_per_layer = scatter_add(
            torch.ones_like(layer_idx, dtype=torch.float),
            layer_idx,
            dim=0,
            dim_size=MAX_LAYERS
        )

        sparsity = 1.0 - hits_per_layer / 23000.0

        layer_features = torch.stack(
            [layer_loge, x_center, y_center, x_width, y_width, sparsity],
            dim=1
        )

        # 🔥 HARD CLEAN (very important)
        layer_features = torch.nan_to_num(
            layer_features,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0
        )

        g.layer_x = layer_features
        g.layer_edge_index = build_layer_chain_graph(MAX_LAYERS)
        g.layer_idx = layer_idx

        # -------------------------------
        # NORMALIZATION (SAFE)
        # -------------------------------
        if self.stats is not None:
            (x_mean, x_std, y0_mean, y0_std,
             E_sum_mean, E_sum_std,
             E_ratio_mean, E_ratio_std,
             layer_mean, layer_std) = self.stats

            x_std = torch.clamp(x_std, min=1e-6)
            layer_std = torch.clamp(layer_std, min=1e-3)

            g.x = (g.x - x_mean) / x_std
            g.y[0] = (g.y[0] - y0_mean) / y0_std
            g.E_sum = (g.E_sum - E_sum_mean) / E_sum_std
            g.E_ratio = (g.E_ratio - E_ratio_mean) / E_ratio_std

            g.layer_x = (g.layer_x - layer_mean) / layer_std
            g.layer_x = torch.clamp(g.layer_x, -5, 5)

        # 🔥 FINAL SAFETY
        g.layer_x = torch.nan_to_num(g.layer_x, nan=0.0)

        return g


# -------------------------------
# SAFE STATS COMPUTATION
# -------------------------------
def compute_stats(dataset):
    xs, y0s, E_sums, E_ratios, layer_feats = [], [], [], [], []

    for i, g in enumerate(dataset):

        if torch.isnan(g.layer_x).any():
            print(f"[WARNING] Skipping NaN graph {i}")
            continue

        xs.append(g.x)
        y0s.append(g.y[0])
        E_sums.append(g.E_sum)
        E_ratios.append(g.E_ratio)
        layer_feats.append(g.layer_x)

    x = torch.cat(xs)
    x = torch.nan_to_num(x)

    x_mean = x.mean(0)
    x_std = x.std(0).clamp_min(1e-6)

    y0 = torch.stack(y0s)
    y0 = torch.nan_to_num(y0)

    y0_mean = y0.mean()
    y0_std = y0.std().clamp_min(1e-6)

    E_sum = torch.stack(E_sums)
    E_sum = torch.nan_to_num(E_sum)

    E_sum_mean = E_sum.mean()
    E_sum_std = E_sum.std().clamp_min(1e-6)

    E_ratio = torch.stack(E_ratios)
    E_ratio = torch.nan_to_num(E_ratio)

    E_ratio_mean = E_ratio.mean()
    E_ratio_std = E_ratio.std().clamp_min(1e-6)

    layer = torch.cat(layer_feats)
    layer = torch.nan_to_num(layer)

    layer_mean = layer.mean(0)
    layer_std = layer.std(0).clamp_min(1e-3)

    return (x_mean, x_std, y0_mean, y0_std,
            E_sum_mean, E_sum_std,
            E_ratio_mean, E_ratio_std,
            layer_mean, layer_std)


def build_layer_chain_graph(num_layers):
    edges = []
    for i in range(num_layers - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def preprocess(train_paths):
    temp_train = ConcatDataset([GraphFolderDataset(p) for p in train_paths])
    stats = compute_stats(temp_train)

    train_dataset = ConcatDataset([
        GraphFolderDataset(p, stats=stats) for p in train_paths
    ])

    return train_dataset, stats

train_paths = [
    "/eos/user/a/abkumar/train_graphs",
    "/eos/user/a/abkumar/train_graphs_2",
    "/eos/user/a/abkumar/train_graphs_3",
    "/eos/user/a/abkumar/train_graphs_4",
    "/eos/user/a/abkumar/train_graphs_5",
    "/eos/user/a/abkumar/train_graphs_6",
    "/eos/user/a/abkumar/train_graphs_7",
    "/eos/user/a/abkumar/train_graphs_8",
    "/eos/user/a/abkumar/train_graphs_9",
    "/eos/user/a/abkumar/train_graphs_10",
    "/eos/user/a/abkumar/train_graphs_11",
    "/eos/user/a/abkumar/train_graphs_12",
    "/eos/user/a/abkumar/train_graphs_13",
    "/eos/user/a/abkumar/train_graphs_14",
    "/eos/user/a/abkumar/train_graphs_15",
    "/eos/user/a/abkumar/train_graphs_16",
    "/eos/user/a/abkumar/train_graphs_17",
    "/eos/user/a/abkumar/train_graphs_18",
    "/eos/user/a/abkumar/train_graphs_19",
    "/eos/user/a/abkumar/train_graphs_20",
    "/eos/user/a/abkumar/train_graphs_21",
    "/eos/user/a/abkumar/train_graphs_22",
    "/eos/user/a/abkumar/train_graphs_23",
    "/eos/user/a/abkumar/train_graphs_24",
    "/eos/user/a/abkumar/train_graphs_25",
    "/eos/user/a/abkumar/train_graphs_26",
    "/eos/user/a/abkumar/train_graphs_27",
    "/eos/user/a/abkumar/train_graphs_28",
    "/eos/user/a/abkumar/train_graphs_29",
    "/eos/user/a/abkumar/train_graphs_30",
    "/eos/user/a/abkumar/train_graphs_31",
    "/eos/user/a/abkumar/train_graphs_32",
    "/eos/user/a/abkumar/train_graphs_33",
    "/eos/user/a/abkumar/train_graphs_34",
]

train_dataset, stats = preprocess(
    train_paths
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
#val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2)
#test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2)

x_m, x_s, y0_m, y0_s, E_tot_mean, E_tot_std, E_ratio_mean, E_ratio_std, layer_mean, layer_std = stats

class TotalEnergyFlow(nn.Module):
    def __init__(self, y_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, r, y):
        mu_logstd = self.net(y)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = 0.5 * torch.tanh(log_std)
        std = torch.exp(log_std).clamp(min=1e-3, max=10.0)

        z = (r - mu) / std
        log_det = -log_std.squeeze(-1)
        LOG_2PI = np.log(2 * np.pi)

        log_pz = 0.5 * z.pow(2).squeeze(-1) + 0.5 * LOG_2PI
        return log_pz - log_det

    @torch.no_grad()
    def sample(self, y):
        mu_logstd = self.net(y)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = 0.5 * torch.tanh(log_std)
        std = torch.exp(log_std)
        z = torch.randn_like(mu)
        return mu + std * z

class GNNConditioner(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.conv1 = GATConv(in_dim, hidden_dim, heads= 4 , concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim,  heads= 4 , concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim,  heads= 4 , concat=False)
        #self.conv1 = GCNConv(in_dim, hidden_dim)
        #self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        return self.out(h)

class ConditionalCoupling(nn.Module):
    def __init__(self, feat_dim, cond_dim, hidden_dim, mask):
        super().__init__()

        self.mask = mask

        self.conditioner = GNNConditioner(in_dim=feat_dim + cond_dim, hidden_dim=hidden_dim, out_dim=2 * feat_dim)

    def forward(self, x, edge_index, cond):

        x_masked = x * self.mask.unsqueeze(-1)

        h_input = torch.cat([x_masked, cond], dim=-1)

        s_t = self.conditioner(h_input, edge_index)
        s, t = s_t.chunk(2, dim=-1)

        s = torch.tanh(s)

        x_out = x_masked + (1 - self.mask.unsqueeze(-1)) * (x * torch.exp(s) + t)
        log_det = ((1 - self.mask.unsqueeze(-1)) * s).sum(dim=-1)

        return x_out, log_det

    def inverse(self, z, edge_index, cond):

        z_masked = z * self.mask.unsqueeze(-1)

        h_input = torch.cat([z_masked, cond], dim=-1)

        s_t = self.conditioner(h_input, edge_index)
        s, t = s_t.chunk(2, dim=-1)

        s = torch.tanh(s)

        x = z_masked + (1 - self.mask.unsqueeze(-1)) * ((z - t) * torch.exp(-s))

        return x

class LayerGraphFlow(nn.Module):
    def __init__(self, feat_dim=6, cond_dim=4, hidden_dim=64, num_layers=4):
        super().__init__()

        self.feat_dim = feat_dim

        self.couplings = nn.ModuleList()

        for i in range(num_layers):
            mask = None
            self.couplings.append(ConditionalCoupling(feat_dim, cond_dim, hidden_dim, mask=None))

    def create_mask(self, batch, parity, device):
        mask = torch.zeros(batch.size(0), dtype=torch.float, device=device)
        for g in batch.unique():
            idx = (batch == g).nonzero(as_tuple=True)[0]
            mask[idx[parity::2]] = 1.0
        return mask

    def forward(self, x, edge_index, batch, y, e_tot):

        cond = torch.cat([y, e_tot], dim=-1)
        cond = cond[batch]

        log_det_total = 0
        z = x

        for i, coupling in enumerate(self.couplings):

            mask = self.create_mask(batch, i % 2, z.device)
            coupling.mask = mask

            z, log_det = coupling(z, edge_index, cond)
            log_det_total += log_det

        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi))
        log_pz = log_pz.sum(dim=-1)

        log_prob = log_pz + log_det_total
        log_prob = global_add_pool(log_prob.unsqueeze(-1), batch).squeeze(-1)

        nll = -log_prob

        return nll

    @torch.no_grad()
    def sample(self, num_nodes, edge_index, batch, y, e_tot):

        cond = torch.cat([y, e_tot], dim=-1)
        cond = cond[batch]

        z = torch.randn(num_nodes, self.feat_dim, device=edge_index.device)

        x = z

        for i, coupling in reversed(list(enumerate(self.couplings))):

            mask = self.create_mask(batch, i % 2, x.device)
            coupling.mask = mask

            x = coupling.inverse(x, edge_index, cond)

        return x

class EdgeModel(nn.Module):
    def __init__(self, z_dim, cond_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * z_dim + cond_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, src, dst, edge_attr, u, batch):
        
        cond_node = u[batch]
        x = torch.cat([src, dst, cond_node], dim=-1)

        return self.mlp(x)

class NodeModel(nn.Module):
    def __init__(self, z_dim, cond_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(z_dim + hidden_dim + cond_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, z_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):

        row, col = edge_index

        msg = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce='mean')
        cond_node = u[batch]

        out = torch.cat([x, msg, cond_node], dim=-1)

        return self.mlp(out)

class LayerEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden=64):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads= 4 , concat=False)
        self.conv2 = GATConv(hidden, hidden,  heads= 4 , concat=False)
        #self.conv1 = GCNConv(in_dim, hidden)
        #self.conv2 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return global_mean_pool(h, batch)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t):
        
        half_dim = self.emb_dim // 2
        device = t.device

        freq = torch.exp(torch.arange(half_dim, device=device) *(-(math.log(10000.0) / (half_dim - 1))))

        args = t * freq
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return emb

class NodeGraphFlow(nn.Module):
    def __init__(self, z_dim, y_dim, hidden_dim, layer_hidden=64, time_dim=32, num_layers=4):
        super().__init__()
        self.layer_encoder = LayerEncoder(in_dim=6, hidden=layer_hidden)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.cond_dim = y_dim + 1 + layer_hidden + time_dim
        self.layers = nn.ModuleList([MetaLayer(EdgeModel(z_dim, self.cond_dim, hidden_dim), NodeModel(z_dim, self.cond_dim, hidden_dim), None) for _ in range(num_layers)])

    def forward(self, z, edge_index, batch, y, e_tot, layer_x, layer_edge_index, layer_batch, t):
        
        layer_embedding = self.layer_encoder(layer_x, layer_edge_index, layer_batch)
        
        cond_graph = torch.cat([y, e_tot, layer_embedding], dim=-1)
        
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t)

        u = torch.cat([cond_graph, t_emb], dim=-1)

        edge_attr = None
        
        #z_input = z
        for layer in self.layers:
            #z_res = z
            z, edge_attr, _ = layer(z, edge_index, edge_attr, u, batch)
            #z = z + z_res

        return z

class GraphEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(x_dim+y_dim, hidden_dim, heads= 4 , concat=False)
        self.conv3 = GATConv(hidden_dim, z_dim,  heads= 4 , concat=False)

    def forward(self, x, edge_index, y, batch):
        y_node = y[batch]
        h = torch.cat([x, y_node], dim=-1)
        h = F.relu(self.conv1(h, edge_index))
        z = self.conv3(h, edge_index)
        return z


class GraphDecoder(nn.Module):
    def __init__(self, z_dim, y_dim, x_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(z_dim+y_dim, hidden_dim,  heads= 4 , concat=False)
        self.out = nn.Linear(hidden_dim, x_dim)

    def forward(self, z, edge_index, y, batch):
        y_node = y[batch]
        h = torch.cat([z, y_node], dim=-1)
        h = F.relu(self.conv1(h, edge_index))
        return self.out(h)

#z_dim = 28
x_dim = 4       
y_dim = 3
hidden_dim = 128

#encoder = GraphEncoder(x_dim, y_dim, z_dim, hidden_dim).to(device)
#decoder = GraphDecoder(z_dim, y_dim, x_dim, hidden_dim).to(device)
#opt_encdec = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

E_flow = TotalEnergyFlow(y_dim).to(device)
layer_flow = LayerGraphFlow(feat_dim=6, cond_dim=y_dim+1, hidden_dim= 64, num_layers=4).to(device)
node_flow = NodeGraphFlow(x_dim, y_dim, hidden_dim, layer_hidden=64, time_dim=32, num_layers=4).to(device)
opt_ef = torch.optim.Adam(E_flow.parameters(), lr=1e-4)
opt_lf = torch.optim.Adam(layer_flow.parameters(), lr=1e-4)
opt_nf = torch.optim.Adam(node_flow.parameters(), lr=1e-4)
#scheduler = MultiStepLR(opt_nf, milestones=[20, 30], gamma=0.1)
scheduler = StepLR(opt_nf, step_size=20, gamma=0.1)
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")
'''
for epoch in range(10):
    encoder.train()
    decoder.train()
    total = 0
    for batch in train_loader:
        batch = batch.to(device)
        y = batch.y.view(batch.num_graphs, y_dim)

        z = encoder(batch.x, batch.edge_index, y, batch.batch)
        x_hat = decoder(z, batch.edge_index, y, batch.batch)

        loss = F.mse_loss(x_hat, batch.x)

        opt_encdec.zero_grad()
        loss.backward()
        opt_encdec.step()
        total += loss.item()

    print(f"[AE] Epoch {epoch} loss {total/len(train_loader):.4f}")

encoder.eval()
decoder.eval()
for p in encoder.parameters():
    p.requires_grad_(False)
for p in decoder.parameters():
    p.requires_grad_(False)

def test(E_flow, layer_flow, node_flow, loader, opt_ef, opt_lf, opt_nf, device, alpha):
    E_flow.eval()
    layer_flow.eval()
    node_flow.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            data = batch.to(device)
            x = data.x
            y = data.y.view(data.num_graphs, y_dim)
            edge_index = data.edge_index
            batch = data.batch
            e_tot = data.E_sum.view(data.num_graphs, 1)
            e_r = data.E_ratio.view(data.num_graphs, 1)
            layer_x = data.layer_x
            layer_edge_index = data.layer_edge_index
            layer_idx = data.layer_idx
            num_graphs = data.num_graphs
            nodes_per_graph_l = MAX_LAYERS

            layer_batch = torch.arange(num_graphs, device=device).repeat_interleave(nodes_per_graph_l)

        
            L_r = E_flow(e_r, y).mean()
            
            E_r_gen = E_flow.sample(y)
            E_r_gen_un = E_r_gen*E_ratio_std + E_ratio_mean
            logE_inc = y[:,0].unsqueeze(-1) * y0_s + y0_m
            E_inc = torch.exp(logE_inc)
            E_tot_gen_un = torch.log(E_r_gen_un*E_inc)
            #E_tot_gen_un = E_r_gen_un*(y[:,0].unsqueeze(-1)*y0_s + y0_m)
            e_tot_gen = (E_tot_gen_un - E_tot_mean)/E_tot_std
            
            L_layer = layer_flow(layer_x, layer_edge_index, layer_batch, y, e_tot).mean()
            
            layer_gen = layer_flow.sample(layer_x.size(0), layer_edge_index, layer_batch, y, e_tot)
            
            E_cond = (1 - alpha) * e_tot + alpha * e_tot_gen.detach()
            l_cond = (1 - alpha) * layer_x + alpha * layer_gen.detach()
        
            #with torch.no_grad():
            #    z0 = encoder(x, edge_index, y, batch)
            z0 = x
            #z0 = (z0 - z_mean) / z_std
            z1 = torch.randn_like(z0)
            t = torch.rand(y.size(0), 1, device=device)
            t_safe = t.clamp(max=1.0 - 0.001)
            t_node = t_safe[batch]
            z_t = (1.0 - t_node) * z1 + t_node * z0
            v_target = (z0 - z_t)/ (1.0 - t_node + 1e-6)
            v_pred = node_flow(z_t, edge_index, batch, y, E_cond, l_cond, layer_edge_index, layer_batch, t_safe)
        
            L_node = ((v_pred - v_target)**2).mean()
            total_loss+= (L_node.item()+L_layer.item()+L_r.item())
    return total_loss/len(loader)            
'''
max_epoch=20
t_loss = []
v_loss = []
tes_loss = []

for epochs in range(1, max_epoch+1):
    if epochs < 6:
        alpha = 0
    elif epochs > 5 and epochs < 16:
        alpha = (epochs - 5)/10
    elif epochs > 15:
        alpha = 1
    E_flow.train()
    layer_flow.train()
    node_flow.train()
    freeze_conditioners = (epochs > 15)

    if freeze_conditioners:
        for p in E_flow.parameters():
            p.requires_grad_(False)
        for p in layer_flow.parameters():
            p.requires_grad_(False)
    else:
        for p in E_flow.parameters():
            p.requires_grad_(True)
        for p in layer_flow.parameters():
            p.requires_grad_(True)
    train_loss = 0
    for batch in train_loader:
        data = batch.to(device)
        x = data.x
        y = data.y.view(data.num_graphs, y_dim)
        edge_index = data.edge_index
        batch = data.batch
        e_tot = data.E_sum.view(data.num_graphs, 1)
        e_r = data.E_ratio.view(data.num_graphs, 1)
        layer_x = data.layer_x
        layer_edge_index = data.layer_edge_index
        layer_idx = data.layer_idx
        num_graphs = data.num_graphs
        nodes_per_graph_l = MAX_LAYERS

        layer_batch = torch.arange(num_graphs, device=device).repeat_interleave(nodes_per_graph_l)

                
        L_r = E_flow(e_r, y).mean()
        if not freeze_conditioners:
            opt_ef.zero_grad()
            L_r.backward()
            opt_ef.step()
        
        E_flow.eval()
        with torch.no_grad():
            E_r_gen = E_flow.sample(y)
            E_r_gen_un = E_r_gen*E_ratio_std + E_ratio_mean
            logE_inc = y[:,0].unsqueeze(-1) * y0_s + y0_m
            E_inc = torch.exp(logE_inc)
            safe_val = (E_r_gen_un * E_inc).clamp_min(1e-8)
            E_tot_gen_un = torch.log(safe_val)
            #E_tot_gen_un = E_r_gen_un*(y[:,0].unsqueeze(-1)*y0_s + y0_m)
            e_tot_gen = (E_tot_gen_un - E_tot_mean)/E_tot_std
        E_flow.train()
        
        L_layer = layer_flow(layer_x, layer_edge_index, layer_batch, y, e_tot).mean()
        if not freeze_conditioners:
            opt_lf.zero_grad()
            L_layer.backward()
            opt_lf.step()
        
        layer_flow.eval()
        with torch.no_grad():
            layer_gen = layer_flow.sample(layer_x.size(0), layer_edge_index, layer_batch, y, e_tot)
        layer_flow.train()
        
        E_cond = (1 - alpha) * e_tot + alpha * e_tot_gen.detach()
        l_cond = (1 - alpha) * layer_x + alpha * layer_gen.detach()
        
        #with torch.no_grad():
        #    z0 = encoder(x, edge_index, y, batch)
        z0 = x
        #z0 = (z0 - z_mean) / z_std
        z1 = torch.randn_like(z0)
        t = torch.rand(y.size(0), 1, device=device)
        t_safe = t.clamp(max=1.0 - 0.001)
        t_node = t_safe[batch]
        z_t = (1.0 - t_node) * z1 + t_node * z0
        v_target = (z0 - z_t)/ (1.0 - t_node + 1e-3)
        v_pred = node_flow(z_t, edge_index, batch, y, E_cond, l_cond, layer_edge_index, layer_batch, t_safe)
        
        L_node = ((v_pred - v_target)**2).mean()
        opt_nf.zero_grad()
        L_node.backward()
        opt_nf.step()
        train_loss+= (L_node.item()+L_layer.item()+L_r.item())
    train_loss/=len(train_loader)
    t_loss.append(train_loss)
    #val_loss = test(E_flow, layer_flow, node_flow, val_loader, opt_ef, opt_lf, opt_nf, device, alpha)
    #v_loss.append(val_loss)
    #test_loss = test(E_flow, layer_flow, node_flow, test_loader, opt_ef, opt_lf, opt_nf, device, alpha)
    #tes_loss.append(test_loss)
    #print(f"Epoch {epochs:03d} TrainLoss {train_loss:.4f} val_loss {val_loss:.4f} Test_loss {test_loss:.4f}")
    print(f"Epoch {epochs:03d} TrainLoss {train_loss:.4f}")
    #scheduler.step()

plt.plot(t_loss, label="Train")
#plt.plot(v_loss, label="Validation")
#plt.plot(tes_loss, label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.show()
