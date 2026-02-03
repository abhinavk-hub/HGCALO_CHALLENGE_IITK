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
from torch_geometric.data import Data
from torch.cuda.amp import autocast, GradScaler
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import GATConv


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphFolderDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".pt") and os.path.isfile(os.path.join(root_dir, f))
        ])
        #self.files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pt")])

    def load_all(self):
        """Load all graphs once into memory."""
        graphs = []
        for fname in self.files:
            path = os.path.join(self.root_dir, fname)
            g = torch.load(path, weights_only=False)      # Data(x, edge_index, edge_attr, y)
            graphs.append(g)
        return graphs


# ---------- Simple in-memory dataset wrapper ----------

class ListDataset(Dataset):
    def __init__(self, graph_list):
        self.graphs = graph_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# ---------- Helper preprocessing functions ----------

def remove_redundancy(mat, drop_indices):
    """Mat is tensor [N, F]. Drop columns in drop_indices."""
    keep = [i for i in range(mat.size(1)) if i not in drop_indices]
    return mat[:, keep]

def remove_constant(y):
    """Keep only y[0] and y[2]."""
    return y[[0, 2]]

def compute_normalization_stats(graph_list, attr_name):
    feats = []
    for g in graph_list:
        v = getattr(g, attr_name)
        if v is not None:
            feats.append(v)
    feats = torch.cat(feats, dim=0)

    mean = feats.mean(dim=0)
    std = feats.std(dim=0)
    std[std < 1e-8] = 1.0
    return mean, std

def apply_norm(graph_list, attr_name, mean, std):
    for g in graph_list:
        v = getattr(g, attr_name)
        setattr(g, attr_name, (v - mean) / std)

def layer_feature_sum(
    feature,        # (N,)
    layer_idx,     # (N,)
    batch_idx      # (N,)
):

    num_graphs = batch_idx.max().item() + 1
    max_layers = 47

    # combine (graph, layer) into a single index
    combined_idx = batch_idx * max_layers + layer_idx

    total_bins = num_graphs * max_layers

    layer_feature_flat = scatter_add(
        feature,
        combined_idx,
        dim=0,
        dim_size=total_bins
    )

    layer_feature = layer_feature_flat.view(num_graphs, max_layers)

    # mask: which (graph, layer) actually exists
    layer_mask = layer_feature > 0

    return layer_feature, layer_mask

@torch.no_grad()
def compute_energy_stats(graph_list, max_layers=47):
    """
    Computes dataset-level statistics for:
    - log total energy per graph
    - log total energy per layer
    """

    log_E_tot_all = []
    log_E_layer_all = []

    for g in graph_list:
        batch = torch.zeros(g.x.size(0), dtype=torch.long)

        # node energy (already log in preprocessing)
        E_node_log = g.x[:, 3]
        E_node = torch.exp(E_node_log) - 1e-6

        # ----- total graph energy -----
        E_tot = E_node.sum()
        log_E_tot_all.append(torch.log(E_tot + 1e-6))

        # ----- total layer energy -----
        L_phys = torch.round(g.x[:, 2]).long().clamp(0, max_layers - 1)

        layer_E, layer_mask = layer_feature_sum(
            feature=E_node,
            layer_idx=L_phys,
            batch_idx=batch
        )

        log_layer_E = torch.log(layer_E[layer_mask] + 1e-6)
        log_E_layer_all.append(log_layer_E)

    log_E_tot_all = torch.stack(log_E_tot_all)
    log_E_layer_all = torch.cat(log_E_layer_all)

    E_tot_mean = log_E_tot_all.mean()
    E_tot_std  = log_E_tot_all.std().clamp_min(1e-6)

    E_layer_mean = log_E_layer_all.mean()
    E_layer_std  = log_E_layer_all.std().clamp_min(1e-6)

    return E_tot_mean, E_tot_std, E_layer_mean, E_layer_std

# ---------- Main preprocessing ----------

def preprocess(train_path, val_path, test_path):
    
    # Load raw graphs (NOT modifying .pt files)
    train_raw = GraphFolderDataset(train_path).load_all()
    val_raw   = GraphFolderDataset(val_path).load_all()
    test_raw  = GraphFolderDataset(test_path).load_all()

    # ---------- STEP 1: Drop redundant columns ----------
    for dataset in [train_raw, val_raw, test_raw]:
        for g in dataset:
            if g.x is not None:
                g.x[:,3]=torch.log(g.x[:, 3] + 1e-6)
            if g.y is not None:
                g.y[0] = torch.log(g.y[0] + 1e-6)
                #g.y = remove_constant(g.y)             # keep y0, y2

    # ---------- STEP 2: Normalize with train stats ----------
    (
        E_tot_mean,
        E_tot_std,
        E_layer_mean,
        E_layer_std
    ) = compute_energy_stats(train_raw)
    x_mean, x_std = compute_normalization_stats(train_raw, 'x')
    #e_mean, e_std = compute_normalization_stats(train_raw, 'edge_attr')

    apply_norm(train_raw, 'x', x_mean, x_std)
    apply_norm(val_raw,   'x', x_mean, x_std)
    apply_norm(test_raw,  'x', x_mean, x_std)

    #apply_norm(train_raw, 'edge_attr', e_mean, e_std)
    #apply_norm(val_raw,   'edge_attr', e_mean, e_std)
    #apply_norm(test_raw,  'edge_attr', e_mean, e_std)

    # ---------- STEP 3: Normalize only y[0] ----------
    y0_train = torch.stack([g.y[0] for g in train_raw])
    y0_mean = y0_train.mean()
    y0_std  = y0_train.std()
    if y0_std < 1e-8:
        y0_std = 1.0

    for dataset in [train_raw, val_raw, test_raw]:
        for g in dataset:
            g.y[0] = (g.y[0] - y0_mean) / y0_std

    # ---------- STEP 4: Return as PyTorch datasets ----------
    
    return (
        ListDataset(train_raw),
        ListDataset(val_raw),
        ListDataset(test_raw), x_mean, x_std, y0_mean, y0_std, E_tot_mean, E_tot_std, E_layer_mean, E_layer_std
    )


# ---------- Run preprocessing ----------
train_dataset, val_dataset, test_dataset, x_m, x_s, y0_m, y0_s, E_tot_mean, E_tot_std, E_layer_mean, E_layer_std = preprocess(
    "/eos/user/a/abkumar/train_graphs",
    "/eos/user/a/abkumar/val_graphs",
    "/eos/user/a/abkumar/test_graphs"
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)


# In[3]:


graphs = [train_dataset[i] for i in range(len(train_dataset))]

def collect_features(dataset):
    all_x = torch.cat([g.x for g in dataset], dim=0)
    all_y = torch.cat([g.y.reshape(-1, 3) for g in dataset], dim=0)  # reshape global label
    return all_x,  all_y

def plot_feature_histograms(x, feature_names=None, bins=100):
    num_features = x.shape[1]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(num_features)]

    plt.figure(figsize=(4 * num_features, 4))

    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        plt.hist(x[:, i].cpu().numpy(), bins=bins)
        plt.title(feature_names[i])
        plt.xlabel("Value")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

x_before, y_before = collect_features(graphs)
feature_names = ["x", "y", "layer", "energy"]
plot_feature_histograms(x_before, feature_names)


# In[4]:


class TotalEnergyFlow(nn.Module):
    def __init__(self, y_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # mean, log_std
        )

    def forward(self, E_tot, y):
        """
        E_tot: [B, 1]
        y:     [B, y_dim]
        """
        mu_logstd = self.net(y)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = 0.5 * torch.tanh(log_std)
        std = torch.exp(log_std)

        z = (E_tot - mu) / std
        log_det = -log_std.squeeze(-1)
        LOG_2PI = np.log(2 * np.pi)

        log_pz = 0.5 * z.pow(2).squeeze(-1) + 0.5 * LOG_2PI
        return log_pz - log_det

    def sample(self, y):
        mu_logstd = self.net(y)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = 0.5 * torch.tanh(log_std)
        std = torch.exp(log_std)
        z = torch.randn_like(mu)
        return mu + std * z

class LayerFeatureFlow(nn.Module):
    def __init__(self, y_dim, hidden_dim=128, max_layers=47):
        super().__init__()

        self.layer_embed = nn.Embedding(max_layers, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(y_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # mu, log_std (shared across layers)
        )

    def forward(self, L_x, y, layer_mask):
        """
        L_x:        [B, L]
        y:          [B, y_dim]
        layer_mask: [B, L] (bool or 0/1)
        """

        B, L = L_x.shape
        device = L_x.device

        layer_idx = torch.arange(L, device=device)
        layer_emb = self.layer_embed(layer_idx)      # [L, H]
        layer_emb = layer_emb.unsqueeze(0).expand(B, L, -1)

        y_exp = y.unsqueeze(1).expand(B, L, -1)

        h = torch.cat([y_exp, layer_emb], dim=-1)    # [B, L, *]
        mu_logstd = self.net(h)

        mu, log_std = mu_logstd[..., 0], mu_logstd[..., 1]
        log_std = 0.5 * torch.tanh(log_std)
        std = torch.exp(log_std)

        z = (L_x - mu) / std

        LOG_2PI = np.log(2 * np.pi)
        log_prob = 0.5 * (z ** 2 + LOG_2PI) - log_std

        log_prob = log_prob * layer_mask

        return log_prob.sum(dim=1).mean()

    @torch.no_grad()
    def sample(self, y, num_layers=47):
        """
        y:          [B, y_dim]
        num_layers: int (max layers in this batch)
        """

        B = y.size(0)
        device = y.device

        layer_idx = torch.arange(num_layers, device=device)
        layer_emb = self.layer_embed(layer_idx).unsqueeze(0).expand(B, -1, -1)

        y_exp = y.unsqueeze(1).expand(B, num_layers, -1)

        h = torch.cat([y_exp, layer_emb], dim=-1)
        mu_logstd = self.net(h)

        mu, log_std = mu_logstd[..., 0], mu_logstd[..., 1]
        log_std = 0.5 * torch.tanh(log_std)
        std = torch.exp(log_std)

        z = torch.randn_like(mu)
        return mu + std * z


class ConditioningGNN(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim):
        super().__init__()

        self.conv1 = GATConv(x_dim+y_dim, hidden_dim, heads=2, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.dropout = nn.Dropout(0.2)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * x_dim)  # scale + shift
        )

    def forward(self, x, edge_index, y, batch):
        """
        x: [N, Dx]
        y: [Dy] or [N, Dy]
        """
        y_node=y[batch]

        h = torch.cat([x, y_node], dim=-1)
        h = F.relu(self.conv1(h, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)

        s_t = self.mlp(h)
        s, t = s_t.chunk(2, dim=-1)
        return s, t

class GraphAffineCoupling(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = ConditioningGNN(x_dim, y_dim, hidden_dim)
        #self.net = ConditioningMLP(x_dim, y_dim, hidden_dim)

    def forward(self, x, edge_index, y, batch):
        x_masked = x * self.mask
        s, t = self.net(x_masked, edge_index, y, batch)
        #s, t = self.net(x_masked, y, batch)

        s = 0.5*torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)

        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)

        return z, log_det

    def inverse(self, z, edge_index, y, batch):
        z_masked = z * self.mask
        #batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        s, t = self.net(z_masked, edge_index, y, batch)

        s = 0.5*torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)

        x = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return x
class GraphNormalizingFlow(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, num_layers):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            mask = self.create_feature_mask(x_dim, (i % 2))
            self.layers.append(
                GraphAffineCoupling(x_dim, y_dim, hidden_dim, mask)
            )

    def create_feature_mask(self, x_dim, flip):
        mask = torch.zeros(x_dim)
        mask[::2] = 1.0
        #mask[:3] = 1.0
        if flip:
            mask = 1.0 - mask
        return mask
    def forward(self, x, edge_index, y, batch):
        log_det_total = torch.zeros(x.size(0), device=x.device)
        #log_det_total = 0
        z = x

        for layer in self.layers:
            z, log_det = layer(z, edge_index, y, batch)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z, edge_index, y, batch):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, edge_index, y, batch)
        return x

#true for all features; although the variable are specific to energy
def layer_feature_sum(
    feature,        # (N,)
    layer_idx,     # (N,)
    batch_idx      # (N,)
):

    num_graphs = batch_idx.max().item() + 1
    max_layers = 47

    # combine (graph, layer) into a single index
    combined_idx = batch_idx * max_layers + layer_idx

    total_bins = num_graphs * max_layers

    layer_feature_flat = scatter_add(
        feature,
        combined_idx,
        dim=0,
        dim_size=total_bins
    )

    layer_feature = layer_feature_flat.view(num_graphs, max_layers)

    # mask: which (graph, layer) actually exists
    layer_mask = layer_feature > 0

    return layer_feature, layer_mask


def test(E_flow,layer_flow, graph_flow, loader, opt_E, opt_L, opt_graph, device):
    E_flow.eval()
    layer_flow.eval()
    graph_flow.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(batch.num_graphs, y_dim)
        batch_idx = batch.batch
        E_node_log = batch.x[:,3]*x_s[3] + x_m[3]
        E_node = torch.exp(E_node_log)-1e-6
        E_tot = torch.zeros(batch.num_graphs, device=device)
        E_tot.index_add_(0, batch_idx, E_node)
        E_log = torch.log(E_tot)
        E_log_n = (E_log - E_tot_mean)/ E_tot_std

        loss_E = 0.1*E_flow(E_log_n.unsqueeze(-1), y).mean()
        L_phys = torch.round(batch.x[:,2]).long()
        L_phys = torch.clamp(L_phys, 0, 46)
        L_e , l_mask = layer_feature_sum(E_node, layer_idx = L_phys, batch_idx= batch.batch)
        L_e_log = torch.log(L_e+ 1e-6)
        L_e_log_n = (L_e_log- E_layer_mean) / E_layer_std
        #L_x = layer_feature_sum(batch.x[:,0], layer_idx = L_phys, batch_idx= batch.batch)
        #L_y = layer_feature_sum(batch.x[:,1], layer_idx = L_phys, batch_idx= batch.batch)
        #L_L = layer_feature_sum(batch.x[:,2], layer_idx = L_phys, batch_idx= batch.batch)
        #L_e = layer_feature_sum(batch.x[:,3], layer_idx = L_phys, batch_idx= batch.batch).to(device)
        #loss_layerx = layer_flow(L_x,y=E_log[batch_idx].unsqueeze(-1))
        #loss_layery = layer_flow(L_y,y=E_log[batch_idx].unsqueeze(-1))
        #loss_layerl = layer_flow(L_L,y=E_log[batch_idx].unsqueeze(-1))
        loss_layere = layer_flow(L_e_log_n,E_log_n.unsqueeze(-1), l_mask)
        #L_loss = (loss_layerx+loss_layery+loss_layerl+loss_layere)
        L_loss = 0.1*loss_layere
        cond_noise = 0.05
        L_e_log_noisy = L_e_log_n + cond_noise * torch.randn_like(L_e_log_n)
        

        z, log_det = graph_flow(batch.x,batch.edge_index,torch.cat([E_log_n[batch_idx].unsqueeze(-1).detach(),L_e_log_noisy[batch_idx].detach()], dim=-1),batch_idx)

        log_pz = 0.5 * (z ** 2).sum(dim=-1)
        loss_graph = (log_pz - log_det).mean()
        beta = min(1.0, epoch / 5.0)
        loss_graph = beta * loss_graph
        '''
        x_phys = batch.x.clone()
        e_log = batch.x[:,3]*x_s[3] + x_m[3]
        x_phys[:,3] = torch.exp(e_log) 
        x_phys[:, 0] = batch.x[:, 0]*x_s[0] + x_m[0]
        x_phys[:, 1] = batch.x[:, 1]*x_s[1] + x_m[1]
        x_phys[:, 2] = batch.x[:, 2]*x_s[2] + x_m[2]
        L_phys = torch.round(x_phys[:,2]).long()
        L_phys = torch.clamp(L_phys, 0, 47)
        x_gen = graph_flow.inverse(z, batch.edge_index, y[batch_idx], batch_idx) # note: energy is fraction, and rest normalized.
        eg_log = x_gen[:,3]*x_s[3] + x_m[3]
        x_gen[:,3] = torch.exp(eg_log) 
        x_gen[:, 0] = x_gen[:, 0]*x_s[0] + x_m[0]
        x_gen[:, 1] = x_gen[:, 1]*x_s[1] + x_m[1]
        x_gen[:, 2] = x_gen[:, 2]*x_s[2] + x_m[2]
        lambda_layer = 1.0  # tune this
        loss_layer1 = layer_feature_mse_loss(energy_real = x_phys[:,3],energy_gen = x_gen[:,3],layer_idx = L_phys,batch_idx = batch.batch)
        loss_layer2 = layer_feature_mse_loss(energy_real = x_phys[:,0],energy_gen = x_gen[:,0],layer_idx = L_phys,batch_idx = batch.batch)
        loss_layer3 = layer_feature_mse_loss(energy_real = x_phys[:,1],energy_gen = x_gen[:,1],layer_idx = L_phys,batch_idx = batch.batch)
        loss_layer4 = layer_feature_mse_loss(energy_real = x_phys[:,2],energy_gen = x_gen[:,2],layer_idx = L_phys,batch_idx = batch.batch)
        loss = loss_graph + lambda_layer * (loss_layer1+loss_layer2+loss_layer3+loss_layer4)
        '''
        total_loss += (loss_graph.item()+loss_E.item()+L_loss.item())
        del batch, y , E_node, E_tot, E_log, loss_E, z, log_det, loss_graph
    return total_loss/len(loader)

x_dim = 4       
y_dim = 3 
lcon_dim = 1
gcon_dim = 47
hidden_dim = 64
num_layers = 8
#num_nodes = 47*2200
E_flow = TotalEnergyFlow(y_dim).to(device)
layer_flow = LayerFeatureFlow(lcon_dim).to(device)
graph_flow = GraphNormalizingFlow(x_dim, gcon_dim+lcon_dim, hidden_dim, num_layers).to(device)
opt_E = torch.optim.Adam(E_flow.parameters(), lr=1e-4)
opt_L = torch.optim.Adam(layer_flow.parameters(), lr=1e-4)
opt_graph = torch.optim.Adam(graph_flow.parameters(), lr=1e-4, weight_decay=1e-4)
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")
epochs = 10
#scaler = GradScaler()
t_loss = []
v_loss = []
tes_loss = []
for epoch in range(1, epochs + 1):
    E_flow.train()
    layer_flow.train()
    graph_flow.train()
    freeze_conditioners = (epoch > 3)

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
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        y = batch.y.view(batch.num_graphs, y_dim)
        batch_idx = batch.batch
        E_node_log = batch.x[:,3]*x_s[3] + x_m[3]
        E_node = torch.exp(E_node_log)-1e-6
        E_tot = torch.zeros(batch.num_graphs, device=device)
        E_tot.index_add_(0, batch_idx, E_node)
        E_log = torch.log(E_tot)
        E_log_n = (E_log - E_tot_mean)/ E_tot_std
        loss_E = 0.1*E_flow(E_log_n.unsqueeze(-1), y).mean()
        if not freeze_conditioners:
            opt_E.zero_grad()
            loss_E.backward()
            opt_E.step()
        L_phys = torch.round(batch.x[:,2]).long()
        L_phys = torch.clamp(L_phys, 0, 46)
        #L_x = layer_feature_sum(batch.x[:,0], layer_idx = L_phys, batch_idx= batch.batch)
        #L_y = layer_feature_sum(batch.x[:,1], layer_idx = L_phys, batch_idx= batch.batch)
        #L_L = layer_feature_sum(batch.x[:,2], layer_idx = L_phys, batch_idx= batch.batch)
        L_e, L_mask = layer_feature_sum(E_node, layer_idx = L_phys, batch_idx= batch.batch)
        L_e_log = torch.log(L_e+ 1e-6)
        L_e_log_n = (L_e_log- E_layer_mean) / E_layer_std
        #loss_layerx = layer_flow(L_x,y=E_log[batch_idx].unsqueeze(-1))
        #loss_layery = layer_flow(L_y,y=E_log[batch_idx].unsqueeze(-1))
        #loss_layerl = layer_flow(L_L,y=E_log[batch_idx].unsqueeze(-1))
        loss_layere = layer_flow(L_e_log_n,E_log_n.unsqueeze(-1), L_mask)
        #L_loss = (loss_layerx+loss_layery+loss_layerl+loss_layere)
        L_loss = 0.1*loss_layere
        if not freeze_conditioners:
            opt_L.zero_grad()
            L_loss.backward()
            opt_L.step()
        cond_noise = 0.05
        L_e_log_noisy = L_e_log_n + cond_noise * torch.randn_like(L_e_log_n)
        #z, log_det = graph_flow(batch.x,batch.edge_index,torch.cat([E_log[batch_idx].unsqueeze(-1)],L_x[batch_idx].unsqueeze(-1),L_y[batch_idx].unsqueeze(-1),L_L[batch_idx].unsqueeze(-1),L_e[batch_idx].unsqueeze(-1), dim=-1),batch_idx)
        z, log_det = graph_flow(batch.x,batch.edge_index,torch.cat([E_log_n[batch_idx].unsqueeze(-1).detach(),L_e_log_noisy[batch_idx].detach()], dim=-1),batch_idx)


        log_pz = 0.5 * (z ** 2).sum(dim=-1)
        loss_graph = (log_pz - log_det).mean()
        beta = min(1.0, epoch / 5.0)
        loss_graph = beta * loss_graph
        '''
        x_phys = batch.x.clone()
        e_log = batch.x[:,3]*x_s[3] + x_m[3]
        x_phys[:,3] = torch.exp(e_log) 
        x_phys[:, 0] = batch.x[:, 0]*x_s[0] + x_m[0]
        x_phys[:, 1] = batch.x[:, 1]*x_s[1] + x_m[1]
        x_phys[:, 2] = batch.x[:, 2]*x_s[2] + x_m[2]
        L_phys = torch.round(x_phys[:,2]).long()
        L_phys = torch.clamp(L_phys, 0, 47)
        x_gen = graph_flow.inverse(z, batch.edge_index, y[batch_idx], batch_idx) # note: energy is fraction, and rest normalized.
        eg_log = x_gen[:,3]*x_s[3] + x_m[3]
        x_gen[:,3] = torch.exp(eg_log) 
        x_gen[:, 0] = x_gen[:, 0]*x_s[0] + x_m[0]
        x_gen[:, 1] = x_gen[:, 1]*x_s[1] + x_m[1]
        x_gen[:, 2] = x_gen[:, 2]*x_s[2] + x_m[2]
        lambda_layer = 1.0  # tune this
        loss_layer1 = layer_feature_mse_loss(energy_real = x_phys[:,3],energy_gen = x_gen[:,3],layer_idx = L_phys,batch_idx = batch.batch)
        loss_layer2 = layer_feature_mse_loss(energy_real = x_phys[:,0],energy_gen = x_gen[:,0],layer_idx = L_phys,batch_idx = batch.batch)
        loss_layer3 = layer_feature_mse_loss(energy_real = x_phys[:,1],energy_gen = x_gen[:,1],layer_idx = L_phys,batch_idx = batch.batch)
        loss_layer4 = layer_feature_mse_loss(energy_real = x_phys[:,2],energy_gen = x_gen[:,2],layer_idx = L_phys,batch_idx = batch.batch)
        loss = loss_graph + lambda_layer * (loss_layer1+loss_layer2+loss_layer3+loss_layer4)
        '''

        opt_graph.zero_grad()
        loss_graph.backward()
        opt_graph.step()
        total_loss += (loss_graph.item()+loss_E.item()+L_loss.item())
        del batch, y , E_node, E_tot, E_log, loss_E, z, log_det, loss_graph
    total_loss/=len(train_loader)
    t_loss.append(total_loss)
    val_loss = test(E_flow, layer_flow, graph_flow, val_loader, opt_E, opt_L, opt_graph, device)
    v_loss.append(val_loss)
    test_loss = test(E_flow, layer_flow, graph_flow, test_loader, opt_E, opt_L, opt_graph, device)
    tes_loss.append(test_loss)
    print(f"Epoch {epoch:03d} TrainLoss {total_loss:.4f} val_loss {val_loss:.4f} Test_loss {test_loss:.4f}")

plt.plot(t_loss, label="Train")
plt.plot(v_loss, label="Validation")
plt.plot(tes_loss, label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
#plt.savefig("gen_loss.png")
plt.show()


# In[13]:


def total_energy_per_graph(x, e_idx=3):
    E = x[:, e_idx]
    return E.sum()
E_flow.eval()
layer_flow.eval()
graph_flow.eval()
E_real_list = []
E_gen_list = []
with torch.no_grad():
    for batch in test_dataset:
        batch = batch.to(device)
        y = batch.y.view(1, y_dim)
        batch_vec = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        E_log_gen = E_flow.sample(y)
        L_loge_gen = layer_flow.sample(E_log_gen[batch_vec])
        T = torch.ones(x_dim, device=batch.x.device)
        T[3] = 0.8   # energy
        T[0] = 0.8 # spatial
        T[1] = 1.2 # spatial
        T[2] = 0.8   # layer-like

        z = torch.randn_like(batch.x)*T
        #batch_vec = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        x_gen = graph_flow.inverse(z, batch.edge_index, torch.cat([E_log_gen[batch_idx].detach(),L_loge_gen[batch_idx].detach()], dim=-1), batch_vec) # note: energy is fraction, and rest normalized.
        x_phys = batch.x.clone()
        
        e_log = batch.x[:,3]*x_s[3] + x_m[3]
        x_phys[:,3] = torch.exp(e_log)
        eg_log = x_gen[:,3]*x_s[3] + x_m[3]
        x_gen[:,3] = torch.exp(eg_log)
        
        E_r = total_energy_per_graph(x_phys).cpu()
        E_g  = total_energy_per_graph(x_gen).cpu()
        E_real_list.append(E_r)
        E_gen_list.append(E_g)

        del batch, x_phys, z, x_gen, E_r, E_g, batch_vec
    E_real = np.array(E_real_list)
    E_gen  = np.array(E_gen_list)


# In[14]:


print(E_real.min())
print(E_gen.min())
print(E_real.max())
print(E_gen.max())


# In[15]:


# --- common binning ---
bins = np.linspace(
    min(E_real.min(), E_gen.min()),
    max(E_real.max(), E_gen.max()),
    40
)
# --- histogram values ---
hist_real, _ = np.histogram(E_real, bins=bins)
hist_gen,  _ = np.histogram(E_gen,  bins=bins)

# avoid division by zero
ratio = np.divide(
    hist_gen,
    hist_real,
    out=np.ones_like(hist_gen, dtype=float),
    where=hist_real > 0
)

bin_centers = 0.5 * (bins[1:] + bins[:-1])

# --- figure with 2 subplots ---
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    figsize=(8, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

# ===== TOP: ENERGY DISTRIBUTIONS =====
ax_top.step(
    bins[:-1],
    hist_real,
    where="post",
    label="Real",
    linewidth=2
)

ax_top.step(
    bins[:-1],
    hist_gen,
    where="post",
    label="Generated",
    linewidth=2
)

ax_top.set_ylabel("Counts")
ax_top.legend()
ax_top.grid(alpha=0.3)

# ===== BOTTOM: RATIO =====
ax_bot.axhline(1.0, color="black", linestyle="--", linewidth=1)

ax_bot.plot(
    bin_centers,
    ratio
)

ax_bot.set_ylabel("Gen / Real")
ax_bot.set_xlabel("Total shower energy")
#ax_bot.set_ylim(0, 2)

ax_bot.grid(alpha=0.3)


plt.tight_layout()
plt.savefig("energy_comparison.png", dpi=150)
plt.show()


# In[48]:


print(E_gen)
print(E_real)


# In[16]:


E_flow.eval()
layer_flow.eval()
graph_flow.eval()
real_list = []
gen_list = []

with torch.no_grad():
    for batch in test_dataset:
        batch = batch.to(device)
        y = batch.y.view(1, y_dim)
        x_phys = batch.x.clone()
        e_log = batch.x[:,3]*x_s[3] + x_m[3]
        x_phys[:,3] = torch.exp(e_log) 
        x_phys[:, 0] = batch.x[:, 0]*x_s[0] + x_m[0]
        x_phys[:, 1] = batch.x[:, 1]*x_s[1] + x_m[1]
        x_phys[:, 2] = batch.x[:, 2]*x_s[2] + x_m[2]
        L_phys = torch.round(x_phys[:,2]).long()
        L_phys = torch.clamp(L_phys, 0, 46)

        # --- generate ---
        T = torch.ones(x_dim, device=batch.x.device)
        T[3] = 0.8   # energy
        T[0] = 0.8 # spatial
        T[1] = 1.2 # spatial
        T[2] = 0.8   # layer-like
        batch_vec = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        E_log_gen = E_flow.sample(y)
        L_loge_gen = layer_flow.sample(E_log_gen[batch_vec])

        z = torch.randn_like(batch.x)*T
        x_gen = graph_flow.inverse(z, batch.edge_index, torch.cat([E_log_gen[batch_idx].detach(),L_loge_gen[batch_idx].detach()], dim=-1), batch_vec) # note: energy is fraction, and rest normalized.
        eg_log = x_gen[:,3]*x_s[3] + x_m[3]
        x_gen[:,3] = torch.exp(eg_log) 
        x_gen[:, 0] = x_gen[:, 0]*x_s[0] + x_m[0]
        x_gen[:, 1] = x_gen[:, 1]*x_s[1] + x_m[1]
        x_gen[:, 2] = x_gen[:, 2]*x_s[2] + x_m[2]
        L_gen = torch.round(x_gen[:,2]).long()
        L_gen = torch.clamp(L_gen, 0, 46)
        gen_res = {}
        for layer in range(48):
            mask = L_gen == layer
            if mask.sum() == 0:
                continue
            x_l = x_gen[:, 0][mask]
            y_l = x_gen[:, 1][mask]
            e_l = x_gen[:, 3][mask]
            tot_l_energy = e_l.sum()
            x_center = (x_l * e_l).sum() / (tot_l_energy + 1e-12)
            y_center = (y_l * e_l).sum() / (tot_l_energy + 1e-12)
            x_width = torch.sqrt(((x_l - x_center) ** 2 * e_l).sum() / (tot_l_energy + 1e-12))
            y_width = torch.sqrt(((y_l - y_center) ** 2 * e_l).sum() / (tot_l_energy + 1e-12))
            sparsity = (2200 - len(e_l))/2200
            gen_res[layer] = {
                "energy": tot_l_energy,
                "x_center": x_center,
                "y_center": y_center,
                "x_width": x_width,
                "y_width": y_width,
                "sparsity": sparsity,
            }
        gen_list.append(gen_res)
        phys_res = {}
        for layer in range(48):
            mask = L_phys == layer
            if mask.sum() == 0:
                continue
            x_l = x_phys[:, 0][mask]
            y_l = x_phys[:, 1][mask]
            e_l = x_phys[:, 3][mask]
            tot_l_energy = e_l.sum()
            x_center = (x_l * e_l).sum() / (tot_l_energy + 1e-12)
            y_center = (y_l * e_l).sum() / (tot_l_energy + 1e-12)
            x_width = torch.sqrt(((x_l - x_center) ** 2 * e_l).sum() / (tot_l_energy + 1e-12))
            y_width = torch.sqrt(((y_l - y_center) ** 2 * e_l).sum() / (tot_l_energy + 1e-12))
            #sparsity = (2200 - len(e_l))/2200"sparsity": torch.tensor(sparsity, device=device)
            sparsity = (e_l == 0).float().mean()
            phys_res[layer] = {
                "energy": tot_l_energy,
                "x_center": x_center,
                "y_center": y_center,
                "x_width": x_width,
                "y_width": y_width,
                "sparsity": sparsity,
            }
        real_list.append(phys_res)


# In[17]:


print(x_gen[:,0].min())
print(x_gen[:,0].max())
print(x_phys[:,0].min())
print(x_phys[:,0].max())
print(x_gen[:,1].min())
print(x_gen[:,1].max())
print(x_phys[:,1].min())
print(x_phys[:,1].max())
print(x_gen[:,2].min())
print(x_gen[:,2].max())
print(x_phys[:,2].min())
print(x_phys[:,2].max())
print(x_gen[:,3].min())
print(x_gen[:,3].max())
print(x_phys[:,3].min())
print(x_phys[:,3].max())


# In[18]:


def collect_feature(results, layer, feature):
    vals = []
    for event in results:
        if layer in event:
            v = event[layer][feature]
            if torch.is_tensor(v):
                v = v.item()
            vals.append(v)
        else:
            vals.append(0.0)  # empty layer
    return torch.tensor(vals)

def plot_hist_with_ratio(real, gen, bins, xlabel, title):
    hist_r, edges = np.histogram(real, bins=bins)
    hist_g, _     = np.histogram(gen, bins=edges)

    centers = 0.5 * (edges[1:] + edges[:-1])

    ratio = np.divide(
        hist_g, hist_r,
        out=np.ones_like(hist_g, dtype=float),
        where=hist_r > 0
    )

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(6, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # --- top: histogram comparison
    ax.step(centers, hist_r, where="mid", linewidth=2, label="Real")
    ax.step(centers, hist_g, where="mid", linewidth=2, label="Generated")
    ax.set_ylabel("Events")
    ax.set_title(title)
    ax.legend()

    # --- bottom: ratio
    rax.axhline(1.0, color="k", linestyle="--")
    rax.step(centers, ratio, where="mid", linewidth=2)
    rax.set_ylabel("Gen / Real")
    rax.set_xlabel(xlabel)
    rax.set_ylim(0, 2)

    plt.tight_layout()
    plt.show()

layers_to_plot = range(5)
features = ["energy","x_center","y_center","x_width","y_width","sparsity",]
for layer in layers_to_plot:
    print(f"\n=== Layer {layer} ===")

    for feature in features:
        real_vals = collect_feature(real_list, layer, feature)
        gen_vals  = collect_feature(gen_list,  layer, feature)

        # remove NaNs (centers can be NaN for empty layers)
        mask = ~torch.isnan(real_vals) & ~torch.isnan(gen_vals)
        real_vals = real_vals[mask]
        gen_vals  = gen_vals[mask]

        # choose bins intelligently
        if feature == "energy":
            bins = np.linspace(0, real_vals.max().item() * 1.1 + 1e-6, 40)
        else:
            bins = 40

        plot_hist_with_ratio(
            real_vals.numpy(),
            gen_vals.numpy(),
            bins=bins,
            xlabel=feature,
            title=f"Layer {layer} — {feature}"
        )


# In[ ]:




