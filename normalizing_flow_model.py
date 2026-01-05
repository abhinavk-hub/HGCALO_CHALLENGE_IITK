# for using torch on lxplus: source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh
# for using torch with cuda on lxplus: source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphFolderDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        data = torch.load(path, weights_only=False)  # Data(x, edge_index, y)
        return data

train_dataset = GraphFolderDataset("/eos/user/a/abkumar/gen_train_pp")
val_dataset   = GraphFolderDataset("/eos/user/a/abkumar/gen_val_pp")
test_dataset  = GraphFolderDataset("/eos/user/a/abkumar/gen_test_pp")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

class ConditioningGNN(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim):
        super().__init__()

        self.conv1 = GCNConv(x_dim + y_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * x_dim)  # scale + shift
        )

    def forward(self, x, edge_index, y, batch):
        y_node=y[batch]

        h = torch.cat([x, y_node], dim=-1)
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.conv2(h, edge_index))

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

        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)

        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)

        return z, log_det

    def inverse(self, z, edge_index, y, batch):
        z_masked = z * self.mask
        #batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        s, t = self.net(z_masked, edge_index, y, batch)

        s = torch.tanh(s) * (1 - self.mask)
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
def train_flow(flow, loader, optimizer, device):
    flow.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        x = batch.x
        y = batch.y.view(len(batch),2)
        #y = batch.y.reshape(len(batch), 2)
        edge_index = batch.edge_index
        
        z, log_det = flow(x, edge_index, y, batch.batch)

        log_pz = 0.5 * (z ** 2).sum(dim=-1)
        loss = (log_pz - log_det).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        del x, edge_index, y, z, log_det, loss, batch
        #torch.cuda.empty_cache()

    return total_loss / len(loader)

x_dim = 4       
y_dim = 2    
hidden_dim = 128
num_layers = 8
#num_nodes = 47*2200

flow = GraphNormalizingFlow(
    x_dim, y_dim, hidden_dim, num_layers
).to(device)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")
epochs = 3
#scaler = GradScaler()
for epoch in range(1, epochs + 1):
    train_loss = train_flow(flow, train_loader,optimizer, device)
    #val_loss = train_flow(flow, val_loader, optimizer, device)
    print(f"Epoch {epoch:03d} TrainLoss {train_loss:.4f}")

def total_energy_per_graph(x,e_mu, e_std, e_idx=3):
    
    #num_graphs = batch.max().item() + 1
    E = x[:, e_idx]*e_std+e_mu

    #E_sum = torch.zeros(1, device=x.device)
    #E_sum.scatter_add_(0,1, E)
    E_sum = sum(E)

    return E_sum
flow.eval()

#energy_ratios = []
E_real_list = []
E_gen_list = []
stats = torch.load("/eos/user/a/abkumar/geometry/normalization_stats.pt", weights_only=False)
x_mu = stats["x_mean"]
x_std = stats["x_std"]
E_mu = x_mu[3].item()
E_std = x_std[3].item()
with torch.no_grad():
    N_EVAL = 100
    indices = random.sample(range(len(test_dataset)), N_EVAL)
    #for batch in test_dataset:
    for i in indices:
        batch = test_dataset[i]
        batch = batch.to(device)

        # --- generate ---
        z = torch.randn_like(batch.x)
        batch_vec = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        x_gen = flow.inverse(z, batch.edge_index, batch.y.view(1, y_dim), batch_vec)
        #x_gen = flow.inverse(z, batch.edge_index, batch.y)

        # --- energies ---
        E_r = total_energy_per_graph(batch.x, E_mu, E_std).cpu()
        E_g  = total_energy_per_graph(x_gen, E_mu, E_std).cpu()
        E_real_list.append(E_r)
        E_gen_list.append(E_g)

        #ratio = (E_gen / E_real).detach().cpu()
        #energy_ratios.append(ratio)
        del batch, z, x_gen, E_r, E_g, batch_vec
        #torch.cuda.empty_cache()
    #energy_ratios = np.array(energy_ratios)
    E_real = np.array(E_real_list)
    E_gen  = np.array(E_gen_list)

'''
x = np.arange(len(energy_ratios))
plt.figure(figsize=(10, 4))

# Ratio points
plt.plot(x, energy_ratios, alpha=0.7, label="Generated / Real")

# Reference line
plt.axhline(
    y=1.0,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Ideal (ratio = 1)"
)

plt.xlabel("Graph index")
plt.ylabel("Total energy ratio")
#plt.ylim(0, max(2.0, ratios.max() * 1.1))

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("energy_ratio.png")
plt.show()
'''


bins = np.linspace(
    min(E_real.min(), E_gen.min()),
    max(E_real.max(), E_gen.max()),
    40
)

hist_real, _ = np.histogram(E_real, bins=bins)
hist_gen,  _ = np.histogram(E_gen,  bins=bins)

# avoid division by zero
ratio = np.divide(
    hist_gen,
    hist_real,
    out=np.zeros_like(hist_gen, dtype=float),
    where=hist_real > 0
)

bin_centers = 0.5 * (bins[1:] + bins[:-1])

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    figsize=(8, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]}
)

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
