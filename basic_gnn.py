import os
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch.nn import Linear, Sequential, ReLU
import numpy as np

class GraphFolderDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pt")])

    def load_all(self):
        graphs = []
        for fname in self.files:
            path = os.path.join(self.root_dir, fname)
            g = torch.load(path)      # Data(x, edge_index, edge_attr, y)
            graphs.append(g)
        return graphs

class ListDataset(Dataset):
    def __init__(self, graph_list):
        self.graphs = graph_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def remove_redundancy(mat, drop_indices):
    keep = [i for i in range(mat.size(1)) if i not in drop_indices]
    return mat[:, keep]

def remove_constant(y):
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


def preprocess(train_path, val_path, test_path):
    
    train_raw = GraphFolderDataset(train_path).load_all()
    val_raw   = GraphFolderDataset(val_path).load_all()
    test_raw  = GraphFolderDataset(test_path).load_all()

    for dataset in [train_raw, val_raw, test_raw]:
        for g in dataset:
            if g.x is not None:
                g.x = remove_redundancy(g.x, [3])      
            if g.edge_attr is not None:
                g.edge_attr = remove_redundancy(g.edge_attr, [4])  
            if g.y is not None:
                g.y = remove_constant(g.y)             

    x_mean, x_std = compute_normalization_stats(train_raw, 'x')
    e_mean, e_std = compute_normalization_stats(train_raw, 'edge_attr')

    apply_norm(train_raw, 'x', x_mean, x_std)
    apply_norm(val_raw,   'x', x_mean, x_std)
    apply_norm(test_raw,  'x', x_mean, x_std)

    apply_norm(train_raw, 'edge_attr', e_mean, e_std)
    apply_norm(val_raw,   'edge_attr', e_mean, e_std)
    apply_norm(test_raw,  'edge_attr', e_mean, e_std)

    y0_train = torch.stack([g.y[0] for g in train_raw])
    y0_mean = y0_train.mean()
    y0_std  = y0_train.std()
    if y0_std < 1e-8:
        y0_std = 1.0

    for dataset in [train_raw, val_raw, test_raw]:
        for g in dataset:
            g.y[0] = (g.y[0] - y0_mean) / y0_std
    
    aa = [y0_mean, y0_std]

    return (
        ListDataset(train_raw),
        ListDataset(val_raw),
        ListDataset(test_raw), aa
    )


train_dataset, val_dataset, test_dataset, bb = preprocess(
    "/eos/user/a/abkumar/gnn_train",
    "/eos/user/a/abkumar/gnn_val",
    "/eos/user/a/abkumar/gnn_test"
)


class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, edge_channels, hidden, out_classes):
        super().__init__()

        nn1 = Sequential(Linear(in_channels, hidden), ReLU(), Linear(hidden, hidden))
        nn2 = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden))
        
        self.edge_encoder = Sequential(Linear(edge_channels, hidden),ReLU(),Linear(hidden, hidden))

        self.conv1 = GINEConv(nn1, edge_dim=hidden)
        self.conv1.edge_encoder = self.edge_encoder
        self.conv2 = GINEConv(nn2, edge_dim=hidden)
        self.conv2.edge_encoder = self.edge_encoder

        self.lin = Linear(hidden, out_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = self.edge_encoder(edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

model = SimpleGNN(
    in_channels=train_dataset[0].x.size(1),
    edge_channels=train_dataset[0].edge_attr.size(1),
    hidden=64,
    out_classes=len(torch.unique(train_dataset[0].y))
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()

train_losses = []
val_losses = []
test_losses = []
true_energy = []
pred_energy = []
def val_loss(loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            data.y = data.y.reshape(len(data), 2)
            loss = criterion(out, data.y)
            total += loss.item()
    return total / len(loader)
    
def test_loss(loader):
    model.eval()
    total = 0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            data.y = data.y.reshape(len(data), 2)
            true_e = data.y[:,0]*bb[1]+bb[0]
            pred_e = out[:,0]*bb[1]+bb[0]
            all_true.append(true_e.cpu())
            all_pred.append(pred_e.cpu())
            loss = criterion(out, data.y)
            total += loss.item()
    return (total / len(loader), torch.cat(all_true), torch.cat(all_pred))

for epoch in range(15):
    model.train()
    total = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        #print(out.shape)
        #print(data.y.shape)
        data.y = data.y.reshape(len(data), 2)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total += loss.item()

    train_losses.append(total / len(train_loader))
    val_losses.append(val_loss(val_loader))
    t_loss, t_e, p_e = test_loss(test_loader)
    test_losses.append(t_loss)
    true_energy.append(t_e)
    pred_energy.append(p_e)
    

    print(f"Epoch {epoch:02d} | Train {train_losses[-1]:.4f} | Val {val_losses[-1]:.4f} | Test {test_losses[-1]:.4f}")

plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.plot(test_losses, label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.savefig("loss.png")
plt.show()

true_energy = torch.cat(true_energy)
pred_energy = torch.cat(pred_energy)
plt.figure(figsize=(7,7))
plt.scatter(true_energy, pred_energy, s=10, alpha=0.5)

# Plot ideal diagonal y = x
min_val = min(true_energy.min(), pred_energy.min())
max_val = max(true_energy.max(), pred_energy.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--')

plt.xlabel("True GEN Energy")
plt.ylabel("Predicted GEN Energy")
plt.title("GEN vs Predicted Energy")
plt.grid(True)
plt.tight_layout()
plt.savefig("gen_vs_pred.png")
plt.show()

ratio= (pred_energy-true_energy)/true_energy
mu = torch.mean(ratio).item()
sigma = torch.std(ratio).item()
plt.figure(figsize=(8,6))
#counts, bins, _ = plt.hist(ratio, bins=60, density=True, alpha=0.6, color='g')
plt.hist(ratio, bins=60, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = (1/(sigma * np.sqrt(2*np.pi))) * np.exp( - (x - mu)**2 / (2*sigma**2) )
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel("(Pred - True) / True")
plt.ylabel("Density")
plt.title("Residual Distribution with Gaussian Fit")
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_hist_gaussian.png")
plt.show()
