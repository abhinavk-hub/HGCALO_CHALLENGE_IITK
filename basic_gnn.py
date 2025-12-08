import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch.nn import Linear, Sequential, ReLU

class GraphFolderDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith(".pt")])

    def len(self):
        return len(self.files)

    def get(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        data = torch.load(path)  # Data(x, edge_index, edge_attr, y)
        return data

train_dataset = GraphFolderDataset("/eos/user/a/abkumar/gnn_train")
val_dataset   = GraphFolderDataset("/eos/user/a/abkumar/gnn_val")
test_dataset  = GraphFolderDataset("/eos/user/a/abkumar/gnn_test")

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

def val_loss(loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            data.y = data.y.reshape(len(data), 3)
            loss = criterion(out, data.y)
            total += loss.item()
    return total / len(loader)
    
def test_loss(loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            data.y = data.y.reshape(len(data), 3)
            loss = criterion(out, data.y)
            total += loss.item()
    return total / len(loader)

for epoch in range(15):
    model.train()
    total = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        #print(out.shape)
        #print(data.y.shape)
        data.y = data.y.reshape(len(data), 3)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total += loss.item()

    train_losses.append(total / len(train_loader))
    val_losses.append(val_loss(val_loader))
    test_losses.append(test_loss(test_loader))

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
