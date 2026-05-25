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
