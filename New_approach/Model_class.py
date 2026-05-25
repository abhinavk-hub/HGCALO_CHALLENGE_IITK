device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LAYERS = 47

class EnergyFlow(nn.Module):
    def __init__(self, cond_dim=1, hidden_dim=64, num_flows=4):
        super().__init__()
        
        self.num_flows = num_flows

        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cond_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),   nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
            for _ in range(num_flows)
        ])
 
    def forward(self, e_t, e_inc):        
        z = e_t
        log_det = 0.0
        for net in self.nets:
            st = net(e_inc)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s)
            z = z*torch.exp(s) + t
            log_det = log_det + s.squeeze(-1)

        log_pz  = -0.5 * (z ** 2 + math.log(2 * math.pi)) 
        nll = -(log_pz.squeeze(-1) + log_det)
        return nll

    @torch.no_grad()
    def sample(self, e_inc):
        z = torch.randn(e_inc.size(0), 1, device=e_inc.device)
        for net in reversed(self.nets):
            st   = net(e_inc)
            s, t = st.chunk(2, dim=-1)
            s    = torch.tanh(s)
            z    = (z - t) * torch.exp(-s)
        return z

class LongGraphFlow(nn.Module):
    def __init__(self, cond_dim=2, hidden_dim=64, num_conv_layers=3, el_flow = 4):
        super().__init__()
        
        self.pos_embed = nn.Embedding(MAX_LAYERS, hidden_dim)
        self.conv=nn.ModuleList()
        self.conv.append(GATConv(hidden_dim + cond_dim, hidden_dim, heads=4, concat=False))
        for _ in range(num_conv_layers -1):
            self.conv.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        self.h_norm = nn.LayerNorm(hidden_dim)
        self.el_flow = el_flow
        self.energy_head= nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),   nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
            for _ in range(el_flow)
        ])
        self.poisson_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,1))
        
    def encode(self, long_edge_index, long_batch, E_total, E_inc):
        B = E_total.size(0)
        layer_ids = torch.arange(MAX_LAYERS, device=long_edge_index.device)
        pos       = self.pos_embed(layer_ids).repeat(B, 1)   
        cond = torch.stack([E_total, E_inc], dim=-1)
        cond = cond[long_batch]
        h=torch.cat([pos, cond], dim =-1)
        
        for conv in self.conv:
            h=F.relu(conv(h, long_edge_index))
        
        return h
    
    def forward(self, long_x, long_edge_index, long_batch, E_total, E_inc, raw_nhits):
        h =self.encode(long_edge_index, long_batch, E_total, E_inc)
        
        log_E_l = long_x[:, 0]
        z = log_E_l.unsqueeze(-1)
        log_det = 0.0
        h_normed = self.h_norm(h)
        for net in self.energy_head:
            st = net(h_normed)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s)
            t = torch.tanh(t) * 3.0 
            z = z*torch.exp(s) + t
            log_det = log_det + s.squeeze(-1)

        log_pz  = -0.5 * (z ** 2 + math.log(2 * math.pi)) 
        nll_e_node = -(log_pz.squeeze(-1) + log_det)
        nll_energy = global_mean_pool(nll_e_node.unsqueeze(-1), long_batch).squeeze(-1)
        
        log_lambda = self.poisson_head(h).squeeze(-1)
        lam = torch.exp(log_lambda).clamp(min=1e-6)
        nll_p_node = lam - raw_nhits.float()*log_lambda
        nll_poisson = global_mean_pool(nll_p_node.unsqueeze(-1), long_batch).squeeze(-1)
        
        return nll_energy, nll_poisson
    
    @torch.no_grad()
    def sample(self, long_edge_index, long_batch, E_total, E_inc):
        h= self.encode(long_edge_index, long_batch, E_total , E_inc)
        
        z = torch.randn(long_batch.size(0), 1, device=h.device)
        h_normed = self.h_norm(h)
        for net in reversed(self.energy_head):
            st   = net(h_normed)
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s)
            t = torch.tanh(t) * 3.0 
            z    = (z - t) * torch.exp(-s)
        log_E_l_norm = z.squeeze(-1)
        
        log_lambda = self.poisson_head(h).squeeze(-1)
        lam = torch.exp(log_lambda).clamp(min=1e-6)
        nhits_l = torch.poisson(lam).long()
        
        active = nhits_l > 0
        
        return log_E_l_norm, nhits_l, active

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

class TransRFM(nn.Module):
    def __init__(self, node_dim=5, cond_dim=2, layer_embedding_dim =8, time_dim=16, hidden_dim=128, num_layers=6, num_layers_ids=47):
        super().__init__()
        
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.layer_embed = nn.Embedding(num_layers_ids, layer_embedding_dim)
        
        in_dim = node_dim + cond_dim + layer_embedding_dim + time_dim
        #in_dim = node_dim + cond_dim + time_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([GATConv(hidden_dim, hidden_dim, heads=4, concat=False) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, node_dim)
        
    def forward(self, z_t, edge_index, trans_batch, E_l, nhits, layer_ids, t):
        E_l_node = E_l[trans_batch]
        nhits_node = nhits[trans_batch]
        layer_id_node = layer_ids[trans_batch]
        t_node = t[trans_batch]
        
        t_emb = self.time_embed(t_node)
        l_emb = self.layer_embed(layer_id_node)
        
        cond = torch.cat([E_l_node.unsqueeze(-1), nhits_node.unsqueeze(-1), l_emb, t_emb], dim=-1)
        #cond = torch.cat([E_l_node.unsqueeze(-1), nhits_node.unsqueeze(-1), t_emb], dim=-1)
        
        h = torch.cat([z_t, cond], dim =-1)
        h = F.relu(self.input_proj(h))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        
        return self.out(h)
    
    @torch.no_grad()
    def sample(self, n_nodes, trans_batch, E_l, nhits, layer_ids, steps=50, t_threshold= 0.4, edge_builder = None):
        z = torch.randn( n_nodes, 5, device=E_l.device)
        dt = 1.0/steps
        empty_ei   = torch.zeros((2, 0), dtype=torch.long, device=E_l.device)
        edge_index = empty_ei
        phase1_steps = int(t_threshold * steps)
        phase2_steps = steps - phase1_steps
        for i in range(phase1_steps):
            t_val = i*dt
            t_node = torch.full((E_l.size(0),1), t_val, device=E_l.device)
            v = self.forward(z, edge_index, trans_batch, E_l, nhits, layer_ids, t_node)
            z = z + v*dt
        
        if edge_builder is not None:
            edge_index = edge_builder(z, trans_batch)
        else:
            edge_index = empty_ei
        
        for i in range(phase1_steps, steps):
            t_val = i*dt
            t_node = torch.full((E_l.size(0),1), t_val, device=E_l.device)
            v = self.forward(z, edge_index, trans_batch, E_l, nhits, layer_ids, t_node)
            z = z + v*dt
            
        return z

energy_flow = EnergyFlow(cond_dim=1, hidden_dim=64, num_flows=4).to(device)
long_flow = LongGraphFlow(cond_dim=2, hidden_dim=64, num_conv_layers=3, el_flow = 4).to(device)
trans_rfm = TransRFM(node_dim=5, cond_dim=2, layer_embedding_dim=8, time_dim=16, hidden_dim=128, num_layers=4).to(device)        
