def to_phys_E_total(val_norm, stats):
    return inverse_transform(stats, 'E_total', val_norm)

def to_phys_E_l(val_norm, stats):
    return inverse_transform(stats, 'long_E', val_norm)

def to_phys_node(tx_norm, stats):
    tx = tx_norm.clone()
    tx[:, 0] = inverse_transform(stats, 'te', tx_norm[:, 0])   
    tx[:, 1] = inverse_transform(stats, 'tx', tx_norm[:, 1])   
    tx[:, 2] = inverse_transform(stats, 'ty', tx_norm[:, 2])   
    tx[:, 3] = inverse_transform(stats, 'tr', tx_norm[:, 3])   
    tx[:, 4] = inverse_transform(stats, 'phi', tx_norm[:, 4])
    return tx

def build_edges_from_z(z_norm, trans_batch, k=4):
    device   = z_norm.device
    z_x = inverse_transform(stats, 'tx', z_norm[:,1])
    z_y = inverse_transform(stats, 'ty', z_norm[:,2])
    #x_approx = z_norm[:, 1].cpu().numpy()
    #y_approx = z_norm[:, 2].cpu().numpy()
    x_approx = z_x.cpu().numpy()
    y_approx = z_y.cpu().numpy()
 
    all_edges  = []
    node_offset = 0
 
    for lg in trans_batch.unique():
        mask = (trans_batch == lg).cpu()
        xs   = x_approx[mask]
        ys   = y_approx[mask]
        n    = int(mask.sum())
 
        if n <= 1:
            node_offset += n
            continue
 
        coords = np.stack([xs, ys], axis=1)
        k_eff  = min(k + 1, n)
        nbrs   = NearestNeighbors(n_neighbors=k_eff, algorithm='auto').fit(coords)
        _, indices = nbrs.kneighbors(coords)
 
        src_list, dst_list = [], []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if i != j:                    
                    src_list.append(i)
                    dst_list.append(j)
    
        if len(src_list) == 0:
            node_offset += n
            continue 
        ei = torch.tensor(np.stack([src_list, dst_list]), dtype=torch.long)
        ei = to_undirected(ei, num_nodes=n)
        if ei.shape[1] > 0:
            ei = ei + node_offset
        all_edges.append(ei)
        node_offset += n
 
    if all_edges:
        return torch.cat(all_edges, dim=1).to(device)
    return torch.zeros((2, 0), dtype=torch.long, device=device)

@torch.no_grad()
def generate_shower(E_inc_batch, stats, rfm_steps=50):
    
    B = E_inc_batch.size(0)
    E_total_norm   = energy_flow.sample(E_inc_batch)              
    
    src_b, dst_b = [], []
    for b in range(B):
        off = b * MAX_LAYERS
        for i in range(MAX_LAYERS - 1):
            src_b += [off+i, off+i+1]
            dst_b += [off+i+1, off+i]
    long_ei = torch.tensor([src_b, dst_b], dtype=torch.long, device=device)
    long_bv = torch.arange(B, device=device).repeat_interleave(MAX_LAYERS)
    log_E_l_norm, nhits_l, active = long_flow.sample(long_ei,long_bv, E_total_norm.squeeze(-1), E_inc_batch.squeeze(-1))
    E_l_phys = to_phys_E_l(log_E_l_norm, stats)
    E_l_phys = E_l_phys.clamp(min=0)
    E_total_phys_list = []
 
    for b in range(B):
        sl    = slice(b * MAX_LAYERS, (b+1) * MAX_LAYERS)
        E_t   = float(to_phys_E_total(E_total_norm[b].squeeze(), stats))
        E_total_phys_list.append(E_t)
        act_b = active[sl]
        e_sum = E_l_phys[sl][act_b].sum()
        if e_sum > 0:
            E_l_phys[sl][act_b] *= (E_t / e_sum.item())
    
    all_n_nodes, all_E_l, all_nhits, all_lids, shower_of_lg = [], [], [], [], []
 
    for b in range(B):
        for l in range(MAX_LAYERS):
            nidx = b * MAX_LAYERS + l
            if not active[nidx]:
                continue
            n = int(nhits_l[nidx].item())
            if n == 0:
                continue
            E_l_v     = float(log_E_l_norm[nidx].item())
            nhits_v   = (math.log(n + 1) - stats['tn_mean']) / stats['tn_std']
            all_n_nodes.append(n)
            all_E_l.append(E_l_v)
            all_nhits.append(nhits_v)
            all_lids.append(l)
            shower_of_lg.append(b)
 
    n_lg = len(all_n_nodes)
    if n_lg == 0:
        return [{'gen_nodes': None, 'gen_layers': None,
                 'E_total_phys': E_total_phys_list[b]} for b in range(B)]

    total_nodes  = sum(all_n_nodes)
    trans_bv     = torch.repeat_interleave(torch.arange(n_lg, device=device),torch.tensor(all_n_nodes, device=device))
    E_l_t        = torch.tensor(all_E_l,   dtype=torch.float32, device=device)
    nhits_t      = torch.tensor(all_nhits, dtype=torch.float32, device=device)
    lids_t       = torch.tensor(all_lids,  dtype=torch.long,    device=device)
    empty_ei     = torch.zeros((2, 0), dtype=torch.long, device=device)
    edge_index = empty_ei

    z = torch.randn(total_nodes, 5, device=device)
    Phase1_steps = int(0.4 * rfm_steps)
    Phase2_steps = rfm_steps - Phase1_steps    
    dt = 1.0 / rfm_steps

    for step in range(Phase1_steps):
        t_val  = step * dt
        t_node = torch.full((n_lg, 1), t_val, device=device)

        v = trans_rfm(z, edge_index, trans_bv, E_l_t, nhits_t, lids_t, t_node)
        z = z + v * dt
        
    edge_index = build_edges_from_z(z, trans_bv, k=4)
    
    for step in range(Phase1_steps, rfm_steps):
        t_val  = step * dt
        t_node = torch.full((n_lg, 1), t_val, device=device)

        v = trans_rfm(z, edge_index, trans_bv, E_l_t, nhits_t, lids_t, t_node)
        z = z + v * dt

    nodes_phys = to_phys_node(z, stats)
    node_ptr = 0
    for lg_i in range(n_lg):
        n     = all_n_nodes[lg_i]
        b     = shower_of_lg[lg_i]
        l     = all_lids[lg_i]
        E_l_v = float(E_l_phys[b * MAX_LAYERS + l].item())
        seg   = nodes_phys[node_ptr : node_ptr + n]
        e_seg = seg[:, 0].clamp(min=0)
        e_sum = e_seg.sum()
        if e_sum > 0 and E_l_v > 0:
            nodes_phys[node_ptr : node_ptr + n, 0] = e_seg * (E_l_v / e_sum.item())
        node_ptr += n
    
    results  = []
    node_ptr = 0
    lg_ptr   = 0
 
    for b in range(B):
        s_nodes, s_layers = [], []
        n_lgs_b = sum(1 for s in shower_of_lg if s == b)
 
        for _ in range(n_lgs_b):
            n   = all_n_nodes[lg_ptr]
            l   = all_lids[lg_ptr]
            seg = nodes_phys[node_ptr : node_ptr + n]
            s_nodes.append(seg)
            s_layers.append(torch.full((n,), l, dtype=torch.long, device=device))
            node_ptr += n
            lg_ptr   += 1
 
        if s_nodes:
            results.append({
                'gen_nodes':    torch.cat(s_nodes,  dim=0),
                'gen_layers':   torch.cat(s_layers, dim=0),
                'E_total_phys': E_total_phys_list[b],
            })
        else:
            results.append({
                'gen_nodes': None, 'gen_layers': None,
                'E_total_phys': E_total_phys_list[b],
            })
 
    return results

def compute_observables(nodes_phys, layer_ids):
    
    obs     = {}
    e_all   = nodes_phys[:, 0]
    x_all   = nodes_phys[:, 1]
    y_all   = nodes_phys[:, 2]
    E_total = float(e_all.sum().item())

    for l in range(MAX_LAYERS):
        mask = layer_ids == l
        if mask.sum() == 0:
            continue

        e_l  = e_all[mask]
        x_l  = x_all[mask]
        y_l  = y_all[mask]
        nhit = int(mask.sum().item())

        e_sum = e_l.sum() + 1e-12
        xc    = (x_l * e_l).sum() / e_sum
        yc    = (y_l * e_l).sum() / e_sum
        xw    = torch.sqrt(((x_l - xc) ** 2 * e_l).sum() / e_sum)
        yw    = torch.sqrt(((y_l - yc) ** 2 * e_l).sum() / e_sum)

        obs[l] = {
            'energy':   float(e_l.sum().item()),
            'x_center': float(xc.item()),
            'y_center': float(yc.item()),
            'x_width':  float(xw.item()),
            'y_width':  float(yw.item()),
            'nhits':    nhit,
        }

    return obs, E_total
'''
def compute_real_observables(data, stats):

    tx    = to_phys_node(data.trans_x, stats)
    layer_ids = data.trans_layer_ids             
    batch_vec = data.trans_batch                 
    layer_per_node = layer_ids[batch_vec]
    
    E_total_phys = float(to_phys_E_total(data.E_total, stats))
    obs, _ = compute_observables(tx, layer_per_node)

    return obs, E_total_phys


long_edge_index = torch.zeros((2, 2*(MAX_LAYERS-1)), dtype=torch.long)
src, dst = [], []
for i in range(MAX_LAYERS - 1):
    src += [i, i+1]
    dst += [i+1, i]
long_edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)
test_paths   = ["/eos/user/a/abkumar/a_freshlook_at_HGCALO/test_graphs_0"]
test_dataset = ShowerDataset(test_paths, stats=stats) # we only need this code snippet if we running this code on a separate jupyter notebook
'''

real_obs_list  = []
gen_obs_list   = []
E_total_real   = []
E_total_gen    = []
E_inc_phys_list = []
n_processed     = 0
#MAX_SHOWERS = 2000
print(f"Evaluating on {len(test_dataset)} test showers...")

with torch.no_grad():
    for batch_data in test_loader:
        #if n_processed >= MAX_SHOWERS:
        #    break
 
        batch_data = batch_data.to(device)
        #B = batch_data.num_graphs
        B = batch_data.E_inc.view(-1,1).size(0)

        n_act_cs = torch.cat([torch.tensor([0], device=device), batch_data.n_active.cumsum(0)])
        for i in range(B):
            node_mask      = batch_data.trans_x_batch == i
            tx_i           = batch_data.trans_x[node_mask]
            tb_i           = batch_data.trans_batch[node_mask]
            lg_s           = n_act_cs[i].item()
            lg_e           = n_act_cs[i+1].item()
            lids_i         = batch_data.trans_layer_ids[lg_s:lg_e]
            tb_local       = tb_i - tb_i.min()
            layer_per_node = lids_i[tb_local]
            tx_phys        = to_phys_node(tx_i, stats)
            obs_r, E_r     = compute_observables(tx_phys, layer_per_node)
            real_obs_list.append(obs_r)
            E_total_real.append(E_r)
            E_inc_phys_list.append(float(inverse_transform(stats, 'E_inc', batch_data.E_inc[i].item())))

        E_inc_b = batch_data.E_inc.view(-1, 1)
        results = generate_shower(E_inc_b, stats)
 
        for res in results:
            if res['gen_nodes'] is None:
                gen_obs_list.append({})
                E_total_gen.append(0.0)
            else:
                obs_g, _ = compute_observables(
                    res['gen_nodes'], res['gen_layers'])
                gen_obs_list.append(obs_g)
                E_total_gen.append(res['E_total_phys'])
 
        n_processed += B
        if n_processed % 500 == 0:
            print(n_processed, " Done!")
 
print("Generation complete.\n")

def collect_layer_feature(obs_list, layer, feature):
    vals = []
    for obs in obs_list:
        if layer in obs:
            vals.append(obs[layer][feature])
        else:
            vals.append(0.0)
    return np.array(vals)

def plot_hist_ratio(real_vals, gen_vals, xlabel, title, bins=40):
    real_vals = np.array(real_vals)
    gen_vals  = np.array(gen_vals)

    lo  = min(real_vals.min(), gen_vals.min())
    hi  = max(real_vals.max(), gen_vals.max()) + 1e-8
    bns = np.linspace(lo, hi, bins + 1) if isinstance(bins, int) else bins

    hist_r, edges = np.histogram(real_vals, bins=bns)
    hist_g, _     = np.histogram(gen_vals,  bins=bns)
    centers       = 0.5 * (edges[1:] + edges[:-1])

    ratio = np.divide(hist_g, hist_r,
                      out=np.ones_like(hist_g, dtype=float),
                      where=hist_r > 0)

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(7, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    ax.step(edges[:-1], hist_r, where="post", linewidth=2, label="Real")
    ax.step(edges[:-1], hist_g, where="post", linewidth=2, label="Generated")
    ax.set_ylabel("Events")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    rax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    rax.step(centers, ratio, where="mid", linewidth=2)
    rax.set_ylabel("Gen / Real")
    rax.set_xlabel(xlabel)
    rax.set_ylim(0, 2)
    rax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_hist_ratio(
    E_total_real, E_total_gen,
    xlabel="Total shower energy",
    title="Total shower energy — Real vs Generated",
    bins=40
)

'''
E_inc_phys_list = []
for data in test_dataset:
    data = data.to(device)
    E_inc_phys = float(inverse_transform(stats, 'E_inc', data.E_inc.item()))
    E_inc_phys_list.append(E_inc_phys)
we already have a E_inc_phy_list, this repitition is slow and has other issues as well
'''
E_ratio_real = [e / (ei + 1e-12)
                for e, ei in zip(E_total_real, E_inc_phys_list)]
E_ratio_gen  = [e / (ei + 1e-12)
                for e, ei in zip(E_total_gen,  E_inc_phys_list)]

plot_hist_ratio(
    E_ratio_real, E_ratio_gen,
    xlabel="E_reco / E_inc",
    title="Energy Ratio (Reco / Incident)",
    bins=40
)

layers_to_plot = range(30)
features = ['energy', 'x_center', 'y_center', 'x_width', 'y_width', 'nhits']

for layer in layers_to_plot:
    for feature in features:
        real_vals = collect_layer_feature(real_obs_list, layer, feature)
        gen_vals  = collect_layer_feature(gen_obs_list,  layer, feature)

        if real_vals.sum() == 0 and gen_vals.sum() == 0:
            continue
        mask = ~((real_vals == 0) & (gen_vals == 0))
        real_vals = real_vals[mask]
        gen_vals  = gen_vals[mask]

        if len(real_vals) < 5:
            continue

        plot_hist_ratio(
            real_vals, gen_vals,
            xlabel=feature,
            title=f"Layer {layer} — {feature}",
            bins=40
        )

# Longitudinal profiles (energy + nhits)

mean_E_real = []
mean_E_gen  = []

for layer in range(MAX_LAYERS):
    real_e = collect_layer_feature(real_obs_list, layer, 'energy')
    gen_e  = collect_layer_feature(gen_obs_list,  layer, 'energy')
    mean_E_real.append(real_e.mean())
    mean_E_gen.append(gen_e.mean())

fig, (ax, rax) = plt.subplots(
    2, 1, figsize=(10, 6),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True
)
layers = np.arange(MAX_LAYERS)
ax.plot(layers, mean_E_real, 'o-', linewidth=2, label="Real",      markersize=4)
ax.plot(layers, mean_E_gen,  's-', linewidth=2, label="Generated", markersize=4)
ax.set_ylabel("Mean energy per layer")
ax.set_title("Longitudinal energy profile")
ax.legend()
ax.grid(alpha=0.3)

ratio_long = np.divide(
    mean_E_gen, mean_E_real,
    out=np.ones_like(mean_E_gen),
    where=np.array(mean_E_real) > 0
)
rax.axhline(1.0, color='k', linestyle='--', linewidth=1)
rax.plot(layers, ratio_long, 'o-', linewidth=2, markersize=4)
rax.set_ylabel("Gen / Real")
rax.set_xlabel("Layer index")
rax.set_ylim(0, 2)
rax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

mean_n_real = []
mean_n_gen  = []

for layer in range(MAX_LAYERS):
    real_n = collect_layer_feature(real_obs_list, layer, 'nhits')
    gen_n  = collect_layer_feature(gen_obs_list,  layer, 'nhits')
    mean_n_real.append(real_n.mean())
    mean_n_gen.append(gen_n.mean())

fig, (ax, rax) = plt.subplots(
    2, 1, figsize=(10, 6),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True
)
ax.plot(layers, mean_n_real, 'o-', linewidth=2, label="Real",      markersize=4)
ax.plot(layers, mean_n_gen,  's-', linewidth=2, label="Generated", markersize=4)
ax.set_ylabel("Mean nhits per layer")
ax.set_title("Occupancy profile")
ax.legend()
ax.grid(alpha=0.3)

ratio_n = np.divide(
    mean_n_gen, mean_n_real,
    out=np.ones_like(mean_n_gen),
    where=np.array(mean_n_real) > 0
)
rax.axhline(1.0, color='k', linestyle='--', linewidth=1)
rax.plot(layers, ratio_n, 'o-', linewidth=2, markersize=4)
rax.set_ylabel("Gen / Real")
rax.set_xlabel("Layer index")
rax.set_ylim(0, 2)
rax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
