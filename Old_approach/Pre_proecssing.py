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
