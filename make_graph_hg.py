import numpy as np
import h5py
import glob
import re
import torch
from torch_geometric.data import Data
import os
import pickle as pkl
from HGCalGeo import *
from sklearn.neighbors import NearestNeighbors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_files = glob.glob("/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s/HGCal_showers*.h5")
#all_files = glob.glob("/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePion_E-5_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s/HGCal_showers*.h5")

def numerical_sort(name):
    return int(re.findall(r"(\d+)\.h5$", name)[0])

all_files = sorted(all_files, key=numerical_sort)

#selected_files = [all_files[300]]
selected_files = all_files[216:245]
#print(selected_files)
       
#GEOM_PKL = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s/HGCal_geo_2024_large.pkl"
GEOM_PKL = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s/HGCal_geo_2024.pkl"

#OUT_DIR = "/eos/user/a/abkumar/train_graphs_pions/"
BATCH_SIZE = 3
BASE_OUT_DIR = "/eos/user/a/abkumar/"



with open(GEOM_PKL, "rb") as f:
    geom = pkl.load(f)

nlayers = getattr(geom, "nlayers", None)
max_cells = getattr(geom, "max_cells", None)
xmap = getattr(geom, "xmap", None)
ymap = getattr(geom, "ymap", None)

def build_nodes(shower_energy):
    node_features = []
    node_layer = []

    NL, NV = shower_energy.shape

    for L in range(NL):
        for V in range(NV):
            E = shower_energy[L, V]
            if E <= 0:
                continue

            x = xmap[L][V]
            y = ymap[L][V]

            node_features.append([x, y, L, E])
            node_layer.append(L)
    
    if len(node_features) == 0:
        return torch.empty((0, 4), dtype=torch.float), []

    x=torch.tensor(node_features, dtype=torch.float)
    return x, node_layer

def knn_edges(src_idx, dst_idx, coords, k):
    if len(dst_idx) == 0:
        return []

    k_eff = min(k, len(dst_idx))

    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(coords[dst_idx])
    _, indices = nbrs.kneighbors(coords[src_idx])

    edges = []
    for i, neigh in enumerate(indices):
        for j in neigh:
            edges.append([src_idx[i], dst_idx[j]])

    return edges

def build_edges(x, node_layer, k_intra=4, k_inter=4):
    coords = x[:, :2].numpy()
    edges = []

    node_layer = np.array(node_layer)
    layers = sorted(set(node_layer))

    nodes_by_layer = {
        L: np.where(node_layer == L)[0]
        for L in layers
    }

    for L in layers:
        idx_L = nodes_by_layer[L]

        # same layer
        edges += knn_edges(idx_L, idx_L, coords, k_intra)

        # layer above
        if L + 1 in nodes_by_layer:
            idx_up = nodes_by_layer[L + 1]
            edges += knn_edges(idx_L, idx_up, coords, k_inter)

        # layer below
        if L - 1 in nodes_by_layer:
            idx_dn = nodes_by_layer[L - 1]
            edges += knn_edges(idx_L, idx_dn, coords, k_inter)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def make_graph(shower_energy, geninfo):
    x, node_layer = build_nodes(shower_energy)
    if x.shape[0] == 0:
        return None 
    edge_index = build_edges(x, node_layer)

    data = Data(
        x=x,                 # [N_nodes, 4]
        edge_index=edge_index,
        y=torch.tensor(geninfo, dtype=torch.float)
    )

    return data

for batch_idx in range(0, len(selected_files), BATCH_SIZE):
    batch_files = selected_files[batch_idx:batch_idx + BATCH_SIZE]

    # Create a separate folder for this batch
    batch_number = (batch_idx // BATCH_SIZE) + 35
    batch_out_dir = os.path.join(BASE_OUT_DIR, f"train_graphs_{batch_number}")
    os.makedirs(batch_out_dir, exist_ok=True)

    print(f"\nProcessing batch {batch_number}")
    print(batch_files)
    counter = 0
    for path in batch_files:
        with h5py.File(path, "r") as f:
            showers = f["showers"][:]
            gen_info = f["gen_info"][:]
        N = len(showers)
        for i in range(N):
            graph = make_graph(showers[i], gen_info[i])
            if graph is None:
                continue
            out_name = f"graph_{counter}.pt"
            torch.save(graph, os.path.join(batch_out_dir, out_name))
            counter += 1
    print("Done!")
    print(f"Total graphs stored: {counter}")
