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

def numerical_sort(name):
    return int(re.findall(r"(\d+)\.h5$", name)[0])

all_files = sorted(all_files, key=numerical_sort)

#selected_files = [all_files[200]]
selected_files = all_files[: 3]
#print(selected_files)
       
GEOM_PKL = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s/HGCal_geo_2024.pkl"

OUT_DIR = "/eos/user/a/abkumar/gnn_train/"


with open(GEOM_PKL, "rb") as f:
    geom = pkl.load(f)

nlayers = getattr(geom, "nlayers", None)
max_cells = getattr(geom, "max_cells", None)
xmap = getattr(geom, "xmap", None)
ymap = getattr(geom, "ymap", None)

def make_graph(shower_energy, geninfo, geom):
    """
    shower_energy: [NLayers, NVox] energy array
    geninfo: [3] = (E_gen, eta, phi)
    """

    node_features = []
    coordinates = []
    layer_ids = []
    
    NL, NV = shower_energy.shape

    # --- Build nodes ---
    for L in range(NL):
        for V in range(NV):
            E = shower_energy[L, V]
            x = xmap[L][V]
            y = ymap[L][V]
            z = L
            if E <= 0:
                continue

            # Node features: x,y,z,layer,E
            node_features.append([x, y, L, E])
            coordinates.append([x, y, z])
            layer_ids.append(L)

    node_features = np.array(node_features, dtype=np.float32)
    coordinates = np.array(coordinates, dtype=np.float32)
    layer_ids = np.array(layer_ids, dtype=np.int32)

    if len(node_features) == 0:
        return None  # empty shower

    # --- Build edges using kNN ---
    #N = len(node_features)
    #coords = torch.tensor(coordinates, dtype=torch.float, device='cuda')  # (N,3)

    # All-pair distances (massively parallel)
    #dist_matrix = torch.cdist(coords, coords)

    # Get kNN (skip self-node)
    #k = min(KNN_K + 1, N)
    #_, knn_idx = torch.topk(-dist_matrix, k=k, dim=1)

    # move back to CPU for Python processing
    #knn_idx = knn_idx[:, 1:].cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=min(8, len(node_features)),
                            algorithm='kd_tree').fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)
    senders = []
    receivers = []

    for i in range(len(node_features)):
        for j in indices[i]:
            if i == j:
                continue
            senders.append(i)
            receivers.append(j)

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)

    # --- Compute edge features ---
    edge_attr_list = []
    coords = torch.tensor(coordinates, dtype=torch.float)
    layer_tensor = torch.tensor(layer_ids, dtype=torch.float)

    for s, r in zip(senders, receivers):
        dx = coords[r, 0] - coords[s, 0]
        dy = coords[r, 1] - coords[s, 1]
        dz = coords[r, 2] - coords[s, 2]
        dist = torch.sqrt(dx*dx + dy*dy + dz*dz)
        dlayer = layer_tensor[r] - layer_tensor[s]

        edge_attr_list.append([dx.item(), dy.item(), dist.item(), dlayer.item()])

    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # Build PyG graph
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(geninfo, dtype=torch.float)
    )

    return data
counter = 0
for path in selected_files:
    with h5py.File(path, "r") as f:
        showers = f["showers"][:]
        gen_info = f["gen_info"][:]
    N = len(showers)
    for i in range(N):
        graph = make_graph(showers[i], gen_info[i], geom)
        if graph is None:
            continue
        out_name = f"graph_{counter}.pt"
        torch.save(graph, os.path.join(OUT_DIR, out_name))
        counter += 1
print("Done!")
print(f"Total graphs stored: {counter}")
