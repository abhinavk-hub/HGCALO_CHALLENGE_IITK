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
from torch_geometric.utils import to_undirected

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_files = glob.glob("/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s/HGCal_showers*.h5")
def numerical_sort(name):
    return int(re.findall(r"(\d+)\.h5$", name)[0])
all_files = sorted(all_files, key=numerical_sort)
#selected_files = [all_files[300]]
selected_files = all_files[0:9]

GEOM_PKL = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePhoton_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/h5s/HGCal_geo_2024.pkl"

BATCH_SIZE = 3
BASE_OUT_DIR = "/eos/user/a/abkumar/a_freshlook_at_HGCALO"

with open(GEOM_PKL, "rb") as f:
    geom = pkl.load(f)
nlayers = getattr(geom, "nlayers", None)
max_cells = getattr(geom, "max_cells", None)
xmap = getattr(geom, "xmap", None)
ymap = getattr(geom, "ymap", None)

MAX_LAYERS = 47
EPS        = 1e-8

def build_long_edge(num_layers = MAX_LAYERS):
    edges = []
    for i in range(num_layers - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
long_edge_index = build_long_edge()

def build_long_node(shower_energy):
    node_features = []
    active_l = []
    NL, NV = shower_energy.shape
    for L in range(NL):
        E_l = 0
        Nhit_l = 0
        for V in range(NV):
            e = shower_energy[L,V]
            if e > EPS:
                Nhit_l += 1
                E_l += e
        node_features.append([E_l, Nhit_l])
        if Nhit_l == 0:
            active_l.append(0)
        else:
            active_l.append(1)
    while len(node_features) < MAX_LAYERS:
        node_features.append([0.0, 0.0])
        active_l.append(0)
    n_f = torch.tensor(node_features, dtype=torch.float)
    return n_f, active_l

def build_intra_edges(xs, ys, k=4):
    N = len(xs)
    if N <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
 
    coords = np.stack([xs, ys], axis=1)   
    k_eff  = min(k + 1, N)          
 
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm='auto').fit(coords)
    _, indices = nbrs.kneighbors(coords)
 
    src_list, dst_list = [], []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:                    
                src_list.append(i)
                dst_list.append(j)
 
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long)
 
    ei = torch.tensor([src_list, dst_list], dtype=torch.long)
    return ei

def make_graph(shower_energy, geninfo):
    
    E_inc  = float(geninfo[0])
    eta    = float(geninfo[1])
    phi    = float(geninfo[2])
    
    NL, NV = shower_energy.shape
    E_total = 0 
    for L in range(NL):
        for V in range(NV):
            E = shower_energy[L, V]
            if E > EPS:
                E_total += E
    if E_total <= 0:
        return None
    
    long_node, active_l = build_long_node(shower_energy)
    
    #making tranverse graph
    all_trans_x        = []
    all_trans_edge     = []   
    all_trans_batch    = []   
    all_layer_ids      = []   
    all_E_l            = []  
    all_nhits          = []
    
    node_offset      = 0
    layer_graph_idx  = 0 
    
    NL, NV = shower_energy.shape
    for L in range(NL):
        if active_l[L] == 0:
            continue
        E_l = 0
        Nhit_l = 0
        node_f =[]
        xs =[]
        ys =[]
        for V in range(NV):
            e = shower_energy[L,V]
            if e > EPS:
                Nhit_l += 1
                x = xmap[L][V]
                y = ymap[L][V]
                r   = np.sqrt(x**2 + y**2)
                phi_cell = np.arctan2(y, x)
                node_f.append([e,x,y,r,phi_cell])
                xs.append(x)
                ys.append(y)
                E_l += e
        all_trans_x.append(torch.tensor(node_f, dtype=torch.float32))
        ei = build_intra_edges(xs, ys, k=4)
        ei = to_undirected(ei)  
        if ei.shape[1] > 0:
            ei = ei + node_offset
        all_trans_edge.append(ei)
        all_trans_batch.append(torch.full((Nhit_l,), layer_graph_idx, dtype=torch.long))
        all_layer_ids.append(L)
        all_E_l.append(E_l)
        all_nhits.append(Nhit_l)

        node_offset     += Nhit_l
        layer_graph_idx += 1

    data = Data(
        E_inc    = torch.tensor(E_inc,   dtype=torch.float32),
        eta      = torch.tensor(eta,     dtype=torch.float32),
        phi      = torch.tensor(phi,     dtype=torch.float32),
        E_total  = torch.tensor(E_total, dtype=torch.float32),
        n_active = torch.tensor(layer_graph_idx, dtype=torch.long),

        long_x          = long_node,     
        long_edge_index = long_edge_index.clone(),   
        active          = torch.tensor(active_l, dtype=torch.bool),

        trans_x         = torch.cat(all_trans_x,   dim=0),  
        trans_edge_index= torch.cat(all_trans_edge, dim=1),  
        trans_batch     = torch.cat(all_trans_batch,dim=0),   
        trans_layer_ids = torch.tensor(all_layer_ids, dtype=torch.long),  
        trans_E_l       = torch.tensor(all_E_l,       dtype=torch.float32),
        trans_nhits     = torch.tensor(all_nhits,      dtype=torch.long), 
    )

    return data

for batch_idx in range(0, len(selected_files), BATCH_SIZE):
    batch_files = selected_files[batch_idx:batch_idx + BATCH_SIZE]

    # Create a separate folder for this batch
    batch_number = (batch_idx // BATCH_SIZE)
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
