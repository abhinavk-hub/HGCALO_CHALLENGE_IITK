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
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import GATConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import global_add_pool
import math
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LAYERS = 47
