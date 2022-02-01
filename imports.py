import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.nn.dense.dense_gcn_conv import DenseGCNConv
 
from sklearn.metrics import auc, roc_curve

from parameters import *
