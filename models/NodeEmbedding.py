""" import os, sys

import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imports import *
 """

import torch_geometric
""" class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)


model = GCN(hidden_channels=16)

print(model)
 """
