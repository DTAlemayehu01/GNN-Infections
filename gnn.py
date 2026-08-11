#!./.venv/bin/python

import importlib
from infection_propagation import Graph
from infection_propagation import Figures
from infection_propagation import Generators

importlib.reload(Graph)
importlib.reload(Figures)
importlib.reload(Generators)

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as dists
import scipy.spatial as sp
import networkx as nx
import pandas as pd

from itertools import product, combinations
from collections import defaultdict
import random
import json
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset # DataLoader, TensorDataset

import torch_geometric.nn as gnn
import torch_geometric.data as gnn_data
import torch_geometric.utils as gnn_utils
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_observer_vector(times, observers, n):
    t = defaultdict(int)
    times = np.array([times]).flatten()
    observers = np.array([observers]).flatten()
    for time, node in zip(times, observers):
        t[node] = time
    return [[t[i]] for i in range(n)]

def get_source_vector(test_src, n):
    return [[(i == test_src)] for i in range(n)]

def get_edge_features(graph):
    graph.graph.sim_all()
    
    gdf = graph.graph._adjency_matrix.stack()
    edges = gdf[gdf != np.inf].index
    features = gdf[gdf != np.inf]
    edge_list = [[],[]]
    feature_list = []
    
    for u, v in edges:
        edge_list[0].append(u)
        edge_list[1].append(v)
        feature_list.append([features[(u,v)]])

    return edge_list, feature_list

def create_datum_object(x, y, edge_list, edge_feature_list):
    data = gnn_data.Data()
    data.x = torch.tensor(x, dtype=torch.float32)
    data.y = torch.tensor(y, dtype=torch.float32)
    data.edge_index = torch.tensor(edge_list, dtype=torch.long)
    data.edge_attr = torch.tensor(edge_feature_list, dtype=torch.float32)
    return data

def make_data(srcs, dsts, graph_class, *args, **kwargs):
    g = graph_class(*args, **kwargs)
    n = len(g.graph.vertices())
    test_src, observers = g.get_source_observer_pairs(srcs, dsts, **kwargs)
    times = g.simulation_trial(test_src, observers, iters=1, fixed_graph=True)
    x = get_observer_vector(times, observers, n)
    y = get_source_vector(test_src, n)
    edge_list, edge_feature_list = get_edge_features(g)
    return create_datum_object(x, y, edge_list, edge_feature_list)

dataset = []
observers = 2 # or 1
graph_type = Generators.CircleIIDExpGraph
graph_size = 20
iters = 100

end_points = True
break_symmetry = True

for _ in range(iters):
    x = make_data(1, observers, graph_type, graph_size, end_points=end_points, break_symmetry=break_symmetry)
    dataset.append(x)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = gnn.GCNConv(1, 16)
        self.conv2 = gnn.GCNConv(16, 16)
        self.conv3 = gnn.GCNConv(16, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.sigmoid()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.sigmoid()
        x = self.conv3(x, edge_index, edge_weight)
        return x

ratio = 0.8
n = int(len(dataset)*0.8)
train_data = DataLoader(dataset[:n], batch_size=10)
test_data = DataLoader(dataset[n:])

model = GCN()

# pos_weight = torch.tensor([len(y)/sum(y)])
pos_weight = torch.tensor([20/1])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(model, data):
      optimizer.zero_grad() 
      out = model(
            data.x, data.edge_index, data.edge_attr.squeeze()
      )  
      loss = criterion(out, data.y)  
      loss.backward()  
      optimizer.step() 
      return loss

def test(model, dataset):
      all_preds = []
      all_truths = []
      for data in dataset:
          out = model(
                data.x, data.edge_index, data.edge_attr.squeeze()
          )
          preds = torch.sigmoid(out)
          print(preds.flatten().data)
          preds = (preds > 0.5).float()
          # print(preds)
          all_preds.append(preds.squeeze())
          all_truths.append(data.y.flatten())
      preds = torch.cat(all_preds)
      truths = torch.cat(all_truths)

      tp = ((preds == 1) & (truths == 1)).sum().float()
      tn = ((preds == 0) & (truths == 0)).sum().float()
      fp = ((preds == 1) & (truths == 0)).sum().float()
      fn = ((preds == 0) & (truths == 1)).sum().float()

      accuracy = (tp + tn) / (tp + tn + fp + fn)
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = 2 * precision * recall / (precision + recall)
      return accuracy, precision, recall, f1

model.train()
for epoch in range(1000):
    epoch_loss = 0
    print("Individual Losses: ", end="")
    for data in train_data:
        loss = train(model, data)
        epoch_loss = epoch_loss + loss
        print(f"{loss.item():.4f}", end=", ")
    print("")
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

model.eval()
err = test(model, test_data)
print(f'Accuracy: {err[0]:.4f}, Precision: {err[1]:.4f}, Recall: {err[2]:.4f}, F1: {err[3]:.4f}')
