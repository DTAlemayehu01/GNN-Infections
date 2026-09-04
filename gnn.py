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

def fix_arc(graph, observers, arc=0.56):
    length = graph.n
    arc_min = int((arc-0.05)*length)
    arc_max = int((arc+0.05)*length)
    arc_len = int(random.uniform(arc_min, arc_max)) 
    while True:
        new_arc = abs(observers[0] - observers[1])
        if new_arc != arc_len and new_arc != (length - arc_len):
            observers = [observers[0], (observers[0] + arc_len)%length]
            return observers
        else:
            return observers

def break_symmetry(graph, observers):
    length = self.n
    while True:
        if abs(observers[0] - observers[1]) == length / 2:
            observers = graph.sample_nodes(2)
        else:
            break
    return observers

def force_end_points(graph, observers):
    graph = graph.graph.graph
    vertex_end_points = {
        x for x in graph.nodes() if graph.degree(x)==1 
    }
    return vertex_end_points

def force_end_point(graph, observers):
    graph = graph.graph.graph
    vertex_end_points = {
        x for x in graph.nodes() if graph.degree(x)==1 
    }
    observers = [random.choice(vertex_end_points)]
    return observers

def get_observer_vector(vertices, times, observers, n):
    t = defaultdict(int)
    times = np.array([times]).flatten()
    observers = list(observers)
    for time, node in zip(times, observers):
        t[node] = time
    return [[t[node]] for node in vertices]

def get_observer_parent(times, observers, parents, n):
    t = defaultdict(lambda: [-1])
    times = np.array([times]).flatten()
    observers = np.array([observers]).flatten()
    for time, node in zip(times, observers):
        node_key = node
        t[node] = [parents[node_key]]
    return [t[node] for node in range(n)]

# Expecting Single SRC
def get_line_expected_infection_times(n):
    exp_time = lambda node, src: abs(node-src)
    infection_times_feature = []
    for src in range(n):
        expected_times = [
            exp_time(node, src) for node in range(n)
        ]
        infection_times_feature.append(expected_times)
    return infection_times_feature

def get_circle_expected_infection_times(n):
    exp_time = lambda x: -abs(x - n/2) + n/2
    infection_times_feature = []
    for src in range(n):
        expected_times = [
            exp_time(src - node) for node in range(n)
        ]
        infection_times_feature.append(expected_times)
    return infection_times_feature
    

def get_source_vector(vertices, test_src, n):
    return [[(node == test_src)] for node in vertices]

def get_edge_features(graph, extra_features=False):
    graph.graph.sim_all()
    
    gdf = graph.graph.get_adjacency(unweighted=False).stack()
    edges = gdf[gdf != np.inf].index
    features = gdf[gdf != np.inf]
    edge_list = [[],[]]
    feature_list = []
    
    vertices_map = graph.graph.enumerated_nodes()
    for start, end in edges:
        u = vertices_map[start]
        v = vertices_map[end]
        if features[(start,end)] > 0:
            edge_list[0].append(u)
            edge_list[1].append(v)
            feature = [features[(start,end)]]
            if extra_features:
                x = np.linspace(0,2,10)
                y = dists.expon.pdf(x)
                feature = np.array(feature)
                feature = np.concat((feature, y))
            
            feature_list.append(feature)

    edge_list = np.array(edge_list)
    feature_list = np.array(feature_list)

    return edge_list, feature_list

def create_datum_object(x, y, edge_list, edge_feature_list):
    data = gnn_data.Data()
    data.x = x
    data.y = torch.tensor(y, dtype=torch.float32)
    data.edge_index = torch.tensor(edge_list, dtype=torch.long)
    data.edge_attr = torch.tensor(edge_feature_list, dtype=torch.float32)
    return data

def make_data(
        srcs, dsts, graph_class, *args,
        extra_edge_features=False, parent_node_feature=False,
        extra_node_features=False, **kwargs
):
    g = graph_class(*args, **kwargs)
    vertices = g.graph.vertices()
    n = len(vertices)
    test_src, observers = g.get_source_observer_pairs(srcs, dsts, **kwargs)
    times = g.simulation_trial(test_src, observers, iters=1, fixed_graph=True)
    x = get_observer_vector(vertices, times, observers, n)
    x = torch.tensor(x, dtype=torch.float32)
    if parent_node_feature:
        xp = torch.tensor(
            get_observer_parent(times, observers, g.graph._parent, n),
            dtype=torch.float32
        )
        x = torch.cat((x, xp), 1)
    if extra_node_features:
        if graph_class == Generators.LineIIDExpGraph:
            xp = torch.tensor(
                get_line_expected_infection_times(n),
                dtype=torch.float32
            )
        else: # Generators.CircleIIDExpGraph
            xp = torch.tensor(
                get_circle_expected_infection_times(n),
                dtype=torch.float32
            )
        x = torch.cat((x, xp), 1)
    y = get_source_vector(vertices, test_src, n)
    edge_list, edge_feature_list = get_edge_features(
        g, extra_features=extra_edge_features
    )
    return create_datum_object(x, y, edge_list, edge_feature_list)

dataset = []
observers = 2 # or 1
# graph_type = Generators.CircleIIDExpGraph
graph_type = Generators.LineIIDExpGraph
# observer_constraint = fix_arc
observer_constraint = force_end_points
# more_features=False
graph_size = 30
iters = 100

end_points = True

for _ in range(iters):
    n = random.randint(20, graph_size)
    x = make_data(
        1, observers, graph_type, n,
        extra_edge_features=True,
        parent_node_feature=False,
        extra_node_features=False,
        observer_constraints=observer_constraint
)
    
    dataset.append(x)

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data = DataLoader(train_data, batch_size=10)
test_data = DataLoader(test_data)

#num_features, num_predictions = dataset[0].x.shape[1], dataset[0].y.shape[1]
#class GCN(torch.nn.Module):
#    def __init__(self):
#        super().__init__()
#        torch.manual_seed(1234)
#        self.conv1 = gnn.GCNConv(num_features, 16)
#        self.conv2 = gnn.GCNConv(16, 16)
#        self.conv3 = gnn.GCNConv(16, num_predictions)
#
#    def forward(self, x, edge_index, edge_weight):
#        x = self.conv1(x, edge_index, edge_weight)
#        x = x.relu()
#        #x = F.dropout(x, p=0.5, training=self.training)
#        x = self.conv2(x, edge_index, edge_weight)
#        x = x.relu()
#        #x = F.dropout(x, p=0.5, training=self.training)
#        x = self.conv3(x, edge_index, edge_weight)
#        return x
#
#model = GCN()
#pos_weight = torch.tensor([20/1])
#criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#y_post_processing = False
num_features, num_predictions = dataset[0].x.shape[1], dataset[0].y.shape[1]
edge_dim = dataset[0].edge_attr.shape[1]
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        # self.conv1 = gnn.GINEConv(num_features, 16)
        # self.conv2 = gnn.GINEConv(16, 16)
        # self.conv3 = gnn.GINEConv(16, num_predictions)
        self.conv1 = gnn.GINEConv(nn.Sequential(nn.Linear(num_features, 16), nn.ReLU(), nn.Linear(16, 16)), edge_dim=edge_dim)
        self.conv2 = gnn.GINEConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16)), edge_dim=edge_dim)
        self.conv3 = gnn.GINEConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, num_predictions)), edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)
        return x

model = GCN()
pos_weight = torch.tensor([20/1])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
y_post_processing = False
#num_features, num_predictions = dataset[0].x.shape[1], dataset[0].y.shape[1]
#class softGCN(torch.nn.Module):
#    def __init__(self):
#        super().__init__()
#        torch.manual_seed(1234)
#        self.conv1 = gnn.GCNConv(num_features, 16)
#        self.conv2 = gnn.GCNConv(16, 16)
#        self.conv3 = gnn.GCNConv(16, num_predictions)
#        
#    def forward(self, x, edge_index, edge_weight):
#        x = self.conv1(x, edge_index, edge_weight)
#        x = x.relu()
#        x = self.conv2(x, edge_index, edge_weight)
#        x = x.relu()
#        x = self.conv3(x, edge_index, edge_weight)
#        # x = gnn_utils.softmax(x, dim=0)
#        return x
#
#model = softGCN()
#pos_weight = torch.tensor([20/1])
#criterion = torch.nn.CrossEntropyLoss()
#y_post_processing = True
#num_features, num_predictions = dataset[0].x.shape[1], dataset[0].y.shape[1]
#class GAT(torch.nn.Module):
#    def __init__(self):
#        super().__init__()
#        torch.manual_seed(1234)
#        self.conv1 = gnn.GATConv(num_features, 16)
#        self.conv2 = gnn.GATConv(16, 16)
#        self.conv3 = gnn.GATConv(16, num_predictions)
#        
#    def forward(self, x, edge_index, edge_weight):
#        x = self.conv1(x, edge_index, edge_weight)
#        x = x.tanh()
#        x = self.conv2(x, edge_index, edge_weight)
#        x = x.tanh()
#        x = self.conv3(x, edge_index, edge_weight)
#        return x
#
#model = GAT()
#pos_weight = torch.tensor([20/1])
#criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#y_post_processing = False

# pos_weight = torch.tensor([len(y)/sum(y)])
# pos_weight = torch.tensor([20/1])
# criterion = torch.nn.BCELoss()
learning_rate=0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

def train(model, data):
      optimizer.zero_grad() 
      # print(data.x)
      # print(data.y)
      # print(data.edge_attr)
      # print(data)
      out = model(
            data.x, data.edge_index, data.edge_attr.squeeze()
            # data.x, data.edge_index, data.edge_attr.squeeze(-2)
      )  
      if y_post_processing:
            loss = criterion(out.squeeze(-1), data.y.squeeze(-1).argmax())
      else:
            loss = criterion(out, data.y)  
      loss.backward()  
      optimizer.step() 
      return loss

def test(model, dataset):
      all_preds = []
      all_truths = []
      for data in dataset:
          preds = model(
                data.x, data.edge_index, data.edge_attr.squeeze()
          )
          print(f"Data: {data.x[:,:1].flatten()}")
          if y_post_processing:
                preds = torch.softmax(preds, dim=0)
                print(f"Prob: {preds.flatten().data}")
          else:
                preds = torch.sigmoid(preds)
                print(f"Conf: {preds.flatten().data}")
          if y_post_processing:
                pred = preds.argmax().squeeze()
                preds = torch.zeros(preds.shape)
                preds[pred] = 1
          else:
                preds = (preds > 0.5).float()
          print(f"True: {data.y.flatten()}")
          print(f"Pred: {preds.flatten().data}")
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
    print(f'Epoch: {epoch:03d}, Loss: {epoch_loss:.4f}')

model.eval()
err = test(model, test_data)
print(f'Accuracy: {err[0]:.4f}, Precision: {err[1]:.4f}, Recall: {err[2]:.4f}, F1: {err[3]:.4f}')
