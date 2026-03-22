#!/usr/bin/env python3
"""
Generate architecture diagrams for DNN, GCN, and GAT models using torchviz.
Output: figures/dnn_architecture.*, figures/gcn_architecture.*, figures/gat_architecture.*
Requires: torchviz, graphviz (brew install graphviz)
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure graphviz 'dot' is on PATH (e.g. from brew install graphviz)
if sys.platform == 'darwin':
    brew_bin = '/opt/homebrew/bin'
    if os.path.exists(brew_bin) and brew_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = brew_bin + os.pathsep + os.environ.get('PATH', '')

from torchviz import make_dot

os.makedirs('figures', exist_ok=True)

# Optional: GNN models (skip if torch_geometric not installed)
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, BatchNorm
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


# -----------------------------------------------------------------------------
# Model definitions (match 03_neural_network_classification.ipynb and 04_graph_neural_network_classification.ipynb)
# -----------------------------------------------------------------------------

class NNClassifier(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hd
        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class GCNClassifier(nn.Module):
    def __init__(self, node_features=4, global_features=10, hidden_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        combined_dim = hidden_dim * 2 + global_features
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        global_features = data.u.view(-1, 10)
        combined = torch.cat([x_mean, x_max, global_features], dim=1)
        return self.classifier(combined).squeeze()


class GATClassifier(nn.Module):
    def __init__(self, node_features=4, global_features=10, hidden_dim=32, num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(node_features, hidden_dim, heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_dim * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_dim * heads))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))
        self.bns.append(BatchNorm(hidden_dim))
        combined_dim = hidden_dim * 2 + global_features
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        global_features = data.u.view(-1, 10)
        combined = torch.cat([x_mean, x_max, global_features], dim=1)
        return self.classifier(combined).squeeze()


def make_dnn_diagram():
    """DNN: Input(32) → Linear(128) → ... → Sigmoid"""
    model = NNClassifier(input_dim=32, hidden_dims=[128, 64, 32], dropout=0.3)
    model.eval()
    x = torch.randn(2, 32)  # batch of 2 (BatchNorm needs >1 in eval, or use 2 for safety)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    dot.render('figures/dnn_architecture', format='png', cleanup=True)
    dot.render('figures/dnn_architecture', format='pdf', cleanup=True)
    print("Saved figures/dnn_architecture.png and .pdf")


def make_gnn_dummy_data():
    """Create minimal graph batch for GNN forward pass (1 graph, 5 nodes)."""
    node_features = torch.randn(5, 4)
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
    ], dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)
    global_features = torch.randn(1, 10)
    return Data(x=node_features, edge_index=edge_index, batch=batch, u=global_features)


def make_gcn_diagram():
    """GCN: 3× GCNConv(64) → Pool → MLP → Sigmoid"""
    if not HAS_TORCH_GEOMETRIC:
        print("Skipping GCN (torch_geometric not installed)")
        return
    model = GCNClassifier(node_features=4, global_features=10, hidden_dim=64, num_layers=3, dropout=0.3)
    model.eval()
    data = make_gnn_dummy_data()
    y = model(data)
    dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    dot.render('figures/gcn_architecture', format='png', cleanup=True)
    dot.render('figures/gcn_architecture', format='pdf', cleanup=True)
    print("Saved figures/gcn_architecture.png and .pdf")


def make_gat_diagram():
    """GAT: 2× GATConv(32,4h) + GATConv(32,1h) → Pool → MLP → Sigmoid"""
    if not HAS_TORCH_GEOMETRIC:
        print("Skipping GAT (torch_geometric not installed)")
        return
    model = GATClassifier(node_features=4, global_features=10, hidden_dim=32, num_layers=3, heads=4, dropout=0.3)
    model.eval()
    data = make_gnn_dummy_data()
    y = model(data)
    dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    dot.render('figures/gat_architecture', format='png', cleanup=True)
    dot.render('figures/gat_architecture', format='pdf', cleanup=True)
    print("Saved figures/gat_architecture.png and .pdf")


if __name__ == '__main__':
    try:
        torch.manual_seed(42)
        make_dnn_diagram()
        if HAS_TORCH_GEOMETRIC:
            make_gcn_diagram()
            make_gat_diagram()
        print("\nDone. Diagrams in figures/")
    except Exception as e:
        if 'dot' in str(e).lower() or 'graphviz' in str(e).lower():
            print("Error: graphviz 'dot' not found. Install with: brew install graphviz", file=sys.stderr)
        raise
