import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

# GCN
class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        self.layers.append(pyg_nn.GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))
        self.layers.append(pyg_nn.GCNConv(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in range(self.num_layers - 1):
            x = F.relu(self.layers[layer](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

# GAT
class GAT(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout, heads):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        self.layers.append(pyg_nn.GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(pyg_nn.GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.layers.append(pyg_nn.GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in range(self.num_layers - 1):
            x = F.elu(self.layers[layer](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

# GraphSAGE
class GraphSAGE(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        self.layers.append(pyg_nn.SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(pyg_nn.SAGEConv(hidden_dim, hidden_dim))
        self.layers.append(pyg_nn.SAGEConv(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in range(self.num_layers - 1):
            x = F.relu(self.layers[layer](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

# GIN
class GIN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers
        self.mlp1 = nn.ModuleList()

        self.mlp1.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        ))
        
        for _ in range(num_layers - 2):
            self.mlp1.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ))
        
        self.mlp1.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in range(self.num_layers - 1):
            x = self.mlp1[layer](x)
        x = self.mlp1[-1](x)
        return x
