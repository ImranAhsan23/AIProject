import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: node features (N x F), adj: adjacency matrix (N x N)
        h = torch.mm(adj, x)  # Aggregate neighbors
        h = self.linear(h)    # Apply linear transformation
        return h


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # Add GCN layers
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers.append(GCNLayer(hidden_dim, output_dim))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj)
        return F.log_softmax(x, dim=1)


class GraphSageLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSageLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # Aggregate neighbors' features
        neighbor_agg = torch.mm(adj, x)
        # Combine with self node features
        h = torch.cat([x, neighbor_agg], dim=1)
        h = self.linear(h)  # Apply transformation
        return h


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        self.layers.append(GraphSageLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GraphSageLayer(hidden_dim, hidden_dim))
        self.layers.append(GraphSageLayer(hidden_dim, output_dim))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj)
        return F.log_softmax(x, dim=1)


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads):
        super(GATLayer, self).__init__()
        self.heads = heads
        self.linear = nn.Linear(in_features, out_features * heads)
        self.attention = nn.Parameter(torch.Tensor(heads, out_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x, adj):
        h = self.linear(x)  # Apply linear transformation
        h = h.view(h.size(0), self.heads, -1)  # (N, H, F)
        attn_scores = torch.matmul(h, self.attention)  # Attention scores
        attn_scores = attn_scores * adj.unsqueeze(1)  # Mask with adjacency
        attn_weights = F.softmax(attn_scores, dim=-1)  # Normalize
        h = torch.einsum('nhf,nhf->nf', attn_weights, h)  # Weighted sum
        return h


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, heads, dropout):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        self.layers.append(GATLayer(input_dim, hidden_dim, heads))
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_dim * heads, hidden_dim, heads))
        self.layers.append(GATLayer(hidden_dim * heads, output_dim, 1))  # Single head output

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj)
        return F.log_softmax(x, dim=1)


class GINLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj):
        neighbor_agg = torch.mm(adj, x)
        h = (1 + self.eps) * x + neighbor_agg
        return self.mlp(h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        self.layers.append(GINLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GINLayer(hidden_dim, hidden_dim))
        self.layers.append(GINLayer(hidden_dim, output_dim))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj)
        return F.log_softmax(x, dim=1)
