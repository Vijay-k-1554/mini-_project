import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- KNN Graph Construction ----------
def build_knn_adjacency(x, k=5):
    """
    x: (B, N, D)
    returns: (B, N, N)
    """
    B, N, D = x.shape

    # Normalize features for cosine similarity
    x_norm = F.normalize(x, p=2, dim=-1)

    # Cosine similarity matrix
    sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, N, N)

    # Remove self-similarity
    mask = torch.eye(N, device=x.device).unsqueeze(0)
    sim = sim.masked_fill(mask.bool(), -1)

    # Get top-k neighbors
    knn_vals, knn_idx = torch.topk(sim, k=k, dim=-1)

    # Build adjacency matrix
    adj = torch.zeros_like(sim)

    for b in range(B):
        for i in range(N):
            adj[b, i, knn_idx[b, i]] = 1.0

    # Add self-loops
    adj = adj + torch.eye(N, device=x.device).unsqueeze(0)

    return adj


# ---------- GCN Layer ----------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: (B, N, D)
        adj: (B, N, N)
        """
        # Degree matrix
        deg = adj.sum(dim=-1, keepdim=True)

        # Safe inverse
        deg_inv = deg.clamp(min=1.0).pow(-1)

        # Normalize adjacency
        adj_norm = adj * deg_inv

        # Message passing
        out = torch.bmm(adj_norm, x)

        # Linear transform
        out = self.linear(out)

        return out


# ---------- GNN Model ----------
class GNNModel(nn.Module):
    """
    GNN for relational reasoning on ViT tokens
    """

    def __init__(self, in_dim=128, hidden_dim=128, num_layers=2, k=5, pool="none"):
        super().__init__()

        if pool not in ("none", "mean"):
            raise ValueError('pool must be "none" or "mean"')

        self.k = k
        self.pool = pool

        layers = []
        for i in range(num_layers):
            layers.append(
                GCNLayer(
                    in_dim if i == 0 else hidden_dim,
                    hidden_dim
                )
            )

        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        x: (B, N, D)
        returns:
          - (B, N, hidden_dim) if pool="none"
          - (B, hidden_dim) if pool="mean"
        """

        # ---------- Build graph ----------
        adj = build_knn_adjacency(x, k=self.k)

        # ---------- GNN layers ----------
        for layer in self.layers:
            x = x + self.activation(layer(x, adj))  # residual connection

        # ---------- Pooling ----------
        if self.pool == "mean":
            return x.mean(dim=1)

        return x
