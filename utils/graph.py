import torch
import torch.nn.functional as F


def build_knn_adjacency(x, k=5):
    """
    x: (B, N, D)
    returns: (B, N, N)
    """
    B, N, D = x.shape

    # Normalize features
    x_norm = F.normalize(x, p=2, dim=-1)

    # Cosine similarity
    sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, N, N)

    # Remove self similarity
    mask = torch.eye(N, device=x.device).unsqueeze(0)
    sim = sim.masked_fill(mask.bool(), -1)

    # Top-k neighbors
    knn_vals, knn_idx = torch.topk(sim, k=k, dim=-1)

    # Build adjacency matrix
    adj = torch.zeros_like(sim)

    for b in range(B):
        for i in range(N):
            adj[b, i, knn_idx[b, i]] = 1.0

    # Add self loops
    adj = adj + torch.eye(N, device=x.device).unsqueeze(0)

    return adj