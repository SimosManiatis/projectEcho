# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv

from config import IN_FEATS, HIDDEN_FEATS, OUT_FEATS, DROPOUT_RATE, INPUT_FEATURE_NAMES
from config import logger

class NodeNorm(nn.Module):
    """Normalize each node's feature vector to zero mean, unit variance."""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)

class GraphRegressor(nn.Module):
    """
    GNN with:
      1) Node-wise MLP pre-embedding
      2) GAT attention layers
      3) NodeNorm to avoid over-smoothing
      4) Hard monotonic-decreasing skip on DoorCount, WindowCount, OpeningArea
    """
    def __init__(self,
                 in_feats: int = IN_FEATS,
                 hidden_feats: int = HIDDEN_FEATS,
                 out_feats: int = OUT_FEATS,
                 dropout_rate: float = DROPOUT_RATE,
                 num_heads:    int = 8):
        super().__init__()

        # 1) Pre-embed with MLP
        self.input_mlp = nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, hidden_feats)
        )

        # 2) Attention layers
        per_head = hidden_feats // num_heads
        self.gat1 = GATConv(hidden_feats, per_head, num_heads, allow_zero_in_degree=True)
        self.gat2 = GATConv(hidden_feats, per_head, num_heads, allow_zero_in_degree=True)
        self.gat3 = GATConv(hidden_feats, per_head, num_heads, allow_zero_in_degree=True)

        # 3) NodeNorm after each layer + skip + output
        self.norm1   = NodeNorm()
        self.norm2   = NodeNorm()
        self.norm3   = NodeNorm()
        self.fc_skip = nn.Linear(hidden_feats, hidden_feats)
        self.fc_out  = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(p=dropout_rate)

        # 4) Monotonic-decreasing skip weights for [DoorCount, WindowCount, OpeningArea]
        #    raw parameters, to be constrained ≤ 0 via -softplus()
        self.neg_raw = nn.Parameter(torch.zeros(3))

        logger.info("Initialized GraphRegressor with hard monotonic-decreasing skip")

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        # Pre-embed each node
        h0 = self.input_mlp(x)

        # Layer 1
        h1 = self.gat1(g, h0).flatten(1)
        h1 = self.norm1(h1); h1 = F.relu(h1); h1 = self.dropout(h1)
        # Layer 2
        h2 = self.gat2(g, h1).flatten(1)
        h2 = self.norm2(h2 + h1); h2 = F.relu(h2); h2 = self.dropout(h2)
        # Layer 3
        h3 = self.gat3(g, h2).flatten(1)
        h3 = self.norm3(h3 + h2); h3 = F.relu(h3); h3 = self.dropout(h3)

        # Skip from pre-embedded
        skip = F.relu(self.fc_skip(h0))
        out  = self.fc_out(h3 + skip)  # [N,2]

        # Hard monotonic skip on raw features
        idxs      = [INPUT_FEATURE_NAMES.index(f) for f in ("DoorCount", "WindowCount", "OpeningArea")]
        neg_feats = x[:, idxs]                      # [N,3]
        neg_w     = -F.softplus(self.neg_raw)       # ensure weights ≤ 0
        extra     = (neg_feats * neg_w).sum(dim=1, keepdim=True)  # [N,1]
        out[:,1:2] = out[:,1:2] + extra              # adjust only ObjectVolume

        return out
