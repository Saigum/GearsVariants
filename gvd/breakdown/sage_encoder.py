import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SAGEEncoder(nn.Module):
    """
    Small GraphSAGE node encoder.
    """
    def __init__(self, x_dim, hidden, out, layers=2, dropout=0.0, normalize=False):
        super().__init__()
        dims = [x_dim] + [hidden]*(layers-1) + [out]
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.dropout = dropout
        self.normalize = normalize
        self.bns = nn.ModuleList([nn.BatchNorm1d(d) for d in dims[1:-1]]) if layers > 1 else None
        print(f"Expected Dimensions of activations through encoder: {dims}")
    def forward(self, x, edge_index):
        h = x
        L = len(self.convs)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < L - 1:
                h = F.relu(h)
                if self.bns is not None:
                    h = self.bns[i](h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1,eps=1e-8)
        return h