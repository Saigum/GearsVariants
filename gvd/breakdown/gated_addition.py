import torch
from torch import nn

class Gated_Addition(nn.Module):
    def __init__(self,vector_dim:int):
        super().__init__()
        self.gating = nn.Linear(2*vector_dim,vector_dim)
    def forward(self,x1,x2):
        g = self.gating(torch.concat((x1,x2),dim=-1))
        return g*x1 + (1-g)*x2