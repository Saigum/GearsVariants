import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce
from sage_encoder import SAGEEncoder # Assuming sage_encoder.py is in the same directory

class AttentionalPerturber(nn.Module):
    """
    Predicts edge deltas using a GAT-like additive attention mechanism.

      h = SAGE(x, edge_index)
      h_z = concat(h, z_broadcasted_to_nodes)
      
      score_u = AttnLinear_Left(h_z)
      score_v = AttnLinear_Right(h_z)
      
      Δ_{uv} = LeakyReLU(score_u[u] + score_v[v])
      w' = softplus( w + λ * Δ )
    """
    def __init__(self,
                 x_dim: int,
                 z_dim: int,
                 enc_hidden: int = 128,
                 enc_out: int = 128,
                 enc_layers: int = 2,
                 enc_dropout: float = 0.0,
                 enc_l2norm: bool = False,
                 nonneg: bool = True,
                 symmetric: bool = True,
                 lambda_init: float = 0.5):
        super().__init__()
        self.encoder = SAGEEncoder(x_dim, enc_hidden, enc_out,
                                   layers=enc_layers,
                                   dropout=enc_dropout,
                                   normalize=enc_l2norm)

        # Input to attention heads is node embedding + global z
        attn_in_dim = enc_out + z_dim
        
        # We use separate linear layers for source (left) and target (right)
        # This allows for asymmetric interactions (e.g., GATv2 style)
        self.attn_l = nn.Linear(attn_in_dim, 1, bias=False)
        self.attn_r = nn.Linear(attn_in_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.nonneg = nonneg
        self.symmetric = symmetric
        self.lambda_scale = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        
        # Note: We ignore the original weight `w` in this formulation,
        # but you could easily add it to `attn_in_dim` if needed
        # by first creating edge features. This is the simpler node-centric version.

    @staticmethod
    def _symmetrize(edge_index, w, reduce: str = "mean"):
        # (Copied from your original class)
        ei, w_sum = coalesce(edge_index, w, reduce="add")
        ones = torch.ones_like(w)
        _, count = coalesce(edge_index, ones, reduce="add")
        if reduce == "mean":
            w_sym = w_sum / count.clamp_min(1)
        elif reduce == "sum":
            w_sym = w_sum
        else:
            raise ValueError("reduce must be 'mean' or 'sum'")
        return ei, w_sym

    def forward(self,
                x: torch.Tensor,             # [N, x_dim]
                edge_index: torch.Tensor,     # [2, E]
                w: torch.Tensor,              # [E]
                z: torch.Tensor               # [z_dim]
                ):
        # 1) Node embeddings
        h = self.encoder(x, edge_index)       # [N, enc_out]
        N = h.size(0)

        # 2) Combine node features with global perturbation z
        z_b = z.expand(N, -1)                 # [N, z_dim]
        h_z = torch.cat([h, z_b], dim=-1)     # [N, enc_out + z_dim]

        # 3) Get attention scores for all nodes
        scores_l = self.attn_l(h_z)           # [N, 1]
        scores_r = self.attn_r(h_z)           # [N, 1]

        # 4) Compute edge deltas by summing endpoint scores
        u, v = edge_index[0], edge_index[1]   # [E]
        delta_scores = scores_l[u] + scores_r[v]  # [E, 1]
        delta = self.leaky_relu(delta_scores).squeeze(-1) # [E]

        # 5) Apply and post-process
        w_prime = w + self.lambda_scale * delta
        if self.nonneg:
            w_prime = F.softplus(w_prime)

        if self.symmetric:
            edge_index, w_prime = self._symmetrize(edge_index, w_prime, reduce="mean")
            _, delta = self._symmetrize(edge_index, delta, reduce="mean")
            return edge_index, w_prime, delta

        return edge_index, w_prime, delta