# test_perturbation_diffusion.py
import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Sequence, Union

# ---------- Fixed version of your layer (minimal edits) ----------
class PerturbationDiffusion(MessagePassing):
    def __init__(
        self,
        num_nodes: int = 5045,
        node_feature_dim: int = 64,
        K: int = 10,
        alpha: float = 0.1,
        learn_alpha: bool = False,
        device: str = "cuda",
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)  # IMPORTANT: init MessagePassing

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = {
            "num_nodes": num_nodes,
            "node_feature_dim": node_feature_dim,
            "K": K,
            "device": str(self.device),
        }

        # Learnable perturbation embeddings, one per node
        self.pert_emb = nn.Embedding(num_nodes, node_feature_dim)

        # Handle alpha (fixed or learnable in (0,1))
        if learn_alpha:
            # initialize around desired alpha via logit
            init = torch.tensor(float(alpha))
            init = torch.clamp(init, 1e-6, 1-1e-6)
            self._alpha_param = nn.Parameter(torch.log(init/(1.0-init)))
            self.register_parameter("_alpha_fixed", None)
        else:
            self.register_parameter("_alpha_param", None)
            self.register_buffer("_alpha_fixed", torch.tensor(float(alpha)))
        self.norm = None

        # Move params/buffers to device
        self.to(self.device)

    @property
    def alpha(self) -> Tensor:
        if self._alpha_param is None:
            return self._alpha_fixed
        return torch.sigmoid(self._alpha_param)

    def _compute_norm(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Optional[Tensor] = None,
    ):
        # If no weights, start with ones
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        if self.norm is not None:
            return self.norm
        
        # Add self-loops with weight=1.0
        ei, ew = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1.0, num_nodes=num_nodes)

        row, col = ei
        deg = degree(col, num_nodes=num_nodes, dtype=torch.float32)
        deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
        self.norm = deg_inv_sqrt[row] * ew * deg_inv_sqrt[col]
        return self.norm

    def _build_injection(self, inj_index: Union[int, Sequence[int], Tensor]) -> Tensor:
        delta = torch.zeros(self.config["num_nodes"], self.config["node_feature_dim"], device=self.device)
        if isinstance(inj_index, int):
            inj_index = torch.tensor([inj_index], device=self.device)
        elif isinstance(inj_index, (list, tuple)):
            inj_index = torch.tensor(inj_index, device=self.device)
        else:
            inj_index = inj_index.to(self.device)

        p = self.pert_emb(inj_index)  # (m, F)
        delta[inj_index] += p
        return delta

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,                      # to compute the graph norm
        edge_weight: Optional[Tensor] = None,    # to compute the graph norm
        inj_index: Optional[Union[int, Sequence[int], Tensor]] = None,
    ) -> Tensor:
        node_features = node_features.to(self.device)

        if inj_index is not None:
            delta = self._build_injection(inj_index)
            x0 = node_features + delta
        else:
            x0 = node_features

        out = x0
        a = self.alpha
        ei, ew = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1.0, num_nodes=self.config["num_nodes"])
        norm = self._compute_norm(edge_index=edge_index,num_nodes=self.config["num_nodes"],
                                  edge_weight=edge_weight)
        for _ in range(self.config["K"]):
            out = self.propagate(ei, x=out, norm=norm)
            out = (1.0 - a) * out + a * x0
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        # GCN-style message; swap this for higher-order diffusion if desired
        return norm.view(-1, 1) * x_j


# run_test_refactored_pd.py
import torch
from torch import nn, Tensor
from torch_geometric.utils import to_undirected
from typing import Optional, Sequence, Union

def make_cycle_graph(n: int, device: torch.device) -> torch.Tensor:
    """Return 2xE undirected cycle edge_index with both directions."""
    src = torch.arange(n, device=device)
    dst = (src + 1) % n
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return ei  # already bidirectional

def make_random_graph(n: int, m: int, device: torch.device) -> torch.Tensor:
    """n nodes, m undirected edges (without self-loops)."""
    i = torch.randint(0, n, (m,), device=device)
    j = torch.randint(0, n, (m,), device=device)
    mask = i != j
    i, j = i[mask], j[mask]
    ei = torch.stack([i, j], dim=0)
    ei = to_undirected(ei, num_nodes=n)  # both directions
    return ei

@torch.no_grad()
def print_stats(name, x: torch.Tensor):
    print(f"{name:>18s}: shape={tuple(x.shape)}, mean={x.mean().item():.5f}, std={x.std().item():.5f}")

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- synthetic graph ---
    num_nodes = 12
    feat_dim  = 8
    K         = 6
    edge_index = make_cycle_graph(num_nodes, device=device)
    edge_weight = None  # or torch.ones(edge_index.size(1), device=device)

    # --- node features ---
    x = torch.randn(num_nodes, feat_dim, device=device, requires_grad=True)

    # --- layer under test (your refactored class) ---
    layer = PerturbationDiffusion(
        num_nodes=num_nodes,
        node_feature_dim=feat_dim,
        K=K,
        alpha=0.2,
        learn_alpha=True,
        device=str(device),
    )

    print(f"Initial alpha: {layer.alpha.item():.4f}")

    # --- forward: baseline (no injection) ---
    y_base = layer(
        node_features=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        inj_index=None
    )
    print_stats("y_base", y_base)

    # --- forward: single-node injection ---
    y_inj_3 = layer(
        node_features=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        inj_index=3
    )
    print_stats("y_inj (node 3)", y_inj_3)

    # sanity: shape & effect
    assert y_base.shape == (num_nodes, feat_dim)
    assert y_inj_3.shape == (num_nodes, feat_dim)
    delta_3 = (y_inj_3 - y_base).norm(dim=1)
    print("Per-node L2 change (inj @3):", [round(v, 4) for v in delta_3.tolist()])
    assert delta_3[3] > 0, "Injected node should change."

    # --- forward: multi-node injection ---
    inj_nodes = [2, 5, 9]
    y_inj_multi = layer(
        node_features=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        inj_index=inj_nodes
    )
    print_stats("y_inj (2,5,9)", y_inj_multi)
    delta_m = (y_inj_multi - y_base).norm(dim=1)
    print("Per-node L2 change (inj @2,5,9):", [round(v, 4) for v in delta_m.tolist()])
    assert delta_m[inj_nodes].min() > 0, "All injected nodes should change."

    # --- simple loss & backprop to test grads (use the multi-injection output) ---
    loss = (y_inj_multi ** 2).mean()
    loss.backward()

    # check grads exist
    print(f"Grad ||x||: {x.grad.norm().item():.6f}")
    pe_grad = layer.pert_emb.weight.grad
    print(f"Grad ||pert_emb||: {0.0 if pe_grad is None else pe_grad.norm().item():.6f}")
    if layer._alpha_param is not None and layer._alpha_param.grad is not None:
        print(f"Grad alpha_param: {layer._alpha_param.grad.item():.6f}")

    # --- tiny optimization loop to verify learning works ---
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    for step in range(5):
        opt.zero_grad()
        y = layer(x, edge_index=edge_index, edge_weight=edge_weight, inj_index=inj_nodes)
        loss = (y ** 2).mean()
        loss.backward()
        opt.step()
        print(f"step {step}: loss={loss.item():.6f}, alpha={layer.alpha.item():.4f}")

    # --- (optional) test with edge weights ---
    # weights = torch.ones(edge_index.size(1), device=device)
    # y_w = layer(x, edge_index=edge_index, edge_weight=weights, inj_index=3)
    # print_stats("y (weighted)", y_w)

    # --- (optional) test on a different graph ---
    # NOTE: your class caches norm; if you switch graphs in the same layer instance,
    # reset the cache:
    # layer.norm = None
    # new_ei = make_random_graph(num_nodes, m=18, device=device)
    # y_new = layer(x, edge_index=new_ei, edge_weight=None, inj_index=1)
    # print_stats("y (new graph)", y_new)

if __name__ == "__main__":
    main()
