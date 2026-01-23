from torch import nn
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import coalesce
from torch_geometric.data import Data, Batch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce
from gears.model import GEARS_Model
    
class Gated_Addition(nn.Module):
    def __init__(self,vector_dim:int):
        super().__init__()
        self.gating = nn.Linear(2*vector_dim,vector_dim)
    def forward(self,x1,x2):
        g = self.gating(torch.concat((x1,x2),dim=-1))
        return g*x1 + (1-g)*x2

# lowrank_perturber_sage.py
class GEARS_No_Coexpress(GEARS_Model):
    def __init__(self, args):
        super().__init__(args)
        self.layers_emb_pos = torch.nn.ModuleList() # Empty module list

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

# Assuming SAGEEncoder is defined elsewhere
# from your_project.encoders import SAGEEncoder

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
    

class GEARS_MECH(GEARS_Model):
    def __init__(self, args):
        super().__init__(args)
        self.pretrain_phase = None
        hidden_size = args['hidden_size']
        self.return_graphs = False
        low_rank_percent = 0.2 ## make this something thats in the config.
        ## adding the gated expression embedding module
        self.expression_projection = nn.Linear(1,args["hidden_size"])
        self.gating = Gated_Addition(vector_dim=args["hidden_size"])
        ## adding self.graph_perturber
        r = int(low_rank_percent*self.G_coexpress.shape[1])
        print(f"Downprojected Rank {r}")
        ## it takes the gene-embeddings as an input
        ## i mean it uses the gener graph
        ## so x_dim is embedding_dim, ie: hidden_size in the args , same for the pert_dim ie: z_dim
        self.graph_perturber = AttentionalPerturber(
            x_dim=hidden_size, z_dim= hidden_size,
        )
        ## other args used in the above init:
        # enc_hidden: int = 128,
        # enc_out: int = 128,
        # enc_layers: int = 2,
        # enc_dropout: float = 0.0,
        # enc_l2norm: bool = False,
        # nonneg: bool = True,
        # symmetric: bool = True,
        # lambda_init: float = 0.5


        ## will have to be a module, that takes in an edge_weights, a learnable perturbation embedding, and outputs a changed/perturbed graph index of the same dimension.

    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        num_graphs = len(data.batch.unique()) # each cell has its own graph
        ## get base gene embeddings, num_batch of the same graph.
        emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
        emb = self.bn_emb(emb)
        base_emb = self.emb_trans(emb)

        ## EXPRESSION EMBEDDING MODULE, TO EMBED CELL EXPRESSION.
        x_cells = x.view(num_graphs, self.num_genes, 1)
        means   = x_cells.mean(dim=0, keepdim=True)  # (1, G, 1)
        centered= x_cells - means                    # (B, G, 1)
        # back to (B*G, 1) for linear
        centered = centered.view(-1, 1)
        expr_emb = self.expression_projection(centered)  # (B*G, H)
        base_emb = self.gating(base_emb,expr_emb)

        #
        if not self.pretrain_phase:
            ### START OF BLOCK FOR THIS
            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))
            ## augment global perturbation embedding with GNN
            ## pass the perturbation global embedding through gnn
            for idx, layer in enumerate(self.sim_layers):
                # GCN with Perturbation graph constructed via Gene-Ontology Network
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()
            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        ## j = -1 is the control perturbation index, dunno why they arent appending it
                        ## idx indicates which cell, j corresponds to the perturbation number.
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T ## 2 x numgraohs with (cell_id, pert_id)
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                ## i is index of that cell in the batch, j is the perturbation index/type ?
                ## pert_index[0] is the cell_id
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        ## to pert_track, add the perturbation embedding for that perturbation, which is in pert_index[1][i]
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                        ## pert_track[cell_id] = pert_track[cell_id] + pert_global[pert_id corresponding to cell_id ]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                # print(f"This is pert_track{pert_track}")

                ## Edge index will remain the same, only the edge weights will change.

                all_edge_weights = {}

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        ## pass this specific perturbation embedding set through an mlp for it to add to the base embedding
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        ## j is cell_id, ie: which graph we're adding this to, and emb_total[idx] has the perturbation embedding that we're adding to that corresponding cell
                        ##
                        node_embs = base_emb[j]
                        pert_emb = emb_total[idx]
                        edge_index, w_prime, delta = self.graph_perturber(node_embs, self.G_coexpress, self.G_coexpress_weight, pert_emb)
                        all_edge_weights[j] = w_prime

        pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
        data_list = []
        # print(f"This is all_edge_weights {all_edge_weights}")
        ## one hot vector to tell if its a control or not
        ctrl_no_ctrl = torch.zeros(num_graphs,1).to(self.args['device'])
        for i in range(num_graphs):
            node_features = pos_emb[i * self.num_genes : (i + 1) * self.num_genes]
            # print(f"Node feature dim: {node_features.shape}")
            if i in all_edge_weights:
                ## special graph corresponding to perturbed edges
                data_list.append(
                    Data(
                        x=node_features,  # Node features for graph i
                        edge_index=self.G_coexpress, # Same structure
                        edge_weight=all_edge_weights[i] # DIFFERENT weights
                    )
                )
                ctrl_no_ctrl[i] = 1.0
            else:
                ## control cells 
                data_list.append(
                    Data(
                        x=node_features,  # Node features for graph i
                        edge_index=self.G_coexpress, # Same structure
                        edge_weight= self.G_coexpress_weight
                    )
                )
        
        batch = Batch.from_data_list(data_list)
        batch = batch.to(self.args["device"])
        for idx, layer in enumerate(self.layers_emb_pos):
            batch.x = layer(batch.x, batch.edge_index, batch.edge_weight)
            if idx < len(self.layers_emb_pos) - 1:
                batch.x = batch.x.relu()
        final_pos_emb = batch.x.view(num_graphs, self.num_genes, -1)
        base_emb = base_emb + 0.7 * final_pos_emb
        base_emb = base_emb.reshape(num_graphs*self.num_genes,-1)
        base_emb = self.emb_trans_v2(base_emb)
        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        base_emb = self.bn_pert_base(base_emb)
        base_emb = self.transform(base_emb)
        out = self.recovery_w(base_emb)
        out = out.reshape(num_graphs, self.num_genes, -1)
        out = out.unsqueeze(-1) * self.indv_w1
        w = torch.sum(out, axis = 2)
        out = w + self.indv_b1
        cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
        cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)
        cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
        cross_gene_out = torch.cat([out, cross_gene_embed], 2)
        cross_gene_out = cross_gene_out * self.indv_w2
        cross_gene_out = torch.sum(cross_gene_out, axis=2)
        out = cross_gene_out + self.indv_b2
        out = out.reshape(num_graphs * self.num_genes, -1)  + x.reshape(-1,1) 
        out = torch.split(torch.flatten(out), self.num_genes)
        if self.return_graphs:
            graph_data =   {"graphs": data_list,
                            "graph_metadata": ctrl_no_ctrl
                            }
            return out, graph_data
            
        return torch.stack(out)
    def compute_graphs(self,data):
        self.return_graphs = True
        with torch.no_grad():
            out, graph_data = self.forward(data)
        self.return_graphs = False
        return out,graph_data
