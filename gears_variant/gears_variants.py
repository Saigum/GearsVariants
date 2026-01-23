from torch import nn
from gears.model import GEARS_Model
import torch
from gears.model import MLP
from torch_geometric.nn import SGConv
from torch_geometric.nn import GravNetConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import TransformerConv

class GEARS_No_Perturb(GEARS_Model):
    def __init__(self, args):
        super().__init__(args)
        self.sim_layers = torch.nn.ModuleList()


class GEARS_SelfAttn(GEARS_Model):
    def __init__(self, args):
        super().__init__(args)
        self.cross_gene_attn = nn.MultiheadAttention(embed_dim=self.num_genes,num_heads=args["num_heads"])
    def forward(self,data):
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique()) # each cell has its own graph
            ## get base gene embeddings, num_batch of the same graph.
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device'])) 
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        

            ## positional embeddings to differentiate each cell's embedding
            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                # pass in the positional embegginfs through a gcn with the co-expression graph.
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    # relu till the last layer.
                    pos_emb = pos_emb.relu()

            # combine base embeddings and positional embeddings
            base_emb = base_emb + 0.2 * pos_emb
            # pass embeddings for each cell through an mlp
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        ## idx indicates which cell, j corresponds to the perturbation number.
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T
            ## perturbation embeddings for total number of perturbations to be considered.
            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                # GCN with Perturbation graph constructed via Gene-Ontology Network
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            outpass = out.reshape(num_graphs, self.num_genes, -1).squeeze(2)
            cross_gene_op,cross_gene_attn = self.cross_gene_attn(outpass,outpass,outpass)
            cross_gene_embed = self.cross_gene_state(cross_gene_op)
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            return torch.stack(out)

        
class GEARS_No_Coexpress(GEARS_Model):
    def __init__(self, args):
        super().__init__(args)
        self.layers_emb_pos = torch.nn.ModuleList() # Empty module list
        
class GEARS_Transformer(GEARS_Model):
    def __init__(self, args):
        super().__init__(args)
        
        # Transformer layers for co-expression GNN
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(TransformerConv(args['hidden_size'], args['hidden_size'], heads=1))
            
        # Transformer layers for perturbation similarity GNN
        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(TransformerConv(args['hidden_size'], args['hidden_size'], heads=1))
    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique()) # each cell has its own graph
            ## get base gene embeddings, num_batch of the same graph.
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device'])) 
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        

            ## positional embeddings to differentiate each cell's embedding
            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                # pass in the positional embegginfs through a gcn with the co-expression graph.
                pos_emb = layer(pos_emb, self.G_coexpress)
                if idx < len(self.layers_emb_pos) - 1:
                    # relu till the last layer.
                    pos_emb = pos_emb.relu()

            # combine base embeddings and positional embeddings
            base_emb = base_emb + 0.2 * pos_emb
            # pass embeddings for each cell through an mlp
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        ## idx indicates which cell, j corresponds to the perturbation number.
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T
            ## perturbation embeddings for total number of perturbations to be considered.
            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                # GCN with Perturbation graph constructed via Gene-Ontology Network
                pert_global_emb = layer(pert_global_emb, self.G_sim)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            return torch.stack(out)
  

class GEARS_GAT(GEARS_Model):
    def __init__(self, args):
        super().__init__(args)
        
        # GAT layers for co-expression GNN
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(GATConv(args['hidden_size'], args['hidden_size'], heads=1))
            
        # GAT layers for perturbation similarity GNN
        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(GATConv(args['hidden_size'], args['hidden_size'], heads=1))


class Gated_Addition(nn.Module):
    def __init__(self,vector_dim:int):
        super().__init__()
        self.gating = nn.Linear(2*vector_dim,vector_dim)
    def forward(self,x1,x2):
        g = self.gating(torch.concat((x1,x2),dim=-1))
        return g*x1 + (1-g)*x2   
       
class GEARS_EMBED(GEARS_Model):
    def __init__(self, args):
        super().__init__(args)
        self.expression_projection = nn.Linear(1,args["hidden_size"])
        if(args["gated"] is True):
            self.gating = Gated_Addition(vector_dim=args["hidden_size"])
        self.gated = args["gated"]
    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique()) # each cell has its own graph
            ## get base gene embeddings, num_batch of the same graph.
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        
            
             # ────── 2) per‑gene mean‑centering + projection ────────────────
            # reshape x → (B, G, 1)
            x_cells = x.view(num_graphs, self.num_genes, 1)
            means   = x_cells.mean(dim=0, keepdim=True)  # (1, G, 1)
            centered= x_cells - means                    # (B, G, 1)
            # back to (B*G, 1) for linear
            centered = centered.view(-1, 1)
            expr_emb = self.expression_projection(centered)  # (B*G, H)
            # fuse expression offset into base
            # print(f"shapes of base embedding {base_emb.shape} ")
            # print(f"shapes of expression embedding {expr_emb}")
            if self.gated:
                base_emb = self.gating(base_emb,expr_emb)
            else:
                base_emb = base_emb + expr_emb
            # positional embeddings to differentiate each cell's embedding
            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                # pass in the positional embegginfs through a gcn with the co-expression graph.
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    # relu till the last layer.
                    pos_emb = pos_emb.relu()
                    
            # combine base embeddings and positional embeddings
            base_emb = base_emb  + 0.2 * pos_emb
            # pass embeddings for each cell through an mlp
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        ## idx indicates which cell, j corresponds to the perturbation number.
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T
            ## perturbation embeddings for total number of perturbations to be considered.
            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                # GCN with Perturbation graph constructed via Gene-Ontology Network
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            return torch.stack(out)
  
  

class Variant(nn.Module):
    """
    GEARS variant:
      1) Gene embedding via local 1→H MLP + global sparse-code projector (whole-vector → [G,H]).
      2) Perturbation context fused BEFORE co-expression GNN (no separate perturbation graph).
      3) Cross-gene fusion via TransformerEncoder; final Linear(H→1) per gene.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']  # reusing this for coexpr depth
        self.no_perturb = args['no_perturb']

        # ----- hyperparams with fallbacks -----
        device = args['device']
        self.device = device
        self.code_dim = int(args.get('sparse_code_dim', min(64, max(16, hidden_size))))  # K
        self.local_global_mix = float(args.get('local_global_mix', 0.5))  # weight for local MLP (0..1)
        nheads = int(args.get('num_attn_heads', max(1, min(8, hidden_size // 64))))
        xenc_layers = int(args.get('num_xformer_layers', 2))
        attn_dropout = float(args.get('attn_dropout', 0.1))
        ffn_mult = int(args.get('ffn_mult', 4))

        # ----- co-expression graph (provided) -----
        self.G_coexpress = args['G_coexpress'].to(device)
        self.G_coexpress_weight = args['G_coexpress_weight'].to(device)

        # =========================
        # 1) Gene "embedding" stack
        # =========================
        # Local per-gene projector: (value: R¹) → (embedding: R^H)
        # Applied pointwise to each gene's scalar value.
        self.gene_value_mlp = MLP([1, hidden_size, hidden_size], last_layer_act='ReLU')

        # Global sparse-code projector:
        #   x_b ∈ R^G  →  a_b = Enc(x_b) ∈ R^K  →  E_b = Σ_k a_bk * Basis_k  with Basis ∈ R^{K×G×H}
        self.global_encoder = nn.Sequential(
            nn.Linear(self.num_genes, self.code_dim),
            nn.ReLU(),
            nn.Linear(self.code_dim, self.code_dim),
            nn.ReLU(),
        )
        # Learned basis (initialized small for stability)
        self.gene_basis = nn.Parameter(0.02 * torch.randn(self.code_dim, self.num_genes, hidden_size))

        # Normalizations / transforms around pre-GNN features
        self.bn_pre = nn.BatchNorm1d(hidden_size)
        self.pre_transform = nn.ReLU()

        # ========================================
        # 2) Perturbation encoding fused pre-GNN
        # ========================================
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')

        # =============================
        # Co-expression GNN (SGConv K=1)
        # =============================
        self.coexpr_layers = nn.ModuleList([SGConv(hidden_size, hidden_size, K=1) for _ in range(self.num_layers_gene_pos)])

        # Light "recovery" transform (kept conceptually from original)
        self.recovery_w = MLP([hidden_size, hidden_size * 2, hidden_size], last_layer_act='linear')

        # =========================================
        # 3) Cross-gene fusion via Transformer block
        # =========================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nheads,
            dim_feedforward=ffn_mult * hidden_size,
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True,  # input as [B, G, H]
            norm_first=True,
        )
        self.xformer = nn.TransformerEncoder(encoder_layer, num_layers=xenc_layers)

        # Final per-gene projection to scalar
        self.gene_out_proj = nn.Linear(hidden_size, 1)

        # Batch norms mirroring the original names (for continuity)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)

        # Uncertainty head (per gene)
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size * 2, hidden_size, 1], last_layer_act='linear')

    def forward(self, data):
        """
        Inputs (unchanged):
          - data.x: (num_graphs * num_genes, 1) float tensor of base gene values
          - data.pert_idx: iterable per-graph with perturbation indices (−1 for controls)
          - data.batch: graph index per node (length: num_graphs * num_genes)

        Outputs (unchanged):
          - If self.no_perturb: [num_graphs, num_genes]
          - Else: [num_graphs, num_genes]  (and if self.uncertainty: also [num_graphs, num_genes] log-variance)
        """
        x, pert_idx = data.x, data.pert_idx

        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)

        # How many graphs in the batch
        num_graphs = len(data.batch.unique())
        G, H = self.num_genes, self.gene_out_proj.in_features
        device = self.device

        # ---------------------------
        # Build pre-GNN gene embeddings
        # ---------------------------
        # x_flat: [B*G, 1] → local per-gene embeddings [B*G, H]
        x_flat = x.to(device)
        local_emb = self.gene_value_mlp(x_flat)  # [B*G, H]

        # global sparse-code: reshape x to [B, G]
        x_b_g = x_flat.view(num_graphs, G)
        code = self.global_encoder(x_b_g)  # [B, K]
        # E_sparse: einsum over code and basis → [B, G, H]
        E_sparse = torch.einsum('bk,kgh->bgh', code, self.gene_basis).contiguous()
        # mix local & global
        E_local = local_emb.view(num_graphs, G, H)
        pre_gnn = self.local_global_mix * E_local + (1.0 - self.local_global_mix) * E_sparse  # [B, G, H]

        # -----------------------------------------------
        # Fuse perturbation context BEFORE coexpr message passing
        # -----------------------------------------------
        # Build fused per-graph perturbation embedding (like original logic)
        pert_index = []
        for b_idx, perts in enumerate(pert_idx):
            for p in perts:
                if p != -1:
                    pert_index.append([b_idx, p])
        if len(pert_index) > 0:
            pert_index = torch.tensor(pert_index, device=device).T  # [2, M]
            pert_track = {}
            for i, b in enumerate(pert_index[0]):
                p = int(pert_index[1][i])
                if b.item() in pert_track:
                    pert_track[b.item()] = pert_track[b.item()] + self.pert_emb.weight[p]
                else:
                    pert_track[b.item()] = self.pert_emb.weight[p]

            if len(pert_track) > 0:
                # fuse multiple perts per graph
                vals = torch.stack(list(pert_track.values()), dim=0)  # [B_eff, H]
                if vals.size(0) == 1:
                    fused = self.pert_fuse(torch.cat([vals, vals], dim=0))[:1]  # handle B_eff==1
                else:
                    fused = self.pert_fuse(vals)  # [B_eff, H]

                # add to each gene in the corresponding graph
                for k, b in enumerate(pert_track.keys()):
                    pre_gnn[b] = pre_gnn[b] + fused[k].unsqueeze(0).expand(G, H)
        # If all controls, pre_gnn stays as-is.

        # ---------------------------------
        # Normalize / pre-transform then GNN
        # ---------------------------------
        pre = self.bn_pre(pre_gnn.reshape(num_graphs * G, H))
        pre = self.pre_transform(pre)  # [B*G, H]

        # Co-expression GNN over (potentially batched) graph
        z = pre
        for idx, layer in enumerate(self.coexpr_layers):
            z = layer(z, self.G_coexpress, self.G_coexpress_weight)
            if idx < len(self.coexpr_layers) - 1:
                z = z.relu()
        # light recovery mapping (like original)
        z = self.recovery_w(z)  # [B*G, H]

        # ---------------------------------
        # Cross-gene Transformer fusion
        # ---------------------------------
        z_tokens = z.view(num_graphs, G, H)  # [B, G, H]
        z_tokens = self.xformer(z_tokens)    # [B, G, H] with residuals inside

        # Project per gene to scalar
        y_delta = self.gene_out_proj(z_tokens).view(num_graphs * G, 1)  # [B*G, 1]

        # Residual to base signal (as in original)
        y = y_delta + x_flat  # [B*G, 1]

        # Split back to [B, G]
        out = torch.split(torch.flatten(y), G)
        out = torch.stack(out)

        if self.uncertainty:
            # per-gene log-variance from the same fused representation
            logvar = self.uncertainty_w(z_tokens.reshape(num_graphs * G, H))
            logvar = torch.split(torch.flatten(logvar), G)
            logvar = torch.stack(logvar)
            return out, logvar

        return out

class VariantTiny(nn.Module):
    """
    Smaller + faster GEARS variant:
      - Width scaled by `width_mult`
      - Shallower Transformer encoder & FFN
      - Fewer GNN layers by default
      - LayerNorm instead of BatchNorm on [B*G, H]
    API/outputs match your Variant.
    """

    def __init__(self, args, width_mult: float = 0.5):
        super().__init__()
        self.args = args
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        base_hidden = args['hidden_size']
        self.uncertainty = args['uncertainty']
        # keep the same arg but we’ll override with a smaller default if not explicitly set
        self.num_layers_gene_pos = int(args.get('num_gene_gnn_layers', 1))
        self.no_perturb = args['no_perturb']
        device = args['device']
        self.device = device

        # ----- scaled hyperparams -----
        H = max(32, int(base_hidden * width_mult))
        code_dim_base = int(args.get('sparse_code_dim', min(64, max(16, base_hidden))))
        self.code_dim = max(8, int(code_dim_base * width_mult))
        # transformer heads/layers/ffn
        req_heads = int(args.get('num_attn_heads', max(1, base_hidden // 64)))
        nheads = max(1, min(4, req_heads // 2 if req_heads > 1 else 1))
        # ensure divisibility
        while H % nheads != 0 and nheads > 1:
            nheads -= 1
        self.nheads = nheads
        xenc_layers = max(1, int(args.get('num_xformer_layers', 2) // 2))
        attn_dropout = float(args.get('attn_dropout', 0.1))
        ffn_mult = 2  # leaner than 4

        # co-expression graph
        self.G_coexpress = args['G_coexpress'].to(device)
        self.G_coexpress_weight = args['G_coexpress_weight'].to(device)

        # =========================
        # 1) Gene "embedding" stack
        # =========================
        # Local projector: 1 -> H (single hidden for speed)
        self.gene_value_mlp = nn.Sequential(
            nn.Linear(1, H),
            nn.GELU()
        )

        # Global sparse-code projector (smaller K)
        self.global_encoder = nn.Sequential(
            nn.Linear(self.num_genes, self.code_dim),
            nn.ReLU(),
            nn.Linear(self.code_dim, self.code_dim),
            nn.ReLU(),
        )
        self.gene_basis = nn.Parameter(0.02 * torch.randn(self.code_dim, self.num_genes, H))

        # Lighter normalization/activation pre-GNN
        self.ln_pre = nn.LayerNorm(H)
        self.pre_transform = nn.GELU()

        # ========================================
        # 2) Perturbation encoding fused pre-GNN
        # ========================================
        self.pert_emb = nn.Embedding(self.num_perts, H, max_norm=True)
        self.pert_fuse = MLP([H, H], last_layer_act='ReLU')  # thinner

        # =============================
        # Co-expression GNN (SGConv K=1)
        # =============================
        gnn_layers = max(1, self.num_layers_gene_pos)  # default 1
        self.coexpr_layers = nn.ModuleList([SGConv(H, H, K=1) for _ in range(gnn_layers)])

        # Light "recovery" transform
        self.recovery_w = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, H),
        )

        # =========================================
        # 3) Cross-gene fusion via Transformer block
        # =========================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=self.nheads,
            dim_feedforward=ffn_mult * H,
            dropout=attn_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.xformer = nn.TransformerEncoder(encoder_layer, num_layers=xenc_layers)

        # Final per-gene projection to scalar
        self.gene_out_proj = nn.Linear(H, 1)

        # For continuity with original names (unused now, but retained)
        self.bn_pert_base = nn.BatchNorm1d(H, affine=False, track_running_stats=False)

        # Uncertainty head (lean)
        if self.uncertainty:
            self.uncertainty_w = nn.Linear(H, 1)

        # Mixing weight (keep same default behavior)
        self.local_global_mix = float(args.get('local_global_mix', 0.5))

    def forward(self, data):
        x, pert_idx = data.x, data.pert_idx

        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)

        num_graphs = int(data.batch.max().item()) + 1  # faster than unique()
        G = self.num_genes
        H = self.gene_out_proj.in_features
        device = self.device

        # ----- local embedding -----
        x_flat = x.to(device)                     # [B*G, 1]
        E_local = self.gene_value_mlp(x_flat)     # [B*G, H]
        E_local = E_local.view(num_graphs, G, H)  # [B, G, H]

        # ----- global sparse-code -----
        x_b_g = x_flat.view(num_graphs, G)
        code = self.global_encoder(x_b_g)         # [B, K]
        E_sparse = torch.einsum('bk,kgh->bgh', code, self.gene_basis).contiguous()  # [B, G, H]

        # mix
        pre_gnn = self.local_global_mix * E_local + (1.0 - self.local_global_mix) * E_sparse  # [B, G, H]

        # ----- fuse perturbations pre-GNN -----
        if len(pert_idx) > 0:
            # accumulate per-graph embeddings
            fused_store = {}
            for b_idx, perts in enumerate(pert_idx):
                acc = None
                for p in perts:
                    if p != -1:
                        emb = self.pert_emb.weight[int(p)]
                        acc = emb if acc is None else acc + emb
                if acc is not None:
                    fused_store[b_idx] = acc
            if fused_store:
                vals = torch.stack(list(fused_store.values()), dim=0)  # [B_eff, H]
                fused = self.pert_fuse(vals)
                for k, b in enumerate(fused_store.keys()):
                    pre_gnn[b] = pre_gnn[b] + fused[k].unsqueeze(0)

        # ----- pre-norm + GNN -----
        pre = pre_gnn.reshape(num_graphs * G, H)  # [B*G, H]
        pre = self.ln_pre(pre)
        pre = self.pre_transform(pre)

        z = pre
        for idx, layer in enumerate(self.coexpr_layers):
            z = layer(z, self.G_coexpress, self.G_coexpress_weight)
            if idx < len(self.coexpr_layers) - 1:
                z = z.relu()

        z = self.recovery_w(z)                    # [B*G, H]

        # ----- cross-gene Transformer -----
        z_tokens = z.view(num_graphs, G, H)       # [B, G, H]
        z_tokens = self.xformer(z_tokens)         # [B, G, H]

        # ----- head(s) -----
        y_delta = self.gene_out_proj(z_tokens).view(num_graphs * G, 1)  # [B*G, 1]
        y = y_delta + x_flat

        out = torch.split(torch.flatten(y), G)
        out = torch.stack(out)

        if self.uncertainty:
            logvar = self.uncertainty_w(z_tokens.reshape(num_graphs * G, H))
            logvar = torch.split(torch.flatten(logvar), G)
            logvar = torch.stack(logvar)
            return out, logvar

        return out

class GEARS_CELL(GEARS_EMBED):
    def __init__(self, args):
        super().__init__(args)
        self.cell_gcn = GravNetConv(in_channels=args["num_genes"],out_channels=args["num_genes"],
                                    space_dimensions=args["hidden_size"],propagate_dimensions=4,k=6)
    def forward(self, data):
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique()) # each cell has its own graph
            ## get base gene embeddings, num_batch of the same graph.
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        
            
             # ────── 2) per‑gene mean‑centering + projection ────────────────
            # reshape x → (B, G, 1)
            x_cells = x.view(num_graphs, self.num_genes, 1)
            means   = x_cells.mean(dim=0, keepdim=True)  # (1, G, 1)
            centered= x_cells - means                    # (B, G, 1)
            # back to (B*G, 1) for linear
            # centered = centered.view(-1, 1)
            ## instead of projecting it, we pass this through the gravnet, which does knn-graph message passing.
            cellbatch_idx = torch.zeros(
                centered.shape[0],           # number of rows in centered.squeeze(-1)
                dtype=torch.long,            # must be integer type
                device=centered.device       # must live on the same GPU/CPU as x
            )

            message_passed = self.cell_gcn(centered.squeeze(-1),cellbatch_idx) ## should be off same shape
            message_passed = message_passed.view(-1,1)
            expr_emb = self.expression_projection(message_passed)
            # print(expr_emb.shape)
            # expr_emb = self.expression_projection(centered)  # (B*G, H)
            # fuse expression offset into base
            # print(f"shapes of base embedding {base_emb.shape} ")
            # print(f"shapes of expression embedding {expr_emb}")
            base_emb = base_emb + expr_emb
            # positional embeddings to differentiate each cell's embedding
            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                # pass in the positional embegginfs through a gcn with the co-expression graph.
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    # relu till the last layer.
                    pos_emb = pos_emb.relu()
                    
            # combine base embeddings and positional embeddings
            base_emb = base_emb  + 0.2 * pos_emb
            # pass embeddings for each cell through an mlp
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        ## idx indicates which cell, j corresponds to the perturbation number.
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T
            ## perturbation embeddings for total number of perturbations to be considered.
            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                # GCN with Perturbation graph constructed via Gene-Ontology Network
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            return torch.stack(out)
  
        