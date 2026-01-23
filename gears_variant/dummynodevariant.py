import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import ModuleList, Sequential, Linear, ReLU, Parameter
from torch_geometric.nn import GPSConv, GCNConv, SGConv
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import ModuleList, Sequential, Linear, ReLU, Parameter
from torch_geometric.nn import GPSConv, GCNConv, SGConv
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from tqdm import tqdm
from gears.data import PertData
from gears_changes import *

class ZeroOutVariant(nn.Module):
    def __init__(self, pert_data: PertData_, args):
        super(ZeroOutVariant, self).__init__()
        self.pert_data = pert_data
        self.device = args.get('device', 'cuda')
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2        
        ### TORCH MODULES ###### 
        self.gene_embeddings = nn.Embedding(self.num_genes, hidden_size)
        
        self.scalar_projector = nn.Sequential(nn.Linear(self.num_genes, self.num_genes * hidden_size // 2),
                                              nn.ReLU(),
                                              nn.Linear(self.num_genes * hidden_size // 2, self.num_genes * hidden_size),
                                              nn.ReLU(),
                                             )
        self.deviation_weight = nn.Parameter(torch.randn(self.num_genes, 1))
        self.bn_emb = nn.BatchNorm1d(self.num_genes * hidden_size)
        
        self.G_coexpress = args['G_coexpress'].to(self.device)
        self.G_coexpress_weight = args['G_coexpress_weight'].to(self.device)
        
        self.coexpress_encoder = ModuleList()
        for _ in range(self.num_layers_gene_pos):
            local_gnn = SGConv(hidden_size, hidden_size, 1)
            conv = GPSConv(hidden_size, 
                           conv=local_gnn, 
                           heads=4, 
                           attn_type='multihead')
            self.coexpress_encoder.append(conv)
            
        self.decoder_mlp = nn.Sequential(nn.Linear(hidden_size, self.indv_out_hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.indv_out_hidden_size, 1)
                                         )
        self.coexpression_weight = nn.Parameter(torch.randn(self.num_genes, 1))

    def forward(self, data):
        x, pert_idx = data.x, data.pert_idx
        batch_tensor = data.batch
        num_graphs = len(batch_tensor.unique())

        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            gene_embeddings_base = self.gene_embeddings(
                torch.arange(self.num_genes, device=self.device)
            ).expand(num_graphs, -1, -1)

            x_cells = x.view(num_graphs, self.num_genes, 1)
            means = x_cells.mean(dim=0, keepdim=True)
            stds = x_cells.std(dim=0, keepdim=True) + 1e-6
            z_x_cells = (x_cells - means) / stds
            z_x_cells = z_x_cells.reshape(num_graphs, self.num_genes)
            
            expression_emb = self.scalar_projector(z_x_cells)
            expression_emb = expression_emb.reshape(num_graphs, self.num_genes, -1)
            
            h = gene_embeddings_base + self.deviation_weight * expression_emb
            h = h.reshape(num_graphs * self.num_genes, -1)
            h_in = h.clone()

            # === NEW DYNAMIC GRAPH CONSTRUCTION ===
            # We build a new batched edge_index and edge_weight for each batch
            # based on the perturbation status of each cell.
            
            base_edge_index = self.G_coexpress
            base_edge_weight = self.G_coexpress_weight

            new_edge_indices = []
            new_edge_weights = []
            node_offset = 0

            for i in range(num_graphs):
                # Get the perturbed gene index for the i-th graph in the batch
                pert_gene_idx = pert_idx[i].item()                
                # NOTE: We assume -1 is the sentinel value for control cells.
                if pert_gene_idx == -1:
                    current_edges = base_edge_index
                    current_weights = base_edge_weight
                else:                    
                    edge_mask = (base_edge_index[0] == pert_gene_idx) | \
                                (base_edge_index[1] == pert_gene_idx)                    
                    keep_mask = ~edge_mask ## mask of positions on edge_index that do not involve perturbed gene
                    current_edges = base_edge_index[:, keep_mask] 
                    current_weights = base_edge_weight[keep_mask]
                new_edge_indices.append(current_edges + node_offset)
                new_edge_weights.append(current_weights)
                node_offset += self.num_genes
                
            batched_edge_index = torch.cat(new_edge_indices, dim=1)
            batched_edge_weight = torch.cat(new_edge_weights, dim=0)
            for layer in self.coexpress_encoder:
                h = layer(h, batched_edge_index, batch_tensor, edge_weight=batched_edge_weight)
                h = h.relu()
            
            h_out = h.reshape(num_graphs, self.num_genes, -1)
            h_in_res = h_in.reshape(num_graphs, self.num_genes, -1)
            
            gate = torch.sigmoid(self.coexpression_weight)
            gene_embeddings_out = (1 - gate) * h_in_res + gate * h_out
            
            out = self.decoder_mlp(gene_embeddings_out.reshape(num_graphs * self.num_genes, -1))
            
        return out


def train(
              pert_data: PertData_,
              model: nn.Module,
              device='cuda',
              config: dict = None,
              epochs=20,
              lr=1e-3,
              weight_decay=5e-4,
              # pretrain_epochs=20, # REMOVED
              wandb=None,
              loss_fct=None,
              print_sys=print
             ):
         
        train_loader = pert_data.dataloader['train_loader']
        val_loader = pert_data.dataloader['val_loader']
        ctrl_expression = torch.tensor(
            np.mean(pert_data.adata.X[pert_data.adata.obs.condition == 'ctrl'],
                    axis=0)).reshape(-1, ).to(device)
        pert_full_id2pert = dict(pert_data.adata.obs[['condition_name', 'condition']].values)
        dict_filter = {pert_full_id2pert[i]: j for i, j in
                            pert_data.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        # === PRETRAINING LOOP REMOVED ===
        # print_sys('Start Pre-training...')
        # model.pretraining_phase = True
        # ... (loop removed) ...
        # print_sys('Start Perturbation Training...')
        # model.pretraining_phase = False
        
        print_sys('Start Training...') # Updated log
        best_model = deepcopy(model)
        min_val = np.inf
        
        with tqdm(range(epochs), desc="Training Epochs") as pbar:
            for epoch in pbar:
                model.train()
                total_loss = 0
                
                for step, batch in enumerate(train_loader):
                    batch.to(device)
                    optimizer.zero_grad()
                    
                    # We assume batch.y is the target (perturbed or control expression)
                    # and batch.x is the input (e.g., control expression)
                    y = batch.y 
                    pred = model(batch)
                    
                    loss = loss_fct(pred, y, batch.pert,
                                    ctrl=ctrl_expression, 
                                    dict_filter=dict_filter,
                                    direction_lambda=config['direction_lambda'])
                    
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    if wandb:
                        wandb.log({'training_loss': loss.item()})

                    if step % 50 == 0:
                        log = "Epoch {} Step {} Train Loss: {:.4f}" 
                        print_sys(log.format(epoch + 1, step + 1, loss.item()))
                
                avg_train_loss = total_loss / len(train_loader)
                pbar.set_postfix(train_loss=avg_train_loss)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch.to(device)
                        y = batch.y
                        pred = model(batch)
                        loss = loss_fct(pred, y, batch.pert,
                                        ctrl=ctrl_expression, 
                                        dict_filter=dict_filter,
                                        direction_lambda=config['direction_lambda'])
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                if wandb:
                    wandb.log({'val_loss': avg_val_loss})
                
                log = "Epoch {} Val Loss: {:.4f}"
                print_sys(log.format(epoch + 1, avg_val_loss))

                if avg_val_loss < min_val:
                    min_val = avg_val_loss
                    best_model = deepcopy(model)
                    print_sys('Best model saved!')

                scheduler.step()

        print_sys('Finished Training.')
        return best_model