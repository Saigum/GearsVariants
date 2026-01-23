import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import ModuleList, Sequential, Linear, ReLU, Parameter
from torch_geometric.nn import GPSConv, SGConv
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from tqdm import tqdm
from lightning.fabric import Fabric
from torch_geometric.data import Data, DataLoader
import pickle
import os
import scanpy as sc
import warnings
from torch.utils.data.distributed import DistributedSampler
from gears.data_utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter
from gears.utils import print_sys, zip_data_download_wrapper, dataverse_download,\
                  filter_pert_in_go, get_genes_from_perts, tar_data_download_wrapper
from gears_changes import *
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


class PertDataDistributed(PertData_):
    def get_dataloader(self, batch_size, test_batch_size = None):
        """
        Get dataloaders for distributed training and testing. This method 
        overrides the base class method to use DistributedSampler.
        """
        if test_batch_size is None:
            test_batch_size = batch_size
            
        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name
       
        # Create cell graphs lists (which act as datasets)
        cell_graphs = {}
        if self.split == 'no_split':
            i = 'test'
            cell_graphs[i] = []
            for p in self.set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(self.dataset_processed[p])
                
            print_sys("Creating distributed dataloaders....")
            
            # Create a sampler for the test set
            test_sampler = DistributedSampler(cell_graphs['test'], shuffle=False)
            
            # Set up dataloader
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=test_batch_size, sampler=test_sampler)

            print_sys("Dataloaders created...")
            self.dataloader = {'test_loader': test_loader}
            return self.dataloader
        else:
            if self.split =='no_test':
                splits = ['train','val']
            else:
                splits = ['train','val','test']
            
            for i in splits:
                cell_graphs[i] = []
                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating distributed dataloaders....")
            
            # Create samplers for each split
            train_sampler = DistributedSampler(cell_graphs['train'], shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(cell_graphs['val'], shuffle=False)

            # Set up dataloaders. shuffle must be False when using a sampler.
            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=batch_size, sampler=val_sampler)
            
            if self.split !='no_test':
                test_sampler = DistributedSampler(cell_graphs['test'], shuffle=False)
                test_loader = DataLoader(cell_graphs['test'],
                                batch_size=test_batch_size, sampler=test_sampler)
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader}
            else: 
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader}
            print_sys("Done!")

class ZeroOutVariant(nn.Module):
    def __init__(self, pert_data: PertData_, args):
        super(ZeroOutVariant, self).__init__()
        self.pert_data = pert_data
        # device is now managed by Fabric, but we keep it in args for potential single-device use
        self.device = args.get('device', 'cpu') 
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
        self.gene_embeddings = nn.Embedding(self.num_genes, hidden_size)
        
        self.scalar_projector = nn.Sequential(nn.Linear(self.num_genes, self.num_genes * hidden_size // 2),
                                              nn.ReLU(),
                                              nn.Linear(self.num_genes * hidden_size // 2, self.num_genes * hidden_size),
                                              nn.ReLU(),
                                             )
        self.deviation_weight = nn.Parameter(torch.randn(self.num_genes, 1))
        self.bn_emb = nn.BatchNorm1d(self.num_genes * hidden_size)
        
        # We expect G_coexpress and its weight to be on the correct device later
        self.G_coexpress = args['G_coexpress']
        self.G_coexpress_weight = args['G_coexpress_weight']
        
        self.coexpress_encoder = ModuleList()
        for _ in range(self.num_layers_gene_pos):
            # Using a simple GCNConv as a replacement for the example's SGConv with GPSConv
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

            # The graph tensors are moved to the correct device in the train function
            base_edge_index = self.G_coexpress
            base_edge_weight = self.G_coexpress_weight

            new_edge_indices = []
            new_edge_weights = []
            node_offset = 0

            for i in range(num_graphs):
                pert_gene_idx = pert_idx[i].item()                
                if pert_gene_idx == -1: # Control cells
                    current_edges = base_edge_index
                    current_weights = base_edge_weight
                else: # Perturbed cells                 
                    edge_mask = (base_edge_index[0] == pert_gene_idx) | \
                                (base_edge_index[1] == pert_gene_idx)                    
                    keep_mask = ~edge_mask
                    current_edges = base_edge_index[:, keep_mask] 
                    current_weights = base_edge_weight[keep_mask]
                
                new_edge_indices.append(current_edges + node_offset)
                new_edge_weights.append(current_weights)
                node_offset += self.num_genes
                
            batched_edge_index = torch.cat(new_edge_indices, dim=1)
            batched_edge_weight = torch.cat(new_edge_weights, dim=0)
            
            for layer in self.coexpress_encoder:
                h = layer(h, batched_edge_index, batch_tensor, edge_attr=batched_edge_weight)
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
    config: dict = None,
    epochs=20,
    lr=1e-3,
    weight_decay=5e-4,
    wandb=None,
    loss_fct=None,
    # Use fabric.print instead of the raw print function
    print_sys=print
    ):

    # 1. Initialize Fabric
    # The DDP strategy is recommended for multi-GPU training.
    fabric = Fabric(accelerator="auto", strategy="ddp_find_unused_parameters_true", devices="auto")
    fabric.launch()

    device = fabric.device
    
    # Move graph data to the correct device once
    model.G_coexpress = model.G_coexpress.to(device)
    model.G_coexpress_weight = model.G_coexpress_weight.to(device)
    model.device = device # Update model's internal device attribute

    train_loader = pert_data.dataloader['train_loader']
    val_loader = pert_data.dataloader['val_loader']
    
    # This tensor must be on the correct device for the current process
    ctrl_expression = torch.tensor(
        np.mean(pert_data.adata.X[pert_data.adata.obs.condition == 'ctrl'],
                axis=0)).reshape(-1, ).to(device)
    
    pert_full_id2pert = dict(pert_data.adata.obs[['condition_name', 'condition']].values)
    dict_filter = {pert_full_id2pert[i]: j for i, j in
                        pert_data.adata.uns['non_zeros_gene_idx'].items() if
                        i in pert_full_id2pert}
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    # 2. Setup model, optimizer, and dataloaders with Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    
    fabric.print('Start Training...')
    min_val = np.inf
    best_model_path = "best_model_state.pt"
    
    # Disable tqdm on non-main processes to avoid multiple progress bars
    is_main_process = fabric.global_rank == 0
    
    with tqdm(range(epochs), desc="Training Epochs", disable=not is_main_process) as pbar:
        for epoch in pbar:
            model.train()
            total_loss = 0
            
            # Set epoch for sampler
            train_loader.sampler.set_epoch(epoch)

            for step, batch in enumerate(train_loader):
                # Data is automatically moved to the correct device by fabric.setup_dataloaders
                optimizer.zero_grad()
                
                y = batch.y 
                pred = model(batch)
                
                loss = loss_fct(pred, y, batch.pert,
                                ctrl=ctrl_expression, 
                                dict_filter=dict_filter,
                                direction_lambda=config['direction_lambda'])
                
                # 3. Use fabric.backward() for distributed backpropagation
                fabric.backward(loss)
                nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()

                total_loss += loss.item()
                if wandb and is_main_process:
                    wandb.log({'training_loss': loss.item()})

                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    fabric.print(log.format(epoch + 1, step + 1, loss.item()))
            
            avg_train_loss = total_loss / len(train_loader)
            if is_main_process:
                pbar.set_postfix(train_loss=avg_train_loss)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    y = batch.y
                    pred = model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                    ctrl=ctrl_expression, 
                                    dict_filter=dict_filter,
                                    direction_lambda=config['direction_lambda'])
                    total_val_loss += loss.item()
            
            avg_val_loss_tensor = torch.tensor(total_val_loss / len(val_loader), device=device)
            
            # 4. Gather validation loss from all processes for consistent checkpointing
            fabric.all_reduce(avg_val_loss_tensor, op="mean")
            avg_val_loss = avg_val_loss_tensor.item()
            
            if wandb and is_main_process:
                wandb.log({'val_loss': avg_val_loss})
            
            log = "Epoch {} Val Loss: {:.4f}"
            fabric.print(log.format(epoch + 1, avg_val_loss))

            if avg_val_loss < min_val:
                min_val = avg_val_loss
                # 5. Save checkpoint only from the main process
                if is_main_process:
                    # Use model.module to access the original nn.Module
                    state = {"model": model.module.state_dict()}
                    fabric.save(best_model_path, state)
                    fabric.print('Best model saved!')

            scheduler.step()

    fabric.print('Finished Training.')
    
    # 6. Load the best model on the main process and return it
    # This synchronizes all processes before loading the final model
    fabric.barrier()

    if is_main_process:
        checkpoint = fabric.load(best_model_path)
        # Unwrap the model to load the state dict
        model.module.load_state_dict(checkpoint["model"])
        return model.module
    
    return None

