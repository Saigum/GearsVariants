import torch
import numpy as np
from gears import GEARS
from gears.utils import loss_fct
from torch import nn
from gears.utils import get_similarity_network, GeneSimNetwork
from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis
from copy import deepcopy
import os
import csv
import torch.nn.functional as F
import networkx as nx # Assuming you need this for the GeneSimNetwork

# Assuming these classes/modules are in the same directory or accessible via appropriate imports
from gears_mech import GEARS_MECH
from gears_no_coexpress import GEARS_No_Coexpress
from gene_sim_network_khops import GeneSimNetworkKHops
from gears.model import GEARS_Model
from torch import optim
from torch.optim.lr_scheduler import LRScheduler,StepLR
from gears.utils import print_sys

class GEARS_PRETRAIN(GEARS):
    def __init__(self, pert_data, device='cuda', weight_bias_track=False, proj_name='GEARS', exp_name='GEARS'):

        self.weight_bias_track = weight_bias_track

        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)
            self.wandb = wandb
        else:
            self.wandb = None

        self.device = device
        self.config = None

        self.dataloader = pert_data.dataloader ##
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred = {}
        self.saved_logvar_sum = {}

        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition.values == 'ctrl'],
                    axis=0)).reshape(-1, ).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']

        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                            enumerate(self.pert_list) if pert in self.gene_list}
        self.hvg_idx = getattr(pert_data,"hvg_idx", None)

    def update(self,
                pert_data):
        ''' Function to update the pert_data the model is using, ie to switch from control pert_data to perturbed pert_data. '''
        self.dataloader = pert_data.dataloader ##
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition.values == 'ctrl'],
                    axis=0)).reshape(-1, ).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']

        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                            enumerate(self.pert_list) if pert in self.gene_list}
        self.hvg_idx = getattr(pert_data,"hvg_idx", None)


        ## calculating co expression similarity graph
        edge_list = {}
        edge_list["coexpress"] = get_similarity_network(network_type='co-express',
                                            adata=self.adata,
                                            threshold=self.graph_config["coexpress_threshold"],
                                            k=self.graph_config["num_similar_genes_co_express_graph"],
                                            data_path=self.data_path,
                                            data_name=self.dataset_name,
                                            split=self.split, seed=self.seed,
                                            train_gene_set_size=self.train_gene_set_size,
                                            set2conditions=self.set2conditions)
        ## checking if multi-hop
        if self.config["num_hops"] > 1:
            sim_network = GeneSimNetworkKHops(edge_list["coexpress"],self.pert_list,self.node_map)
            sim_network.add_zero_weight_khop_edges(
                k=self.config["num_hops"],
                m=self.config["num_add"],)
        else:
            sim_network = GeneSimNetwork(edge_list["coexpress"], self.pert_list, node_map = self.node_map_pert)
        self.config['G_coexpress'] = sim_network.edge_index
        self.config['G_coexpress_weight'] = sim_network.edge_weight
        edge_list["coexpress"] = get_similarity_network(network_type='go',
                                        adata=self.adata,
                                        threshold=self.graph_config["coexpress_threshold"],
                                        k=self.graph_config["num_similar_genes_go_graph"],
                                        pert_list=self.pert_list,
                                        data_path=self.data_path,
                                        data_name=self.dataset_name,
                                        split=self.split, seed=self.seed,
                                        train_gene_set_size=self.train_gene_set_size,
                                        set2conditions=self.set2conditions,
                                        default_pert_graph=self.default_pert_graph)

        self.config['G_go'] = sim_network.edge_index
        self.config['G_go_weight'] = sim_network.edge_weight




    def model_initialize(self, hidden_size = 64,
                            num_go_gnn_layers = 1,
                            num_gene_gnn_layers = 1,
                            decoder_hidden_size = 16,
                            num_similar_genes_go_graph = 20,
                            num_similar_genes_co_express_graph = 20,
                            coexpress_threshold = 0.4,
                            uncertainty = False,
                            uncertainty_reg = 1,
                            direction_lambda = 1e-1,
                            G_go = None,
                            G_go_weight = None,
                            G_coexpress = None,
                            G_coexpress_weight = None,
                            no_perturb = False,
                            gears_model=0,
                            num_hops=1,
                            num_add=1,
                            num_heads=4,
                            gated=False,
                            **kwargs
                        ):

        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        uncertainty: bool
            whether or not to turn on uncertainty mode, default False
        uncertainty_reg: float
            regularization term to balance uncertainty loss and prediction loss, default 1
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False
        gears_model: int
            0- original model, 1- expression embedding, 2 - GAT, 3 - TransformerConv, 4- No Coexpression 5- No perturbation.
        num_hops:int
            1- No additional zero weight edges will be added, else num_hops represents the max number of hops a target node is away from the source node, for us to consider adding a zero-weight edge between source-target
        num_add:int,
            1- Max number of edges of the above type we add for each node
        gated: bool
            Boolean expression that represents as to whether we're gating the expression embedding being added to the base embedding.

        Returns
        -------
        None
        """

        self.config = {
            'hidden_size': hidden_size,
            'num_go_gnn_layers' : num_go_gnn_layers,
            'num_gene_gnn_layers' : num_gene_gnn_layers,
            'decoder_hidden_size' : decoder_hidden_size,
            'num_similar_genes_go_graph' : num_similar_genes_go_graph,
            'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
            'coexpress_threshold': coexpress_threshold,
            'uncertainty' : uncertainty,
            'uncertainty_reg' : uncertainty_reg,
            'direction_lambda' : direction_lambda,
            'G_go': G_go,
            'G_go_weight': G_go_weight,
            'G_coexpress': G_coexpress,
            'G_coexpress_weight': G_coexpress_weight,
            'device': self.device,
            'num_genes': self.num_genes,
            'num_perts': self.num_perts,
            'no_perturb': no_perturb,
            'gears_model': gears_model,
            "num_hops":num_hops,
            "num_add":num_add,
            'num_heads': num_heads,
            'gated':gated
        }

        self.graph_config = {
            "threshold":coexpress_threshold,
            "k_coexpress":num_similar_genes_co_express_graph,
            "k_go":num_similar_genes_go_graph
        }
        if self.wandb:
            self.wandb.config.update(self.config)

        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                                adata=self.adata,
                                                threshold=coexpress_threshold,
                                                k=num_similar_genes_co_express_graph,
                                                data_path=self.data_path,
                                                data_name=self.dataset_name,
                                                split=self.split, seed=self.seed,
                                                train_gene_set_size=self.train_gene_set_size,
                                                set2conditions=self.set2conditions)
            ## checking if multi-hop
            if self.config["num_hops"] > 1:
                sim_network = GeneSimNetworkKHops(edge_list,self.pert_list,self.node_map)
                sim_network.add_zero_weight_khop_edges(
                    k=self.config["num_hops"],
                    m=self.config["num_add"],)
            else:
                sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map)

            # sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight

        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                                adata=self.adata,
                                                threshold=coexpress_threshold,
                                                k=num_similar_genes_go_graph,
                                                pert_list=self.pert_list,
                                                data_path=self.data_path,
                                                data_name=self.dataset_name,
                                                split=self.split, seed=self.seed,
                                                train_gene_set_size=self.train_gene_set_size,
                                                set2conditions=self.set2conditions,
                                                default_pert_graph=self.default_pert_graph)

            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight

        if self.config["gears_model"] == 0 :
            self.model = GEARS_Model(self.config).to(self.device)
        elif self.config["gears_model"] == 1:
            self.model = GEARS_MECH(self.config).to(self.device)
        #     self.model = GEARS_EMBED(self.config).to(self.device)
        # elif self.config["gears_model"] == 2:
        #     self.model = GEARS_GAT(self.config).to(self.device)
        # elif self.config["gears_model"] == 3:
        #     self.model = GEARS_Transformer(self.config).to(self.device)
        elif self.config["gears_model"] == 4:
            self.model = GEARS_No_Coexpress(self.config).to(self.device)
        # elif self.config["gears_model"] == 5:
        #     self.model = GEARS_No_Perturb(self.config).to(self.device)
        # elif self.config["gears_model"] == 6:
        #     self.model = GEARS_SelfAttn(self.config).to(self.device)
        # elif self.config["gears_model"] == 7:
        #     self.model = GEARS_CELL(self.config).to(self.device)
        # elif self.config["gears_model"] == 8:
        #     self.model = VariantTiny(self.config).to(self.device)
        # elif self.config["gears_model"] == 9:
        #     self.model = Variant(self.config).to(self.device)

        self.best_model = deepcopy(self.model)

    def pretrain_ctrl(self, epochs = 20,
                lr = 1e-3,
                weight_decay = 5e-4
                ):
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']

        print(f"Length of train_loader: {len(train_loader)}")
        print(f"Length of val_loader {len(val_loader)}")

        ## This will change the model's forward, to reconstruction only

        self.model.pretrain_phase = True

        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        min_val = np.inf
        print_sys('Start Training...')
        hvg_idx = self.hvg_idx
        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                x = batch.x
                pred = self.model(batch)
                ## autoencoder style loss with directionality enforced.
                ## Consider switching to MSE instead ?
                loss = loss_fct(pred, x, batch.pert,
                                ctrl = self.ctrl_expression,
                                dict_filter = self.dict_filter,
                                direction_lambda = self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()
                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})
                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}"
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))
            scheduler.step()
            train_run = evaluate(train_loader,self.model,False,self.device)
            val_run = evaluate(val_loader,self.model,False,self.device)
            ## computing metrics :
            try:
                train_metrics, _ = compute_metrics(train_run)
                val_metrics, _ = compute_metrics(val_run)
                ## print epoch performance
                log = "Epoch {}: Train Overall MSE: {:.4f} " \
                        "Validation Overall MSE: {:.4f}. "
                print_sys(log.format(epoch + 1, train_metrics['mse'],
                                    val_metrics['mse']))
                if(min_val > val_metrics["mse"]):
                    print(f"New Best model has val_mse: {val_metrics['mse']}")
                    self.best_model = deepcopy(self.model)
            except Exception as e:
                print(f"Error while computing metrics: {e}")
            ## computing loss on the highly variable gene set.
            ## computing validation loss of the validation dataloader ?
            ## consider adding evaluation for reconstruction via the autoencoder over here ?
        print(f"Best Val Loss: {min_val}")
        print("Done Training ....")

    def train(self, epochs = 20,
                lr = 1e-3,
                weight_decay = 5e-4
                ):
        print("Training Phase with perturbing graph .....")

        self.model.pretrain_phase = False

        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
        print(f"Length of train_loader: {len(train_loader)}")
        print(f"Length of val_loader {len(val_loader)}")
        from torch import optim
        from torch.optim.lr_scheduler import StepLR
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        min_val = np.inf
        print_sys('Start Training...')
        hvg_idx = self.hvg_idx
        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                x = batch.x
                pred = self.model(batch)
                ## autoencoder style loss with directionality enforced.
                ## Consider switching to MSE instead ?
                loss = loss_fct(pred, y, batch.pert,
                                ctrl = self.ctrl_expression,
                                dict_filter = self.dict_filter,
                                direction_lambda = self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()
                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})
                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}"
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))
            scheduler.step()
            train_run = evaluate(train_loader,self.model,False,self.device)
            val_run = evaluate(val_loader,self.model,False,self.device)
            ## computing metrics :
            try:
                train_metrics, _ = compute_metrics(train_run)
                val_metrics, _ = compute_metrics(val_run)
                log = "Epoch {}: Train Overall MSE: {:.4f} " \
                        "Validation Overall MSE: {:.4f}. "
                print_sys(log.format(epoch + 1, train_metrics['mse'],
                                    val_metrics['mse']))
                if(min_val > val_metrics["mse"]):
                    print(f"New Best model has val_mse: {val_metrics['mse']}")
                    min_val = val_metrics["mse"]
                    self.best_model = deepcopy(self.model)
            except Exception as e:
                print(f"Error while computing metrics: {e}")
            ## computing loss on the highly variable gene set.
            ## computing validation loss of the validation dataloader ?
            ## consider adding evaluation for reconstruction via the autoencoder over here

        ### TODO: Add test metrics.
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        test_res = evaluate(test_loader, self.best_model,
                            self.config['uncertainty'], self.device)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        print_sys(log.format(test_metrics['mse_de']))
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        metrics = ['pearson_delta']
        # accumulate sums and counts per metric
        metric_sums = {}
        metric_counts = {}
        
        for pert, metrics in out.items():
            for metric, value in metrics.items():
                metric_sums[metric] = metric_sums.get(metric, 0) + value
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

        # compute averages
        metric_avgs = {m: metric_sums[m] / metric_counts[m] for m in metric_sums}

        print(metric_avgs)



        print(f"Best Val Loss: {min_val}")
        print("Done Training ....")
        return out,out_non_dropout
    def _load_ground_truth_graphs(self, 
                                  base_path="/kaggle/input/coexpressiongraphs/Downloads/coexpression_graphs", 
                                  filename="/_42_5045_0.4_20_co_expression_network.csv"): # Corrected filename
        """
        Loads ground truth co-expression graphs from CSV files.

        This method builds a dictionary: {pert_name: 1D_weight_tensor}
        
        CRITICAL ASSUMPTIONS:
        - `self.node_map`: A dict {'gene_name': gene_idx} must exist.
        - `self.model.G_coexpress`: The edge_index tensor [2, num_edges] must exist.
        - `self.node_map_pert`: A dict {'pert_name': pert_idx} must exist.
        - Files are located at: {base_path}/pert_ctrl+{pert_name}/{filename}
        """
        print("Attempting to load ground truth co-expression graphs...")
        
        # --- 1. Check for required attributes ---
        if not hasattr(self, 'node_map'):
            print("  Error: `self.node_map` (for genes) not found. Cannot load ground truth graphs.")
            return {}
        if not hasattr(self, 'model') or not hasattr(self.model, 'G_coexpress'):
            print("  Error: `self.model.G_coexpress` (edge_index) not found. Cannot load ground truth graphs.")
            return {}
        if not hasattr(self, 'node_map_pert'):
            print("  Error: `self.node_map_pert` not found. Cannot load ground truth graphs.")
            return {}

        # --- 2. Build Edge Index Map for fast lookup ---
        # This maps (u_idx, v_idx) -> position in the edge_weight tensor
        try:
            edge_index = self.model.G_coexpress.cpu()
            num_edges = edge_index.shape[1]
            edge_index_map = {}
            for i in range(num_edges):
                u_idx = edge_index[0, i].item()
                v_idx = edge_index[1, i].item()
                edge_index_map[(u_idx, v_idx)] = i
            
            gene_name_to_idx = self.node_map # Use self.node_map for gene names
            print(f"  Built edge index map for {num_edges} edges.")
        except Exception as e:
            print(f"  Error building edge_index_map: {e}. Aborting GT load.")
            return {}

        # --- 3. Get all perturbation names, including 'ctrl' ---
        # (Based on the logic in your original function)
        pert_names = list(self.node_map_pert.keys())
        if 'ctrl' not in pert_names:
            pert_names.append('ctrl')

        gt_graph_store = {}
        
        # --- 4. Iterate and load each GT graph file ---
        for pert_name in pert_names:
            # Format: coexpression_graphs/pert_ctrl+GATA1/edges.csv
            # Adjusted to match the `_42_5045_0.4_20_co_expression_network.csv` pattern
            file_path = os.path.join(base_path, f"pert_{pert_name}+ctrl", filename)
            
            if not os.path.exists(file_path):
                # print(f"  Info: No GT graph file found for '{pert_name}' at {file_path}. Skipping.")
                continue

            try:
                # Initialize a zero-vector for this GT graph's weights
                gt_weights_tensor = torch.zeros(num_edges, device=self.device)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader) # Skip header
                    
                    edges_found = 0
                    edges_not_in_index = 0
                    
                    for row in reader:
                        if len(row) < 3: continue # Skip malformed rows
                        source_name, target_name, weight_str = row[0], row[1], row[2]
                        
                        u_idx = gene_name_to_idx.get(source_name)
                        v_idx = gene_name_to_idx.get(target_name)
                        
                        if u_idx is None or v_idx is None:
                            continue # Gene name not in our map
                        
                        try:
                            weight = float(weight_str)
                        except ValueError:
                            continue # Invalid weight
                        
                        # Check for the edge in our model's edge_index
                        edge_pos = edge_index_map.get((u_idx, v_idx))
                        
                        if edge_pos is not None:
                            gt_weights_tensor[edge_pos] = weight
                            edges_found += 1
                        else:
                            # Optional: Check for reverse edge if graph is undirected
                            edge_pos_rev = edge_index_map.get((v_idx, u_idx))
                            if edge_pos_rev is not None:
                                gt_weights_tensor[edge_pos_rev] = weight
                                edges_found += 1
                            else:
                                edges_not_in_index += 1
                
                gt_graph_store[pert_name] = gt_weights_tensor.detach()
                print(f"  Loaded GT graph for '{pert_name}'. "
                      f"Mapped {edges_found} edges. ({edges_not_in_index} edges from file were not in model's G_coexpress).")

            except Exception as e:
                print(f"  Error loading GT graph for '{pert_name}' from {file_path}: {e}")

        print(f"Successfully loaded {len(gt_graph_store)} ground truth graphs.")
        return gt_graph_store

    # --- NEW HELPER METHOD 2: Pearson Correlation ---
    
    def calculate_pearson(self, x, y):
        """Calculates the Pearson correlation coefficient between two 1D tensors."""
        # Center the vectors
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)
        
        # Use cosine similarity on centered vectors
        # Add epsilon for numerical stability in case of zero variance
        epsilon = 1e-8
        cos_sim = F.cosine_similarity(x_centered.unsqueeze(0), 
                                      y_centered.unsqueeze(0), 
                                      eps=epsilon)
        return cos_sim.item()

    # --- UPDATED Main Function ---
    
    def look_cell_graphs(self, dataloader):
        """
        Analyzes and compares predicted graph edge weights for different
        perturbations against:
        1. The baseline control graph.
        2. The corresponding ground truth co-expression graph.
        
        Assumes:
        - self.model.G_coexpress_weight: The baseline control edge weights (1D Tensor).
        - self.model.compute_graphs() returns: (pred, {"graphs": [Data, Data, ...]}).
        - The input `batch.pert` corresponds 1-to-1 with the list of output graphs.
        - `self.node_map_gene` exists for loading GT graphs.
        """
        
        print("Starting graph comparison analysis...")
        
        # --- 1. Get Baseline Control Graph ---
        try:
            control_weights = self.model.G_coexpress_weight.to(self.device).detach()
            num_edges = control_weights.shape[0]
            
            if self.model.G_coexpress_weight.shape[0] != self.model.G_coexpress.shape[1]:
                 print(f"Warning: Control weights shape ({control_weights.shape[0]}) "
                       f"does not match edge_index shape ({self.model.G_coexpress.shape[1]}).")

            print(f"Loaded baseline control graph with {num_edges} edges.")
            
        except AttributeError:
            print("Error: `self.model.G_coexpress_weight` or `self.model.G_coexpress` not found. Aborting.")
            return
        
        # --- 1.5. [NEW] Load Ground Truth Graphs ---
        # This calls the new helper method and stores GT graphs in a dictionary
        # self.gt_graph_store = { 'pert_name_1': tensor, 'pert_name_2': tensor, ... }
        try:
            # This assumes _load_ground_truth_graphs is a method of the same class
            self.gt_graph_store = self._load_ground_truth_graphs() 
        except Exception as e:
            print(f"Error during Ground Truth graph loading: {e}. Proceeding without GT comparison.")
            self.gt_graph_store = {} # Ensure it's a dict
        
        # --- 2. Create Perturbation Name Map ---
        try:
            idx_to_pert_name = {v: k for k, v in self.node_map_pert.items()}
            idx_to_pert_name[-1] = 'ctrl'
        except AttributeError:
            print("Error: `self.node_map_pert` not found. Aborting.")
            return

        # Dictionary to store aggregated results
        results_store = {}

        # --- 3. Iterate Through Dataloader ---
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                batch.to(self.device)
                
                # Run the model
                pred, graph_data = self.model.compute_graphs(batch)

                # --- 4. Get Predicted Graphs and Labels ---
                predicted_graphs_list = graph_data.get("graphs")
                if predicted_graphs_list is None or not isinstance(predicted_graphs_list, list):
                    print(f"Error: 'graphs' key not in model output or is not a list. Aborting batch {step}.")
                    continue
                
                num_graphs_in_batch = len(predicted_graphs_list)
                pert_indices = np.array(batch.pert_idx) 

                if num_graphs_in_batch != pert_indices.shape[0]:
                    print(f"Warning: Mismatch in batch {step}. "
                          f"Model output {num_graphs_in_batch} graphs, but input batch had {pert_indices.shape[0]} labels. Skipping.")
                    continue

                # --- 5. Compare Each Graph in Batch ---
                for i in range(num_graphs_in_batch):
                    
                    pert_idx = pert_indices[i][0].item() # Accessing the first element of the tensor
                    pert_name = idx_to_pert_name.get(pert_idx, f"Unknown_idx_{pert_idx}")
                    
                    data_graph_i = predicted_graphs_list[i]
                    predicted_weights_graph_i = data_graph_i.edge_weight.detach()
                    
                    if predicted_weights_graph_i.shape[0] != num_edges:
                        print(f"Warning: Shape mismatch in batch {step}, graph {i} (pert: {pert_name}). "
                              f"Expected {num_edges} edges, but predicted graph has {predicted_weights_graph_i.shape[0]}. Skipping graph.")
                        continue
                    
                    # --- 5a. Comparison vs. Control (Original Logic) ---
                    
                    edge_diff = predicted_weights_graph_i - control_weights
                    mean_abs_diff = torch.mean(torch.abs(edge_diff)).item()
                    
                    epsilon = 1e-6
                    control_is_zero = torch.abs(control_weights) < epsilon
                    pred_is_zero = torch.abs(predicted_weights_graph_i) < epsilon
                    
                    new_edges_mask = control_is_zero & (~pred_is_zero)
                    num_new_edges = torch.sum(new_edges_mask).item()
                    
                    avg_new_edge_strength = 0.0
                    if num_new_edges > 0:
                        avg_new_edge_strength = torch.mean(torch.abs(predicted_weights_graph_i[new_edges_mask])).item()

                    lost_edges_mask = (~control_is_zero) & pred_is_zero
                    num_lost_edges = torch.sum(lost_edges_mask).item()

                    # --- 5b. [NEW] Comparison vs. Ground Truth ---
                    
                    # Get the pre-loaded GT graph tensor for this perturbation
                    gt_weights = self.gt_graph_store.get(pert_name)
                    
                    gt_pearson_corr = np.nan
                    gt_cos_sim = np.nan
                    
                    if gt_weights is not None:
                        # Check for shape mismatch (should not happen if loading was correct)
                        if gt_weights.shape[0] == num_edges:
                            # Use new helper function for Pearson
                            gt_pearson_corr = self.calculate_pearson(predicted_weights_graph_i, gt_weights)
                            
                            # Calculate Cosine Similarity
                            gt_cos_sim = F.cosine_similarity(
                                predicted_weights_graph_i.unsqueeze(0), 
                                gt_weights.unsqueeze(0)
                            ).item()
                        else:
                            print(f"Warning: GT graph for {pert_name} has shape {gt_weights.shape[0]} but expected {num_edges}. Skipping GT comparison.")
                    # else: No GT graph was loaded for this pert_name, metrics will remain np.nan

                    # --- 6. Store Results (Updated) ---
                    if pert_name not in results_store:
                        results_store[pert_name] = {
                            'count': 0,
                            'mean_abs_diff': [],
                            'num_new_edges': [],
                            'avg_new_edge_strength': [],
                            'num_lost_edges': [],
                            'gt_pearson_corr': [],  # New
                            'gt_cos_sim': []        # New
                        }
                    
                    results_store[pert_name]['count'] += 1
                    results_store[pert_name]['mean_abs_diff'].append(mean_abs_diff)
                    results_store[pert_name]['num_new_edges'].append(num_new_edges)
                    if num_new_edges > 0:
                        results_store[pert_name]['avg_new_edge_strength'].append(avg_new_edge_strength)
                    results_store[pert_name]['num_lost_edges'].append(num_lost_edges)
                    
                    # Append GT metrics (will append np.nan if not found)
                    results_store[pert_name]['gt_pearson_corr'].append(gt_pearson_corr)
                    results_store[pert_name]['gt_cos_sim'].append(gt_cos_sim)

        # --- 7. Print Final Aggregated Report (Updated) ---
        print("\n--- 📊 Graph Comparison Report ---")
        print(f"Analyzed {sum(v['count'] for v in results_store.values())} total graphs across {len(results_store)} perturbations.")
        
        for pert_name in sorted(results_store.keys(), key=lambda x: (x == 'ctrl', x)): # Sort, put 'ctrl' first
            stats = results_store[pert_name]
            count = stats['count']
            
            # --- Original Averages ---
            avg_mean_diff = np.mean(stats['mean_abs_diff'])
            avg_new_edges = np.mean(stats['num_new_edges'])
            avg_lost_edges = np.mean(stats['num_lost_edges'])
            
            if stats['avg_new_edge_strength']:
                avg_new_edge_strength = np.mean(stats['avg_new_edge_strength'])
            else:
                avg_new_edge_strength = 0.0

            # --- [NEW] GT Averages ---
            # Use np.nanmean to safely ignore 'nan' values if a GT graph was missing
            avg_gt_pearson = np.nanmean(stats['gt_pearson_corr'])
            avg_gt_cos_sim = np.nanmean(stats['gt_cos_sim'])

            print(f"\n**Perturbation: {pert_name} (n={count})**")
            print(f"  --- vs. Control Graph ---")
            print(f"    Avg. Mean Absolute Edge Diff: {avg_mean_diff:.5f}")
            print(f"    Avg. New Edges Created (0 -> non-0): {avg_new_edges:.2f}")
            print(f"    Avg. Strength of New Edges: {avg_new_edge_strength:.5f}")
            print(f"    Avg. Edges Lost (non-0 -> 0): {avg_lost_edges:.2f}")
            
            print(f"  --- vs. Ground Truth Graph ---")
            # Only print GT stats if they are not all 'nan'
            if not np.isnan(avg_gt_pearson):
                print(f"    Avg. Pearson Correlation: {avg_gt_pearson:.4f}")
                print(f"    Avg. Cosine Similarity: {avg_gt_cos_sim:.4f}")
            else:
                print(f"    (No Ground Truth graph loaded for this perturbation)")

        print("-----------------------------------")
        
        return results_store
    def get_perturbation_graphs(self,
                                dataloader,
                                base_path="/kaggle/input/coexpressiongraphs/Downloads/coexpression_graphs",
                                filename="/_42_5045_0.4_20_co_expression_network.csv"): # Corrected filename
        
        pert_names = list(self.node_map_pert.keys())
        if 'ctrl' not in pert_names:
            pert_names.append('ctrl')

        gt_graph_store = {}
        
        # --- 3. Iterate and load each GT graph file ---
        graphs = {}
        for pert_name in pert_names:
            file_path = os.path.join(base_path, f"pert_{pert_name}+ctrl", filename)
            ## its in source,target form, so loading the entire graph using networkX
            if not os.path.exists(file_path):
                print(f"  Info: No GT graph file found for '{pert_name}' at {file_path}. Skipping.")
                continue
            # Source - https://stackoverflow.com/a
            # Posted by ducminh, modified by community. See post 'Timeline' for change history
            # Retrieved 2025-11-13, License - CC BY-SA 3.0

            Data = open(file_path, "r")
            next(Data, None)  # skip the first line in the input file
            Graphtype = nx.Graph()
            G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                                nodetype=int, data=(('weight', float),))
            ## computing graph metrics wrt the control graph 
            graphs[pert_name] = G ##
            
            ## remapping via node_map_gene
            mapped_G = nx.relabel_nodes(G, self.node_map) # Use self.node_map for gene names