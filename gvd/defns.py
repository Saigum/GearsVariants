# --- Standard library ---
import os
import csv
import random
import pickle
import operator
from functools import reduce
from copy import deepcopy

# --- Core scientific stack ---
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# --- Graph + geometry ---
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
import networkx as nx

# --- Single-cell / bio ---
import anndata as ad
import scanpy as sc

# --- Utils ---
from tqdm.auto import tqdm

# --- GEARS ecosystem ---
from gears import PertData, GEARS
from gears.model import GEARS_Model
from gears.data_utils import DataSplitter, print_sys
from gears.utils import (
    loss_fct,
    get_similarity_network,
    GeneSimNetwork,
    np_pearson_cor,
)
from gears.inference import (
    deeper_analysis,
    non_dropout_analysis,
    evaluate,
    compute_metrics,
)
from accelerate import Accelerator
from newclasses import GEARS_MECH,GEARS_No_Coexpress

import os
import csv
from collections import defaultdict


from torch.utils.data import Dataset
from gears.data_utils import get_genes_from_perts,get_DE_genes,get_dropout_non_zero_genes,DataSplitter,print_sys
from gears.utils import zip_data_download_wrapper,dataverse_download
import torch


class PertData_(PertData):
    def prepare_split(self, split = 'simulation',
                      seed = 1,
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None,
                      only_test_set_perts = False,
                      test_pert_genes = None,
                      split_dict_path=None,
                      val_size=0.1):
        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split', 'custom']
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None
        self.val_size = val_size

        if split == 'custom':
            try:
                with open(split_dict_path, 'rb') as f:
                    self.set2conditions = pickle.load(f)
            except:
                    raise ValueError('Please set split_dict_path for custom split')
            return

        self.train_gene_set_size = train_gene_set_size
        split_folder = os.path.join(self.dataset_path, 'splits')
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' \
                                       +  str(train_gene_set_size) + '.pkl'
        split_path = os.path.join(split_folder, split_file)

        if test_perts:
            split_path = split_path[:-4] + '_' + test_perts + '.pkl'

        if os.path.exists(split_path):
            print('here1')
            print_sys("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            print_sys("Creating new splits....")
            if test_perts:
                test_perts = test_perts.split('_')

            if split in ['simulation', 'simulation_single']:
                # simulation split
                DS = DataSplitter(self.adata, split_type=split)

                adata, subgroup = DS.split_data(train_gene_set_size = train_gene_set_size,
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup

            elif split[:5] == 'combo':
                # combo perturbation
                split_type = 'combo'
                seen = int(split[-1])

                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split('_')

                DS = DataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)

            elif split == 'single':
                # single perturbation
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,val_size=val_size,
                                      seed=seed)

            elif split == 'no_test':
                # no test set
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(seed=seed)

            elif split == 'no_split':
                # no split
                adata = self.adata
                adata.obs['split'] = 'test'

            set2conditions = dict(adata.obs.groupby('split').agg({'condition':
                                                        lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()}
            pickle.dump(set2conditions, open(split_path, "wb"))
            print_sys("Saving new splits at " + split_path)

        self.set2conditions = set2conditions

        if split == 'simulation':
            print_sys('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print_sys(i + ':' + str(len(j)))
        print_sys("Done!")


    def load(self, data_name=None, data_path=None):
        ## void return anyways
        print(data_name)
        super().load(data_name, data_path)
        ## finding out hvg index set
        # sc.pp.neighbors(self.adata, n_neighbors=15, use_rep='X')
        # sc.pp.highly_variable_genes(self.adata, n_top_genes=2000, subset=False, flavor='seurat_v3')
        # self.hvg_idx = self.adata.var['highly_variable'].to_numpy().nonzero()[0]
    def get_dataloader(self, batch_size, test_batch_size = None):
        """
        Get dataloaders for training and testing

        Parameters
        ----------
        batch_size: int
            Batch size for training
        test_batch_size: int
            Batch size for testing

        Returns
        -------
        dict
            Dictionary of dataloaders

        """
        if test_batch_size is None:
            test_batch_size = batch_size

        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name

        # Create cell graphs
        cell_graphs = {}
        if self.split == 'no_split':
            i = 'test'
            cell_graphs[i] = []
            for p in self.set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating dataloaders....")
            # Set up dataloaders
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)

            print_sys("Dataloaders created...")
            return {'test_loader': test_loader}
        else:
            if self.split =='no_test':
                splits = ['train','val']
            else:
                splits = ['train','val','test']
            print(self.set2conditions)
            for i in splits:
                cell_graphs[i] = []
                if i in self.set2conditions:
                    for p in self.set2conditions[i]:
                        cell_graphs[i].extend(self.dataset_processed[p])
            # print(cell_graphs)
            print_sys("Creating dataloaders....")

            # Set up dataloaders
            if(len(cell_graphs["val"]) == 0):
                ## give a subset of entries to val from train.
                shuffled_list = cell_graphs["train"].copy()
                random.shuffle(shuffled_list)
                split_index = int(self.val_size*len(shuffled_list))
                cell_graphs["val"] = shuffled_list[:split_index]
                cell_graphs["train"] = shuffled_list[split_index:]



            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, shuffle=True, drop_last = True)
            if len(cell_graphs["val"])>0:
                val_loader = DataLoader(cell_graphs['val'],
                                    batch_size=batch_size, shuffle=True)
            else:
                val_loader = None
            if len(cell_graphs["test"])>0:
                test_loader = DataLoader(cell_graphs['val'],
                                    batch_size=batch_size, shuffle=True)
            else:
                test_loader = None

            if self.split !='no_test':
                test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader}

            else:
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader}
            print_sys("Done!")

def rbf_kernel(x, y, gamma=None):
    """Computes the RBF kernel between x and y."""
    if gamma is None:
        gamma = 1.0 / x.size(1)

    
    ## assume x and y are of same batch_size, here they are FeatDim,1 and 1,FeatDim, so FeatDim,FeatDim 
    diff = x - y
    dist_sq = torch.sum(diff ** 2, dim=-1)
    K = torch.exp(-gamma * dist_sq)
    return K

def maximum_mean_discrepancy(x, y, gamma=None):
    """Computes the Maximum Mean Discrepancy (MMD) between x and y."""
    ## need to obtain pairwise_distance 
    batch_size = x.size(0)
    
    ## quick reshaping for broadcasting to compute pair-wise differences.
    K_xx = rbf_kernel(x[:,None], x[None,:], gamma)
    K_yy = rbf_kernel(y[:,None], y[None,:], gamma)
    K_xy = rbf_kernel(x[:,None], y[None,:], gamma)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd

def lfc_mmd(pred, y,perts ,ctrl,dict_filter ,direction_lambda):
    "Calculates the loss between predicted and actual expression changes. MMD here is weighted by the logfoldchange of the ctrl wrt ground truth perturbed"
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    batch_size = pred.shape[0]
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in set(perts):
        perturbed_indices = np.where(perts == p)[0]
        
        if p != "ctrl":
            retain_idx = dict_filter[p]
        else:
            retain_idx = np.arange(pred.shape[1])
        
        pred_p = pred[perturbed_indices][:, retain_idx]
        y_p = y[perturbed_indices][:, retain_idx]
        ctrl = ctrl[retain_idx]
        ## log-foldchange wrt average control
        logfoldchange_pred = torch.log2((y+ 1)/(ctrl + 1))
        
        ## scaling both pred and y  by lfc magnitude of ctrl and groundt truth
        y_p = y_p * torch.abs(logfoldchange_pred).mean(dim=0)
        pred_p = pred_p * torch.abs(logfoldchange_pred).mean(dim=0)
        
        losses = losses + maximum_mean_discrepancy(pred_p,y_p)
    
    return losses/len(set(perts)) 


        
    
    
class GeneSimNetworkKHops():
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices
    """
    def __init__(self, edge_list, gene_list, node_map):
        """
        Initialize GeneSimNetwork class
        """
        self.edge_list = edge_list
        self.gene_list = gene_list
        self.node_map = node_map
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        self._update_tensors()

    def _update_tensors(self):
        """Helper to regenerate tensors from the current nx.Graph state."""
        if not self.G.edges:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_weight = torch.empty((0,), dtype=torch.float)
            return

        # print(self.node_map)
        # print(self.G.edges)
        edge_index_ = [(self.node_map[e[0]], self.node_map[e[1]]) for e in self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T

        edge_attr = nx.get_edge_attributes(self.G, 'importance')
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

    # --- UPDATED METHOD ---

    def add_zero_weight_khop_edges(self, k, m):
        """
        For each node, finds all nodes reachable within k-hops. From this set,
        it calculates the max-multiplicative-strength path for each.
        It then adds 'm' zero-weight edges to the unconnected nodes with
        the highest strength.

        Args:
            k (int): The maximum hop distance to search.
            m (int): The number of new edges to add per node.
        """
        if k <= 0:
            print("k must be a positive integer.")
            return

        # Changed print statement
        print(f"Graph modification in progress. Original edge count: {len(self.edge_index[0])}")
        new_edges_to_add = []

        # Added tqdm wrapper to the main loop
        for start_node in tqdm(list(self.G.nodes()), desc="Processing nodes"):
            potential_edges = []

            # 1. OPTIMIZATION: Get the subgraph of all nodes reachable
            #    within k hops using nx.ego_graph.
            ego_graph = nx.ego_graph(self.G, n=start_node, radius=k)

            # 2. Iterate ONLY over this smaller set of reachable nodes
            for target_node in ego_graph.nodes():
                # Exclude self-loops and existing direct edges
                if start_node == target_node or self.G.has_edge(start_node, target_node):
                    continue

                # 3. Find all simple paths (this is the required slow part)
                #    We still search on the *original graph* (self.G)
                paths = nx.all_simple_paths(self.G,
                                            source=start_node,
                                            target=target_node,
                                            cutoff=k)

                max_strength = 0.0
                for path in paths:
                    if len(path) > 1:
                        # Calculate multiplicative strength
                        strength = reduce(operator.mul,
                                        (self.G[u][v]['importance'] for u, v in zip(path[:-1], path[1:])))
                        if strength > max_strength:
                            max_strength = strength

                # If a path was found, store this as a potential edge
                if max_strength > 0:
                    potential_edges.append((target_node, max_strength))

            # 4. Sort potential edges by their calculated strength
            potential_edges.sort(key=lambda x: x[1], reverse=True)

            # 5. Add the top 'm' new edges
            for target_node, _ in potential_edges[:m]:
                new_edges_to_add.append((start_node, target_node, {'importance': 0.0}))

        # 6. Add all new edges to the graph at once
        self.G.add_edges_from(new_edges_to_add)
        print(f"Added {len(new_edges_to_add)} new zero-weight edges.")

        # 7. Regenerate tensors to reflect the new graph structure
        self._update_tensors()
        print(f"Tensors updated. New edge count: {len(self.edge_index[0])}")



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
                weight_decay = 5e-4,
                distributed=False
                ):
        print("Training Phase with perturbing graph .....")

        self.model.pretrain_phase = False

        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
        print(f"Length of train_loader: {len(train_loader)}")
        print(f"Length of val_loader {len(val_loader)}")
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        min_val = np.inf
        
        if distributed:
            accelerator = Accelerator()
            self.model, optimizer, train_loader, val_loader = accelerator.prepare(
                self.model, optimizer, train_loader, val_loader
            )
            self.device = accelerator.device
            
        print_sys('Start Training...')
        hvg_idx = self.hvg_idx
        optimizer.zero_grad()
        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                # optimizer.zero_grad()
                y = batch.y
                x = batch.x
                pred = self.model(batch)
                ## autoencoder style loss with directionality enforced.
                ## Consider switching to MSE instead ?
                loss = loss_fct(pred, y, batch.pert,
                                ctrl = self.ctrl_expression,
                                dict_filter = self.dict_filter,
                                direction_lambda = self.config['direction_lambda'])
                if distributed:
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                else:
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
                                  filename="edges.csv"):
        """
        Loads ground truth co-expression graphs from CSV files.

        This method builds a dictionary: {pert_name: 1D_weight_tensor}
        
        CRITICAL ASSUMPTIONS:
        - `self.node_map_gene`: A dict {'gene_name': gene_idx} must exist.
        - `self.model.G_coexpress`: The edge_index tensor [2, num_edges] must exist.
        - `self.node_map_pert`: A dict {'pert_name': pert_idx} must exist.
        - Files are located at: {base_path}/pert_ctrl+{pert_name}/{filename}
        """
        print("Attempting to load ground truth co-expression graphs...")
        
        # --- 1. Check for required attributes ---
        if not hasattr(self, 'node_map'):
            print("  Error: `self.node_map_gene` not found. Cannot load ground truth graphs.")
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
            
            gene_name_to_idx = self.node_map_gene
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
            file_path = os.path.join(base_path, f"pert_{pert_name}", filename)
            
            if not os.path.exists(file_path):
                print(f"  Info: No GT graph file found for '{pert_name}' at {file_path}. Skipping.")
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
                pert_indices = np.array(batch.pert) 

                if num_graphs_in_batch != pert_indices.shape[0]:
                    print(f"Warning: Mismatch in batch {step}. "
                          f"Model output {num_graphs_in_batch} graphs, but input batch had {pert_indices.shape[0]} labels. Skipping.")
                    continue

                # --- 5. Compare Each Graph in Batch ---
                for i in range(num_graphs_in_batch):
                    
                    pert_idx = pert_indices[i].item()
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
                                filename="edges.csv"):
        
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
            mapped_G = nx.relabel_nodes(G, self.node_map_gene)
                

    
            
            

            
             

# # F.cosine_similarity and np are not used in this specific function,
# # but torch is needed.

# def standalone_load_ground_truth_graphs(
#     model_G_coexpress,
#     node_map_gene,
#     node_map_pert,
#     device,
#     base_path="/kaggle/input/coexpressiongraphs/Downloads/coexpression_graphs", 
#     filename="/_42_5045_0.4_20_co_expression_network.csv"
# ):
#     """
#     Loads ground truth co-expression graphs from CSV files.

#     This function is isolated and does not depend on a class instance.
#     It builds a dictionary: {pert_name: 1D_weight_tensor}
    
#     Args:
#         node_map_gene (dict): A dict {'gene_name': gene_idx}.
#         node_map_pert (dict): A dict {'pert_name': pert_idx}.
#         device (torch.device or str): The device to create new tensors on.
#         base_path (str): Base directory for GT graphs.
#         filename (str): Name of the edges file (e.g., "edges.csv").
#     """
#     print("Attempting to load ground truth co-expression graphs (standalone)...")
    
#     # --- 1. Build Edge Index Map ---
#     # We use the passed-in arguments directly, no 'self'.
#     try:
#         edge_index = model_G_coexpress.cpu()
#         num_edges = edge_index.shape[1]
#         edge_index_map = {}
#         for i in range(num_edges):
#             u_idx = edge_index[0, i].item()
#             v_idx = edge_index[1, i].item()
#             edge_index_map[(u_idx, v_idx)] = i
        
#         gene_name_to_idx = node_map_gene
#         print(f"  Built edge index map for {num_edges} edges.")
#     except Exception as e:
#         print(f"  Error building edge_index_map: {e}. Aborting GT load.")
#         return {}

#     # --- 2. Get all perturbation names ---
#     pert_names = list(node_map_pert.keys())
#     if 'ctrl' not in pert_names:
#         pert_names.append('ctrl')

#     gt_graph_store = {}
    
#     # --- 3. Iterate and load each GT graph file ---
#     for pert_name in pert_names:
#         file_path = os.path.join(base_path, f"pert_{pert_name}+ctrl", filename)
        
#         if not os.path.exists(file_path):
#             # print(f"  Info: No GT graph file found for '{pert_name}' at {file_path}. Skipping.")
#             continue

#         try:
#             # Initialize a zero-vector on the specified 'device'
#             gt_weights_tensor = torch.zeros(num_edges, device=device)
            
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 reader = csv.reader(f)
#                 header = next(reader) # Skip header
                
#                 edges_found = 0
#                 edges_not_in_index = 0
                
#                 for row in reader:
#                     if len(row) < 3: continue
#                     source_name, target_name, weight_str = row[0], row[1], row[2]
                    
#                     u_idx = gene_name_to_idx.get(source_name)
#                     v_idx = gene_name_to_idx.get(target_name)
                    
#                     if u_idx is None or v_idx is None:
#                         continue 
                    
#                     try:
#                         weight = float(weight_str)
#                     except ValueError:
#                         continue
                    
#                     # Check for the edge in our model's edge_index
#                     edge_pos = edge_index_map.get((u_idx, v_idx))
                    
#                     if edge_pos is not None:
#                         gt_weights_tensor[edge_pos] = weight
#                         edges_found += 1
#                     else:
#                         # Optional: Check for reverse edge if graph is undirected
#                         edge_pos_rev = edge_index_map.get((v_idx, u_idx))
#                         if edge_pos_rev is not None:
#                             gt_weights_tensor[edge_pos_rev] = weight
#                             edges_found += 1
#                         else:
#                             edges_not_in_index += 1
            
#             gt_graph_store[pert_name] = gt_weights_tensor.detach()
#             print(f"  Loaded GT graph for '{pert_name}'. "
#                   f"Mapped {edges_found} edges. ({edges_not_in_index} edges from file were not in model's G_coexpress).")

#         except Exception as e:
#             print(f"  Error loading GT graph for '{pert_name}' from {file_path}: {e}")

#     print(f"Successfully loaded {len(gt_graph_store)} ground truth graphs.")
#     return gt_graph_store



# ---------------------------------------------------------------------------
# [HELPER FUNCTION FOR PART 1]
# ---------------------------------------------------------------------------

def _load_full_graph_from_file(file_path, gene_name_to_idx, device):
    """
    Internal helper to load one full graph from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        gene_name_to_idx (dict): {'gene_name': gene_idx} mapping.
        device (torch.device): Device to send tensors to.
        
    Returns:
        A torch_geometric.data.Data object (edge_index, edge_weight) or None.
    """
    if not os.path.exists(file_path):
        # print(f"  Info: File not found, skipping: {file_path}")
        return None
    
    edge_list = []
    weight_list = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            
            for row in reader:
                if len(row) < 3: continue
                source_name, target_name, weight_str = row[0], row[1], row[2]
                
                u_idx = gene_name_to_idx.get(source_name)
                v_idx = gene_name_to_idx.get(target_name)
                
                # Skip edge if either gene is not in the map
                if u_idx is None or v_idx is None:
                    continue 
                
                try:
                    weight = float(weight_str)
                    edge_list.append([u_idx, v_idx])
                    weight_list.append(weight)
                except ValueError:
                    continue # Skip if weight is not a valid float

        if not edge_list:
            print(f"  Warning: No valid edges found in {file_path}")
            return None
            
        # Create tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
        edge_weight = torch.tensor(weight_list, dtype=torch.float).to(device)
        
        graph_data = Data(edge_index=edge_index, edge_weight=edge_weight)
        return graph_data

    except Exception as e:
        print(f"  Error reading file {file_path}: {e}")
        return None

# ---------------------------------------------------------------------------
# [REVISED FUNCTION 1]
# ---------------------------------------------------------------------------

def standalone_load_ground_truth_graphs(
    node_map_gene,
    node_map_pert,
    device,
    base_path="/kaggle/input/coexpressiongraphs/Downloads/coexpression_graphs", 
    filename="/_42_5045_0.4_20_co_expression_network.csv"
):
    """
    Loads full ground truth co-expression graphs from CSV files.

    This function loads the *entire* graph from each file, not just
    edges matching a predefined index.
    
    It returns:
    1.  gt_graph_store (dict): {pert_name: Data(edge_index, edge_weight)}
    2.  gt_jaccard_scores (dict): {pert_name: structural_jaccard_vs_ctrl}
    
    Args:
        node_map_gene (dict): A dict {'gene_name': gene_idx}.
        node_map_pert (dict): A dict {'pert_name': pert_idx}.
        device (torch.device or str): The device to create new tensors on.
        base_path (str): Base directory for GT graphs.
        filename (str): Name of the edges file.
    """
    print("Attempting to load full ground truth co-expression graphs...")
    
    gt_graph_store = {}
    gt_jaccard_scores = {}
    gene_name_to_idx = node_map_gene

    # --- 1. Load Control Graph First (as baseline) ---
    ctrl_file_path = os.path.join(base_path, "pert_ctrl+ctrl", filename)
    ctrl_graph = _load_full_graph_from_file(ctrl_file_path, gene_name_to_idx, device)
    
    if ctrl_graph is None:
        print(f"  FATAL ERROR: Control graph not found at {ctrl_file_path}. Aborting.")
        return {}, {}
    
    gt_graph_store['ctrl'] = ctrl_graph
    # Create a canonical set of edges (u, v) where u <= v
    ctrl_edge_set = set(
        tuple(sorted(edge)) for edge in ctrl_graph.edge_index.t().tolist()
    )
    print(f"  Loaded 'ctrl' graph with {len(ctrl_edge_set)} unique edges.")

    # --- 2. Load all other perturbation graphs ---
    pert_names = list(node_map_pert.keys())
    
    for pert_name in pert_names:
        if pert_name == 'ctrl': # Already loaded
            continue
            
        file_path = os.path.join(base_path, f"pert_{pert_name}+ctrl", filename)
        pert_graph = _load_full_graph_from_file(file_path, gene_name_to_idx, device)
        
        if pert_graph is not None:
            gt_graph_store[pert_name] = pert_graph
            
            # --- 3. Compute Structural Jaccard Similarity ---
            pert_edge_set = set(
                tuple(sorted(edge)) for edge in pert_graph.edge_index.t().tolist()
            )
            
            intersection = len(ctrl_edge_set.intersection(pert_edge_set))
            union = len(ctrl_edge_set.union(pert_edge_set))
            
            jaccard_sim = intersection / union if union > 0 else 0.0
            gt_jaccard_scores[pert_name] = jaccard_sim
            
            print(f"  Loaded '{pert_name}' graph ({len(pert_edge_set)} edges). "
                  f"Jaccard vs. Ctrl: {jaccard_sim:.4f}")

    print(f"Successfully loaded {len(gt_graph_store)} total ground truth graphs.")
    return gt_graph_store, gt_jaccard_scores

def aggregate_predicted_graphs(model, dataloader, idx_to_pert_name, device):
    """
    Analyzes all cell graphs and returns the *averaged* graph 
    weights for each perturbation.
    
    Returns:
        A dict: { pert_name: {'mean_weights': 1D_Tensor, 'count': N} }
    """
    print("Starting predicted graph aggregation...")
    
    try:
        num_edges = model.G_coexpress_weight.shape[0]
    except AttributeError:
        print("Error: `model.G_coexpress_weight` not found. Aborting.")
        return {}
        
    # Temporary store for *all* weights
    pred_weights_store = defaultdict(list)

    # --- 1. Iterate Through Dataloader to Collect Weights ---
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            batch.to(device)
            pred, graph_data = model.compute_graphs(batch)
            
            predicted_graphs_list = graph_data.get("graphs")
            if predicted_graphs_list is None: continue
            
            pert_indices = np.array(batch.pert)
            if len(predicted_graphs_list) != pert_indices.shape[0]: continue

            for i in range(len(predicted_graphs_list)):
                pert_idx = pert_indices[i].item()
                pert_name = idx_to_pert_name.get(pert_idx, f"Unknown_idx_{pert_idx}")
                
                pred_weights_i = predicted_graphs_list[i].edge_weight.detach()
                
                if pred_weights_i.shape[0] == num_edges:
                    pred_weights_store[pert_name].append(pred_weights_i)
    
    print(f"  Scan complete. Found predicted graphs for {len(pred_weights_store)} perturbations.")

    # --- 2. Aggregate by Averaging ---
    aggregated_pred_store = {}
    for pert_name, weights_list in pred_weights_store.items():
        if not weights_list:
            continue
            
        stacked_weights = torch.stack(weights_list, dim=0)
        mean_pred_weights = torch.mean(stacked_weights, dim=0)
        count = stacked_weights.shape[0]
        
        aggregated_pred_store[pert_name] = {
            'mean_weights': mean_pred_weights,
            'count': count
        }
        
    print(f"  Averaged weights for {len(aggregated_pred_store)} perturbations.")
    return aggregated_pred_store

# --- Helpers for Function 3 ---

def _calculate_pearson(x, y):
    """Calculates Pearson correlation for two 1D tensors."""
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr.item()

def _calculate_weighted_jaccard(pred_weights, gt_weights):
    """Calculates the Tanimoto similarity for weighted, non-negative vectors."""
    pred_abs = torch.abs(pred_weights)
    gt_abs = torch.abs(gt_weights)
    min_sum = torch.sum(torch.minimum(pred_abs, gt_abs))
    max_sum = torch.sum(torch.maximum(pred_abs, gt_abs))
    if max_sum == 0:
        return 1.0 if min_sum == 0 else 0.0
    return (min_sum / max_sum).item()

def _project_gt_graph(gt_data, control_edge_index, device):
    """
    Projects the weights of a full GT graph onto the model's fixed control_edge_index.
    """
    num_control_edges = control_edge_index.shape[1]
    gt_weights_on_control = torch.zeros(num_control_edges, device=device)
    
    gt_map = {}
    gt_edges = gt_data.edge_index.cpu().t().tolist()
    gt_weights = gt_data.edge_weight.cpu().tolist()
    
    # Build a lookup map for the GT graph's weights
    for (u, v), w in zip(gt_edges, gt_weights):
        gt_map[(u, v)] = w
        gt_map[(v, u)] = w # Assume undirected
        
    # Iterate over the *model's* fixed edge structure
    control_edges_list = control_edge_index.cpu().t().tolist()
    for i, (u, v) in enumerate(control_edges_list):
        # Pull the weight from the GT map, default to 0.0
        weight = gt_map.get((u, v), 0.0) 
        gt_weights_on_control[i] = weight
        
    return gt_weights_on_control.to(device)

# --- NEW COMPARISON FUNCTION 3 ---
def compare_aggregated_graphs(
    aggregated_pred_store, 
    gt_graph_store, 
    gt_jaccard_scores, 
    model, 
    device
):
    """
    Compares the averaged predicted graphs against the full ground truth graphs.
    """
    print("\n--- 📊 Aggregated Graph Comparison Report ---")
    final_report = {}
    
    try:
        control_edge_index = model.G_coexpress.to(device)
        control_weights = model.G_coexpress_weight.to(device).detach()
    except AttributeError:
        print("Error: `model.G_coexpress` or `model.G_coexpress_weight` not found. Aborting.")
        return {}
        
    if 'ctrl' not in gt_graph_store:
        print("Error: 'ctrl' graph not found in Ground Truth store. Aborting.")
        return {}
        
    # Project the GT control graph onto the model's edge index once
    gt_ctrl_weights_on_control = _project_gt_graph(
        gt_graph_store['ctrl'], control_edge_index, device
    )

    for pert_name, pred_data in aggregated_pred_store.items():
        
        mean_pred_weights = pred_data['mean_weights']
        count = pred_data['count']
        
        # --- Get Corresponding Ground Truth Graph ---
        gt_full_graph = gt_graph_store.get(pert_name)
        
        if gt_full_graph is None:
            print(f"\n**Perturbation: {pert_name} (n={count})**")
            print(f"  (No Ground Truth graph loaded for this perturbation)")
            continue

        # --- Project GT Graph onto Model's Edge Index ---
        gt_weights_on_control = _project_gt_graph(
            gt_full_graph, control_edge_index, device
        )
        
        # --- Compute Metrics vs. GT ---
        gt_pearson_corr = _calculate_pearson(mean_pred_weights, gt_weights_on_control)
        gt_cos_sim = F.cosine_similarity(
            mean_pred_weights.unsqueeze(0), 
            gt_weights_on_control.unsqueeze(0)
        ).item()
        gt_weighted_jaccard = _calculate_weighted_jaccard(
            mean_pred_weights, gt_weights_on_control
        )
        
        # --- Compute Metrics vs. Control Graphs ---
        diff_vs_learned_ctrl = torch.mean(torch.abs(mean_pred_weights - control_weights)).item()
        diff_vs_gt_ctrl = torch.mean(torch.abs(mean_pred_weights - gt_ctrl_weights_on_control)).item()

        # --- Store and Print Report ---
        structural_jaccard = gt_jaccard_scores.get(pert_name, np.nan)
        final_report[pert_name] = {
            'count': count,
            'gt_pearson': gt_pearson_corr,
            'gt_cosine_sim': gt_cos_sim,
            'gt_weighted_jaccard': gt_weighted_jaccard,
            'gt_structural_jaccard': structural_jaccard,
            'mean_abs_diff_vs_learned_ctrl': diff_vs_learned_ctrl,
            'mean_abs_diff_vs_gt_ctrl': diff_vs_gt_ctrl
        }
        
        print(f"\n**Perturbation: {pert_name} (n={count})**")
        print(f"  --- vs. Ground Truth '{pert_name}' Graph ---")
        print(f"    Pearson Correlation:      {gt_pearson_corr:.4f}")
        print(f"    Cosine Similarity:        {gt_cos_sim:.4f}")
        print(f"    Weighted Jaccard (Tanimoto): {gt_weighted_jaccard:.4f}")
        print(f"    Structural Jaccard vs. Ctrl: {structural_jaccard:.4f}")
        print(f"  --- vs. Control Graphs ---")
        print(f"    Mean Abs. Diff (vs. *Learned* Ctrl): {diff_vs_learned_ctrl:.5f}")
        print(f"    Mean Abs. Diff (vs. *GT* Ctrl):      {diff_vs_gt_ctrl:.5f}")

    print("-----------------------------------")
    return final_report


class PertDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path:str,
        data_name:str):
        super().__init__()
        self.root_path = root_path
        self.data_name = data_name

        self.adata = sc.read_h5ad(os.path.join(root_path, data_name),backed='r')
        self.gene_names = self.adata.var_names.tolist()
        self.pert_names = self.adata.obs['condition'].tolist()
        self.node_map_gene = {gene: idx for idx, gene in enumerate(self.gene_names)}
        self.node_map_pert = {pert: idx for idx, pert in enumerate(set(self.pert_names))}
    def __len__(self):
        return self.adata.n_obs
    def __getitem__(self, idx):
        cell_data = self.adata[idx,:]
        expr = torch.tensor(cell_data.X.toarray(), dtype=torch.float).squeeze()
        pert_idx = self.node_map_pert[cell_data.obs['condition'][0]]
        return {'expr': expr, 'pert': pert_idx}

