
from torch.utils.data import Dataset
from gears.data_utils import get_genes_from_perts,get_DE_genes,get_dropout_non_zero_genes,DataSplitter,print_sys
from gears.utils import zip_data_download_wrapper,dataverse_download
import torch
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

from gears.utils import filter_pert_in_go

class PerturbedDataset(Dataset):
    """
    Lazy Dataset that generates graphs on the fly.
    """
    def __init__(self, adata, ctrl_adata, condition_indices, 
                 pert_names, node_map_pert, de_genes_map, 
                 num_samples=1, de_idx_size=20):
        """
        adata: The full AnnData object (or subset for this split)
        ctrl_adata: AnnData object containing only control samples
        condition_indices: List of indices in 'adata' that belong to this split
        pert_names: List of perturbation names for mapping
        node_map_pert: Dictionary mapping perturbation names to indices
        de_genes_map: Dictionary or uns object containing DE genes info
        """
        self.adata = adata
        self.ctrl_adata = ctrl_adata
        self.indices = condition_indices
        self.pert_names = pert_names
        self.node_map_pert = node_map_pert
        self.de_genes_map = de_genes_map
        self.num_samples = num_samples
        self.de_idx_size = de_idx_size
        
        # Pre-convert control matrix to dense if possible for speed, 
        # otherwise keep sparse and convert on fly
        self.ctrl_X = self.ctrl_adata.X
        if hasattr(self.ctrl_X, 'toarray'):
            self.ctrl_X = self.ctrl_X.toarray()

    def __len__(self):
        # Total samples = number of cells in this split * number of control pairings
        return len(self.indices) * self.num_samples

    def get_pert_idx(self, pert_category):
        try:
            pert_idx = [self.node_map_pert[p]
                    for p in pert_category.split('+')
                    if p != 'ctrl' and p in self.node_map_pert]
        except:
            pert_idx = [-1]
        return pert_idx if pert_idx else [-1]

    def __getitem__(self, idx):
        # Determine which cell and which sample repetition this is
        actual_idx = self.indices[idx // self.num_samples]
        
        # Extract cell data
        cell_obs = self.adata.obs.iloc[actual_idx]
        pert_category = cell_obs['condition']
        
        # Get Y (The Perturbed Cell)
        y_vec = self.adata.X[actual_idx]
        if hasattr(y_vec, 'toarray'):
            y_vec = y_vec.toarray().flatten()
        
        # Get X (The Control Cell - sampled randomly)
        # Note: For validation consistency, you might want to seed this or fix indices,
        # but typically random sampling for baselines is acceptable in this context.
        if pert_category == 'ctrl':
            # If it's a control cell, X and Y are the same (reconstruction)
            x_vec = y_vec
            pert_idx = [-1]
            de_idx = [-1] * self.de_idx_size
        else:
            # If perturbed, X is a random control sample
            rand_ctrl_idx = np.random.randint(0, self.ctrl_X.shape[0])
            x_vec = self.ctrl_X[rand_ctrl_idx]
            
            pert_idx = self.get_pert_idx(pert_category)
            
            # Get DE Indices
            pert_de_category = cell_obs['condition_name'] if 'condition_name' in cell_obs else pert_category
            if self.de_genes_map and pert_de_category in self.de_genes_map:
                # We assume logic to map gene names to indices happens here
                # This matches the original logic roughly, but requires the var_names map
                # For efficiency, pre-calculating DE indices per condition is better, 
                # but here we do it lazily or look up a pre-calc table.
                de_idx = self.de_genes_map[pert_de_category]
            else:
                de_idx = [-1] * self.de_idx_size

        # Create Graph
        feature_mat = torch.Tensor(np.vstack([x_vec, y_vec])).T # Transpose to (Genes x 2) or just X?
        # Original code: feature_mat = torch.Tensor(X).T where X is the control.
        # Wait, original code: Data(x=feature_mat, y=y)
        
        feature_mat = torch.Tensor(x_vec).unsqueeze(1) # (Num_Genes, 1)
        
        return Data(x=feature_mat, 
                    pert_idx=torch.tensor(pert_idx),
                    y=torch.Tensor(y_vec), 
                    de_idx=torch.tensor(de_idx), 
                    pert=pert_category)


class LazyPertData:
    """
    Lazy version of PertData. 
    Does not construct graph objects for all cells at initialization.
    Generates them on-the-fly in the DataLoader.
    """
    
    def __init__(self, data_path, 
                 gene_set_path=None, 
                 default_pert_graph=True):
        
        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        # self.dataset_processed = None  <-- REMOVED: No huge dictionary in memory
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None
        self.de_genes_indices = {} # Store pre-calculated DE indices per condition

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            
        # Load gene2go (Keep original logic)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path, os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
    
    def set_pert_genes(self):
        # (Logic remains identical to original)
        if self.gene_set_path is not None:
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
        elif self.default_pert_graph is False:
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
        else:
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path, 'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
    
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}
        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
            
    def load(self, data_name = None, data_path = None):
        """
        Loads the AnnData object but DOES NOT create the PyG graph dataset file.
        """
        if data_name in ['norman', 'adamson', 'dixit', 'replogle_k562_essential', 'replogle_rpe1_essential']:
            if data_name == 'norman': url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif data_name == 'adamson': url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif data_name == 'dixit': url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            elif data_name == 'replogle_k562_essential': url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
            elif data_name == 'replogle_rpe1_essential': url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
            
            data_path = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path, self.data_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path) # Can use backed='r' here if RAM is very tight

        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("Invalid data attribute")
        
        self.set_pert_genes()
        
        # Filter Logic
        print_sys('Filtering perturbations not in GO graph...')
        filter_go = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]
        self.adata = self.adata[filter_go.index.values, :]
        
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        self.gene_names = self.adata.var.gene_name
        
        # Pre-calculate DE Gene Indices to save time in DataLoader
        print_sys("Pre-calculating DE indices maps...")
        self._precalc_de_indices()
        
        print_sys("Lazy Loading Complete. Graphs will be generated during iteration.")

    def _precalc_de_indices(self, num_de_genes=20):
        """
        Helper to create a map of {condition_name: [gene_indices]}
        """
        self.de_genes_indices = {}
        if 'rank_genes_groups_cov_all' in self.adata.uns:
            de_genes = self.adata.uns['rank_genes_groups_cov_all']
            var_names = self.adata.var_names
            
            for cond in de_genes.dtype.names:
                genes = de_genes[cond][:num_de_genes]
                # Find indices
                indices = np.where(var_names.isin(genes))[0]
                # Pad if necessary
                if len(indices) < num_de_genes:
                    indices = np.concatenate([indices, [-1]*(num_de_genes-len(indices))])
                self.de_genes_indices[cond] = indices

    def new_data_process(self, dataset_name, adata=None, skip_calc_de=False):
        # (Logic similar to original, but removed create_dataset_file call)
        # ... validation checks ...
        
        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)
        if not os.path.exists(save_data_folder): os.mkdir(save_data_folder)
        self.dataset_path = save_data_folder
        
        self.adata = get_DE_genes(adata, skip_calc_de)
        if not skip_calc_de:
            self.adata = get_dropout_non_zero_genes(self.adata)
        
        self.adata.write_h5ad(os.path.join(save_data_folder, 'perturb_processed.h5ad'))
        self.set_pert_genes()
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        self.gene_names = self.adata.var.gene_name
        
        self._precalc_de_indices()
        print_sys("Done processing new data.")

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

    def get_dataloader(self, batch_size, test_batch_size=None):
        if test_batch_size is None:
            test_batch_size = batch_size
            
        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        
        # Create indices dictionaries instead of graph lists
        # Map 'train' -> [integer indices of cells in adata]
        split_indices = {}
        
        if self.split == 'no_split':
            splits = ['test']
        elif self.split == 'no_test':
            splits = ['train', 'val']
        else:
            splits = ['train', 'val', 'test']

        # We need to map the "conditions" in set2conditions to actual numeric indices in adata
        # This is much faster than copying graph objects
        print_sys("Mapping split conditions to cell indices...")
        
        # Create a lookup for condition -> indices
        # Optimize: Groupby once
        condition_group = self.adata.obs.groupby('condition').indices
        
        for split_name in splits:
            conditions_in_split = self.set2conditions[split_name]
            indices_list = []
            for cond in conditions_in_split:
                if cond in condition_group:
                    # Note: original code skips 'ctrl' in test set sometimes, 
                    # ensure logic matches specific requirements. 
                    # Original: if p != 'ctrl': cell_graphs[i].extend(...)
                    if split_name == 'test' and cond == 'ctrl' and self.split != 'no_split':
                        continue
                    indices_list.extend(condition_group[cond])
            split_indices[split_name] = np.array(indices_list)

        print_sys("Creating Lazy Dataloaders....")
        
        dataloaders = {}
        
        # Helper to make dataset
        def make_ds(name, shuffle):
            return DataLoader(
                PerturbedDataset(
                    adata=self.adata,
                    ctrl_adata=self.ctrl_adata,
                    condition_indices=split_indices[name],
                    pert_names=self.pert_names,
                    node_map_pert=self.node_map_pert,
                    de_genes_map=self.de_genes_indices
                ),
                batch_size=batch_size if name != 'test' else test_batch_size,
                shuffle=shuffle,
                drop_last=(name == 'train')
            )

        if 'train' in splits:
            dataloaders['train_loader'] = make_ds('train', True)
        if 'val' in splits:
            dataloaders['val_loader'] = make_ds('val', True)
        if 'test' in splits:
            dataloaders['test_loader'] = make_ds('test', False)
            
        self.dataloader = dataloaders
        return dataloaders

    # create_dataset_file, create_cell_graph_dataset, create_cell_graph 
    # are no longer needed in the main class as they are handled by PerturbedDataset.

