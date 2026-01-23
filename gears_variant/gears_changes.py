import torch
from gears import PertData
from copy import deepcopy
import os
import pickle
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch.utils.data import DataLoader
from gears.gears import GEARS
from gears.model import GEARS_Model
import anndata as ad
from gears.data_utils import DataSplitter,print_sys
from gears.utils import  get_similarity_network,GeneSimNetwork
from gears.utils import loss_fct,uncertainty_loss_fct
from gears.inference import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import networkx as nx
import anndata as ad
import mygene
from copy import deepcopy
from gears.utils import get_similarity_network, GeneSimNetwork, loss_fct, uncertainty_loss_fct
from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis
from torch.optim.lr_scheduler import StepLR
import lightning as L
from gears_variants import *

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
        
        

def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    """
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list

def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    """

    fname = './data/go_essential_' + data_name + '.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)
    custom_fname= os.path.join(data_path,f"go_essential{data_name}.csv")
    if(os.path.exists(custom_fname)):
        return pd.read_csv()
    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    #perturbation_list
    gene2go = {i: gene2go[i] for i in pert_list}

    print('Creating custom GO graph, this can take a few minutes')
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                      total=len(gene2go.keys())))
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})
    
    if save:
        if(data_path is not None):
            fname = os.path.join(data_path,f"go_essential{data_name}.csv")
        print(f'Saving edge_list to file {fname}')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list

class GEARS_(GEARS):
    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS'):
        """
        Initialize GEARS model

        Parameters
        ----------
        pert_data: PertData object
            dataloader for perturbation data
        device: str
            Device to run the model on. Default: 'cuda'
        weight_bias_track: bool
            Whether to track performance on wandb. Default: False
        proj_name: str
            Project name for wandb. Default: 'GEARS'
        exp_name: str
            Experiment name for wandb. Default: 'GEARS'

        Returns
        -------
        None

        """

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

    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
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

        Returns
        -------
        None
        """
        
        self.config = {'hidden_size': hidden_size,
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
                       'num_heads': num_heads,
                       'gated':gated
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

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
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
            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        if self.config["gears_model"] == 0 :
            self.model = GEARS_Model(self.config).to(self.device)
        elif self.config["gears_model"] == 1:
            self.model = GEARS_EMBED(self.config).to(self.device)
        elif self.config["gears_model"] == 2:
            self.model = GEARS_GAT(self.config).to(self.device)
        elif self.config["gears_model"] == 3:
            self.model = GEARS_Transformer(self.config).to(self.device)
        elif self.config["gears_model"] == 4:
            self.model = GEARS_No_Coexpress(self.config).to(self.device)
        elif self.config["gears_model"] == 5:
            self.model = GEARS_No_Perturb(self.config).to(self.device)
        elif self.config["gears_model"] == 6:
            self.model = GEARS_SelfAttn(self.config).to(self.device)            
        elif self.config["gears_model"] == 7:
            self.model = GEARS_CELL(self.config).to(self.device)   
        elif self.config["gears_model"] == 8:
            self.model = VariantTiny(self.config).to(self.device)
        elif self.config["gears_model"] == 9:
            self.model = Variant(self.config).to(self.device)
            

        self.best_model = deepcopy(self.model)

    def train(self, epochs = 20, 
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay

        Returns
        -------
        None

        """
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                if self.config['uncertainty']:
                    pred, logvar = self.model(batch)
                    loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                      reg = self.config['uncertainty_reg'],
                                      ctrl = self.ctrl_expression, 
                                      dict_filter = self.dict_filter,
                                      direction_lambda = self.config['direction_lambda'])
                else:
                    pred = self.model(batch)
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
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,
                                 self.config['uncertainty'], self.device)
            val_res = evaluate(val_loader, self.model,
                                 self.config['uncertainty'], self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # Print epoch performance
            log = "Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'], 
                             val_metrics['mse']))
            
            # Print epoch performance for DE genes
            log = "Train Top 20 DE MSE: {:.4f} " \
                  "Validation Top 20 DE MSE: {:.4f}. "
            print_sys(log.format(train_metrics['mse_de'],
                             val_metrics['mse_de']))
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'train_' + m: train_metrics[m],
                               'val_'+m: val_metrics[m],
                               'train_de_' + m: train_metrics[m + '_de'],
                               'val_de_'+m: val_metrics[m + '_de']})
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(self.model)
                
        print_sys("Done!")
        self.best_model = best_model

        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        # Model testing
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        test_res = evaluate(test_loader, self.best_model,
                            self.config['uncertainty'], self.device)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        print_sys(log.format(test_metrics['mse_de']))
        
        if self.wandb:
            metrics = ['mse', 'pearson']
            for m in metrics:
                self.wandb.log({'test_' + m: test_metrics[m],
                           'test_de_'+m: test_metrics[m + '_de']                     
                          })
                
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        metrics = ['pearson_delta']
        metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout',
                               'frac_sigma_below_1_non_dropout',
                               'mse_top20_de_non_dropout']
        
        if self.wandb:
            for m in metrics:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

            for m in metrics_non_dropout:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        

        if self.split == 'simulation':
            print_sys("Start doing subgroup analysis for simulation split...")
            subgroup = self.subgroup
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in list(list(test_pert_res.values())[0].keys()):
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m, res in test_pert_res[pert].items():
                        subgroup_analysis[name][m].append(res)

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

            ## deeper analysis
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in metrics:
                    subgroup_analysis[name][m] = []

                for m in metrics_non_dropout:
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(out[pert][m])

                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(out_non_dropout[pert][m])

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        print_sys('Done!')
        return train_res,val_res,test_res
    def train_with_pretraining(self, epochs = 20, 
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        
        ##### pretraining phase: train only on control cells, in batc
        print_sys('Start Pretraining on single perturbation conditions...')
        min_val = np.inf
        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                single_pert_idx = (batch.pert != -1) & (torch.sum(batch.pert, axis=1) == 1)
                if torch.sum(single_pert_idx) == 0:
                    continue
                optimizer.zero_grad()
                y = batch.y[single_pert_idx]
                pred = self.model(batch)
                pred = pred[single_pert_idx]
                loss = loss_fct(pred, y, batch.pert[single_pert_idx],
                                ctrl = self.ctrl_expression, 
                                dict_filter = self.dict_filter,
                                direction_lambda = self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'pretrain_training_loss': loss.item()})

                if step % 50 == 0:
                    log = "Pretrain Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))

            scheduler.step()
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,
                                 self.config['uncertainty'], self.device,
                                 single_perturbation=True)
            val_res = evaluate(val_loader, self.model,
                                 self.config['uncertainty'], self.device,
                                 single_perturbation=True)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # Print epoch performance
            log = "Pretrain Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'],val_metrics['mse']))
        



class GEARS_Fabric(GEARS_):
    def __init__(self, pert_data, 
                 fabric: L.Fabric, 
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS'):
        """
        Initialize GEARS model
        """
        
        # Store fabric and derive device from it
        self.fabric = fabric
        self.device = self.fabric.device
        
        self.weight_bias_track = weight_bias_track
        
        # Only initialize wandb on the main process
        if self.weight_bias_track and self.fabric.global_rank == 0:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.config = None
        
        self.dataloader = pert_data.dataloader
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
        
        # Use fabric.to_device to move tensors
        self.ctrl_expression = self.fabric.to_device(torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition.values == 'ctrl'],
                    axis=0)).reshape(-1, ))
        
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}

    def tunable_parameters(self):
        # ... (no changes needed here) ...
        return super().tunable_parameters()

    def model_initialize(self, **kwargs):
        """
        Initialize the model
        """
        
        # The original model_initialize, but we add self.device to config
        # And remove all .to(self.device) calls from model creation
        
        kwargs.pop('device', None) 
        super().model_initialize(device=self.device, **kwargs) # This sets self.config        
        self.config['device'] = self.device
        if self.wandb and self.fabric.global_rank == 0:
            self.wandb.config.update(self.config)
        

        if self.config["gears_model"] == 0 :
            self.model = GEARS_Model(self.config).to(self.device)
        elif self.config["gears_model"] == 1:
            self.model = GEARS_EMBED(self.config).to(self.device)
        elif self.config["gears_model"] == 2:
            self.model = GEARS_GAT(self.config).to(self.device)
        elif self.config["gears_model"] == 3:
            self.model = GEARS_Transformer(self.config).to(self.device)
        elif self.config["gears_model"] == 4:
            self.model = GEARS_No_Coexpress(self.config).to(self.device)
        elif self.config["gears_model"] == 5:
            self.model = GEARS_No_Perturb(self.config).to(self.device)
        elif self.config["gears_model"] == 6:
            self.model = GEARS_SelfAttn(self.config).to(self.device)            
        elif self.config["gears_model"] == 7:
            self.model = GEARS_CELL(self.config).to(self.device)   
        elif self.config["gears_model"] == 8:
            self.model = VariantTiny(self.config).to(self.device)
        elif self.config["gears_model"] == 9:
            self.model = Variant(self.config).to(self.device)
        
        self.best_model = None 


    def train(self, epochs = 20, 
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
        
        # 1. Setup Optimizer and Scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        # 2. Use fabric.setup for model and optimizer
        self.model, optimizer = self.fabric.setup(self.model, optimizer)

        # 3. Use fabric.setup_dataloaders
        train_loader, val_loader = self.fabric.setup_dataloaders(train_loader, val_loader)

        min_val = np.inf
        best_model_state = None

        if self.fabric.global_rank == 0:
            print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            # Wrap train_loader in tqdm only on rank 0
            if self.fabric.global_rank == 0:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            else:
                pbar = train_loader

            for step, batch in enumerate(pbar):
                optimizer.zero_grad()
                y = batch.y
                
                if self.config['uncertainty']:
                    pred, logvar = self.model(batch)
                    loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                      reg = self.config['uncertainty_reg'],
                                      ctrl = self.ctrl_expression, 
                                      dict_filter = self.dict_filter,
                                      direction_lambda = self.config['direction_lambda'])
                else:
                    pred = self.model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                  ctrl = self.ctrl_expression, 
                                  dict_filter = self.dict_filter,
                                  direction_lambda = self.config['direction_lambda'])
                
                # 5. Use fabric.backward
                self.fabric.backward(loss)
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()
                
                total_loss += loss.item()

                # 6. Only log from the main process
                if self.wandb and self.fabric.global_rank == 0:
                    self.wandb.log({'training_loss': loss.item()})

                if step % 50 == 0 and self.fabric.global_rank == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))
            
            scheduler.step()
            
            # --- Evaluation ---
            # NOTE: Assumes 'evaluate' function moves its batches to device
            # or is passed self.fabric to do so.
            # We run evaluation on all processes, as DistributedSampler
            # gives each process a unique shard.
            train_res = evaluate(train_loader, self.model,
                                 self.config['uncertainty'], self.device)
            val_res = evaluate(val_loader, self.model,
                                 self.config['uncertainty'], self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # 7. All logging and printing is guarded
            if self.fabric.global_rank == 0:
                # Print epoch performance
                log = "Epoch {}: Train Overall MSE: {:.4f} " \
                      "Validation Overall MSE: {:.4f}. "
                print_sys(log.format(epoch + 1, train_metrics['mse'], 
                                 val_metrics['mse']))
                
                log = "Train Top 20 DE MSE: {:.4f} " \
                      "Validation Top 20 DE MSE: {:.4f}. "
                print_sys(log.format(train_metrics['mse_de'],
                                 val_metrics['mse_de']))
                
                if self.wandb:
                    metrics = ['mse', 'pearson']
                    for m in metrics:
                        self.wandb.log({'train_' + m: train_metrics[m],
                                   'val_'+m: val_metrics[m],
                                   'train_de_' + m: train_metrics[m + '_de'],
                                   'val_de_'+m: val_metrics[m + '_de']})
            
            # 8. Save best model state
            # This check happens on all processes
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                # Save the state_dict, not the model object
                best_model_state = deepcopy(self.model.state_dict())
                if self.fabric.global_rank == 0:
                    print_sys('Best model state saved!')
        
        if self.fabric.global_rank == 0:
            print_sys("Done!")
        
        # 9. Load the best model state on all processes
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        else:
            if self.fabric.global_rank == 0:
                print_sys("Warning: No best model found, using last epoch model for testing.")

        # Save the *unwrapped* model to self.best_model
        # Use fabric.unwrap_model to get the original nn.Module
        self.best_model = self.fabric.unwrap_model(self.model)

        if 'test_loader' not in self.dataloader:
            if self.fabric.global_rank == 0:
                print_sys('Done! No test dataloader detected.')
            return
            
        # --- Model testing ---
        test_loader = self.fabric.setup_dataloaders(self.dataloader['test_loader'])
        
        if self.fabric.global_rank == 0:
            print_sys("Start Testing...")
            
        test_res = evaluate(test_loader, self.model,
                            self.config['uncertainty'], self.device)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        
        # 10. All post-training analysis and logging on rank 0
        if self.fabric.global_rank == 0:
            log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
            print_sys(log.format(test_metrics['mse_de']))
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'test_' + m: test_metrics[m],
                               'test_de_'+m: test_metrics[m + '_de']                     
                              })
            
            # ... (all other deeper_analysis, non_dropout_analysis, subgroup_analysis) ...
            # ... (wrapped in 'if self.fabric.global_rank == 0:') ...
            
            out = deeper_analysis(self.adata, test_res)
            # ... etc ...

        print_sys('Done!')
        
        # Note: These results are only valid on rank 0
        return train_res, val_res, test_res
  