import os
import scanpy as sc
from gears.utils import zip_data_download_wrapper
import argparse
from torch_geometric.data import DataLoader
import pickle
from gears.data_utils import DataSplitter, print_sys
from gears.pertdata import PertData
def data_download(data_name,data_dir,extract_dir):
    if data_name == 'norman':
        url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
    elif data_name == 'adamson':
        url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
    elif data_name == 'dixit':
        url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
    elif data_name == 'replogle_k562_essential':
        ## Note: This is not the complete dataset and has been filtered
        url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
    elif data_name == 'replogle_rpe1_essential':
        ## Note: This is not the complete dataset and has been filtered
        url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
    else:
        print("None of these datasets exist")
        return
    zip_data_download_wrapper(url, data_dir, extract_dir)

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



if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Download dataset.')
    parser.add_argument('--data_name', type=str, required=True,
                        help='Name of the dataset to download (e.g., norman, adamson, dixit, replogle_k562_essential, replogle_rpe1_essential)')
    os.makedirs('data', exist_ok=True)
    # os.makedirs(f'data/{parser.parse_args().data_name}', exist_ok=True)
    data_download(data_name=parser.parse_args().data_name, data_dir=os.path.join('data',parser.parse_args().data_name), extract_dir='data')
    print("Performing ")