#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import random

import numpy as np
import torch

# from defns import GEARS_PRETRAIN, PertData_, standalone_load_ground_truth_graphs
from lazy_dataset import LazyPertData
import scanpy as sc
from gears.utils import zip_data_download_wrapper
from newclasses import GEARS_MECH
from gears.utils import get_similarity_network,GeneSimNetwork
from defns import GeneSimNetworkKHops
from tqdm import tqdm
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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GEARS on perturbation data and compute ground-truth graphs."
    )

    # -----------------------
    # Data / paths
    # -----------------------
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="norman",
        help="Dataset name subfolder under data-root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="model_output",
        help="Directory to save model checkpoints and outputs (default: %(default)s)",
    )

    # -----------------------
    # Splits / dataloader
    # -----------------------
    parser.add_argument(
        "--split",
        type=str,
        default="single",
        help="Split type for PertData_.prepare_split (default: %(default)s)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Fraction of test set for combo_single_split_test_set_fraction (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        help="Test batch size (default: %(default)s)",
    )

    # -----------------------
    # Training hyperparams
    # -----------------------
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: %(default)s)",
    )

    # -----------------------
    # Model hyperparams
    # -----------------------
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size for GEARS encoder (default: %(default)s)",
    )
    parser.add_argument(
        "--decoder-hidden-size",
        type=int,
        default=16,
        help="Hidden size for GEARS decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--num-go-gnn-layers",
        type=int,
        default=1,
        help="Number of GO graph GNN layers (default: %(default)s)",
    )
    parser.add_argument(
        "--num-gene-gnn-layers",
        type=int,
        default=1,
        help="Number of gene graph GNN layers (default: %(default)s)",
    )
    parser.add_argument(
        "--num-similar-genes-go-graph",
        type=int,
        default=20,
        help="Number of neighbors in GO graph (default: %(default)s)",
    )
    parser.add_argument(
        "--num-similar-genes-coexpress-graph",
        type=int,
        default=20,
        help="Number of neighbors in coexpression graph (default: %(default)s)",
    )
    parser.add_argument(
        "--coexpress-threshold",
        type=float,
        default=0.4,
        help="Co-expression threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--direction-lambda",
        type=float,
        default=0.1,
        help="Direction regularization lambda (default: %(default)s)",
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Enable uncertainty modeling (default: False)",
    )
    parser.add_argument(
        "--uncertainty-reg",
        type=float,
        default=1.0,
        help="Uncertainty regularization strength (default: %(default)s)",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=3,
        help="Number of hops for message passing (default: %(default)s)",
    )
    parser.add_argument(
        "--num-add",
        type=int,
        default=6,
        help="Extra edges / hops param (model-specific, default: %(default)s)",
    )

    # -----------------------
    # Misc / device / logging
    # -----------------------
    parser.add_argument(
        "--proj-name",
        type=str,
        default="GEARS",
        help="Project name for logging (default: %(default)s)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="GEARS",
        help="Experiment name for logging (default: %(default)s)",
    )
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Flag for distributed training (currently unused, placeholder)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-bias-track",
        action="store_true",
        help="Enable Weights & Biases (or other) tracking via GEARS_PRETRAIN",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: %(default)s)",
    )
    

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(no_cuda: bool) -> torch.device:
    if torch.cuda.is_available() and not no_cuda:
        return torch.device("cuda")
    return torch.device("cpu")


def main(args: argparse.Namespace) -> None:
    # -----------------------
    # Logging & seeds
    # -----------------------
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Parsed arguments: %s", args)
    set_seed(args.seed)

    device = get_device(args.no_cuda)
    logger.info("Using device: %s", device)

    # -----------------------
    # Paths and data loading
    # -----------------------
    data_root = args.data_root
    dataset_name = args.dataset_name

    dataset_path = os.path.join(data_root, dataset_name)
    # if accelerator.is_main_process:
    #         if not os.path.isdir(dataset_path):
    #             logger.error("Dataset path does not exist: %s", dataset_path)
    #             logger.info("Downloading dataset (main process only)...")
    #             os.makedirs(data_root, exist_ok=True)
    #             os.makedirs(dataset_path, exist_ok=True)

    #             data_download(
    #                 data_name=dataset_name,
    #                 data_dir=dataset_path,
    #                 extract_dir=data_root,
    #             )

    #     # Everyone waits until download is done
    # accelerator.wait_for_everyone()
        # raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    fname = os.path.join(dataset_path, "perturb_processed.h5ad")
    logger.info("Expecting processed AnnData file at: %s", fname)

    pert_data = LazyPertData(data_path=fname,gene_set_path=None,default_pert_graph=True)
    print("Shape of adata:",pert_data.adata.shape)
    num_genes = pert_data.adata.shape[1]
    num_perts = pert_data.adata.obs['condition'].nunique()
    # -----------------------
    # GEARS model setup
    # -----------------------
    os.makedirs(args.output_path, exist_ok=True)
    num_perts = pert_data.num_perturbations
    # Helps catch weird autograd bugs if something explodes
    config = {
    'hidden_size': args.hidden_size,
    'num_go_gnn_layers' : args.num_go_gnn_layers,
    'num_gene_gnn_layers' : args.num_gene_gnn_layers,
    'decoder_hidden_size' : args.decoder_hidden_size,
    'num_similar_genes_go_graph' : args.num_similar_genes_go_graph,
    'num_similar_genes_co_express_graph' : args.num_similar_genes_co_express_graph,
    'coexpress_threshold': args.coexpress_threshold,
    'uncertainty' : args.uncertainty,
    'uncertainty_reg' : args.uncertainty_reg,
    'direction_lambda' : args.direction_lambda,
    'G_go': args.G_go,
    'G_go_weight': args.G_go_weight,
    'G_coexpress': args.G_coexpress,
    'G_coexpress_weight': args.G_coexpress_weight,
    'device': device,
    'num_genes': num_genes,
    'num_perts': num_perts,
    'no_perturb': args.no_perturb,
    'gears_model': args.gears_model,
    "num_hops":args.num_hops,
    "num_add":args.num_add,
    'num_heads': args.num_heads,
    'gated':args.gated
    }

    model = GEARS_MECH(config)
    torch.autograd.set_detect_anomaly(True)
    
    edge_list = get_similarity_network(
        network_type='co-express',
        adata=pert_data.adata,
        threshold=args.coexpress_threshold,
        k=args.num_similar_genes_co_express_graph,
        data_path=pert_data.data_path,
        data_name=pert_data.dataset_name,
        split=pert_data.split, seed=args.seed,
        train_gene_set_size=pert_data.train_gene_set_size,
        set2conditions=pert_data.set2condition
    )
    
    ## checking if multi-hop
    pert_list = pert_data.pert_names.tolist()
    sim_network = GeneSimNetworkKHops(edge_list,pert_list,pert_data.node_map)
    sim_network.add_zero_weight_khop_edges(
    k=config["num_hops"],
    m=config["num_add"],
    )
    
    model.to(device)

    ### train loop
    with tqdm(range(args.epochs), desc="Training Epochs") as pbar:
        for epoch in pbar:
            model.train()
            # Training code would go here
            
        pbar.set_postfix({"Epoch": epoch + 1})
    
    # -----------------------
    # Training
    # -----------------------
    logger.info("Starting training for %d epochs...", args.epochs)
    


    logger.info("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
