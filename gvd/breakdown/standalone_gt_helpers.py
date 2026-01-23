import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict


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

# --------
def standalone_load_ground_truth_graphs(
    node_map_gene,
    node_map_pert,
    device,
    base_path="/kaggle/input/coexpressiongraphs/Downloads/coexpression_graphs", 
    filename="_42_5045_0.4_20_co_expression_network.csv"
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
    ctrl_file_path = "/kaggle/input/coexpressiongraphs/Downloads/coexpression_graphs/control/_42_5045_0.4_20_co_expression_network.csv"
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
    print(base_path)
    for pert_name in pert_names:
        if pert_name == 'ctrl': # Already loaded
            continue
        
        # file_path2 = os.path.join(base_path, f"pert_{pert_name}+ctrl", filename)   
        file_path = f"{base_path}/pert_{pert_name}+ctrl/{filename}"
        # print(file_path)
        
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
import re
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
        # match = re.search(r'([A-Z0-9]+)', pert_name
        pair = re.findall(r"Unknown_idx_([^ ()]+)", pert_name)
        if(len(pair)==0):
            continue
        parts = [g for g in pair[0].split("+") if g != "ctrl"]

        if(len(parts)==0):
            continue
        true_name = parts[0]
        # print(match.group(1))  # Output: RUNX1T1
        # --- Get Corresponding Ground Truth Graph ---
        gt_full_graph = gt_graph_store.get(true_name)
        
        if gt_full_graph is None:
            print(f"\n**Perturbation: {pert_name} (n={count})**")
            print(true_name)
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