print("whatdafuk")
import os
import torch
import scanpy as sc

# Import your custom classes and functions
from pertdata_extension import PertData_
from gears_pretrain import GEARS_PRETRAIN
from standalone_gt_helpers import standalone_load_ground_truth_graphs, compare_graphs_to_ground_truth, plot_gt_comparison_results, compare_control_to_ground_truth
from utils import data_download
import json # for saving results

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('model_output', exist_ok=True)

DATA_ROOT="data"
DATASET_NAME="norman"
FNAME=os.path.join(DATA_ROOT,DATASET_NAME,"perturb_processed.h5ad")

# Data Download (uncomment if you need to download)
# data_download(data_name=DATASET_NAME, data_dir=os.path.join(DATA_ROOT,DATASET_NAME), extract_dir=DATA_ROOT)

# Initialize PertData
pert_data = PertData_(data_path=DATA_ROOT)
pert_data.load(data_name=DATASET_NAME, data_path=os.path.join(DATA_ROOT,DATASET_NAME))
pert_data.prepare_split(split='single', seed=1, combo_single_split_test_set_fraction=0.3)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

# Initialize GEARS_PRETRAIN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gears_mech = GEARS_PRETRAIN(pert_data=pert_data, device=device, weight_bias_track=False, proj_name='GEARS', exp_name='GEARS')

# Model Initialization
gears_mech.model_initialize(
    hidden_size=64, num_go_gnn_layers=1,
    num_gene_gnn_layers=1, decoder_hidden_size=16,
    num_similar_genes_go_graph=20, num_similar_genes_co_express_graph=20,
    coexpress_threshold=0.4, uncertainty=False, uncertainty_reg=1,
    direction_lambda=0.1, G_go=None, G_go_weight=None,
    G_coexpress=None, G_coexpress_weight=None, no_perturb=False,
    gears_model=1, # Setting to 1 for GEARS_MECH
    num_hops=3, num_add=6
)

# Enable anomaly detection for debugging gradients
torch.autograd.set_detect_anomaly(True)

# Train the model
deeperanalysis, nondropoutanalysis = gears_mech.train(epochs=4, lr=1e-3, weight_decay=5e-4)
gears_mech.save_model("model_output")

# Set node_map_gene for GT graph loading
gears_mech.node_map_gene = gears_mech.node_map

# Load Ground Truth Graphs using the standalone helper
print("\n--- STEP 1: Loading Ground Truth Graphs ---")
ground_truth_data = standalone_load_ground_truth_graphs(
    model_G_coexpress=gears_mech.model.G_coexpress,
    node_map_gene=gears_mech.node_map_gene,
    node_map_pert=gears_mech.node_map_pert,
    device=gears_mech.device
)

# Run the comparison analysis
print("\n--- STEP 2: Running Predicted vs. Ground Truth Comparison ---")
gt_results = compare_graphs_to_ground_truth(gears_mech, gears_mech.dataloader["train_loader"], ground_truth_data)

# Plot the results
print("\n--- STEP 3: Plotting Comparison Results ---")
plot_gt_comparison_results(gt_results)

# Run baseline comparison (Control vs. GT)
print("\n--- STEP 4: Running Baseline Analysis (Control vs. GT) ---")
baseline_results = compare_control_to_ground_truth(gears_mech, ground_truth_data)

# Save results to JSON
with open("pred_pertvs_pert.json", "w") as fp:
    json.dump(gt_results, fp)

print("\nScript finished.")