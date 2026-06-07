# GearsVariant

This repository is a working research fork of GEARS for exploring perturbation-aware architectural changes and mechanistic graph modelling on single-cell perturbation data. The codebase now reflects three main strands of work:

- `gears-graph.ipynb`: GEARS architecture variants and ablations.
- `mech-modelling(3).ipynb`: mechanistic graph-learning experiments with training outputs.
- `mechanistic_pert.ipynb`: a duplicate/export of the same mechanistic workflow and results.

The current experiments are centered on the Norman dataset with `single` split evaluation.

<p align="center"><img src="https://github.com/snap-stanford/GEARS/blob/master/img/gears.png" alt="gears" width="900px" /></p>

## What Has Been Implemented

### 1. GEARS architecture variants

The work in `gears-graph.ipynb` extends the base GEARS model with several alternative perturbation and graph encoders. The notebook defines and instantiates:

- Original GEARS baseline
- Expression-embedding GEARS
- Gated expression embedding
- Graph Attention Network variant
- TransformerConv variant
- No co-expression graph ablation
- No perturbation graph ablation
- Self-attention variant

Most of the reusable implementations live in:

- `gears_variant/gears_variants.py`
- `gears_variant/gears_changes.py`
- `gvd/breakdown/gated_addition.py`
- `gvd/breakdown/attentional_perturber.py`
- `gvd/breakdown/sage_encoder.py`

Note: `gears-graph.ipynb` contains the model-development workflow and training calls, but it does not currently store benchmark outputs in the notebook itself, so the concrete result summary below comes from the mechanistic notebooks.

### 2. Mechanistic perturbation modelling

The mechanistic notebooks introduce a GEARS variant that tries to predict perturbation-specific changes to the gene co-expression graph during expression forecasting. The main additions are:

- Centered expression embeddings projected into the hidden space
- Gated fusion between gene embeddings and cell-expression embeddings
- An attentional graph perturber that modifies edge weights per perturbation
- GraphSAGE-based node encoding inside the graph perturber
- Optional MMD-based training objective with directionality-aware loss
- K-hop graph augmentation with added zero-weight edges before learning perturbation-specific edge updates

The main implementation is in:

- `gvd/breakdown/gears_mech.py`

## Notebook Summary

### `gears-graph.ipynb`

Purpose:
- Prototype and compare GEARS architecture variants.
- Test whether alternate message passing and embedding strategies improve perturbation response prediction.

Work completed:
- Variant definitions were added.
- Norman data loading and `single` split setup were added.
- Training calls for each variant were prepared and executed in the notebook workflow.

### `mech-modelling(3).ipynb` and `mechanistic_pert.ipynb`

Purpose:
- Train a mechanistic GEARS model that predicts both expression outcomes and perturbation-conditioned graph changes.
- Compare learned perturbation graphs against ground-truth co-expression graphs.

Work completed:
- Custom split handling and Norman dataset loading
- MMD and directionality loss functions
- Mechanistic GEARS pretraining/training wrapper
- Ground-truth graph loading from co-expression CSVs
- Aggregation of predicted perturbation graphs across the train loader
- Graph-vs-ground-truth reporting for 129 perturbations

## Current Results

### Expression prediction

Two saved mechanistic runs are recorded in the notebooks.

#### Run A: `num_add=6`, `use_mmd=True`

- Original graph edges: `6359`
- Added zero-weight edges: `1443`
- Final graph edges: `7802`
- Best validation MSE: `0.003423`
- `pearson_delta`: `0.4338`
- `pearson_delta_de`: `0.4225`
- `pearson_delta_top200_de`: `0.5456`
- `pearson_top200_de`: `0.9780`
- `mse_top200_de`: `0.0437`
- `frac_correct_direction_20`: `0.6214`
- `frac_correct_direction_50`: `0.6321`
- `frac_correct_direction_100`: `0.6139`

#### Run B: `num_add=20`, no MMD

- Original graph edges: `6359`
- Added zero-weight edges: `3016`
- Final graph edges: `9375`
- Best validation MSE: `0.009107`
- `pearson_delta`: `0.3193`
- `pearson_delta_de`: `0.4448`
- `pearson_delta_top200_de`: `0.5089`
- `pearson_top200_de`: `0.9761`
- `mse_top200_de`: `0.0480`
- `frac_correct_direction_20`: `0.5518`
- `frac_correct_direction_50`: `0.5271`
- `frac_correct_direction_100`: `0.4896`

Takeaway:
- The `num_add=6` run with MMD produced the strongest overall expression-prediction result in the saved outputs, especially on validation MSE and directionality metrics.

### Graph reconstruction

For both mechanistic runs, predicted perturbation graphs were aggregated for `129` perturbations and compared with ground-truth co-expression graphs.

Saved report averages from notebook outputs:

#### Run A graph report

- Mean Pearson correlation vs ground-truth graph weights: `0.0142`
- Mean cosine similarity: `0.5469`
- Mean weighted Jaccard: `0.2472`
- Mean absolute difference vs learned control graph: `0.49154`
- Mean absolute difference vs ground-truth control graph: `0.66192`

#### Run B graph report

- Mean Pearson correlation vs ground-truth graph weights: `0.0184`
- Mean cosine similarity: `0.4995`
- Mean weighted Jaccard: `0.2130`
- Mean absolute difference vs learned control graph: `0.56361`
- Mean absolute difference vs ground-truth control graph: `0.69580`

### Single-perturbation cosine table

`gvd/tables.md` contains a per-perturbation cosine comparison over `54` single perturbations:

- Mean cosine similarity, model vs ground truth: `0.5338`
- Mean cosine similarity, control baseline vs ground truth: `0.6468`

Best model cosine in the saved table:
- `KLF1`: `0.5921`

Lowest model cosine in the saved table:
- `FEV`: `0.4888`

Takeaway:
- The mechanistic model learns perturbation-conditioned graph structure to some extent, but the current graph outputs are still weaker than a control-graph baseline on the saved cosine comparison table. Expression prediction is currently stronger than graph-fidelity recovery.

## Repository Layout

- `gears/`: upstream GEARS core code
- `gears_variant/`: alternative GEARS model variants and ablations
- `gvd/breakdown/`: mechanistic graph-learning components
- `gvd/tables.md`: saved graph similarity summary table
- `gears-graph.ipynb`: architecture variant notebook
- `mech-modelling(3).ipynb`: mechanistic modelling notebook with outputs
- `mechanistic_pert.ipynb`: mechanistic modelling notebook duplicate/export

## Installation

Install PyTorch Geometric and the GEARS dependencies from `requirements.txt`. The notebooks were run with `cell-gears`, `torch_geometric`, and `scanpy`.

```bash
pip install -r requirements.txt
```

Depending on your PyTorch and CUDA version, you may also need the matching PyG wheels before running the notebooks.

## Interpretation

This repository should currently be read as an experimental GEARS extension rather than a polished package release. The completed work shows:

- several working GEARS architectural variants,
- a mechanistic perturbation model that augments co-expression graphs during prediction,
- promising expression-level results for the MMD-based mechanistic run,
- and a clear remaining gap between expression prediction quality and graph reconstruction quality.
