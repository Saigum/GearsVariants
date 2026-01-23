#!/usr/bin/env python3
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import wandb as _wandb
from copy import deepcopy
from typing import Tuple

# ---- your code/classes (import from your repo layout) ----
from gears_changes import PertData_         # provided by you
from zero_out import ZeroOutVariant, train_with_pretraining # put your class+train fn here
from disvariant import PertDataDistributed,train
from gears.utils import loss_fct as LOSS_FCT

# train_zero.py

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os # Import the os module

import torch.distributed as dist

def print_rank_0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
def setup_distributed():
    dist.init_process_group(backend='nccl') # 'nccl' is standard for NVIDIA GPUs
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)



# ----------------- config loading -----------------
def load_config(path: str) -> dict:
  _, ext = os.path.splitext(path.lower())
  if ext in (".yaml", ".yml"):
    try:
      import yaml
    except ImportError as e:
      raise SystemExit("PyYAML required for YAML configs. `pip install pyyaml`") from e
    with open(path, "r") as f:
      return yaml.safe_load(f)
  elif ext == ".json":
    with open(path, "r") as f:
      return json.load(f)
  else:
    raise SystemExit(f"Unsupported config ext '{ext}'. Use .yaml/.yml or .json.")


# ----------------- utils -----------------
def seed_everything(seed: int):
  random.seed(seed); np.random.seed(seed)
  torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = True

def pick_device(pref: str) -> str:
  if pref.startswith("cuda") and torch.cuda.is_available(): return pref
  if pref == "mps" and torch.backends.mps.is_available(): return "mps"
  return "cpu"

def _to_dense(X):
  # handles scipy.sparse, numpy arrays, or torch tensors from AnnData
  try:
    import scipy.sparse as sp
    if sp.issparse(X): return X.toarray()
  except Exception:
    pass
  if isinstance(X, torch.Tensor): return X.cpu().numpy()
  return np.asarray(X)

def log_memory_usage(label: str):
    """Logs the current process's memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_gb = mem_info.rss / (1024 ** 3)  # Resident Set Size in GB
        print_rank_0(f"[mem_debug] {label}: RSS = {rss_gb:.2f} GB", file=sys.stderr)
    except ImportError:
        print_rank_0("[mem_debug] psutil not found. Run `pip install psutil`", file=sys.stderr)
    except Exception as e:
        print_rank_0(f"[mem_debug] Error logging memory: {e}", file=sys.stderr)

# ----------------- build data -----------------
def build_data(cfg: dict,distributed:bool=True) -> PertData_:
  dcfg = cfg["data"]
  if not distributed:
    pd = PertData_(dcfg["root"])
  else:
    pd = PertDataDistributed(dcfg["root"])
    
  pd.load(dcfg["dataset"])

  scfg = cfg["split"]
  pd.prepare_split(
    split=scfg.get("type", "simulation"),
    seed=scfg.get("seed", 1),
    train_gene_set_size=scfg.get("train_gene_set_size", 0.75),
    combo_seen2_train_frac=scfg.get("combo_seen2_train_frac", 0.75),
    combo_single_split_test_set_fraction=scfg.get("combo_single_test_frac", 0.10),
    test_perts=scfg.get("test_perts"),
    only_test_set_perts=scfg.get("only_test_set_perts", False),
    test_pert_genes=scfg.get("test_pert_genes"),
    split_dict_path=scfg.get("split_dict_path"),
    val_size=scfg.get("val_size", 0.10),
  )

  lcfg = cfg["loader"]
  pd.get_dataloader(
    batch_size=lcfg.get("batch_size", 128),
    test_batch_size=lcfg.get("test_batch_size", 256),
    # num_workers=lcfg.get("num_workers", 4),
    # pin_memory=lcfg.get("pin_memory", True),
    # persistent_workers=lcfg.get("persistent_workers", True),
    # shuffle=lcfg.get("shuffle", True),
  )
  return pd

from gears.utils import get_similarity_network,GeneSimNetwork
# -------------- args for ZeroOutVariant --------------
def assemble_zeroout_args(pert_data: PertData_, cfg: dict, device: str) -> dict:
  mcfg = cfg["model"]
  dcfg = cfg["data"]
  coexcfg = cfg["coexpression"]
  # infer counts
  num_genes = len(pert_data.gene_names.values.tolist())
  num_perts = len(pert_data.pert_names.tolist())
  cxedge_list = get_similarity_network(network_type='co-express',
                                               adata=pert_data.adata,
                                               threshold=coexcfg.get("threshold"),
                                               k=coexcfg.get("k"),
                                               data_path=dcfg.get("root"),
                                               data_name=dcfg.get("name"),
                                               split=cfg["split"].get("type"), seed=cfg.get("seed"),
                                               train_gene_set_size=cfg["split"].get("train_gene_set_size"),
                                               set2conditions=pert_data.set2conditions)
  cxsim_network = GeneSimNetwork(cxedge_list, pert_data.gene_names.values.tolist(), node_map = pert_data.node_map)
  # Package exactly what ZeroOutVariant expects/uses
  args = {
    "device": device,
    "num_genes": num_genes,
    "num_perts": num_perts,
    "hidden_size": mcfg.get("hidden_size", 64),
    "uncertainty": mcfg.get("uncertainty", False),
    "num_go_gnn_layers": mcfg.get("num_go_gnn_layers", 1),   # used for .num_layers (kept for parity)
    "decoder_hidden_size": mcfg.get("decoder_hidden_size", 16),
    "num_gene_gnn_layers": mcfg.get("num_gene_gnn_layers", 1),
    "no_perturb": mcfg.get("no_perturb", False),
    # graph tensors
    "G_coexpress": cxsim_network.edge_index,
    "G_coexpress_weight": cxsim_network.edge_weight.to(device),
    # train-time regularizer consumed via `config['direction_lambda']` in train_with_pretraining
    "direction_lambda": mcfg.get("direction_lambda", 0.1),
  }
  return args


# -------------- save helpers --------------
def save_checkpoint(model: nn.Module, out_dir: str, name: str):
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f"{name}.pt")
  torch.save(model.state_dict(), path)
  print_rank_0(f"[info] saved: {path}")

def save_config(cfg: dict, out_dir: str, name: str = "config_used"):
  os.makedirs(out_dir, exist_ok=True)
  path = os.path.join(out_dir, f"{name}.json")
  with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
  print_rank_0(f"[info] saved: {path}")


# ----------------- main -----------------
def main():
  ap = argparse.ArgumentParser("Train ZeroOutVariant (no GEARS_ API)")
  ap.add_argument("--config", required=True, type=str, help="YAML/JSON config")
  ap.add_argument("--distributed", action="store_true", help="Use distributed training")
  ap.add_argument("--device", default="cuda", type=str, help="Override device (cpu|cuda|cuda:0|mps)")
  ap.add_argument("--seed", default=None, type=int, help="Override seed")
  ap.add_argument("--out_dir", default=None, type=str, help="Override output dir")
  args = ap.parse_args()

  if(args.distributed):
    setup_distributed()
  cfg = load_config(args.config)
  seed = args.seed if args.seed is not None else cfg.get("seed", 1)
  seed_everything(seed)

  device = pick_device(args.device or cfg.get("device", "cuda"))
  print_rank_0(f"[info] device = {device}")

  out_dir = args.out_dir or cfg.get("out_dir", "./runs/zeroout")
  os.makedirs(out_dir, exist_ok=True)

  # data
  log_memory_usage("main: Before build_data")
  pert_data = build_data(cfg,distributed=args.distributed)
  log_memory_usage("main: After build_data")

  # infer args strictly from what ZeroOutVariant/train need
  log_memory_usage("main: Before assemble_zeroout_args")
  zargs = assemble_zeroout_args(pert_data, cfg, device)
  log_memory_usage("main: After assemble_zeroout_args")

  # model
  log_memory_usage("main: Before model init")
  model = ZeroOutVariant(pert_data=pert_data, args=zargs)
  log_memory_usage("main: After model init")

  # wandb (optional)
  wandb = None
  wcfg = cfg.get("wandb", {})
  if wcfg.get("enable", False):
      _wandb.init(project=wcfg.get("project", "GEARS"), name=wcfg.get("run_name", "zeroout_no_api"), config=cfg)
      wandb = _wandb

  # train
  tcfg = cfg["train"]
  log_memory_usage("main: Before train")
  if not args.distributed:
    best_model = train_with_pretraining(
      pert_data=pert_data,
      model=model,
      device=device,
      config=zargs, # contains direction_lambda
      epochs=tcfg.get("epochs", 40),
      lr=tcfg.get("lr", 1e-3),
      weight_decay=tcfg.get("weight_decay", 5e-4),
      pretrain_epochs=tcfg.get("pretrain_epochs", 20),
      wandb=wandb,
      loss_fct=LOSS_FCT,
      print_sys=print_rank_0,
    )
  else:
    best_model = train(
      pert_data=pert_data,
      model=model,
      device=device,
      config=tcfg,
      wandb=wandb,
      loss_fct=LOSS_FCT,
      print_sys=print_rank_0,
    )
  log_memory_usage("main: After train")

  # save
  save_checkpoint(best_model, out_dir, "zeroout_best")
  save_config(cfg, out_dir)

  if wandb is not None:
    try:
      wandb.finish()
    except Exception:
      pass


if __name__ == "__main__":
  main()