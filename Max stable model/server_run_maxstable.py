#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server runner for Max-Stable CNF pipeline (patched with pair-groups).
- Headless-safe (Agg backend)
- Structured outputs with timestamp
- Logs to file + stdout
- Saves trained models, stats, posterior samples
- Records pair_groups in manifest for consistency checks.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
# os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # uncomment to force CPU

import argparse
import logging
import sys
import time
import json
import pathlib
from datetime import datetime
from typing import Dict, Any

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from config import create_range_parameter_config
from train import train_score_network
from cnf import CNF
from sde import get_sde
from nn_model import NCMLP
from dataset import your_prior_sampler, your_simulator, create_2d_grid
from maxstablerevised_副本5 import *

# --------- robust import for helpers from the long script ----------
def prepare_all_maxstable_data_optimized(n_samples=100, key=42, n_groups = 10):
    """Prepare Max-Stable data (revised version) - compute pairwise scores using estimated parameters"""
    print("Preparing Max-Stable data (revised version)...")

    key = jr.PRNGKey(key)
    
    # 1. Generate training data
    theta_train = your_prior_sampler(n_samples, key)
    key, *sim_keys = jr.split(key, n_samples + 1)
    
    print("Generating Max-Stable observation data...")
    x_train_list = []
    valid_theta_list = []
    
    for i in range(n_samples):
        if i % 500 == 0:
            print(f"   Processing sample {i}/{n_samples}")
            
        x_multi = your_simulator(theta_train[i], sim_keys[i])
        
        if not jnp.any(jnp.isnan(x_multi)):
            x_train_list.append(x_multi)
            valid_theta_list.append(theta_train[i])
    
    if len(x_train_list) == 0:
        raise ValueError("Failed to generate any Max-Stable data!")
    
    x_train = jnp.array(x_train_list)
    theta_train = jnp.array(valid_theta_list)
    
    print(f"Max-Stable data generation completed: {len(x_train_list)}/{n_samples} successful")
    print(f"   Parameter shape: {theta_train.shape}")
    print(f"   Observation shape: {x_train.shape}")
    
    # 2. Compute raw data
    print("\nComputing raw data...")
    x_train_raw = x_train.reshape(x_train.shape[0], -1)  # Flatten to (n_samples, 47*79)
    
    # 3. Compute pairwise features - Key fix: use estimated parameters instead of true parameters
    print("\nComputing pairwise features (using estimated parameters)...")
    
    x_train_pairwise_list = []
    x_train_data_scores_list = []
    
    for i in range(len(x_train)):
        if i % 500 == 0:
            print(f"   Processing pairwise features {i}/{len(x_train)}")
        
        # Key fix: estimate parameters for each observation data
        x_obs = x_train[i]  # (47, 79)
        
        # Estimate parameters (instead of using true parameters)
        theta_estimated = estimate_maxstable_params_simple(x_obs)
        
       
        # Compute pairwise scores using estimated parameters
        theta_samples = theta_estimated[None, :]  # (1, 10)
        x_samples = x_obs[None, :, :]  # (1, 47, 79)
        coordinates = create_2d_grid()
        
        try:
            pairwise_scores = compute_maxstable_pairwise_scores(theta_samples, x_samples, coordinates, n_groups=n_groups)
            
            if not jnp.any(jnp.isnan(pairwise_scores)):
                # Modified: pairwise features - flatten but don't normalize
                pairwise_matrix = pairwise_scores[0]  # (30, 10)
                sample_vec = pairwise_matrix.flatten()  # (300,)
                
                # Remove this normalization line:
                # normalized_vec = (sample_vec - jnp.mean(sample_vec)) / (jnp.std(sample_vec) + 1e-8)
                
                # Use raw values directly:
                x_train_pairwise_list.append(sample_vec)  # No normalization
                
                # data score features - sum by group to get 10-dimensional parameter vector (keep unchanged)
                data_score = jnp.sum(pairwise_scores[0], axis=0)  # (10,) - cross-group sum for each parameter
                x_train_data_scores_list.append(data_score)
            else:
                print(f"   WARNING: Sample {i} pairwise computation failed, skipping")
                
        except Exception as e:
            print(f"   WARNING: Sample {i} pairwise computation error: {e}")
    
    if len(x_train_pairwise_list) == 0:
        raise ValueError("Failed to compute any pairwise features!")
    
    # Convert to arrays
    x_train_pairwise_grouped = jnp.array(x_train_pairwise_list)  # (n, 100)
    x_train_data_scores = jnp.array(x_train_data_scores_list)    # (n, 10) - No longer (n, 1)
    
    # Keep only samples with successful pairwise computation
    n_valid_pairwise = len(x_train_pairwise_list)
    theta_train = theta_train[:n_valid_pairwise]
    x_train = x_train[:n_valid_pairwise]
    x_train_raw = x_train_raw[:n_valid_pairwise]
    
    print(f"Pairwise feature computation completed: {n_valid_pairwise} valid samples")
    
    # 4. Compute combined features
    print("Computing combined features...")
    
    # Fix 3: concatenate data_score(10-dim) + pairwise(100-dim) = (110-dim)
    x_train_combined_grouped = jnp.concatenate([x_train_data_scores, x_train_pairwise_grouped], axis=1)
    

    print(f"\nMax-Stable training data preparation completed (revised version):")
    print(f"   - Parameter shape: {theta_train.shape}")
    print(f"   - Raw data shape: {x_train_raw.shape}")
    print(f"   - Data Score shape: {x_train_data_scores.shape}")  # Now (n, 10)
    print(f"   - Pairwise Grouped shape: {x_train_pairwise_grouped.shape}")  # (n, 100)
    print(f"   - Combined Grouped shape: {x_train_combined_grouped.shape}")  # (n, 110)
    
    print(f"\nKey fixes:")
    print(f"   - Use estimated parameters to compute pairwise scores (simulate real inference scenario)")
    print(f"   - Each observation independently estimates parameters (avoid information leakage)")
    print(f"   - Data Score is now 10-dimensional parameter vector")
    print(f"   - Pairwise is 100-dimensional flattened vector")
    print(f"   - Combined is 110-dimensional combined vector")
    
    return (theta_train, x_train, x_train_raw, x_train_data_scores, 
            x_train_pairwise_grouped, x_train_combined_grouped)

def compute_maxstable_pairwise_scores(theta_samples, x_samples, coords=None, n_groups=30):
    """
    Equal distance grouping version of pairwise scores computation
    
    Returns:
    --------
    pairwise_scores : array (n_param_samples, n_groups, 10)
        For each sample, n_groups distance groups, each group has 10-dimensional gradient vector
    """
    if coords is None:
        from dataset import create_2d_grid
        coords = create_2d_grid()
    
    if len(x_samples.shape) == 2:
        x_samples = x_samples[:, None, :]
    
    n_param_samples, n_obs, n_locations = x_samples.shape
    print(f"Equal distance grouping processing {n_param_samples} samples, divided into {n_groups} groups...")
    
    # Create equal distance groups
    equal_groups_flat, group_sizes = create_equal_distance_groups(coords, n_groups)
    
    # Convert data types to ensure Numba compatibility
    theta_samples_np = np.asarray(theta_samples, dtype=np.float64)
    x_samples_np = np.asarray(x_samples, dtype=np.float64)
    coords_np = np.asarray(coords, dtype=np.float64)
    
    start_time = time.time()
    
    # Equal distance grouping computation
    all_grouped_scores = compute_pairwise_scores_equal_distance_numba(
        theta_samples_np, x_samples_np, coords_np, 
        equal_groups_flat, group_sizes
    )
    
    end_time = time.time()
    
    return jnp.array(all_grouped_scores)

# ---------------- utils ----------------
def setup_logger(log_path: pathlib.Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("runner")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger

def save_json(path: pathlib.Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_npy(path: pathlib.Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr))

def save_eqx(path: pathlib.Path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(path), model)

# ------------- training wrapper -------------
def train_method(method_name: str,
                 theta_train: jnp.ndarray,
                 x_input: jnp.ndarray,
                 seed: int,
                 dim_data: int,
                 outdir: pathlib.Path,
                 logger: logging.Logger):
    key = jr.PRNGKey(seed)
    config = create_range_parameter_config()
    config.algorithm.dim_data = int(dim_data)
    config.algorithm.dim_parameters = int(theta_train.shape[1])
    max_iters_env = os.getenv("MAX_ITERS")
    if max_iters_env:
        try:
            config.optim.max_iters = int(max_iters_env)
        except Exception:
            pass

    sde = get_sde(config)
    model = NCMLP(key, config)

    logger.info(f"[{method_name}] Start training: x={tuple(x_input.shape)}, theta={tuple(theta_train.shape)}, "
                f"dim_data={dim_data}, max_iters={config.optim.max_iters}")
    t0 = time.time()
    trained_model, ds_mean, ds_std = train_score_network(
        config, model, sde, theta_train, x_input, key
    )
    dur = time.time() - t0
    logger.info(f"[{method_name}] Training done in {dur:.2f}s")

    mdir = outdir / method_name
    save_eqx(mdir / "model.eqx", trained_model)
    save_npy(mdir / "ds_mean.npy", ds_mean)
    save_npy(mdir / "ds_std.npy", ds_std)
    save_json(mdir / "training_meta.json", {
        "method": method_name,
        "dim_data": int(dim_data),
        "duration_sec": float(dur),
        "optim": {
            "max_iters": int(config.optim.max_iters),
            "batch_size": int(config.optim.batch_size),
            "lr": float(config.optim.lr),
        },
        "sde": {
            "name": str(config.sde.name),
            "T": float(config.sde.T),
            "beta_min": float(config.sde.beta_min),
            "beta_max": float(config.sde.beta_max),
            "sigma_min": float(config.sde.sigma_min),
            "sigma_max": float(config.sde.sigma_max),
        },
    })
    return trained_model, sde, ds_mean, ds_std, config, dur

def build_cnf(trained_model, sde, ds_mean, ds_std):
    return CNF(score_network=trained_model, sde=sde, ds_means=ds_mean, ds_stds=ds_std)

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Server runner for Max-Stable CNF")
    parser.add_argument("--methods", type=str, default="raw,data_score,pairwise_grouped,combined_grouped",
                        help="Comma-separated methods to run")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--outdir", type=str, default="runs", help="Output base directory")
    parser.add_argument("--posterior-n", type=int, default=1000, help="# posterior samples per method to draw")
    parser.add_argument("--save-train-data", action="store_true", help="Persist prepared training datasets")
    parser.add_argument("--pair-groups", type=int, default=10,
                        help="Number of equal-distance groups for pairwise scores/features")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = pathlib.Path(args.outdir) / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "run.log")
    logger.info("==== Max-Stable CNF server run ====")
    logger.info(f"Args: {vars(args)}")

    logger.info("Preparing data...")
    t0 = time.time()
    (theta_train, x_train, x_train_raw, x_train_data_scores,
     x_train_pairwise_grouped, x_train_combined_grouped) = prepare_all_maxstable_data_optimized(
        n_samples=args.n_samples, key=args.seed, n_groups=args.pair_groups
    )
    prep_sec = time.time() - t0
    logger.info(f"Data prepared in {prep_sec:.2f}s; theta_train={tuple(theta_train.shape)}")

    if args.save_train_data:
        data_dir = run_dir / "data"
        save_npy(data_dir / "theta_train.npy", theta_train)
        save_npy(data_dir / "x_train.npy", x_train)
        save_npy(data_dir / "x_train_raw.npy", x_train_raw)
        save_npy(data_dir / "x_train_data_scores.npy", x_train_data_scores)
        save_npy(data_dir / "x_train_pairwise.npy", x_train_pairwise_grouped)
        save_npy(data_dir / "x_train_combined.npy", x_train_combined_grouped)
        save_json(data_dir / "meta.json", {"n_samples": int(args.n_samples), "prep_seconds": float(prep_sec),
                                           "pair_groups": int(args.pair_groups)})

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    trained: Dict[str, Dict[str, Any]] = {}

    for m in methods:
        if m == "raw":
            dim = int(x_train_raw.shape[1])
            model, sde, ds_mean, ds_std, config, dur = train_method(m, theta_train, x_train_raw, args.seed + 11, dim, run_dir, logger)
        elif m == "data_score":
            dim = int(x_train_data_scores.shape[1])
            model, sde, ds_mean, ds_std, config, dur = train_method(m, theta_train, x_train_data_scores, args.seed + 22, dim, run_dir, logger)
        elif m == "pairwise_grouped":
            dim = int(x_train_pairwise_grouped.shape[1])
            model, sde, ds_mean, ds_std, config, dur = train_method(m, theta_train, x_train_pairwise_grouped, args.seed + 33, dim, run_dir, logger)
        elif m == "combined_grouped":
            dim = int(x_train_combined_grouped.shape[1])
            model, sde, ds_mean, ds_std, config, dur = train_method(m, theta_train, x_train_combined_grouped, args.seed + 44, dim, run_dir, logger)
        else:
            logger.warning(f"Unknown method {m}, skipping.")
            continue
        trained[m] = dict(model=model, sde=sde, ds_mean=ds_mean, ds_std=ds_std, config=config, dim=dim, train_time=dur)

    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    logger.info("Building CNFs and sampling posteriors...")

    for m, info in trained.items():
        cnf = build_cnf(info["model"], info["sde"], info["ds_mean"], info["ds_std"])
        # Use the first training sample's feature vector as a test input
        if m == "raw":
            test_input = x_train_raw[0]
        elif m == "data_score":
            test_input = x_train_data_scores[0]
        elif m == "pairwise_grouped":
            test_input = x_train_pairwise_grouped[0]
        elif m == "combined_grouped":
            test_input = x_train_combined_grouped[0]
        else:
            continue
        key = jr.PRNGKey(args.seed + 100)
        samples = cnf.batch_sample_fn(args.posterior_n, test_input, key)
        save_npy(results_dir / f"{m}_posterior_samples.npy", samples)
        mu = np.asarray(jnp.mean(samples, axis=0)); sd = np.asarray(jnp.std(samples, axis=0))
        save_json(results_dir / f"{m}_posterior_summary.json", {"mean": mu.tolist(), "std": sd.tolist()})
        logger.info(f"[{m}] Saved posterior samples {samples.shape}")

    manifest = {
        "methods": methods,
        "seed": int(args.seed),
        "n_train_samples": int(args.n_samples),
        "prep_seconds": float(prep_sec),
        "posterior_n": int(args.posterior_n),
        "pair_groups": int(args.pair_groups),
        "artifacts": {
            m: {
                "model": f"{m}/model.eqx",
                "ds_mean": f"{m}/ds_mean.npy",
                "ds_std": f"{m}/ds_std.npy",
                "posterior_samples": f"results/{m}_posterior_samples.npy",
                "posterior_summary": f"results/{m}_posterior_summary.json",
                "dim_data": int(info["dim"]),
                "train_time_sec": float(info["train_time"]),
            } for m, info in trained.items()
        }
    }
    save_json(run_dir / "manifest.json", manifest)
    logger.info(f"Run complete. Artifacts saved in: {run_dir}")

if __name__ == "__main__":
    main()