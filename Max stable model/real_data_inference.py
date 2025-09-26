#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real data inference script for CNF and ABC methods using actual rainfall data.
Loads real rainfall data and applies trained CNF models plus ABC for comparison.
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import pathlib
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time  # <-- added for timing

from config import create_range_parameter_config
from sde import get_sde
from nn_model import NCMLP
from cnf import CNF
from dataset import create_2d_grid

# Import maxstable helpers
from maxstablerevised_副本5 import (
    compute_maxstable_test_input,
    mple_theta,
    MaxStablePrior,
    maxstable_model,
    maxstable_distance,
    compute_maxstable_pairwise_scores,
    adjust_abc_results
)

# Import R for real data loading
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

MPLE_THETA = np.asarray(mple_theta, dtype=float)
PARAM_NAMES = ['cov11','cov12','cov22','loc0','loc1','loc2','scale0','scale1','scale2','shape']

def load_real_rainfall_data():
    """Load real rainfall data from SpatialExtremes package"""
    print("Loading real rainfall data from SpatialExtremes package...")
    
    try:
        ro.r("""
        library(SpatialExtremes)
        data(rainfall)
        """)
        
        # Get real data
        with localconverter(ro.default_converter + numpy2ri.converter):
            rain_data = np.array(ro.r('rain'))
            coord_data = np.array(ro.r('coord[,-3]'))  # Remove altitude column
        
        print(f"Real rainfall data loaded: {rain_data.shape}")
        print(f"Coordinates shape: {coord_data.shape}")
        print(f"Data range: [{np.min(rain_data):.2f}, {np.max(rain_data):.2f}]")
        
        return rain_data, coord_data
        
    except Exception as e:
        print(f"Failed to load real rainfall data: {e}")
        return None, None

def compute_real_mple():
    """Compute real MPLE estimation using R"""
    print("Computing real MPLE estimation...")
    
    try:
        ro.r("""
        mod.spe <- fitmaxstab(rain, coord=coord[,-3], cov.mod="gauss",
                              loc ~ lon + lat, scale ~ lon + lat, shape ~ 1,
                              fit.marge=TRUE)
        real_mple <- mod.spe$fitted.values
        mple_std_err <- mod.spe$std.err
        """)
        
        with localconverter(ro.default_converter + numpy2ri.converter):
            real_mple = np.array(ro.r('real_mple'))
            mple_std_err = np.array(ro.r('mple_std_err'))
        
        print(f"Real MPLE: {real_mple}")
        print(f"MPLE Std Errors: {mple_std_err}")
        
        return real_mple, mple_std_err
        
    except Exception as e:
        print(f"Failed to compute MPLE: {e}")
        return None, None

def load_manifest(run_dir: pathlib.Path):
    """Load training manifest"""
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)

def restore_model(run_dir: pathlib.Path, method: str, dim_data: int):
    """Restore trained CNF model"""
    config = create_range_parameter_config()
    config.algorithm.dim_data = int(dim_data)
    config.algorithm.dim_parameters = 10
    sde = get_sde(config)
    key = jr.PRNGKey(0)
    model = NCMLP(key, config)
    model_path = run_dir / method / "model.eqx"
    model = eqx.tree_deserialise_leaves(str(model_path), model)
    ds_mean = np.load(run_dir / method / "ds_mean.npy")
    ds_std = np.load(run_dir / method / "ds_std.npy")
    cnf = CNF(score_network=model, sde=sde,
              ds_means=jnp.asarray(ds_mean), ds_stds=jnp.asarray(ds_std))
    return cnf

def feature_by_method(x_obs: np.ndarray, method: str, n_groups: int):
    """Compute features for a given method"""
    if method == "raw":
        return x_obs.flatten()
    elif method in ("data_score","pairwise_grouped","combined_grouped"):
        return compute_maxstable_test_input(jnp.asarray(x_obs), method, n_groups=n_groups)
    else:
        raise ValueError(f"Unknown method: {method}")

def run_cnf_inference(cnf_models, rain_data, pair_groups, posterior_n=1000):
    """Run CNF inference on real data"""
    print(f"\nRunning CNF inference with {len(cnf_models)} methods...")
    
    cnf_results = {}
    key = jr.PRNGKey(42)

    total_start = time.perf_counter()  # <-- timing: total CNF

    for method_name, cnf_info in cnf_models.items():
        print(f"\nProcessing {method_name}...")
        method_start = time.perf_counter()  # <-- timing: per method
        
        try:
            # Compute features
            feat = np.asarray(feature_by_method(rain_data, method_name, n_groups=pair_groups))
            expected = int(cnf_info['dim_data'])
            
            if feat.shape[0] != expected:
                raise ValueError(f"Feature dimension mismatch: got {feat.shape[0]} vs expected {expected}")
            
            # Sample from posterior
            key, sample_key = jr.split(key)
            samples = cnf_info['cnf'].batch_sample_fn(posterior_n, jnp.asarray(feat), sample_key)
            samples_np = np.asarray(samples, dtype=float)
            
            # Compute statistics
            post_mean = samples_np.mean(axis=0)
            post_std = samples_np.std(axis=0)
            post_median = np.median(samples_np, axis=0)
            post_q25 = np.percentile(samples_np, 25, axis=0)
            post_q75 = np.percentile(samples_np, 75, axis=0)

            elapsed = time.perf_counter() - method_start  # <-- timing end

            cnf_results[method_name] = {
                'samples': samples_np,
                'mean': post_mean,
                'std': post_std,
                'median': post_median,
                'q25': post_q25,
                'q75': post_q75,
                'success': True,
                'elapsed_sec': elapsed,  # <-- record time
            }
            
            print(f"  Success: {samples_np.shape[0]} posterior samples  |  time: {elapsed:.3f}s")
            
        except Exception as e:
            elapsed = time.perf_counter() - method_start  # <-- even on failure
            print(f"  Failed: {e}  |  time: {elapsed:.3f}s")
            cnf_results[method_name] = {'success': False, 'error': str(e), 'elapsed_sec': elapsed}
    
    total_elapsed = time.perf_counter() - total_start  # <-- total CNF time
    print(f"\nCNF total time: {total_elapsed:.3f}s")
    cnf_results['_total_elapsed_sec'] = total_elapsed
    return cnf_results

def run_abc_inference(rain_data, pair_groups, pop_size=500, populations=5):
    """Run ABC inference on real data"""
    print(f"\nRunning ABC inference...")
    
    try:
        import pyabc
        import tempfile
        
        prior = MaxStablePrior()
        
        def _distance_with_groups(sim, obs):
            return maxstable_distance(sim, obs, n_groups=pair_groups)
        
        total_start = time.perf_counter()  # <-- timing: ABC main

        abc = pyabc.ABCSMC(
            models=maxstable_model,
            parameter_priors=prior,
            distance_function=_distance_with_groups,
            population_size=pyabc.populationstrategy.ConstantPopulationSize(pop_size),
            sampler=pyabc.sampler.SingleCoreSampler(),
            eps=pyabc.epsilon.QuantileEpsilon(alpha=0.7, quantile_multiplier=0.95)
        )
        
        obs_data = {"data": np.asarray(rain_data)}
        db_path = tempfile.mktemp(suffix=".db")
        
        history = abc.new(f"sqlite:///{db_path}", obs_data)
        history = abc.run(max_nr_populations=populations)
        
        df, w = history.get_distribution(m=0, t=history.max_t)
        df = df.reset_index(drop=True)
        
        # Original ABC results
        abc_samples = df[PARAM_NAMES].values
        abc_w = np.asarray(w, dtype=float).reshape(-1)
        abc_w = abc_w / abc_w.sum()
        
        abc_mean = (abc_samples * abc_w[:, None]).sum(axis=0)
        abc_var = np.average((abc_samples - abc_mean)**2, axis=0, weights=abc_w)
        abc_std = np.sqrt(abc_var)
        
        elapsed_main = time.perf_counter() - total_start  # <-- end timing: ABC main

        abc_results = {
            'abc': {
                'samples': abc_samples,
                'mean': abc_mean,
                'std': abc_std,
                'success': True,
                'elapsed_sec': elapsed_main,  # <-- record ABC main time
            }
        }
        
        # Try ABC adjustment
        try:
            adj_start = time.perf_counter()  # <-- timing: ABC adjustment
            adjusted_df = adjust_abc_results(
                history=history,
                obs_data=obs_data,
                param_names=PARAM_NAMES,
                true_params=None,  # Unknown for real data
                mple_theta=MPLE_THETA,
                n_groups=pair_groups,
                simulator=lambda theta, seed: np.asarray(your_simulator(jnp.asarray(theta), jr.PRNGKey(seed))),
                score_fn=compute_maxstable_pairwise_scores,
                coords=create_2d_grid(),
                bandwidth_quantile=0.8,
                apply_to='kept'
            )
            adj_elapsed = time.perf_counter() - adj_start  # <-- end timing: ABC adjustment
            
            adj_samples = adjusted_df[PARAM_NAMES].values
            adj_mean = adj_samples.mean(axis=0)
            adj_std = adj_samples.std(axis=0)
            
            abc_results['abc_adj'] = {
                'samples': adj_samples,
                'mean': adj_mean,
                'std': adj_std,
                'success': True,
                'elapsed_sec': adj_elapsed,  # <-- record ABC adj time
            }
            
            print(f"  ABC adjustment successful  |  time: {adj_elapsed:.3f}s")
            
        except Exception as e:
            adj_elapsed = time.perf_counter() - total_start  # best-effort timing reference
            print(f"  ABC adjustment failed: {e}")
            abc_results['abc_adj'] = {'success': False, 'error': str(e), 'elapsed_sec': adj_elapsed}
        
        print(f"  ABC success: {abc_samples.shape[0]} posterior samples  |  time: {elapsed_main:.3f}s")
        return abc_results
        
    except Exception as e:
        print(f"  ABC failed: {e}")
        return {'abc': {'success': False, 'error': str(e), 'elapsed_sec': 0.0}}

def create_comparison_plots(cnf_results, abc_results, real_mple, mple_std_err, save_path=None):
    """Create comparison plots"""
    print("\nCreating comparison plots...")
    
    # Collect all successful results
    all_results = {}
    for method, result in cnf_results.items():
        if isinstance(result, dict) and result.get('success'):
            all_results[method] = result
    
    for method, result in abc_results.items():
        if isinstance(result, dict) and result.get('success'):
            all_results[method] = result
    
    if not all_results:
        print("No successful results to plot")
        return
    
    # 1. Parameter-wise mean comparison
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, param in enumerate(PARAM_NAMES):
        methods = list(all_results.keys())
        means = [all_results[m]['mean'][i] for m in methods]
        stds = [all_results[m]['std'][i] for m in methods] if 'std' in all_results[methods[0]] else None
        
        bars = axes[i].bar(methods, means, alpha=0.7)
        if stds:
            axes[i].errorbar(range(len(methods)), means, yerr=stds, fmt='none', color='black', capsize=3)
        
        # Add MPLE reference line
        if real_mple is not None:
            axes[i].axhline(y=real_mple[i], color='red', linestyle='--', linewidth=2, label=f'MPLE: {real_mple[i]:.3f}')
            axes[i].legend()
        
        axes[i].set_title(f'{param}', fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle("Posterior Means vs MPLE on Real Rainfall Data", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "real_data_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Posterior distributions (boxplots)
    fig, axes = plt.subplots(2, 5, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(PARAM_NAMES):
        plot_data = []
        
        for method, result in all_results.items():
            if 'samples' in result:
                param_samples = result['samples'][:, i]
                for val in param_samples:
                    plot_data.append({'Method': method, 'Value': val})
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.boxplot(data=plot_df, x='Method', y='Value', ax=axes[i])
            
            # Add MPLE reference line
            if real_mple is not None:
                axes[i].axhline(y=real_mple[i], color='red', linestyle='--', linewidth=2)
            
            axes[i].set_title(f'{param}', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle("Posterior Distributions on Real Rainfall Data", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "real_data_posterior_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(cnf_results, abc_results, real_mple, mple_std_err):
    """Create and print summary table with detailed parameter statistics"""
    print("\n" + "="*100)
    print("REAL DATA INFERENCE RESULTS")
    print("="*100)
    
    # Collect all successful results
    all_results = {}
    for method, result in cnf_results.items():
        if isinstance(result, dict) and result.get('success'):
            all_results[method] = result
    
    for method, result in abc_results.items():
        if isinstance(result, dict) and result.get('success'):
            all_results[method] = result
    
    if real_mple is not None:
        print("\nMPLE Reference Values:")
        for i, param in enumerate(PARAM_NAMES):
            mple_val = real_mple[i]
            std_err = mple_std_err[i] if mple_std_err is not None else 0
            print(f"  {param}: {mple_val:.4f} ± {std_err:.4f}")
        print()
    
    # Status table
    print(f"{'Method':<20} {'Status':<10} {'Sample Size':<12} {'Time(s)':<10}")
    print("-" * 60)
    
    for method, result in cnf_results.items():
        if method == '_total_elapsed_sec':
            continue
        if isinstance(result, dict) and result.get('success'):
            n_samples = result['samples'].shape[0] if 'samples' in result else 0
            tsec = result.get('elapsed_sec', 0.0)
            print(f"{method:<20} {'Success':<10} {n_samples:<12} {tsec:<10.3f}")
        else:
            tsec = result.get('elapsed_sec', 0.0) if isinstance(result, dict) else 0.0
            print(f"{method:<20} {'Failed':<10} {'0':<12} {tsec:<10.3f}")
    cnf_total = cnf_results.get('_total_elapsed_sec', None)
    if cnf_total is not None:
        print(f"{'(CNF total)':<20} {'-':<10} {'-':<12} {cnf_total:<10.3f}")
    
    for method, result in abc_results.items():
        if isinstance(result, dict):
            status = "Success" if result.get('success') else "Failed"
            n_samples = result['samples'].shape[0] if (result.get('success') and 'samples' in result) else 0
            tsec = result.get('elapsed_sec', 0.0)
            print(f"{method:<20} {status:<10} {n_samples:<12} {tsec:<10.3f}")
    
    # Detailed parameter statistics table
    if all_results:
        print("\n" + "="*120)
        print("POSTERIOR MEAN AND STANDARD DEVIATION BY METHOD")
        print("="*120)
        
        # Header
        methods = list(all_results.keys())
        header = f"{'Parameter':<12}"
        for method in methods:
            header += f"{method:<20}"
        if real_mple is not None:
            header += f"{'MPLE':<15}"
        print(header)
        print("-" * len(header))
        
        # Parameter rows
        for i, param in enumerate(PARAM_NAMES):
            row = f"{param:<12}"
            
            # Method results
            for method in methods:
                result = all_results[method]
                if 'mean' in result and 'std' in result:
                    mean_val = result['mean'][i]
                    std_val = result['std'][i]
                    row += f"{mean_val:.3f}±{std_val:.3f}    "
                else:
                    row += f"{'N/A':<20}"
            
            # MPLE reference
            if real_mple is not None:
                mple_val = real_mple[i]
                mple_err = mple_std_err[i] if mple_std_err is not None else 0
                row += f"{mple_val:.3f}±{mple_err:.3f}"
            
            print(row)
        
        print("\n" + "="*120)
        print("POSTERIOR MEDIAN AND QUARTILES BY METHOD")
        print("="*120)
        
        # Header for quartiles
        header = f"{'Parameter':<12}"
        for method in methods:
            header += f"{method + ' (Q25|Med|Q75)':<25}"
        print(header)
        print("-" * len(header))
        
        # Parameter rows with quartiles
        for i, param in enumerate(PARAM_NAMES):
            row = f"{param:<12}"
            
            for method in methods:
                result = all_results[method]
                if all(k in result for k in ['q25', 'median', 'q75']):
                    q25 = result['q25'][i]
                    med = result['median'][i]
                    q75 = result['q75'][i]
                    row += f"{q25:.3f}|{med:.3f}|{q75:.3f}    "
                else:
                    row += f"{'N/A':<25}"
            
            print(row)
    
    print("\n" + "="*100)

def main():
    parser = argparse.ArgumentParser(description="Real data inference using CNF and ABC")
    parser.add_argument("--run-dir", required=True, help="Training run directory")
    parser.add_argument("--pair-groups", type=int, default=10, help="Number of distance groups")
    parser.add_argument("--posterior-n", type=int, default=1000, help="CNF posterior samples")
    parser.add_argument("--with-abc", action="store_true", help="Also run ABC inference")
    parser.add_argument("--abc-pop-size", type=int, default=500, help="ABC population size")
    parser.add_argument("--abc-populations", type=int, default=5, help="ABC generations")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    args = parser.parse_args()
    
    # Load real data
    rain_data, coord_data = load_real_rainfall_data()
    if rain_data is None:
        print("Failed to load real rainfall data. Exiting.")
        return
    
    # Compute real MPLE for reference
    real_mple, mple_std_err = compute_real_mple()
    
    # Load trained models
    run_dir = pathlib.Path(args.run_dir).resolve()
    if not (run_dir / "manifest.json").exists():
        print(f"❌ manifest.json not found in {run_dir}")
        print("请确认你提供的 --run-dir 路径正确，比如：runs/run-20250913-152407")
        return
    manifest = load_manifest(run_dir)
    
    # Restore CNF models
    cnf_models = {}
    methods = [m for m in manifest.get("methods", []) if m in ("raw","data_score","pairwise_grouped","combined_grouped")]
    
    print(f"Loading {len(methods)} trained CNF models...")
    for method in methods:
        try:
            dim_data = int(manifest["artifacts"][method]["dim_data"])
            cnf = restore_model(run_dir, method, dim_data)
            cnf_models[method] = {'cnf': cnf, 'dim_data': dim_data}
            print(f"  {method}: loaded (dim={dim_data})")
        except Exception as e:
            print(f"  {method}: failed to load - {e}")
    
    # Setup output directory
    save_path = None
    if args.save_plots:
        if args.output_dir:
            save_path = pathlib.Path(args.output_dir)
        else:
            save_path = run_dir / "real_data_results"
        save_path.mkdir(exist_ok=True, parents=True)
        print(f"Results will be saved to: {save_path}")
    
    # Run CNF inference
    cnf_results = run_cnf_inference(cnf_models, rain_data, args.pair_groups, args.posterior_n)
    
    # Run ABC inference
    abc_results = {}
    if args.with_abc:
        abc_results = run_abc_inference(rain_data, args.pair_groups, args.abc_pop_size, args.abc_populations)
    
    # Create results summary + plots
    create_summary_table(cnf_results, abc_results, real_mple, mple_std_err)
    create_comparison_plots(cnf_results, abc_results, real_mple, mple_std_err, save_path)
    
    # Save numerical results safely
    if save_path:
        def _jsonable(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            if isinstance(x, dict):
                return {k: _jsonable(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_jsonable(v) for v in x]
            return x

        cnf_results_clean = {}
        cnf_total_time = None
        for k, v in cnf_results.items():
            if k == "_total_elapsed_sec":
                cnf_total_time = float(v)
                continue
            if isinstance(v, dict):
                cnf_results_clean[k] = {}
                for kk, vv in v.items():
                    if kk == "samples":
                        continue
                    cnf_results_clean[k][kk] = _jsonable(vv)
            else:
                cnf_results_clean[k] = _jsonable(v)

        abc_results_clean = {}
        for k, v in abc_results.items():
            if isinstance(v, dict):
                abc_results_clean[k] = {}
                for kk, vv in v.items():
                    if kk == "samples":
                        continue
                    abc_results_clean[k][kk] = _jsonable(vv)
            else:
                abc_results_clean[k] = _jsonable(v)

        results_summary = {
            "cnf_results": cnf_results_clean,
            "abc_results": abc_results_clean,
            "cnf_total_elapsed_sec": cnf_total_time,
            "real_mple": _jsonable(real_mple) if real_mple is not None else None,
            "mple_std_err": _jsonable(mple_std_err) if mple_std_err is not None else None,
        }

        with open(save_path / "real_data_results.json", "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Results saved to: {save_path}")

if __name__ == "__main__":
    main()
