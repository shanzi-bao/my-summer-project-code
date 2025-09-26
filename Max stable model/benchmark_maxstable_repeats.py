#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark CNF and ABC via repeated simulated observations.
Outputs:
  - per_replicate.csv : per replicate per method MSE/SE/SD + posterior mean + elapsed time
  - metrics.json      : aggregate MSE and average posterior SD per method
  - abc_rep{r}_posterior_samples_weighted.npz : ABC weighted posterior samples per replicate
  - abc_adj_rep{r}_posterior_samples.npy      : adjusted ABC posterior samples per replicate
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import pathlib
from datetime import datetime
import time
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from config import create_range_parameter_config
from sde import get_sde
from nn_model import NCMLP
from cnf import CNF
from dataset import your_simulator, your_prior_sampler, create_2d_grid

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

MPLE_THETA = np.asarray(mple_theta, dtype=float)
PARAM_NAMES = ['cov11','cov12','cov22','loc0','loc1','loc2','scale0','scale1','scale2','shape']

def load_manifest(run_dir: pathlib.Path):
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)

def restore_model(run_dir: pathlib.Path, method: str, dim_data: int):
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
    if method == "raw":
        return x_obs.flatten()
    elif method in ("data_score","pairwise_grouped","combined_grouped"):
        return compute_maxstable_test_input(jnp.asarray(x_obs), method, n_groups=n_groups)
    else:
        raise ValueError(f"Unknown method: {method}")

def choose_theta(theta_mode: str, rng: np.random.Generator):
    if theta_mode == "mple":
        return MPLE_THETA.copy()
    elif theta_mode == "prior":
        th = np.asarray(your_prior_sampler(1), dtype=float).reshape(-1)
        return th
    else:
        base = MPLE_THETA.copy()
        noise = rng.normal(0, 0.05, size=base.shape)
        return base * (1.0 + noise)

def run_abc_on_observation(obs_data: np.ndarray, out_db_path: pathlib.Path,
                           pop_size: int, populations: int, seed: int, pair_groups: int):
    import pyabc
    from pyabc import ABCSMC
    import numpy.random as npr
    npr.seed(seed)

    prior = MaxStablePrior()

    def _distance_with_groups(sim, obs):
        return maxstable_distance(sim, obs, n_groups=pair_groups)

    abc = ABCSMC(
        models=maxstable_model,
        parameter_priors=prior,
        distance_function=_distance_with_groups,
        population_size=pyabc.populationstrategy.ConstantPopulationSize(pop_size),
        sampler=pyabc.sampler.SingleCoreSampler(),
        eps=pyabc.epsilon.QuantileEpsilon(alpha=0.7, quantile_multiplier=0.95)
    )

    db_uri = f"sqlite:///{out_db_path}"
    history = abc.new(db_uri, {"data": np.asarray(obs_data)})
    history = abc.run(max_nr_populations=populations)

    df, w = history.get_distribution(m=0, t=history.max_t)
    df = df.reset_index(drop=True)
    samples = df[PARAM_NAMES].values
    w = np.asarray(w, dtype=float).reshape(-1)
    return samples, w, history

def run_abc_adjustment(history, obs_data, true_params, n_groups: int):
    try:
        adjusted_df = adjust_abc_results(
            history=history,
            obs_data={"data": obs_data},
            param_names=PARAM_NAMES,
            true_params=true_params,
            mple_theta=MPLE_THETA,
            n_groups=n_groups,
            simulator=your_simulator,
            score_fn=compute_maxstable_pairwise_scores,
            coords=create_2d_grid(),
            bandwidth_quantile=0.8,
            apply_to='kept'
        )
        return adjusted_df[PARAM_NAMES].values
    except Exception as e:
        print(f"ABC adjustment failed: {e}")
        return None

def main():
    ap = argparse.ArgumentParser(description="Benchmark CNF and ABC via repeated simulated observations")
    ap.add_argument("--run-dir", required=True, help="Existing run directory with trained CNF models")
    ap.add_argument("--repeats", type=int, default=20, help="Number of simulated observation replicates")
    ap.add_argument("--theta-mode", type=str, default="mple",
                    choices=["mple","prior","jitter_mple"],
                    help="How to choose true theta for simulation")
    ap.add_argument("--posterior-n", type=int, default=500, help="CNF posterior samples per replicate")
    ap.add_argument("--seed", type=int, default=2025, help="Base PRNG seed")
    ap.add_argument("--methods", type=str, default="",
                    help="Comma-separated subset of methods; empty=all CNF methods")
    ap.add_argument("--pair-groups", type=int, default=10,
                    help="Number of equal-distance groups for pairwise scores")
    # ABC options
    ap.add_argument("--with-abc", action="store_true", help="Also run ABC and adjusted ABC")
    ap.add_argument("--abc-pop-size", type=int, default=200, help="ABC population size")
    ap.add_argument("--abc-populations", type=int, default=3, help="ABC number of generations")
    ap.add_argument("--abc-sample-cap", type=int, default=2000, help="Cap ABC posterior samples")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    manifest = load_manifest(run_dir)
    all_methods = manifest.get("methods", [])

    if args.methods.strip():
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = [m for m in all_methods if m in ("raw","data_score","pairwise_grouped","combined_grouped")]

    if not methods and not args.with_abc:
        raise RuntimeError("No CNF methods found to benchmark, and --with-abc not set.")

    # Restore CNF models
    cnf_map = {}
    dim_map = {}
    if methods:
        dim_map = {m: int(manifest["artifacts"][m]["dim_data"]) for m in methods}
        cnf_map = {m: restore_model(run_dir, m, dim_map[m]) for m in methods}

    # Output directory
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bench_dir = run_dir / "benchmarks" / f"bench-{ts}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    se_sum = {}
    sd_sum = {}

    # Initialize accumulators (include ABC if requested)
    all_methods_to_track = list(methods)
    if args.with_abc:
        all_methods_to_track += ["abc", "abc_adj"]
    for m in all_methods_to_track:
        se_sum[m] = np.zeros(10, dtype=float)
        sd_sum[m] = np.zeros(10, dtype=float)

    for r in range(args.repeats):
        print(f"Running replicate {r+1}/{args.repeats}")

        theta_true = choose_theta(args.theta_mode, rng)
        key = jr.PRNGKey(int(rng.integers(0, 1_000_000)))
        x_obs = np.asarray(your_simulator(jnp.asarray(theta_true), key))

        # CNF methods
        for m in methods:
            start_m = time.time()
            try:
                feat = np.asarray(feature_by_method(x_obs, m, n_groups=args.pair_groups))
                expected = int(dim_map[m])
                if feat.shape[0] != expected:
                    raise ValueError(f"Dim mismatch for {m}: got {feat.shape[0]} vs {expected}")

                cnf = cnf_map[m]
                key2 = jr.PRNGKey(int(rng.integers(0, 1_000_000)))
                samples = cnf.batch_sample_fn(args.posterior_n, jnp.asarray(feat), key2)
                samples_np = np.asarray(samples, dtype=float)

                post_mean = samples_np.mean(axis=0)
                post_sd   = samples_np.std(axis=0)

                se = (post_mean - theta_true)**2
                se_sum[m] += se
                sd_sum[m] += post_sd

                elapsed_m = time.time() - start_m

                rows.append({
                    "replicate": r, "method": m,
                    "elapsed_sec": float(elapsed_m),
                    "mse_mean": float(se.mean()),
                    **{f"mean_{name}": float(post_mean[i]) for i, name in enumerate(PARAM_NAMES)},
                    **{f"se_{name}": float(se[i])         for i, name in enumerate(PARAM_NAMES)},
                    **{f"sd_{name}": float(post_sd[i])     for i, name in enumerate(PARAM_NAMES)},
                })
            except Exception as e:
                elapsed_m = time.time() - start_m
                print(f"CNF {m} failed for replicate {r}: {e}")
                rows.append({
                    "replicate": r, "method": m,
                    "elapsed_sec": float(elapsed_m),
                    "mse_mean": float("nan"),
                    **{f"mean_{name}": float("nan") for name in PARAM_NAMES},
                    **{f"se_{name}":   float("nan") for name in PARAM_NAMES},
                    **{f"sd_{name}":   float("nan") for name in PARAM_NAMES},
                    "error": str(e)
                })

        # ABC methods
        if args.with_abc:
            abc_db = bench_dir / f"abc_rep{r}.db"
            try:
                # ---- Original ABC ----
                t0 = time.time()
                abc_samples, abc_w, history = run_abc_on_observation(
                    obs_data=x_obs,
                    out_db_path=abc_db,
                    pop_size=args.abc_pop_size,
                    populations=args.abc_populations,
                    seed=int(rng.integers(0, 10**6)),
                    pair_groups=args.pair_groups
                )
                abc_time = time.time() - t0

                # 保存原始 ABC 样本（加权）
                try:
                    abc_npz = bench_dir / f"abc_rep{r}_posterior_samples_weighted.npz"
                    np.savez(abc_npz, samples=abc_samples, weights=abc_w)
                    print(f"[INFO] Saved ABC weighted samples -> {abc_npz.name}  shape={abc_samples.shape}")
                except Exception as e:
                    print(f"[WARN] Failed saving ABC weighted samples for rep {r}: {e}")

                # Cap & 归一化权重（用于统计）
                if abc_samples.shape[0] > args.abc_sample_cap:
                    idx = rng.choice(abc_samples.shape[0], args.abc_sample_cap,
                                     replace=False, p=abc_w/abc_w.sum())
                    abc_samples = abc_samples[idx]
                    abc_w = abc_w[idx]
                abc_w = abc_w / abc_w.sum()

                abc_mean = (abc_samples * abc_w[:, None]).sum(axis=0)
                abc_var  = np.average((abc_samples - abc_mean)**2, axis=0, weights=abc_w)
                abc_sd   = np.sqrt(abc_var)

                se = (abc_mean - theta_true)**2
                se_sum["abc"] += se
                sd_sum["abc"] += abc_sd

                rows.append({
                    "replicate": r, "method": "abc",
                    "elapsed_sec": float(abc_time),
                    "mse_mean": float(se.mean()),
                    **{f"mean_{name}": float(abc_mean[i]) for i, name in enumerate(PARAM_NAMES)},
                    **{f"se_{name}":   float(se[i])       for i, name in enumerate(PARAM_NAMES)},
                    **{f"sd_{name}":   float(abc_sd[i])   for i, name in enumerate(PARAM_NAMES)},
                })

                # ---- Adjusted ABC ----
                t1 = time.time()
                abc_adj_samples = run_abc_adjustment(history, x_obs, theta_true, args.pair_groups)
                abc_adj_time = time.time() - t1

                if abc_adj_samples is not None:
                    # 保存调整后样本（等权）
                    try:
                        out_adj = bench_dir / f"abc_adj_rep{r}_posterior_samples.npy"
                        np.save(out_adj, abc_adj_samples)
                        print(f"[INFO] Saved adjusted ABC samples -> {out_adj.name}  shape={abc_adj_samples.shape}")
                    except Exception as e:
                        print(f"[WARN] Failed saving adjusted ABC samples for rep {r}: {e}")

                    adj_mean = abc_adj_samples.mean(axis=0)
                    adj_sd   = abc_adj_samples.std(axis=0)
                    se_adj   = (adj_mean - theta_true)**2

                    se_sum["abc_adj"] += se_adj
                    sd_sum["abc_adj"] += adj_sd

                    rows.append({
                        "replicate": r, "method": "abc_adj",
                        "elapsed_sec": float(abc_adj_time),
                        "mse_mean": float(se_adj.mean()),
                        **{f"mean_{name}": float(adj_mean[i]) for i, name in enumerate(PARAM_NAMES)},
                        **{f"se_{name}":   float(se_adj[i])   for i, name in enumerate(PARAM_NAMES)},
                        **{f"sd_{name}":   float(adj_sd[i])   for i, name in enumerate(PARAM_NAMES)},
                    })
                else:
                    raise ValueError("ABC adjustment returned None")

            except Exception as e:
                print(f"ABC failed for replicate {r}: {e}")
                for method in ["abc", "abc_adj"]:
                    rows.append({
                        "replicate": r, "method": method,
                        "elapsed_sec": float("nan"),
                        "mse_mean": float("nan"),
                        **{f"mean_{name}": float("nan") for name in PARAM_NAMES},
                        **{f"se_{name}":   float("nan") for name in PARAM_NAMES},
                        **{f"sd_{name}":   float("nan") for name in PARAM_NAMES},
                        "error": str(e)
                    })

    # ---- Aggregate metrics ----
    agg = {}
    for m in all_methods_to_track:
        mse = (se_sum[m] / args.repeats).tolist()
        avg_sd = (sd_sum[m] / args.repeats).tolist()
        agg[m] = {
            "per_param_mse":       {PARAM_NAMES[i]: mse[i]    for i in range(10)},
            "total_mse":           float(np.sum(mse)),
            "avg_posterior_sd":    {PARAM_NAMES[i]: avg_sd[i] for i in range(10)},
            "avg_posterior_sd_mean": float(np.mean(avg_sd)),
            "repeats": int(args.repeats),
            "posterior_n": int(args.posterior_n) if m in methods else "N/A",
            "theta_mode": args.theta_mode
        }

    # ---- Save outputs ----
    import csv
    csv_path = bench_dir / "per_replicate.csv"
    if rows:
        keys = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    with open(bench_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    print(f"\nBenchmark completed. Results saved to {bench_dir}")

    # Summary
    print("\nSummary (Total MSE):")
    for m in sorted(agg.keys()):
        print(f"{m:20s}: {agg[m]['total_mse']:.6f}")

if __name__ == "__main__":
    main()
