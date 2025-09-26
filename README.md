# Composite Likelihood–Enhanced Score-Based Diffusion Models (GRF & Max-Stable)

This repository contains two case studies of likelihood-free Bayesian inference via **score-based diffusion models** enhanced with composite-likelihood features:

1. **Gaussian Random Field (GRF):** range parameter inference on an 8×8 grid.  
2. **Max-Stable (Smith model) for spatial extremes:** 10-parameter inference on simulated and real rainfall data.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Gaussian Random Field](#gaussian-random-field)
  - [Max-Stable – Simulated Repeats](#max-stable--simulated-repeats)
  - [Max-Stable – Real Data](#max-stable--real-data)
- [Project Structure](#project-structure)
- [Output Files and Results](#output-files-and-results)
- [Credits](#credits)
- [License](#license)

## Installation

Make sure you have Python 3.9+ installed. Then install dependencies with:

```bash
pip install -r requirements.txt
```

For Max-Stable experiments, you also need R (≥ 4.0) and the R package `SpatialExtremes`:

```r
install.packages("SpatialExtremes")
```

> Make sure `rpy2` can locate your R installation (set `R_HOME` if needed).

## Usage

### Gaussian Random Field

Run the Jupyter notebook directly:

```bash
jupyter notebook final.ipynb
```

This will train the score-based diffusion model, sample from the posterior, and visualize posterior distributions.

**Output:** Interactive plots and analysis within the notebook

---

### Max-Stable – Simulated Repeats

**Step 1 – Train score-based diffusion models**

```bash
python server_run_maxstable.py \
  --methods raw,data_score,pairwise_grouped,combined_grouped \
  --n-samples 100 \
  --pair-groups 10 \
  --outdir runs
```

**Output:**
```
runs/run-YYYYMMDD-HHMMSS/
├── manifest.json              # Training metadata and file locations
├── run.log                    # Detailed training logs
├── raw/
│   ├── model.eqx             # Trained neural network (Equinox format)
│   ├── ds_mean.npy           # Data standardization parameters
│   └── ds_std.npy            # Data standardization parameters
├── data_score/
│   ├── model.eqx
│   └── ...
├── pairwise_grouped/
│   └── ...
├── combined_grouped/
│   └── ...
└── results/
    ├── raw_posterior_samples.npy      # Test posterior samples (1000×10)
    ├── raw_posterior_summary.json     # Mean and std summaries
    └── ...
```

**Step 2 – Benchmark on repeated simulated observations**

```bash
python benchmark_maxstable_repeats.py \
  --run-dir runs/run-YYYYMMDD-HHMMSS \
  --repeats 20 \
  --posterior-n 500 \
  --pair-groups 10 \
  --with-abc
```

**Output:**
```
runs/run-YYYYMMDD-HHMMSS/benchmarks/bench-YYYYMMDD-HHMMSS/
├── per_replicate.csv          # Per-experiment MSE and posterior SD by method
├── metrics.json               # Aggregated performance metrics
├── abc_rep0.db                # ABC database files (if --with-abc)
├── abc_rep1.db
└── ...
```

**Step 3 – Visualize benchmark results**

```bash
python benchmark_visualization.py \
  --bench-dir runs/run-YYYYMMDD-HHMMSS/benchmarks/bench-YYYYMMDD-HHMMSS \
  --save-plots
```

**Output:**
```
runs/.../benchmarks/.../plots/
├── parameter_mse_heatmap.png         # MSE comparison across methods/parameters
├── posterior_uncertainty.png         # Average posterior standard deviations
└── parameter_posterior_boxplots.png  # Distribution comparisons
```

---

### Max-Stable – Real Data

```bash
python real_data_inference.py \
  --run-dir runs/run-YYYYMMDD-HHMMSS \
  --pair-groups 10 \
  --posterior-n 1000 \
  --with-abc \
  --save-plots
```

**Output:**
```
runs/run-YYYYMMDD-HHMMSS/real_data_results/
├── real_data_results.json                    # Posterior summaries and timing info
├── real_data_comparison.png                  # Posterior means vs MPLE reference
└── real_data_posterior_distributions.png     # Method comparison boxplots
```

**Console Output:** Detailed parameter comparison tables showing:
- Posterior means and standard deviations for each method
- Comparison with MPLE benchmark values from SpatialExtremes
- Method performance rankings
- Timing information for each approach

---

## Project Structure

```text
Gaussian Random Field/
├── dataset.py
├── config.py
├── nn_model.py
├── train.py
├── cnf.py
├── final.ipynb              # run directly

Max stable model/
├── dataset.py               # R-based simulator + priors
├── maxstablerevised_副本5.py  # feature construction, MPLE, ABC helpers
├── server_run_maxstable.py  # full training pipeline
├── benchmark_maxstable_repeats.py   # repeated-simulation benchmark
├── benchmark_visualization.py       # visualization for simulated benchmarks
├── real_data_inference.py           # real data inference (plots directly)
├── config.py, sde.py, nn_model.py, train.py, cnf.py
```

## Output Files and Results

### Key Output Files

| File Type | Description | Usage |
|-----------|-------------|--------|
| `manifest.json` | Training metadata, file paths, and configuration | Reproducibility and run identification |
| `model.eqx` | Trained neural network weights | Loading models for inference |
| `ds_mean.npy`, `ds_std.npy` | Data normalization parameters | Required for proper model inference |
| `per_replicate.csv` | Individual experiment results | Statistical analysis of method performance |
| `metrics.json` | Aggregated benchmark statistics | Method comparison and ranking |
| `real_data_results.json` | Real data inference