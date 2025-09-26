#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlined visualization script for CNF and ABC benchmark results.
Creates 3 focused plots: parameter-wise MSE heatmap, posterior uncertainty, and posterior distribution boxplots.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_benchmark_results(bench_dir):
    """Load benchmark results from directory"""
    bench_path = Path(bench_dir)
    
    # Load metrics
    with open(bench_path / "metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Load per-replicate data
    per_rep = pd.read_csv(bench_path / "per_replicate.csv")
    
    return metrics, per_rep

def filter_methods(metrics, per_rep):
    """Remove 'abc' method if both 'abc' and 'abc_adj' exist, keep only abc_adj"""
    available_methods = list(metrics.keys())
    
    if 'abc' in available_methods and 'abc_adj' in available_methods:
        # Remove 'abc' and keep 'abc_adj'
        metrics = {k: v for k, v in metrics.items() if k != 'abc'}
        per_rep = per_rep[per_rep['method'] != 'abc']
        print("Removed 'abc' method, keeping 'abc_adj' only")
    
    return metrics, per_rep

def plot_parameter_wise_mse(metrics, save_path=None):
    """Plot MSE heatmap for each parameter across methods"""
    param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2', 
                   'scale0', 'scale1', 'scale2', 'shape']
    
    methods = list(metrics.keys())
    n_methods = len(methods)
    n_params = len(param_names)
    
    # Prepare data matrix
    mse_matrix = np.zeros((n_methods, n_params))
    for i, method in enumerate(methods):
        for j, param in enumerate(param_names):
            mse_matrix[i, j] = metrics[method]["per_param_mse"][param]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(mse_matrix, cmap='YlOrRd', aspect='auto')
    
    plt.colorbar(im, label='MSE')
    plt.yticks(range(n_methods), methods)
    plt.xticks(range(n_params), param_names, rotation=45)
    plt.title("Parameter-wise MSE Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Parameters")
    plt.ylabel("Methods")
    
    # Add text annotations
    for i in range(n_methods):
        for j in range(n_params):
            plt.text(j, i, f'{mse_matrix[i, j]:.4f}', 
                    ha='center', va='center', fontsize=9,
                    color='white' if mse_matrix[i, j] > np.max(mse_matrix)*0.6 else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "parameter_mse_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_posterior_uncertainty(metrics, save_path=None):
    """Plot average posterior standard deviations for each parameter"""
    param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2', 
                   'scale0', 'scale1', 'scale2', 'shape']
    
    methods = list(metrics.keys())
    n_params = len(param_names)
    
    # Create subplots
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(methods))
    
    for i, param in enumerate(param_names):
        method_names = []
        std_values = []
        
        for method in methods:
            method_names.append(method)
            std_values.append(metrics[method]["avg_posterior_sd"][param])
        
        bars = axes[i].bar(method_names, std_values, alpha=0.7, color=colors)
        axes[i].set_title(f'{param}', fontweight='bold')
        axes[i].set_ylabel('Avg Posterior SD')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, std_values):
            axes[i].text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + 0.02*max(std_values), 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle("Average Posterior Standard Deviations by Parameter", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "posterior_uncertainty.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_posterior_boxplots(per_rep, save_path=None):
    """Plot boxplots of posterior samples for each parameter across methods"""
    param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2', 
                   'scale0', 'scale1', 'scale2', 'shape']
    
    # We need to get posterior samples, but the CSV only has sd values
    # So we'll create boxplots based on the standard deviations across replicates
    fig, axes = plt.subplots(2, 5, figsize=(16, 10))
    axes = axes.flatten()
    
    methods = per_rep['method'].unique()
    
    for i, param in enumerate(param_names):
        sd_col = f'sd_{param}'
        
        if sd_col in per_rep.columns:
            # Prepare data for boxplot
            plot_data = []
            for method in methods:
                method_data = per_rep[per_rep['method'] == method][sd_col]
                for sd_val in method_data.dropna():
                    plot_data.append({'Method': method, 'Posterior SD': sd_val})
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                sns.boxplot(data=plot_df, x='Method', y='Posterior SD', ax=axes[i])
                axes[i].set_title(f'{param}', fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, 'No data available', ha='center', va='center',
                           transform=axes[i].transAxes)
                axes[i].set_title(f'{param}', fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, 'Column not found', ha='center', va='center',
                       transform=axes[i].transAxes)
            axes[i].set_title(f'{param}', fontweight='bold')
    
    plt.suptitle("Posterior Standard Deviation Distribution by Parameter", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / "parameter_posterior_boxplots.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(metrics):
    """Print summary table to console"""
    methods = list(metrics.keys())
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<15} {'Total MSE':<12} {'Avg Posterior SD':<15} {'Repeats':<8}")
    print("-" * 80)
    
    # Sort methods by total MSE
    sorted_methods = sorted(methods, key=lambda x: metrics[x]['total_mse'])
    
    for method in sorted_methods:
        total_mse = metrics[method]['total_mse']
        avg_sd = metrics[method]['avg_posterior_sd_mean']
        repeats = metrics[method]['repeats']
        print(f"{method:<15} {total_mse:<12.6f} {avg_sd:<15.4f} {repeats:<8}")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Create focused benchmark visualizations")
    parser.add_argument("--bench-dir", required=True, 
                       help="Path to benchmark results directory")
    parser.add_argument("--save-plots", action="store_true", 
                       help="Save plots to files")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for saved plots (default: bench-dir/plots)")
    args = parser.parse_args()
    
    # Load and filter data
    print(f"Loading benchmark results from {args.bench_dir}")
    metrics, per_rep = load_benchmark_results(args.bench_dir)
    metrics, per_rep = filter_methods(metrics, per_rep)
    
    # Setup save path
    save_path = None
    if args.save_plots:
        if args.output_dir:
            save_path = Path(args.output_dir)
        else:
            save_path = Path(args.bench_dir) / "plots"
        save_path.mkdir(exist_ok=True, parents=True)
        print(f"Plots will be saved to: {save_path}")
    
    # Create focused plots
    print(f"\nGenerating 3 focused plots for {len(metrics)} methods...")
    
    create_summary_table(metrics)
    
    print("1. Creating parameter-wise MSE heatmap...")
    plot_parameter_wise_mse(metrics, save_path)
    
    print("2. Creating posterior uncertainty analysis...")
    plot_posterior_uncertainty(metrics, save_path)
    
    print("3. Creating parameter posterior boxplots...")
    plot_parameter_posterior_boxplots(args.bench_dir, save_path)
    
    print("\nVisualization complete!")
    if save_path:
        print(f"Plots saved to: {save_path}")

if __name__ == "__main__":
    main()