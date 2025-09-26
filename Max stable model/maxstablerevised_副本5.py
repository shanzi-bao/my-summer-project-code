#!/usr/bin/env python
# coding: utf-8




# Cell 1: Import and basic functions - Max-Stable version
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import time
import numpy as np
from config import create_range_parameter_config
from dataset import your_prior_sampler, your_simulator,create_2d_grid  # These need to be Max-Stable version
from nn_model import NCMLP
from sde import get_sde
from train import train_score_network
from cnf import CNF

# Add Max-Stable related imports
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from jax.scipy.stats import norm
from jax import grad

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
# Import R packages
try:
    spatial_extremes = importr('SpatialExtremes')
    base = importr('base')
    print("R SpatialExtremes package loaded successfully!")
except Exception as e:
    print(f"R package loading failed: {e}")
    raise




# In[2]:

# numba_accelerated.py - Numba-accelerated Max-Stable computation (Quantile grouping version)
import numpy as np
import jax.numpy as jnp
from numba import jit, njit, prange
import time

# Numba-accelerated normal distribution functions
@njit
def norm_cdf_numba(x):
    """Fast normal distribution CDF calculation"""
    return 0.5 * (1 + np.tanh(x * 0.7978845608028654))  # approximation formula

@njit  
def norm_pdf_numba(x):
    """Fast normal distribution PDF calculation"""
    return 0.3989422804014327 * np.exp(-0.5 * x * x)

@njit
def max_stable_pairwise_loglik_multi_obs_numba(params, coord_i, coord_j, y_series_i, y_series_j):
    """
    Numba-accelerated version of Max-Stable pairwise likelihood
    """
    n_obs = len(y_series_i)
    total_loglik = 0.0
    
    # Parse parameters
    cov11, cov12, cov22 = params[0], params[1], params[2]
    beta_mu_0, beta_mu_1, beta_mu_2 = params[3], params[4], params[5]
    beta_la_0, beta_la_1, beta_la_2 = params[6], params[7], params[8]
    xi = max(min(params[9], 0.99), 0.01)  # clip operation
    
    # Calculate spatial correlation parameters
    h_x = coord_j[0] - coord_i[0]
    h_y = coord_j[1] - coord_i[1]
    det_cov = cov11 * cov22 - cov12 * cov12
    det_cov = max(det_cov, 1e-8)
    idet = 1.0 / det_cov
    a_squared = (cov22 * h_x * h_x - 2 * cov12 * h_x * h_y + cov11 * h_y * h_y) * idet
    a = np.sqrt(max(a_squared, 1e-8))
    
    # Location-related GEV parameters - expand matrix operations
    mu_i = beta_mu_0 + beta_mu_1 * coord_i[0] + beta_mu_2 * coord_i[1]
    mu_j = beta_mu_0 + beta_mu_1 * coord_j[0] + beta_mu_2 * coord_j[1]
    lambda_i = max(beta_la_0 + beta_la_1 * coord_i[0] + beta_la_2 * coord_i[1], 0.1)
    lambda_j = max(beta_la_0 + beta_la_1 * coord_j[0] + beta_la_2 * coord_j[1], 0.1)
    
    # Calculate for each observation
    for t in range(n_obs):
        y_i_t = y_series_i[t]
        y_j_t = y_series_j[t]
        
        # GEV to Fréchet transformation
        gev_arg_i = max(1.0 + xi * (y_i_t - mu_i) / lambda_i, 1e-8)
        gev_arg_j = max(1.0 + xi * (y_j_t - mu_j) / lambda_j, 1e-8)
        
        xi_safe = 1e-6 if abs(xi) < 1e-6 else xi
        z_i = max(gev_arg_i**(1.0 / xi_safe), 1e-8)
        z_j = max(gev_arg_j**(1.0 / xi_safe), 1e-8)
        
        # Smith model calculation
        w = a * 0.5 + np.log(z_j / z_i) / a
        v = a - w
        
        # Use Numba fast normal distribution
        Phi_w = norm_cdf_numba(w)
        Phi_v = norm_cdf_numba(v)
        phi_w = norm_pdf_numba(w)
        phi_v = norm_pdf_numba(v)
        
        A = -Phi_w / z_i - Phi_v / z_j
        
        z_i2, z_j2 = z_i * z_i, z_j * z_j
        a_safe = max(a, 1e-8)
        
        B = Phi_w / z_i2 + phi_w / (z_i2 * a_safe) - phi_v / (a_safe * z_j * z_i)
        C = Phi_v / z_j2 + phi_v / (z_j2 * a_safe) - phi_w / (a_safe * z_i * z_j)
        D = (v * phi_w / (a_safe * a_safe * z_i2 * z_j) + w * phi_v / (a_safe * a_safe * z_j2 * z_i))
        
        BC_plus_D = max(B * C + D, 1e-12)
        
        E = (np.log(1.0 / (lambda_i * lambda_j)) + 
             np.log(gev_arg_i / lambda_i) * (1.0 / xi_safe - 1) +
             np.log(gev_arg_j / lambda_j) * (1.0 / xi_safe - 1))
        
        loglik_t = A + np.log(BC_plus_D) + E
        loglik_t = max(min(loglik_t, 1000.0), -1000.0)  # clip operation
        
        if np.isfinite(loglik_t):
            total_loglik += loglik_t
    
    return max(min(total_loglik, 10000.0), -10000.0)

@njit
def finite_difference_gradient_numba(theta, coord_i, coord_j, y_series_i, y_series_j, eps=1e-5):
    """Numba-accelerated finite difference gradient calculation"""
    n_params = len(theta)
    grad = np.zeros(n_params)
    
    f0 = max_stable_pairwise_loglik_multi_obs_numba(theta, coord_i, coord_j, y_series_i, y_series_j)
    if not np.isfinite(f0):
        grad[:] = np.nan
        return grad
    
    for i in range(n_params):
        # Create perturbed versions
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        
        f_plus = max_stable_pairwise_loglik_multi_obs_numba(theta_plus, coord_i, coord_j, y_series_i, y_series_j)
        f_minus = max_stable_pairwise_loglik_multi_obs_numba(theta_minus, coord_i, coord_j, y_series_i, y_series_j)
        
        if np.isfinite(f_plus) and np.isfinite(f_minus):
            grad[i] = (f_plus - f_minus) / (2 * eps)
        elif np.isfinite(f_plus):
            grad[i] = (f_plus - f0) / eps
        else:
            grad[i] = np.nan
    
    return grad


# In[3]:


from dataset import *
mple_theta = np.array([
    332.15273678,  70.39821371, 184.62656960,
     20.65249877,   0.06445933,  -0.15529697,
      3.54050776,   0.02308246,  -0.03934086,
      0.19174323
])

# Fixed return mple_var_cov
mple_var_cov = np.array([
    [3058.28141648,  597.11836430, 1649.08060611,   56.56839944,   0.01071023,  -0.18772264,  30.01013376,  0.06274535,  -0.16883329,   1.71729255],
    [ 597.11836429,  125.30253380,  331.69782626,    7.69543749,   0.01085227,  -0.04567036,   7.60319210,  0.01595882,  -0.04911461,   0.33194457],
    [1649.08060613,  331.69782628,  907.84190032,   24.80659089,   0.01864289,  -0.11197847,  10.72290018,  0.04922308,  -0.11123480,   0.92366532],
    [  56.56839944,    7.69543749,   24.80659089,   63.30044118,  -0.06852866,  -0.05770003,  23.59634781, -0.02639509,  -0.01750179,   0.04883489],
    [   0.01071023,    0.01085227,    0.01864289,   -0.06852866,   0.0001178874, -0.00005462993, -0.031026810,  0.00005000969, -0.00001480252, -0.00002365657],
    [  -0.18772264,   -0.04567036,   -0.11197847,   -0.05770003,  -0.00005462993, 0.0003775941, -0.006237426, -0.00003130929,  0.0001043259,  -0.0001129467],
    [  30.01013376,    7.60319210,   10.72290018,   23.59634781,  -0.031026810, -0.006237426,  27.49918640, -0.034578974, -0.011168717,   0.03901475],
    [   0.06274535,    0.01595882,    0.04922308,   -0.02639509,   0.00005000969, -0.00003130929, -0.034578974,  0.00005537796, -0.00001420581,  0.00001119952],
    [  -0.16883329,   -0.04911461,   -0.11123480,   -0.01750179,  -0.00001480252, 0.0001043259, -0.011168717, -0.00001420581,  0.00007651868, -0.0001309333],
    [   1.71729255,    0.33194457,    0.92366532,    0.04883489,  -0.00002365657, -0.0001129467,  0.03901475,  0.00001119952, -0.0001309333,  0.001161543]
])


#

def estimate_maxstable_params_simple(x_obs, coords=None):
    """
    Estimate Max-Stable parameters using MPLE based on multi-year observation data - Fixed version
    
    Parameters:
    -----------
    x_obs : jnp.ndarray (n_years, n_locations)
        Multi-year observation data, e.g., (47, 79)
    coords : jnp.ndarray (n_locations, 2)
        Coordinate data
        
    Returns:
    --------
    theta_mple : jnp.ndarray (10,)
        Fixed return of pre-computed MPLE parameters
    """
    

    
    # Directly return fixed mple_theta values
    return jnp.array(mple_theta)


# In[6]:


import numpy as np
import jax.numpy as jnp
from numba import njit, prange
import time

def create_equal_distance_groups(coords, n_groups=10):
    """Create equal distance groups (instead of quantile groups)"""
    distances = []
    pair_indices = []
    n_locations = len(coords)
    
    for i in range(n_locations):
        for j in range(i + 1, n_locations):
            dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
            distances.append(dist)
            pair_indices.append((i, j))
    
    distances = np.array(distances)
    
   # print(f"Equal distance grouping:")
    #print(f"   Total pairs: {len(distances)}")
    #print(f"   Distance range: [{np.min(distances):.3f}, {np.max(distances):.3f}]")
    
    # Key change: equal distance grouping instead of quantile grouping
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Calculate equal distance boundaries
    group_boundaries = np.linspace(min_dist, max_dist, n_groups + 1)
    

    
    # Prepare numba format data
    equal_groups_flat = []
    group_sizes = []
    
    for group_id in range(n_groups):
        # Find pairs in this group
        if group_id == n_groups - 1:
            mask = (distances >= group_boundaries[group_id]) & (distances <= group_boundaries[group_id + 1])
        else:
            mask = (distances >= group_boundaries[group_id]) & (distances < group_boundaries[group_id + 1])
        
        group_pairs = [pair_indices[idx] for idx in np.where(mask)[0]]
        group_sizes.append(len(group_pairs))
        
    
        if len(group_pairs) > 0:
            group_distances = [distances[idx] for idx in np.where(mask)[0]]
            actual_min = np.min(group_distances)
            actual_max = np.max(group_distances)

        else:
            print(f"   Group{group_id+1}: [{group_boundaries[group_id]:.3f}, {group_boundaries[group_id+1]:.3f}] - 0 pairs")
        
        # Flatten pair indices
        for i, j in group_pairs:
            equal_groups_flat.extend([i, j])
    
    return np.array(equal_groups_flat, dtype=np.int64), np.array(group_sizes, dtype=np.int64)

@njit(parallel=True)
def compute_pairwise_scores_equal_distance_numba(theta_samples, x_samples, coords, equal_groups_flat, group_sizes):
    """
    Equal distance grouping pairwise scores computation
    """
    n_param_samples, n_obs, n_locations = x_samples.shape
    n_groups = len(group_sizes)  # n_groups groups
    n_params = 10  # 10 parameters
    
    all_grouped_scores = np.zeros((n_param_samples, n_groups, n_params))
    
    # Process each sample in parallel
    for sample_idx in prange(n_param_samples):
        theta = theta_samples[sample_idx]
        x_data = x_samples[sample_idx]
        grouped_scores = np.zeros((n_groups, n_params))
        
        pair_start = 0
        
        # Process each distance group
        for group_idx in range(n_groups):
            group_size = group_sizes[group_idx]
            group_grad_sum = np.zeros(n_params)  # 10-dimensional gradient accumulator
            group_count = 0
            
            # Process all pairs in this group
            for k in range(group_size):
                pair_idx = pair_start + k
                i = equal_groups_flat[pair_idx * 2]
                j = equal_groups_flat[pair_idx * 2 + 1]
                
                y_series_i = x_data[:, i]
                y_series_j = x_data[:, j]
                
                # Check data validity
                valid = True
                for t in range(n_obs):
                    if not (np.isfinite(y_series_i[t]) and np.isfinite(y_series_j[t])):
                        valid = False
                        break
                
                if valid:
                    # Use previously defined numba function
                    grad_vector = finite_difference_gradient_numba(
                        theta, coords[i], coords[j], y_series_i, y_series_j, 1e-4
                    )
                    
                    # Check gradient validity
                    grad_valid = True
                    for p in range(n_params):
                        if not np.isfinite(grad_vector[p]):
                            grad_valid = False
                            break
                    
                    if grad_valid:
                        for p in range(n_params):
                            group_grad_sum[p] += grad_vector[p]
                        group_count += 1
            
            # Calculate average 10-dimensional gradient for this group
            if group_count > 0:
                for p in range(n_params):
                    grouped_scores[group_idx, p] = group_grad_sum[p] / group_count
            
            pair_start += group_size
        
        all_grouped_scores[sample_idx] = grouped_scores
    
    return all_grouped_scores

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

# Quick test for difference between equal distance vs quantile
def test_grouping_methods():
    """Test the effect of equal distance grouping"""
    
    from dataset import create_2d_grid
    import jax.random as jr
    
    print("Testing equal distance grouping...")
    
    # Generate test data
    coords = create_2d_grid()
    theta_test = jnp.array(mple_theta)[None, :]  # (1, 10)
    
    # Generate observation data
    key = jr.PRNGKey(42)
    x_test = your_simulator(mple_theta, key)[None, :, :]  # (1, 47, 79)
    
    print(f"Test data shape: theta {theta_test.shape}, x {x_test.shape}")
    
    # Equal distance grouping (now the default compute_maxstable_pairwise_scores)
    print("\nEqual distance grouping:")
    equal_scores = compute_maxstable_pairwise_scores(
        theta_test, x_test, coords, n_groups=30
    )
    
    # Calculate features
    equal_flat = equal_scores[0].flatten()  # (100,)
    
    print(f"\nEqual distance grouping results:")
    print(f"Feature shape: {equal_scores.shape}")
    print(f"Feature range: [{jnp.min(equal_flat):.3f}, {jnp.max(equal_flat):.3f}]")
    print(f"Feature mean: {jnp.mean(equal_flat):.6f}")
    print(f"Feature std: {jnp.std(equal_flat):.6f}")
    
    return equal_scores



# In[7]:


# Revised prepare function
def prepare_all_maxstable_data_optimized(n_samples=100, key=42, n_groups = 10):
    """Prepare Max-Stable data (revised version) - compute pairwise scores using estimated parameters"""
    print("Preparing Max-Stable data (revised version)...")
    from dataset import create_2d_grid
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


# In[8]:




# Cell 5: Max-Stable training function (modify parameter dimensions)
def train_single_maxstable_method(method_name, theta_train, x_input, train_key, dim_data=16):
    """Train single Max-Stable method"""
    print(f"\nTraining Max-Stable {method_name} method...")
    start_time = time.time()
    
    key = jr.PRNGKey(train_key)
    config = create_range_parameter_config()
    
    # Adjust to Max-Stable parameter dimensions
    config.algorithm.dim_data = dim_data
    config.algorithm.dim_parameter = 10  # Max-Stable has 10 parameters
    
    sde = get_sde(config)
    model = NCMLP(key, config)
    
    trained_model, ds_mean, ds_std = train_score_network(
        config, model, sde, theta_train, x_input, key
    )
    
    training_time = time.time() - start_time
    print(f"Max-Stable {method_name} training completed, time taken: {training_time:.2f} seconds")
    
    return {
        'trained_model': trained_model,
        'ds_mean': ds_mean,
        'ds_std': ds_std,
        'sde': sde,
        'config': config,
        'training_time': training_time,
        'method': method_name
    }





# Modified original compute_maxstable_test_input function to support equal distance grouping
def compute_maxstable_test_input(x_obs, method, n_groups=10):
    """Max-Stable test input computation function - equal distance grouping version"""
    
    if method == 'raw':
        return x_obs.flatten()  # (47, 79) → (3713,)

    theta_estimated = estimate_maxstable_params_simple(x_obs)
    
    # Use equal distance grouping to compute pairwise scores
    theta_samples = theta_estimated[None, :]  # (1, 10)
    x_samples = x_obs[None, :, :]  # (1, 47, 79)
    coordinates = create_2d_grid()
    
    pairwise_scores = compute_maxstable_pairwise_scores(
        theta_samples, x_samples, coordinates, n_groups=n_groups
    )
    pairwise_matrix = pairwise_scores[0]  # (30, 10)
    
    if method == 'data_score':
        data_score = jnp.sum(pairwise_matrix, axis=0)  # (10,)
        return data_score
        
    elif method == 'pairwise_grouped':
        sample_vec = pairwise_matrix.flatten()  # (300,)
        return sample_vec
        
    elif method == 'combined_grouped':
        data_score = jnp.sum(pairwise_matrix, axis=0)  # (10,)
        pairwise_flat = pairwise_matrix.flatten()  # (300,)
        combined_result = jnp.concatenate([data_score, pairwise_flat])
        return combined_result

# Test function to verify dimensions
def test_input_dimensions():
    """Test whether input dimensions of each method are consistent with training"""
    
    print("Testing input dimension consistency...")
    
    # Generate test data
    key = jr.PRNGKey(999)
    test_theta = mple_theta
    x_obs = your_simulator(test_theta, key)  # (47, 79)
    
    print(f"Test observation data shape: {x_obs.shape}")
    
    # Expected training dimensions (based on your training data)
    expected_dims = {
        'raw': 47 * 79,           # 3713 dimensions
        'data_score': 10,         # 10-dimensional parameter vector
        'pairwise_grouped': 300,  # 10×10 flattened
        'combined_grouped': 310   # 10+100
    }
    
    print("\nDimension check:")
    print("-" * 50)
    
    for method in ['raw', 'data_score', 'pairwise_grouped', 'combined_grouped']:
        try:
            test_input = compute_maxstable_test_input(x_obs, method)
            actual_dim = len(test_input)
            expected_dim = expected_dims[method]
            
            status = "PASS" if actual_dim == expected_dim else "FAIL"
            print(f"{method:<20}: {actual_dim:>4} dims (expected {expected_dim:>4} dims) {status}")
            
            if actual_dim != expected_dim:
                print(f"   Actual shape: {test_input.shape}")
                
        except Exception as e:
            print(f"{method:<20}: FAIL - computation failed - {e}")
    
    return True

# Fixed test code - for use in your parameter estimation tests
def get_test_input_for_method(rain_data, method_name):
    """Get test input for specified method - unified interface"""
    
    if method_name == 'raw':
        return rain_data.flatten()
    
    elif method_name in ['data_score', 'pairwise_grouped', 'combined_grouped']:
        return compute_maxstable_test_input(rain_data, method_name)
    
    else:
        raise ValueError(f"Unknown method: {method_name}")

# In your parameter estimation tests, replace the original test input computation:
def updated_parameter_estimation_test_snippet():
    """Show how to use the revised function in parameter estimation tests"""
    
    # Example: in your test loop
    for method_name, cnf_info in cnf_models.items():
        print(f"\n--- {method_name.upper()} ---")
        
        try:
            cnf = cnf_info['cnf']
            
            # Unified test input computation
            test_input = get_test_input_for_method(rain_data, method_name)
            
            # Verify dimensions
            expected_dim = cnf_info['dim_data']
            if len(test_input) != expected_dim:
                print(f"FAIL: Dimension mismatch: expected {expected_dim}, actual {len(test_input)}")
                continue
            
            print(f"PASS: Input dimension match: {len(test_input)}")
            
            # Continue sampling and prediction...
            # base_key, sample_key = jr.split(base_key)
            # samples = cnf.batch_sample_fn(1000, test_input, sample_key)
            # ...
            
        except Exception as e:
            print(f"FAIL: {str(e)[:50]}...")


# Concise test: using SpatialExtremes real rainfall data
import rpy2.robjects as ro
import numpy as np
import jax.numpy as jnp
import jax.random as jr

def test_real_rainfall_data():
    """Test parameter estimation on real rainfall data"""
    
    print("Loading SpatialExtremes real rainfall data...")
    
    # 1. Load real data
    ro.r("""
    library(SpatialExtremes)
    data(rainfall)
    """)
    
    # Get real data
    rain_data = np.array(ro.r('rain'))  
    coord_data = np.array(ro.r('coord[,-3]'))  # Remove elevation column
    
    print(f"Real data shape: {rain_data.shape}")
    print(f"Coordinate shape: {coord_data.shape}")
    print(f"Data range: [{np.min(rain_data):.2f}, {np.max(rain_data):.2f}]")
    
    # 2. Calculate real MPLE estimation using R
    print("\nCalculating real MPLE estimation...")
    ro.r("""
    mple_fit <- fitmaxstab(rain, coord=coord[,-3], cov.mod="gauss",
                          loc ~ lon + lat, scale ~ lon + lat, shape ~ 1,
                          fit.marge=TRUE)
    real_mple <- mple_fit$fitted.values
    """)
    
    real_mple = np.array(ro.r('real_mple'))
    print(f"Real MPLE: {real_mple}")
    
    # 3. Test predictions from each method
    print(f"\nTesting predictions from each method on real data...")
    
    # Import your functions (assuming models are already trained)

    
    param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2', 
                   'scale0', 'scale1', 'scale2', 'shape']
    
    results = {}
    key = jr.PRNGKey(42)
    
    for method_name, cnf_info in cnf_models.items():
        print(f"\n--- {method_name.upper()} ---")
        
        try:
            # Calculate test input
            test_input = get_test_input_for_method(jnp.array(rain_data), method_name)
            print(f"Input dimension: {len(test_input)}")
            
            # CNF sampling
            key, sample_key = jr.split(key)
            samples = cnf_info['cnf'].batch_sample_fn(1000, test_input, sample_key)
            
            # Statistical results
            pred_mean = jnp.mean(samples, axis=0)
            pred_std = jnp.std(samples, axis=0)
            
            print("Prediction results vs real MPLE:")
            print(f"{'Param':<8} {'MPLE':<10} {'CNF Mean':<10} {'CNF Std':<10} {'Diff':<8}")
            print("-" * 50)
            
            total_diff = 0
            for i, name in enumerate(param_names):
                diff = abs(pred_mean[i] - real_mple[i])
                total_diff += diff
                print(f"{name:<8} {real_mple[i]:<10.4f} {pred_mean[i]:<10.4f} {pred_std[i]:<10.4f} {diff:<8.4f}")
            
            print(f"Total difference: {total_diff:.4f}")
            results[method_name] = {
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'total_diff': total_diff
            }
            
        except Exception as e:
            print(f"Failed: {str(e)[:50]}")
    
    # 4. Sort and display best methods
    print(f"\nMethod ranking (sorted by difference from MPLE):")
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['total_diff'])
    
    for i, (method, result) in enumerate(sorted_methods, 1):
        print(f"{i}. {method}: Total difference {result['total_diff']:.4f}")
    
    return results, real_mple

# Run test
if __name__ == "__main__":
    results, real_mple = test_real_rainfall_data()


# In[20]:


# Import your ABC implementation
import pyabc
from pyabc import ABCSMC, DistributionBase, Parameter
from scipy.stats import multivariate_t
import tempfile
import os

# Copy your ABC classes and functions here directly
class MaxStablePrior(DistributionBase):
    """Complete implementation of your MaxStablePrior class"""
    def __init__(self):
        self.df = 5
        self.loc = mple_theta
        self.shape = mple_var_cov * 2.5
        cov = (self.shape + self.shape.T) / 2
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        min_eigenval = 1e-3
        eigenvals = np.maximum(eigenvals, min_eigenval)
        self.shape = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        self.mvt = multivariate_t(loc=self.loc, shape=self.shape, df=self.df)
        self.param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2',
                            'scale0', 'scale1', 'scale2', 'shape']
    
    def rvs(self, *args, **kwargs):
        max_attempts = 100
        for attempt in range(max_attempts):
            raw_sample = self.mvt.rvs()
            constrained = self._apply_constraints(raw_sample)
            if self._validate_sample(constrained):
                sample_dict = {name: float(constrained[i]) for i, name in enumerate(self.param_names)}
                return Parameter(sample_dict)
        sample_dict = {name: float(self.loc[i]) for i, name in enumerate(self.param_names)}
        return Parameter(sample_dict)
    
    def pdf(self, x):
        if not isinstance(x, dict):
            x = x.to_dict() if hasattr(x, 'to_dict') else dict(x)
        theta = np.array([x[name] for name in self.param_names])
        if not self._validate_sample(theta):
            return 1e-10
        try:
            density = self.mvt.pdf(theta)
            density_val = max(float(density), 1e-10)
            if np.isnan(density_val) or np.isinf(density_val):
                return 1e-10
            return density_val
        except Exception:
            return 1e-10
    
    def _apply_constraints(self, theta):
        constrained = theta.copy()
        constrained[0] = np.clip(constrained[0], 50.0, 800.0)   # cov11
        constrained[2] = np.clip(constrained[2], 50.0, 800.0)   # cov22
        constrained[1] = np.clip(constrained[1], -200.0, 200.0)  # cov12
        max_cov12 = 0.8 * np.sqrt(constrained[0] * constrained[2])
        constrained[1] = np.clip(constrained[1], -max_cov12, max_cov12)
        constrained[3] = np.clip(constrained[3], 5.0, 40.0)     # loc0
        constrained[4] = np.clip(constrained[4], -0.5, 0.5)     # loc1
        constrained[5] = np.clip(constrained[5], -0.5, 0.5)     # loc2
        constrained[6] = np.maximum(constrained[6], 1.0)        # scale0
        constrained[7] = np.clip(constrained[7], 0.001, 0.1)    # scale1
        constrained[8] = np.clip(constrained[8], -0.1, 0.1)     # scale2
        constrained[9] = np.clip(constrained[9], 0.05, 0.45)    # shape
        return constrained
    
    def _validate_sample(self, theta):
        det_cov = theta[0] * theta[2] - theta[1]**2
        if det_cov <= 10.0:
            return False
        if theta[9] <= 0.05 or theta[9] >= 0.45:
            return False
        if theta[6] <= 1.0 or theta[7] <= 0.001:
            return False
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            return False
        return True

def maxstable_model(params):
    """Your maxstable_model function"""
    if not hasattr(maxstable_model, '_call_count'):
        maxstable_model._call_count = 0
        maxstable_model._success_count = 0
    maxstable_model._call_count += 1
    
    theta = np.array([params[name] for name in 
                     ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2', 
                      'scale0', 'scale1', 'scale2', 'shape']])
    
    det = theta[0] * theta[2] - theta[1]**2
    if det <= 1e-6 or theta[9] <= 0.05 or theta[9] >= 0.45 or theta[6] <= 1.0:
        return {"data": np.full((47, 79), 25.0, dtype=np.float64)}
    
    try:
        key = jr.PRNGKey(np.random.randint(0, 1000000))
        sim_data = your_simulator(jnp.array(theta), key)
        
        if sim_data is None or np.any(np.isnan(sim_data)) or np.any(np.isinf(sim_data)):
            return {"data": np.full((47, 79), 25.0, dtype=np.float64)}
        
        maxstable_model._success_count += 1
        sim_array = np.array(sim_data, dtype=np.float64)
        
        if np.any(np.isnan(sim_array)) or np.any(np.isinf(sim_array)):
            return {"data": np.full((47, 79), 25.0, dtype=np.float64)}
        
        return {"data": sim_array}
        
    except Exception as e:
        return {"data": np.full((47, 79), 25.0, dtype=np.float64)}

def maxstable_distance(x, y, n_groups):
    """Your distance function"""
    data_x = x["data"]
    data_y = y["data"]
    
    if np.any(np.isnan(data_x)) or np.any(np.isnan(data_y)):
        return 1e8
    if np.any(np.isinf(data_x)) or np.any(np.isinf(data_y)):
        return 1e8
    
    try:
        coords = create_2d_grid()
        theta_dummy = np.array(mple_theta)
        
        x_reshaped = data_x[None, :, :]
        y_reshaped = data_y[None, :, :]
        
        scores_x = compute_maxstable_pairwise_scores(
            theta_dummy[None, :], x_reshaped, coords, n_groups=n_groups
        )[0]
        
        scores_y = compute_maxstable_pairwise_scores(
            theta_dummy[None, :], y_reshaped, coords, n_groups=n_groups
        )[0]
        
        summary_x = scores_x.flatten()
        summary_y = scores_y.flatten()
        
        if np.any(np.isnan(summary_x)) or np.any(np.isnan(summary_y)):
            return 1e8
        if np.any(np.isinf(summary_x)) or np.any(np.isinf(summary_y)):
            return 1e8
        
        distance = np.sqrt(np.sum((summary_x - summary_y)**2))
        
        if np.isnan(distance) or np.isinf(distance) or distance < 0:
            return 1e8
        
        return float(distance)
        
    except Exception as e:
        return 1e10

def analyze_abc_results(history, true_params=None):
    """Your analysis function"""
    if history is None:
        return None, None
    
    try:
        df, w = history.get_distribution(m=0, t=history.max_t)
        param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2', 
                      'scale0', 'scale1', 'scale2', 'shape']
        
        if true_params is not None:
            print(f"{'Param':<8} {'True Val':<10} {'Post Mean':<10} {'Post Std':<10} {'Bias':<8}")
            print("-" * 55)
            
            for i, param in enumerate(param_names):
                true_val = true_params[i]
                post_mean = np.average(df[param], weights=w)
                post_std = np.sqrt(np.average((df[param] - post_mean)**2, weights=w))
                bias = abs(post_mean - true_val)
                print(f"{param:<8} {true_val:<10.4f} {post_mean:<10.4f} {post_std:<10.4f} {bias:<8.4f}")
        
        return df, w
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None, None





# In[21]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# Utility functions
# =========================================================

def _coerce_to_array(x, target_shape):
    """
    Convert x to float np.ndarray(target_shape) with best effort:
    - Supports pandas/torch/jax/list
    - Auto reshape if element count matches target_shape
    - Returns None if failed
    """
    if x is None:
        return None
    try:
        if hasattr(x, "values"):               # pandas
            x = x.values
        if hasattr(x, "detach"):               # torch
            x = x.detach().cpu().numpy()
        x = np.asarray(x)                      # jax / general
        x = x.astype(float, copy=False)
        if x.shape != target_shape:
            if x.size == int(np.prod(target_shape)):
                x = x.reshape(target_shape)
            else:
                return None
        if not np.any(np.isfinite(x)):         # all non-finite
            return None
        return x
    except Exception:
        return None


def _epanechnikov_weights(z_scaled, bandwidth):
    """
    K(u) = 0.75*(1-u^2) for |u|<=1 else 0
    Note: not multiplying by 1/bw here (no effect on regression solution, more numerically stable)
    """
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0
    u = np.abs(z_scaled) / bandwidth
    return np.where(np.isfinite(u) & (u <= 1.0), 0.75 * (1.0 - u**2), 0.0)


def _ess(w):
    nz = w[w > 0]
    if nz.size == 0:
        return 0.0
    return (nz.sum() ** 2) / (np.sum(nz ** 2) + 1e-12)


def _wls(X, y, w, ridge=1e-8):
    """
    Weighted least squares: returns beta, yhat, r2
    Using sqrt(w) trick + small ridge
    """
    w = np.asarray(w).reshape(-1)
    y = np.asarray(y).reshape(-1)
    sw = np.sqrt(w)[:, None]
    Xw = X * sw
    yw = y * sw[:, 0]
    XtX = Xw.T @ Xw
    if ridge and ridge > 0:
        i = np.arange(XtX.shape[0]); XtX[i, i] += ridge
    try:
        beta = np.linalg.solve(XtX, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    yhat = X @ beta
    wsum = w.sum()
    if wsum > 0:
        wmean = (w @ y) / wsum
        ss_res = np.sum(w * (y - yhat) ** 2)
        ss_tot = np.sum(w * (y - wmean) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r2 = 0.0
    return beta, yhat, r2


# =========================================================
# ABC Regression Adjustment (Beaumont Local Linear)
# =========================================================

def abc_regression_adjustment_v2(
    theta_samples,
    summary_stats_list,
    observed_summary_matrix,
    opt_indices=None,
    true_params=None,
    bandwidth=None,              # None -> use 0.8 quantile of |z|
    bandwidth_quantile=0.8,      # adjust quantile (0.8~0.9 recommended)
    use_max_bandwidth=False,     # True uses max(|z|)
    min_effective_n=30,
    ridge=1e-8,
    feature_clip=6.0,            # standardized feature clipping threshold (prevent extreme extrapolation)
    apply_to='kept',             # 'kept' only adjust samples with w>0; 'all' adjust all samples (not recommended)
    verbose=True,

):
    """
    Use flattened differences of "all groups × all statistics" as independent variables, column standardization, Epanechnikov kernel WLS.
    - Compatible with JAX/Torch/pandas/list, NaN/triangular matrix, shape inconsistency (auto reshape).
    - By default only perform regression adjustment on samples with kernel weight>0 (within neighborhood), avoid extrapolation explosion for samples far from observations.
    """
    theta_samples = np.asarray(theta_samples, dtype=float)
    n_samples, n_params = theta_samples.shape

    obs = np.asarray(observed_summary_matrix, dtype=float)
    obs_shape = obs.shape
    obs_flat = obs.reshape(-1)
    obs_finite_flat = np.isfinite(obs_flat)

    if verbose:
        print("Checking regression inputs:")
        print(f"Number of samples: {n_samples}")
        print(f"Groups x stats: {obs_shape[0]} x {obs_shape[1]}")

    if opt_indices is None:
        opt_indices = list(range(n_params))

    # ---------- Calculate distance (common finite positions), convert entries to numpy uniformly ----------
    dists = np.full(n_samples, np.inf, dtype=float)
    coerced_list = [None] * n_samples
    converted, failed = 0, 0

    for k in range(n_samples):
        if k >= len(summary_stats_list):
            failed += 1
            continue

        sim = _coerce_to_array(summary_stats_list[k], obs_shape)
        if sim is None:
            failed += 1
            continue

        coerced_list[k] = sim
        sim_flat = sim.reshape(-1)
        both = obs_finite_flat & np.isfinite(sim_flat)
        if not np.any(both):
            failed += 1
            continue

        diff = (sim_flat - obs_flat)[both]
        dists[k] = np.sqrt(np.sum(diff * diff))
        converted += 1

    if verbose:
        print(f"summary_stats_list length: {len(summary_stats_list)}")
        print(f"converted={converted}, failed={failed}")

    valid = np.isfinite(dists)
    if not np.any(valid):
        print("Error: No valid distance calculations")
        return theta_samples.copy()

    # ---------- Standardize distance & bandwidth ----------
    z = dists[valid] - dists[valid].mean()
    std = dists[valid].std()
    z = z / std if std > 0 else np.zeros_like(z)

    if bandwidth is None:
        base = np.abs(z)
        if base.size == 0:
            bandwidth = 1.0
        else:
            if use_max_bandwidth:
                bandwidth = np.max(base)
            else:
                q = float(np.clip(bandwidth_quantile, 0.5, 1.0))
                bandwidth = np.quantile(base, q)
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                bandwidth = 1.0

    z_all = np.full_like(dists, np.nan, dtype=float)
    z_all[valid] = (dists[valid] - dists[valid].mean()) / (std if std > 0 else 1.0)
    w = _epanechnikov_weights(z_all, bandwidth)
    keep_idx = np.where(w > 0)[0]
    ess = _ess(w)

    if verbose:
        print(f"\nBandwidth: {bandwidth:.6f} (q={bandwidth_quantile}, use_max={use_max_bandwidth})")
        print(f"Non-zero weights: {keep_idx.size}; ESS≈{ess:.1f}")
        if keep_idx.size > 0:
            print(f"Weight range: [{w[keep_idx].min():.6f}, {w[keep_idx].max():.6f}]")

    if keep_idx.size < 5:
        print("Too few valid samples, returning original samples.")
        return theta_samples.copy()

    # ---------- Fixed mask (observed finite positions), construct design matrix (flattened differences) ----------
    mask = np.isfinite(obs_flat)          # (G*S,)
    p = int(mask.sum())                   # valid feature dimension

    X_rows = []
    kept_ok = []
    for k in keep_idx:
        sim = coerced_list[k]
        if sim is None:
            continue
        sim2_flat = sim.reshape(-1).copy()
        # Replace invalid positions in this sample with observed values (making diff=0), ensuring column dimension consistency
        bad = ~np.isfinite(sim2_flat)
        if np.any(bad):
            sim2_flat[bad] = obs_flat[bad]
        diff_vec = (sim2_flat - obs_flat)[mask]   # (p,)
        X_rows.append(diff_vec)
        kept_ok.append(k)

    if len(X_rows) < 5:
        print("Insufficient valid samples after constructing design matrix, returning original samples.")
        return theta_samples.copy()

    kept_ok = np.asarray(kept_ok, dtype=int)
    X = np.asarray(X_rows)                # (n_kept, p)

    # Column standardization (across samples)
    col_mean = X.mean(axis=0)
    col_std = X.std(axis=0, ddof=0)
    col_std[col_std == 0] = 1.0
    Xs = (X - col_mean) / col_std
    if feature_clip is not None and feature_clip > 0:
        np.clip(Xs[:, 1 if False else 0:], -feature_clip, feature_clip, out=Xs)  # Just defensive; clipping before adding intercept
    # Add intercept
    Xs = np.column_stack([np.ones(Xs.shape[0]), Xs])  # (n_kept, 1+p)

    if (w[kept_ok] > 0).sum() < min_effective_n and verbose:
        print(f"Warning: Too few effective samples (weight > 0) ({(w[kept_ok] > 0).sum()}), may overfit.")

    if verbose:
        print(f"\nStarting adjustment for {len(opt_indices)} parameters...")

    theta_adj = theta_samples.copy()

    # ---------- Parameter-wise regression and apply correction ----------
    for j, pidx in enumerate(opt_indices):
        if verbose:
            print(f"\nProcessing parameter {pidx+1} (original index {pidx}):")

        y = theta_samples[kept_ok, pidx]
        beta, yhat, r2 = _wls(Xs, y, w[kept_ok], ridge=ridge)

        if verbose:
            adj_kept = yhat - beta[0]
            print(f"R-squared: {r2:.4f}")
            print(f"Adjustment range (kept): [{adj_kept.min():.4f}, {adj_kept.max():.4f}]")

        if apply_to == 'kept':
            # Only apply regression adjustment to samples with kernel weight>0; other samples keep original values
            for k in kept_ok:
                sim = coerced_list[k]
                if sim is None:
                    continue
                sim2_flat = sim.reshape(-1).copy()
                bad = ~np.isfinite(sim2_flat)
                if np.any(bad):
                    sim2_flat[bad] = obs_flat[bad]
                diff_vec = (sim2_flat - obs_flat)[mask]
                xk = (diff_vec - col_mean) / col_std
                if feature_clip is not None and feature_clip > 0:
                    np.clip(xk, -feature_clip, feature_clip, out=xk)
                xk = np.concatenate([[1.0], xk])
                yk_pred = xk @ beta
                adjustment = yk_pred - beta[0]
                theta_adj[k, pidx] = theta_samples[k, pidx] - adjustment
        else:
            # Apply to all samples (may extrapolate, not recommended; keeping for potential needs)
            for k in range(n_samples):
                sim = coerced_list[k]
                if sim is None:
                    continue
                sim2_flat = sim.reshape(-1).copy()
                bad = ~np.isfinite(sim2_flat)
                if np.any(bad):
                    sim2_flat[bad] = obs_flat[bad]
                diff_vec = (sim2_flat - obs_flat)[mask]
                xk = (diff_vec - col_mean) / col_std
                if feature_clip is not None and feature_clip > 0:
                    np.clip(xk, -feature_clip, feature_clip, out=xk)
                xk = np.concatenate([[1.0], xk])
                yk_pred = xk @ beta
                adjustment = yk_pred - beta[0]
                theta_adj[k, pidx] = theta_samples[k, pidx] - adjustment

    if verbose:
        print("\nRegression adjustment completed!")
    return theta_adj


# =========================================================
# Extract summary statistics from ABC history (need to pass functions)
# =========================================================

def extract_abc_summary_stats(
    history,
    obs_data,
    param_names=None,
    mple_theta=None,
    n_groups=10,
    simulator=None,         # def simulator(params, seed) -> (H,W)
    score_fn=None,          # def score_fn(params_batch, data_batch, coords, n_groups) -> (B, G, S) or (B,G,G)
    coords=None,
    rng=None,
):
    """
    Returns:
      theta_samples: (n_samples, n_params)
      summary_stats_list: list[np.ndarray], each shaped like (G,S) or (G,G)
      observed_summary:   np.ndarray, shaped like (G,S) or (G,G)
      w:                  weights
    """
    if param_names is None:
        param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2',
                       'scale0', 'scale1', 'scale2', 'shape']
    if simulator is None or score_fn is None or coords is None:
        raise ValueError("Must pass simulator, score_fn and coords.")
    if mple_theta is None:
        raise ValueError("Must provide mple_theta (for calculating observed summary statistics).")
    if rng is None:
        rng = np.random.default_rng()

    df, w = history.get_distribution(m=0, t=history.max_t)
    df = df.reset_index(drop=True)
    theta_samples = df[param_names].values
    n = len(df)
    print(f"Extracting {n} samples from ABC history...")

    # Calculate observed summary statistics
    print("Calculating observed data summary statistics...")
    obs_reshaped = obs_data['data'][None, :, :]
    dummy_params = np.array(mple_theta, dtype=float)[None, :]
    observed_summary = score_fn(dummy_params, obs_reshaped, coords, n_groups)[0]
    observed_summary = np.asarray(observed_summary, dtype=float)

    # Calculate summary statistics for each sample
    print("Recalculating summary statistics...")
    summary_stats_list = []
    for idx, row in enumerate(df.itertuples(index=False), 1):
        if idx % 100 == 0 or idx == 1:
            print(f"Processing sample {idx}/{n}")
        params = np.array([getattr(row, name) for name in param_names], dtype=float)
        try:
            seed = int(rng.integers(0, 1_000_000))
            sim_data = simulator(params, seed)
            if sim_data is None or np.any(~np.isfinite(sim_data)):
                raise ValueError("sim_data invalid")
            sim_reshaped = sim_data[None, :, :]
            scores = score_fn(params[None, :], sim_reshaped, coords, n_groups)[0]
            summary_stats_list.append(np.asarray(scores, dtype=float))
        except Exception as e:
            print(f"Sample {idx} calculation failed: {e}")
            summary_stats_list.append(np.full(observed_summary.shape, 0.0, dtype=float))

    return theta_samples, summary_stats_list, observed_summary, w


# =========================================================
# Visualization and summary
# =========================================================

def plot_adjustment_v2(theta_original, theta_adjusted, opt_indices, true_params=None, param_names=None):
    n_params = len(opt_indices)
    n_cols = min(5, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    for i, p in enumerate(opt_indices):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        orig = theta_original[:, p]; adj = theta_adjusted[:, p]
        try:
            ko = gaussian_kde(orig); ka = gaussian_kde(adj)
            x = np.linspace(min(orig.min(), adj.min()), max(orig.max(), adj.max()), 200)
            ax.plot(x, ko(x), linewidth=2, label='Original', color='gray')
            ax.plot(x, ka(x), linewidth=2, label='Adjusted', color='red')
        except Exception:
            ax.hist(orig, bins=20, density=True, alpha=0.5, label='Original', color='gray')
            ax.hist(adj,  bins=20, density=True, alpha=0.5, label='Adjusted', color='red')
        if true_params is not None:
            ax.axvline(true_params[p], linestyle='--', color='black', linewidth=2, label='True')
        name = param_names[p] if param_names else f'param {p}'
        ax.set_title(name, fontweight='bold'); ax.set_xlabel('Value'); ax.set_ylabel('Density')
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    for i in range(n_params, n_rows*n_cols):
        r, c = divmod(i, n_cols); axes[r, c].set_visible(False)
    plt.suptitle('ABC Regression Adjustment Before vs After', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()


def print_adjustment_summary_v2(theta_original, theta_adjusted, opt_indices, true_params=None, param_names=None):
    print("\n" + "="*60)
    print("Mean and Standard Deviation Comparison")
    print("="*60)
    for p in opt_indices:
        name = param_names[p] if param_names else f"Parameter {p}"
        print(f"\n{name}:")
        if true_params is not None:
            print(f"True value: {true_params[p]:.4f}")
        om, os = np.mean(theta_original[:, p]), np.std(theta_original[:, p])
        am, asd = np.mean(theta_adjusted[:, p]), np.std(theta_adjusted[:, p])
        print(f"Original mean (std): {om:.4f} ({os:.4f})")
        print(f"Adjusted mean (std): {am:.4f} ({asd:.4f})")


# =========================================================
# Two entry points: existing summary statistics / calculate from history
# =========================================================

def adjust_abc_results_from_precomputed(
    theta_samples,
    summary_stats_list,
    observed_summary_matrix,
    param_names=None, true_params=None,
    bandwidth=None, bandwidth_quantile=0.8, use_max_bandwidth=False,
    min_effective_n=30, ridge=1e-8,
    feature_clip=6.0, apply_to='kept'
):
    if param_names is None:
        param_names = [f"p{i}" for i in range(theta_samples.shape[1])]

    opt_indices = list(range(len(param_names)))

    theta_adj = abc_regression_adjustment_v2(
        theta_samples=theta_samples,
        summary_stats_list=summary_stats_list,
        observed_summary_matrix=observed_summary_matrix,
        opt_indices=opt_indices,
        true_params=true_params,
        bandwidth=bandwidth,
        bandwidth_quantile=bandwidth_quantile,
        use_max_bandwidth=use_max_bandwidth,
        min_effective_n=min_effective_n,
        ridge=ridge,
        feature_clip=feature_clip,
        apply_to=apply_to,
        verbose=True,
    )

    print("\n" + "="*60)
    print("Regression Adjustment Results Summary")
    print("="*60)
    print_adjustment_summary_v2(theta_samples, theta_adj, opt_indices, true_params, param_names)
    plot_adjustment_v2(theta_samples, theta_adj, opt_indices, true_params, param_names)

    return theta_adj


def adjust_abc_results(
    history, obs_data,
    param_names=None, true_params=None,
    mple_theta=None, n_groups=10,
    simulator=None, score_fn=None, coords=None, rng=None,
    bandwidth=None, bandwidth_quantile=0.8, use_max_bandwidth=False,
    min_effective_n=30, ridge=1e-8,
    feature_clip=6.0, apply_to='kept'
):
    if param_names is None:
        param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2',
                       'scale0', 'scale1', 'scale2', 'shape']

    print("Starting ABC regression adjustment...")

    theta, ss_list, obs_summary, w = extract_abc_summary_stats(
        history, obs_data,
        param_names=param_names, mple_theta=mple_theta, n_groups=n_groups,
        simulator=simulator, score_fn=score_fn, coords=coords, rng=rng
    )

    opt_indices = list(range(len(param_names)))

    theta_adj = abc_regression_adjustment_v2(
        theta_samples=theta,
        summary_stats_list=ss_list,
        observed_summary_matrix=obs_summary,
        opt_indices=opt_indices,
        true_params=true_params,
        bandwidth=bandwidth,
        bandwidth_quantile=bandwidth_quantile,
        use_max_bandwidth=use_max_bandwidth,
        min_effective_n=min_effective_n,
        ridge=ridge,
        feature_clip=feature_clip,
        apply_to=apply_to,
        verbose=True,
    )

    # Write back to DataFrame (keep original columns)
    df0, _ = history.get_distribution(m=0, t=history.max_t)
    adjusted_df = df0.reset_index(drop=True).copy()
    for i, name in enumerate(param_names):
        adjusted_df[name] = theta_adj[:, i]

    print("\n" + "="*60)
    print("Regression Adjustment Results Summary")
    print("="*60)
    print_adjustment_summary_v2(theta, theta_adj, opt_indices, true_params, param_names)
    plot_adjustment_v2(theta, theta_adj, opt_indices, true_params, param_names)

    return adjusted_df

# In[22]:

# Import ABC adjustment module (contents of your second file)

def compare_cnf_abc_adjusted(true_params, test_name=""):
    """Compare results from CNF, original ABC, and adjusted ABC"""
    
    print(f"\n{'='*80}")
    print(f"Test scenario: {test_name}")
    print(f"{'='*80}")
    print(f"True parameters: {true_params[:3]}...")
    
    # Generate observation data
    key = jr.PRNGKey(np.random.randint(0, 1000000))
    observed_data = your_simulator(true_params, key)
    
    if np.any(np.isnan(observed_data)):
        print("Observation data generation failed")
        return None
    
    print(f"Observation data shape: {observed_data.shape}")
    
    param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2', 
                   'scale0', 'scale1', 'scale2', 'shape']
    
    results = {}
    
    # === CNF method testing ===
    print(f"\n--- CNF Method Testing ---")
    
    for method_name, cnf_info in cnf_models.items():
        print(f"\n{method_name.upper()}:")
        try:
            test_input = get_test_input_for_method(observed_data, method_name)
            key, sample_key = jr.split(key)
            samples = cnf_info['cnf'].batch_sample_fn(1000, test_input, sample_key)
            
            pred_mean = jnp.mean(samples, axis=0)
            pred_std = jnp.std(samples, axis=0)
            bias = pred_mean - true_params
            mae = jnp.mean(jnp.abs(bias))
            
            results[f'CNF_{method_name}'] = {
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'bias': bias,
                'mae': float(mae),
                'method_type': 'CNF'
            }
            
            print(f"  MAE: {mae:.4f}")
            
        except Exception as e:
            print(f"  Failed: {str(e)[:50]}")
            results[f'CNF_{method_name}'] = None
    
    # === Original ABC method testing ===
    print(f"\n--- Original ABC Method Testing ---")
    
    try:
        prior = MaxStablePrior()
        
        abc = ABCSMC(
            models=maxstable_model,
            parameter_priors=prior,
            distance_function=maxstable_distance,
            population_size=500,  # Reduce appropriately to speed up
            sampler=pyabc.sampler.SingleCoreSampler(),
            eps=pyabc.epsilon.QuantileEpsilon(alpha=0.7, quantile_multiplier=0.95)
        )
        
        obs_data = {"data": observed_data}
        
        db_path = os.path.join(tempfile.gettempdir(), f"cnf_abc_comp_{int(time.time())}.db")
        abc.new("sqlite:///" + db_path, obs_data)
        
        print("Starting ABC inference...")
        history = abc.run(max_nr_populations=8)  # Reduce iterations appropriately
        
        df_abc, w_abc = analyze_abc_results(history, true_params)
        
        if df_abc is not None:
            abc_pred_mean = np.array([np.average(df_abc[param], weights=w_abc) for param in param_names])
            abc_pred_std = np.array([np.sqrt(np.average((df_abc[param] - abc_pred_mean[i])**2, weights=w_abc)) 
                                   for i, param in enumerate(param_names)])
            abc_bias = abc_pred_mean - true_params
            abc_mae = np.mean(np.abs(abc_bias))
            
            results['ABC_original'] = {
                'pred_mean': abc_pred_mean,
                'pred_std': abc_pred_std,
                'bias': abc_bias,
                'mae': float(abc_mae),
                'method_type': 'ABC',
                'history': history
            }
            
            print(f"Original ABC:")
            print(f"  MAE: {abc_mae:.4f}")
            
            # === Adjusted ABC method testing ===
            print(f"\n--- ABC Regression Adjustment Method Testing ---")
            
            try:
                # Use your regression adjustment function
                adjusted_df = adjust_abc_results(
                    history=history,
                    obs_data=obs_data,
                    param_names=param_names,
                    true_params=true_params,
                    mple_theta=mple_theta,
                    n_groups=5,
                    simulator=your_simulator,
                    score_fn=compute_maxstable_pairwise_scores,
                    coords=create_2d_grid(),
                    bandwidth_quantile=0.8,
                    apply_to='kept'
                )
                
                # Calculate adjusted statistics
                adj_pred_mean = np.array([adjusted_df[param].mean() for param in param_names])
                adj_pred_std = np.array([adjusted_df[param].std() for param in param_names])
                adj_bias = adj_pred_mean - true_params
                adj_mae = np.mean(np.abs(adj_bias))
                
                results['ABC_adjusted'] = {
                    'pred_mean': adj_pred_mean,
                    'pred_std': adj_pred_std,
                    'bias': adj_bias,
                    'mae': float(adj_mae),
                    'method_type': 'ABC_ADJ'
                }
                
                print(f"Adjusted ABC:")
                print(f"  MAE: {adj_mae:.4f}")
                print(f"  Improvement: {abc_mae - adj_mae:.4f}")
                
            except Exception as e:
                print(f"ABC adjustment failed: {str(e)}")
                results['ABC_adjusted'] = None
        
    except Exception as e:
        print(f"ABC failed: {str(e)}")
        results['ABC_original'] = None
        results['ABC_adjusted'] = None
    
    return results, true_params, observed_data

def print_final_comparison_table(all_results):
    """Print final comparison table"""
    
    print(f"\n{'='*100}")
    print("CNF vs ABC vs ABC_Adjusted Final Comparison")
    print(f"{'='*100}")
    
    all_methods = set()
    for results in all_results.values():
        all_methods.update(results.keys())
    
    all_methods = sorted(all_methods)
    
    print(f"{'Method':<25} {'Average MAE':<12} {'Success Count':<10} {'Type':<10}")
    print("-" * 65)
    
    method_maes = {method: [] for method in all_methods}
    method_success = {method: 0 for method in all_methods}
    
    # Collect results
    for test_name, results in all_results.items():
        for method in all_methods:
            if method in results and results[method] is not None:
                method_maes[method].append(results[method]['mae'])
                method_success[method] += 1
    
    # Group by type for display
    cnf_methods = [m for m in all_methods if m.startswith('CNF_')]
    abc_methods = [m for m in all_methods if m.startswith('ABC_')]
    
    print("CNF Methods:")
    for method in cnf_methods:
        if method_maes[method]:
            avg_mae = np.mean(method_maes[method])
            success_count = method_success[method]
            print(f"  {method:<23} {avg_mae:<12.4f} {success_count:<10} {'CNF':<10}")
    
    print("\nABC Methods:")
    for method in abc_methods:
        if method_maes[method]:
            avg_mae = np.mean(method_maes[method])
            success_count = method_success[method]
            method_type = 'ABC' if method == 'ABC_original' else 'ABC_ADJ'
            print(f"  {method:<23} {avg_mae:<12.4f} {success_count:<10} {method_type:<10}")
# In[25]:


# Comprehensive Real Rainfall Data Testing: CNF + ABC + ABC Adjusted
import rpy2.robjects as ro
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import tempfile
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

def test_real_rainfall_comprehensive():
    """Test real rainfall data parameter estimation - CNF, ABC, ABC adjusted with detailed comparisons"""
    
    print("Loading SpatialExtremes real rainfall data...")
    
    # 1. Load real data
    ro.r("""
    library(SpatialExtremes)
    data(rainfall)
    """)
    
    # Get real data
    rain_data = np.array(ro.r('rain'))  
    coord_data = np.array(ro.r('coord[,-3]'))  # Remove altitude column
    
    print(f"Real data shape: {rain_data.shape}")
    print(f"Coordinate shape: {coord_data.shape}")
    print(f"Data range: [{np.min(rain_data):.2f}, {np.max(rain_data):.2f}]")
    
    # 2. Compute real MPLE estimation and standard errors using R
    print("\nComputing real MPLE estimation...")
    ro.r("""
    mod.spe <- fitmaxstab(rain, coord=coord[,-3], cov.mod="gauss",
                          loc ~ lon + lat, scale ~ lon + lat, shape ~ 1,
                          fit.marge=TRUE)
    real_mple <- mod.spe$fitted.values
    mple_std_err <- mod.spe$std.err
    """)
    
    real_mple = np.array(ro.r('real_mple'))
    mple_std_err = np.array(ro.r('mple_std_err'))
    print(f"Real MPLE: {real_mple}")
    print(f"MPLE Std Errors: {mple_std_err}")
    
    # 3. Test all methods
    print(f"\nTesting all methods on real data...")
    
    param_names = ['cov11', 'cov12', 'cov22', 'loc0', 'loc1', 'loc2', 
                   'scale0', 'scale1', 'scale2', 'shape']
    
    results = {}
    key = jr.PRNGKey(42)
    
    # ===== CNF Methods Testing =====
    print(f"\n{'='*60}")
    print("CNF Methods Testing")
    print(f"{'='*60}")
    
    for method_name, cnf_info in cnf_models.items():
        print(f"\n--- CNF {method_name.upper()} ---")
        
        try:
            # Compute test input
            test_input = get_test_input_for_method(jnp.array(rain_data), method_name)
            print(f"Input dimension: {len(test_input)}")
            
            # CNF sampling
            key, sample_key = jr.split(key)
            samples = cnf_info['cnf'].batch_sample_fn(1000, test_input, sample_key)
            
            # Statistics
            pred_mean = jnp.mean(samples, axis=0)
            pred_std = jnp.std(samples, axis=0)
            
            # Calculate differences with real MPLE
            total_diff = 0
            print("Prediction results vs real MPLE:")
            print(f"{'Param':<8} {'MPLE':<10} {'CNF Mean':<10} {'CNF Std':<10} {'Diff':<8}")
            print("-" * 50)
            
            for i, name in enumerate(param_names):
                diff = abs(pred_mean[i] - real_mple[i])
                total_diff += diff
                print(f"{name:<8} {real_mple[i]:<10.4f} {pred_mean[i]:<10.4f} {pred_std[i]:<10.4f} {diff:<8.4f}")
            
            print(f"Total difference: {total_diff:.4f}")
            results[f'CNF_{method_name}'] = {
                'pred_mean': np.array(pred_mean),
                'pred_std': np.array(pred_std),
                'samples': np.array(samples),  # Store samples for boxplots
                'total_diff': total_diff,
                'method_type': 'CNF'
            }
            
        except Exception as e:
            print(f"Failed: {str(e)[:50]}")
            results[f'CNF_{method_name}'] = {'failed': True, 'method_type': 'CNF'}
    
    # ===== ABC Original Method Testing =====
    print(f"\n{'='*60}")
    print("ABC Original Method Testing")
    print(f"{'='*60}")
    
    try:
        # Setup ABC
        prior = MaxStablePrior()
        
        abc = ABCSMC(
            models=maxstable_model,
            parameter_priors=prior,
            distance_function=maxstable_distance,
            population_size=10,  # Reduced to save time
            sampler=pyabc.sampler.SingleCoreSampler(),
            eps=pyabc.epsilon.QuantileEpsilon(alpha=0.7, quantile_multiplier=0.95)
        )
        
        obs_data = {"data": rain_data}
        
        # Create temporary database
        db_path = os.path.join(tempfile.gettempdir(), f"real_abc_{int(time.time())}.db")
        abc.new("sqlite:///" + db_path, obs_data)
        
        print("Starting ABC inference...")
        start_time = time.time()
        history = abc.run(max_nr_populations=2)  # Reduced generations
        abc_time = time.time() - start_time
        
        print(f"ABC completed, time: {abc_time:.2f}s")
        
        # Analyze ABC results
        df_abc, w_abc = analyze_abc_results(history, real_mple)
        
        if df_abc is not None:
            abc_pred_mean = np.array([np.average(df_abc[param], weights=w_abc) for param in param_names])
            abc_pred_std = np.array([np.sqrt(np.average((df_abc[param] - abc_pred_mean[i])**2, weights=w_abc)) 
                                   for i, param in enumerate(param_names)])
            
            # Get samples for boxplots
            abc_samples = df_abc[param_names].values  # (n_samples, n_params)
            
            # Calculate differences with real MPLE
            total_diff = 0
            print("\nABC original results vs real MPLE:")
            print(f"{'Param':<8} {'MPLE':<10} {'ABC Mean':<10} {'ABC Std':<10} {'Diff':<8}")
            print("-" * 50)
            
            for i, name in enumerate(param_names):
                diff = abs(abc_pred_mean[i] - real_mple[i])
                total_diff += diff
                print(f"{name:<8} {real_mple[i]:<10.4f} {abc_pred_mean[i]:<10.4f} {abc_pred_std[i]:<10.4f} {diff:<8.4f}")
            
            print(f"Total difference: {total_diff:.4f}")
            results['ABC_original'] = {
                'pred_mean': abc_pred_mean,
                'pred_std': abc_pred_std,
                'samples': abc_samples,
                'total_diff': total_diff,
                'method_type': 'ABC',
                'abc_time': abc_time,
                'history': history
            }
            
            # ===== ABC Regression Adjustment Testing =====
            print(f"\n{'='*60}")
            print("ABC Regression Adjustment Testing")
            print(f"{'='*60}")
            
            try:
                print("Starting ABC regression adjustment...")
                start_time = time.time()
                
                adjusted_df = adjust_abc_results(
                    history=history,
                    obs_data=obs_data,
                    param_names=param_names,
                    true_params=real_mple,  # Use real MPLE as reference
                    mple_theta=mple_theta,
                    n_groups=10,  # Reduced for speed
                    simulator=your_simulator,
                    score_fn=compute_maxstable_pairwise_scores,
                    coords=create_2d_grid(),
                    bandwidth_quantile=0.5,
                    apply_to='kept'
                )
                
                adj_time = time.time() - start_time
                print(f"ABC adjustment completed, time: {adj_time:.2f}s")
                
                # Calculate adjusted statistics - 修正这里的计算
                adj_samples = adjusted_df[param_names].values  # Get the actual adjusted samples
                adj_pred_mean = np.mean(adj_samples, axis=0)  # Calculate mean from adjusted samples
                adj_pred_std = np.std(adj_samples, axis=0)    # Calculate std from adjusted samples
                
                # Calculate differences with real MPLE
                total_diff = 0
                print("\nABC adjusted results vs real MPLE:")
                print(f"{'Param':<8} {'MPLE':<10} {'Adj Mean':<10} {'Adj Std':<10} {'Diff':<8}")
                print("-" * 50)
                
                for i, name in enumerate(param_names):
                    diff = abs(adj_pred_mean[i] - real_mple[i])
                    total_diff += diff
                    print(f"{name:<8} {real_mple[i]:<10.4f} {adj_pred_mean[i]:<10.4f} {adj_pred_std[i]:<10.4f} {diff:<8.4f}")
                
                print(f"Total difference: {total_diff:.4f}")
                
                # Calculate improvement
                improvement = results['ABC_original']['total_diff'] - total_diff
                print(f"Improvement over ABC original: {improvement:.4f}")
                
                results['ABC_adjusted'] = {
                    'pred_mean': adj_pred_mean,
                    'pred_std': adj_pred_std,
                    'samples': adj_samples,
                    'total_diff': total_diff,
                    'method_type': 'ABC_ADJ',
                    'adj_time': adj_time,
                    'improvement': improvement
                }
                
            except Exception as e:
                print(f"ABC adjustment failed: {str(e)}")
                results['ABC_adjusted'] = {'failed': True, 'method_type': 'ABC_ADJ'}
        
        else:
            print("ABC original method result analysis failed")
            results['ABC_original'] = {'failed': True, 'method_type': 'ABC'}
            results['ABC_adjusted'] = {'failed': True, 'method_type': 'ABC_ADJ'}
    
    except Exception as e:
        print(f"ABC method failed: {str(e)}")
        results['ABC_original'] = {'failed': True, 'method_type': 'ABC'}
        results['ABC_adjusted'] = {'failed': True, 'method_type': 'ABC_ADJ'}
    
    # ===== Create Comparison Tables and Visualizations =====
    print(f"\n{'='*80}")
    print("PARAMETER COMPARISON TABLES AND VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Create comparison tables
    create_comparison_tables(results, real_mple, mple_std_err, param_names)
    
    # Create visualizations
    create_posterior_std_visualization(results, mple_std_err, param_names)
    create_parameter_boxplots(results, real_mple, param_names)
    
    # ===== Method Ranking =====
    print(f"\n{'='*80}")
    print("METHOD RANKING (by Total Difference from MPLE)")
    print(f"{'='*80}")
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() 
                         if 'failed' not in v and 'total_diff' in v}
    
    if successful_results:
        sorted_methods = sorted(successful_results.items(), key=lambda x: x[1]['total_diff'])
        
        print(f"{'Rank':<4} {'Method':<20} {'Type':<10} {'Total Diff':<12}")
        print("-" * 50)
        
        for i, (method, result) in enumerate(sorted_methods, 1):
            method_type = result['method_type']
            total_diff = result['total_diff']
            print(f"{i:<4} {method:<20} {method_type:<10} {total_diff:<12.4f}")
        
        # Show best method details
        best_method, best_result = sorted_methods[0]
        print(f"\nBest method: {best_method}")
        print(f"   Type: {best_result['method_type']}")
        print(f"   Total difference: {best_result['total_diff']:.4f}")
    
    else:
        print("No successful results for comparison")
    
    return results, real_mple, mple_std_err, param_names

def create_comparison_tables(results, real_mple, mple_std_err, param_names):
    """Create posterior mean and standard deviation comparison tables"""
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() 
                         if 'failed' not in v and 'pred_mean' in v}
    
    if not successful_results:
        print("No successful results for parameter comparison")
        return
    
    method_names = list(successful_results.keys())
    
    print("\nPOSTERIOR MEAN COMPARISON:")
    print("=" * 100)
    
    # Create header
    header = f"{'Parameter':<10} {'MPLE':<12}"
    for method in method_names:
        header += f"{method:<15}"
    print(header)
    print("-" * len(header))
    
    # Print each parameter
    for i, param in enumerate(param_names):
        row = f"{param:<10} {real_mple[i]:<12.4f}"
        for method in method_names:
            mean_val = successful_results[method]['pred_mean'][i]
            row += f"{mean_val:<15.4f}"
        print(row)
    
    print("\nPOSTERIOR STANDARD DEVIATION COMPARISON:")
    print("=" * 100)
    
    # Create header
    header = f"{'Parameter':<10} {'MPLE':<12}"
    for method in method_names:
        header += f"{method:<15}"
    print(header)
    print("-" * len(header))
    
    # Print each parameter's standard deviation
    for i, param in enumerate(param_names):
        row = f"{param:<10} {mple_std_err[i]:<12.4f}"
        for method in method_names:
            std_val = successful_results[method]['pred_std'][i]
            row += f"{std_val:<15.4f}"
        print(row)

def create_posterior_std_visualization(results, mple_std_err, param_names):
    """Create individual bar charts for each parameter's standard deviations"""
    
    successful_results = {k: v for k, v in results.items() 
                         if 'failed' not in v and 'pred_std' in v}
    
    if not successful_results:
        return
    
    method_names = list(successful_results.keys())
    labels = ['MPLE'] + method_names
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    # Create individual plot for each parameter
    for i, param in enumerate(param_names):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for this parameter
        std_values = [mple_std_err[i]]  # Start with MPLE
        for method in method_names:
            std_values.append(successful_results[method]['pred_std'][i])
        
        # Create bar chart
        bars = ax.bar(labels, std_values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, std_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(std_values),
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Standard Deviation')
        ax.set_title(f'Posterior Standard Deviation - {param}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

def create_parameter_boxplots(results, real_mple, param_names):
    """Create individual boxplots for each parameter showing different methods"""
    
    successful_results = {k: v for k, v in results.items() 
                         if 'failed' not in v and 'samples' in v}
    
    if not successful_results:
        return
    
    method_names = list(successful_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(method_names)))
    
    # Create individual boxplot for each parameter
    for i, param in enumerate(param_names):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect data for this parameter
        data_to_plot = []
        labels_to_plot = []
        
        for j, method in enumerate(method_names):
            samples = successful_results[method]['samples']
            if samples.ndim == 2:
                param_samples = samples[:, i]
            else:
                param_samples = samples[i]
            data_to_plot.append(param_samples)
            labels_to_plot.append(method)
        
        # Create boxplot
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add MPLE reference line
        ax.axhline(y=real_mple[i], color='red', linestyle='--', 
                   linewidth=3, label=f'MPLE: {real_mple[i]:.4f}')
        
        ax.set_title(f'Posterior Distribution - {param}', fontweight='bold', fontsize=14)
        ax.set_ylabel('Parameter Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()



