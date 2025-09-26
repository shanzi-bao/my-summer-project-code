import jax.numpy as jnp
import jax.random as jr
# One-dimensional Gaussian random field - 4 uniform observation points
def your_prior_sampler(n, key=None):
    """Uniform prior for range parameter"""
    if key is None:
        key = jr.PRNGKey(42)
    
    # range parameter: uniform(0, 100.0)
    return jr.uniform(key, (n, 1), minval=0, maxval=100)

# Changed to: 2D grid arrangement  
def create_2d_grid():
    coords = jnp.linspace(0, 1, 8)  # Changed from 4 to 8: [0, 0.143, 0.286, ..., 1.0]
    X, Y = jnp.meshgrid(coords, coords)
    return jnp.stack([X.flatten(), Y.flatten()], axis=1)  # Changed from (16, 2) to (64, 2)

def your_simulator(theta, key=None):
    locations = create_2d_grid()  # Now (64, 2)
    phi = theta[0]  # Single range parameter
    
    # 2D Euclidean distance
    diff = locations[:, None, :] - locations[None, :, :]  # (64, 64, 2)  
    distances = jnp.sqrt(jnp.sum(diff**2, axis=2))       # (64, 64)
    
    # Exponential covariance: σ² exp(-||si-sj||/φ)
    cov_matrix = 5.0 * jnp.exp(-distances / phi)
    cov_matrix += 1e-6 * jnp.eye(64)  # Changed from eye(16) to eye(64)
    
    return jr.multivariate_normal(key, jnp.zeros(64), cov_matrix)  # Changed from zeros(16) to zeros(64)

# Generate training data
n_samples = 5000
key = jr.PRNGKey(42)

theta_train = your_prior_sampler(n_samples, key)
key, *sim_keys = jr.split(key, n_samples + 1)
x_train = jnp.array([your_simulator(theta_train[i], sim_keys[i]) for i in range(n_samples)])