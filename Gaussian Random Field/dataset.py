import jax.numpy as jnp
import jax.random as jr
# 一维高斯随机场 - 4个均匀观测点
def your_prior_sampler(n, key=None):
    """range参数的均匀先验"""
    if key is None:
        key = jr.PRNGKey(42)
    
    # range参数: uniform(0, 100.0)
    return jr.uniform(key, (n, 1), minval=0, maxval=100)

# 改为：2D网格排列  
def create_2d_grid():
    coords = jnp.linspace(0, 1, 8)  # 🔥 从4改为8: [0, 0.143, 0.286, ..., 1.0]
    X, Y = jnp.meshgrid(coords, coords)
    return jnp.stack([X.flatten(), Y.flatten()], axis=1)  # 🔥 从(16, 2)改为(64, 2)

def your_simulator(theta, key=None):
    locations = create_2d_grid()  # 🔥 现在是(64, 2)
    phi = theta[0]  # 单个range参数
    
    # 2D欧几里得距离
    diff = locations[:, None, :] - locations[None, :, :]  # 🔥 (64, 64, 2)  
    distances = jnp.sqrt(jnp.sum(diff**2, axis=2))       # 🔥 (64, 64)
    
    # 指数协方差：σ² exp(-||si-sj||/φ)
    cov_matrix = 5.0 * jnp.exp(-distances / phi)
    cov_matrix += 1e-6 * jnp.eye(64)  # 🔥 从eye(16)改为eye(64)
    
    return jr.multivariate_normal(key, jnp.zeros(64), cov_matrix)  # 🔥 从zeros(16)改为zeros(64)

# 生成训练数据
n_samples = 5000
key = jr.PRNGKey(42)

theta_train = your_prior_sampler(n_samples, key)
key, *sim_keys = jr.split(key, n_samples + 1)
x_train = jnp.array([your_simulator(theta_train[i], sim_keys[i]) for i in range(n_samples)])


