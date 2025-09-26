import ml_collections
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from nn_model import NCMLP

def create_range_parameter_config():
    config = ml_collections.ConfigDict()

    config.algorithm = algorithm = ml_collections.ConfigDict()
    algorithm.dim_parameters = 10
    algorithm.dim_data = 64
    algorithm.num_rounds = 1
    

    config.sde = sde = ml_collections.ConfigDict()
    sde.name = "vpsde"
    sde.T = 1.0
    sde.beta_min = 0.01
    sde.beta_max = 10.0
    sde.sigma_min = 0.01
    sde.sigma_max = 3
    
    config.score_network = score_network = ml_collections.ConfigDict()
    score_network.width = 256
    score_network.depth = 3
    score_network.activation = jax.nn.silu
    score_network.t_embed_dim = 32
    score_network.theta_embed_dim = 16
    score_network.x_embed_dim = 16
    score_network.use_weighted_loss = True  
    score_network.t_sample_size = 10       
    

    config.optim = optim = ml_collections.ConfigDict()
    optim.max_iters = 1000
    optim.batch_size = 256
    optim.lr = 1e-4
    optim.eval_prop = 0.10
    optim.print_every = 200
    optim.max_patience = 250  
    
 
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.epsilon = 1e-4
    
    return config


config = create_range_parameter_config()



