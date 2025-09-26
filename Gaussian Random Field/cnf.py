# cnf.py - 基于你的代码适配的CNF推断模块
from nn_model import NCMLP
from sde import SDE
import math
import functools as ft
import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

class CNF(eqx.Module):
    score_network: NCMLP
    sde: SDE
    parameter_mean: jnp.ndarray
    parameter_std: jnp.ndarray
    data_mean: jnp.ndarray
    data_std: jnp.ndarray
    t1: float
    t0: float
    dt: float

    def __init__(
        self,
        *,
        score_network,
        sde,
        ds_means,
        ds_stds,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.score_network = score_network
        self.sde = sde
        self.t1 = sde.T
        self.t0 = 1e-5
        self.dt = 0.01

        # 分离参数和数据的标准化参数
        split_indices = [self.sde.dim_parameters, self.sde.dim_parameters + self.sde.dim_data]
        parameter_mean, data_mean, _ = jnp.split(ds_means, indices_or_sections=split_indices)
        parameter_std, data_std, _ = jnp.split(ds_stds, indices_or_sections=split_indices)
        
        self.parameter_mean = parameter_mean
        self.parameter_std = parameter_std
        self.data_mean = data_mean
        self.data_std = data_std

    @eqx.filter_jit
    def batch_sample_fn(self, sample_size, x, key):
        """批量后验采样"""
        # 标准化观测数据
        x = (x - self.data_mean) / self.data_std
        
        # 为每个样本生成不同的随机种子
        sample_keys = jr.split(key, sample_size)
        
        # 创建单样本函数的偏函数
        sample_fn = ft.partial(self.single_sample_fn, self.score_network, self.sde, x)
        
        # 批量采样
        samples = jax.vmap(sample_fn)(sample_keys)
        
        # 反标准化回原始尺度
        samples = self.parameter_mean + self.parameter_std * samples
        return samples

    @eqx.filter_jit
    def single_sample_fn(self, score_network, sde, x, key, epsilon=1e-5, dt=0.01):
        """单个后验样本采样 - 反向ODE"""
        key, base_dist_key = jr.split(key)

        # 定义漂移函数
        drift = ft.partial(sde.drift_ode, score_network, x)
        term = dfx.ODETerm(drift)

        # 从先验噪声开始
        init_theta = sde.base_dist(base_dist_key).reshape(-1,)

        # 求解反向ODE：从t=T积分到t=epsilon
        solver = dfx.Tsit5()
        sol = dfx.diffeqsolve(term, solver, sde.T, epsilon, -dt, init_theta)
        
        return sol.ys[0]

    @eqx.filter_jit
    def batch_logp_fn(self, theta, x, key):
        """批量对数概率计算"""
        # 标准化
        x = (x - self.data_mean) / self.data_std
        theta = (theta - self.parameter_mean) / self.parameter_std
        
        # 批量计算
        logp_keys = jr.split(key, theta.shape[0])
        logp_fn = ft.partial(self.single_logp_fn, self.score_network, self.sde, x)
        logps = jax.vmap(logp_fn)(theta, logp_keys)
        return logps

    @eqx.filter_jit
    def single_logp_fn(self, score_network, sde, x, theta, key):
        """单个样本的对数概率计算 - 前向ODE"""
        # 定义包含Jacobian的漂移函数
        term = ft.partial(sde.drift_dlogp_ode, score_network, x)
        term = dfx.ODETerm(term)

        # 初始状态：(theta, log_det_jacobian)
        delta_log_likelihood = 0.0
        theta_state = (theta, delta_log_likelihood)

        # 前向ODE：从t=0积分到t=T
        solver = dfx.Tsit5()
        sol = dfx.diffeqsolve(term, solver, self.t0, self.t1, self.dt, theta_state)
        
        # 提取最终状态
        (y,), (delta_log_likelihood,) = sol.ys
        
        # 计算对数概率：变量变换 + 基础分布概率
        return delta_log_likelihood + sde.base_dist_logp(y)