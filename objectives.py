from abc import abstractmethod
from dataclasses import dataclass
from jaxtyping import Bool, Float
import gpjax as gpx
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import jax.tree_util as jtu
from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)
import cola
from gpjax.distributions import GaussianDistribution

def __hash__(self):
    # For some reason the hash function needs to be reimplemented.
    leaves = []
    for leaf in jtu.tree_leaves(self):
        try: # Arrays cant be hashed otherwise.
            leaves.append(leaf.tobytes())
        except AttributeError:
            leaves.append(leaf)
    return hash(tuple(leaves)) 
# patch gpx.Dataset
gpx.Dataset.__hash__ = __hash__

@dataclass
class BF_MLL(gpx.objectives.ConjugateMLL):
    # Set compute_bf if you want to recompute the basis functions each iteration -- this allows for optimization of hyperparameter dependent basis functions as well
    compute_bf: Bool = gpx.base.static_field(False)
    def step(self, model, train_data):
        alpha, B = lax.cond(self.compute_bf, lambda D: model.compute_dual(D), lambda D: (model.alpha, model.B), train_data)
        s2 = model.likelihood.obs_stddev**2
        Lambda = model.prior.kernel.Lambda
        Lambdainv = jnp.broadcast_to(jnp.diag(1/Lambda.diagonal()), B.shape)

        Z = s2 * Lambdainv + B
        Z += jnp.broadcast_to(jnp.identity(Z.shape[-1]), Z.shape) * model.prior.jitter
        # Z += cola.ops.I_like(Z) * model.prior.jitter
        # Z = cola.PSD(Z)
        # logdetZ = cola.linalg.logdet(Z, Cholesky())
        ZR, _ = jsp.linalg.cho_factor(Z, lower=True)
        logdetZ = 2 * jnp.log(ZR.diagonal(axis1=-2, axis2=-1)).sum()

        logdetQ = (train_data.n - model.M) * jnp.log(s2) + logdetZ + \
                    jnp.log(Lambda.diagonal()).sum()
        # aZa = alpha.T @ cola.solve(Z, alpha, Cholesky())
        Za = jsp.linalg.cho_solve((ZR, True), alpha.T).T
        # Basically alpha.T @ Za
        aZa = jnp.einsum('i...,i...->...', alpha, Za).sum()
        # Basically y.T @ y
        yTy = jnp.einsum('i...,i...->...', train_data.y, train_data.y).sum()
        # yTy = train_data.y.T @ train_data.y
        yTQy = 1/s2 * (yTy - aZa)
        
        const = train_data.n * jnp.log(2 * jnp.pi)
        return self.constant * (- 1/2 * (logdetQ + yTQy + const).squeeze())

@dataclass
class BF_ELBO(gpx.objectives.AbstractObjective):
    compute_bf: Bool = gpx.base.static_field(False)
    def step(self, model, train_data):
        # alpha, B = lax.cond(self.compute_bf, lambda D: model.compute_dual(D), lambda D: (model.alpha, model.B), train_data)
        # model = model.replace(alpha=alpha, B=B)
        model = lax.cond(self.compute_bf, lambda D: model.update_with_batch(D), lambda D: model, train_data)

        x, y = train_data.X, train_data.y

        m, S = model.mean_parameters

        # Posterior distribution on x given xc
        posterior = model(x)
        # V = posterior.covariance()
        # obs_noise = model.likelihood.obs_stddev**2
        logprob = model.likelihood.expected_log_likelihood(y, 
                                                           posterior.mean()[:,None], 
                                                           posterior.covariance().diagonal()[:,None]).sum()
        # V += cola.ops.I_like(V) * obs_noise
        # V = cola.PSD(V)
        # log2pi = jnp.log(2 * jnp.pi)
        # logsigma = jnp.log(obs_noise)
        # diff = y.squeeze() - posterior.mean()
        # quad = diff.T@diff
        # tr = posterior.covariance().diagonal().sum()

        # logprob = - 1/2 * (train_data.n * log2pi + train_data.n * logsigma + 
        #                    1/obs_noise * (quad + tr))

        # lpd = GaussianDistribution(posterior.mean(), )
        # logprob = lpd.log_prob(jnp.atleast_1d(y.squeeze())).squeeze()
        
        Lambda = model.prior.kernel.Lambda
        # q = GaussianDistribution(m, cola.ops.Dense(S))
        # p = GaussianDistribution(jnp.zeros_like(m), cola.ops.Dense(Lambda))
        # kl = q.kl_divergence(p) # Alternative
        trace = (S.diagonal()/Lambda.diagonal()).sum()
        logdetLambda = jnp.log(Lambda.diagonal()).sum()
        SR, _ = jsp.linalg.cho_factor(S, lower=True)
        logdetS = 2 * jnp.log(SR.diagonal()).sum()
        quad = m.T@(m/Lambda.diagonal())
        kl = 1/2 * (trace - model.M + quad + logdetLambda - logdetS)

        # tr = - 1 / (2*obs_noise) * posterior.covariance().diagonal().sum()
        # return self.constant * (logprob - kl + tr)
        return self.constant * (logprob - kl)
