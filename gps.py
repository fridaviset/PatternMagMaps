from dataclasses import dataclass
import jax.numpy as jnp
import gpjax as gpx
from gpjax.distributions import GaussianDistribution
from gpjax.base import (
    static_field, 
    param_field
)
from gpjax.typing import ScalarFloat
from jaxtyping import (
    Float, 
    Array
)
import cola
from gp_utils import (
    predict,
    vpredict,
    compute_dual,
    dual_to_mean
)

@dataclass
class DualParameters(gpx.base.Module):
    """Convenience module to gather the dual parameters in one container.

    Handles multi-dimensional parameters (i.e., multi-output GPs)

    Parameters
    ----------
    M : int
        Number of parameters
    L : int
        Number of outputs
    alpha : array_like
        First-order dual parameter
    B : array_like
        Second-order dual parameter
    """
    M: int = static_field(default=1)
    L: int = static_field(default=1)
    alpha: Float[Array, "M L"] = param_field(None, trainable=False)
    B: Float[Array, "M M L"] = param_field(None, trainable=False)

    def __post_init__(self):
        if self.alpha is None:
            M = self.M
            L = self.L
            self.alpha = jnp.zeros((M, L)).squeeze() # For scalar models we compress out the L dimension
            self.B = jnp.zeros((L, M, M)).squeeze()

@dataclass
class BFGP(gpx.gps.ConjugatePosterior):
    """Approximate GP based on basis functions.
    
    Can handle multi-output but critically assumes the **same** kernel for each output dimension.
    Designed to be used for magnetic field modeling but applicable in other cases with shared kernel.

    Parameters
    ----------
    L : Number of outputs.
    parameters : DualParameters
        The dual parameters defining the basis function model.
    """    
    L: int = static_field(default=1)
    parameters: DualParameters = param_field(None, trainable=False, repr=False)

    def __post_init__(self):
        if hasattr(self.prior.kernel, "kernels"):
            M = 0
            for kernel in self.prior.kernel.kernels:
                M += kernel.M
        else:
            M = self.prior.kernel.M
        if self.parameters is None:
            self.parameters = DualParameters(M, self.L)

    def predict(self, test_inputs, full_cov=True, extra_data=None):
        """Predicts at the given test inputs.

        Parameters
        ----------
        test_inputs : array_like
            Prediction locations.
        full_cov : bool, optional
            Whether to compute full covariance matrix or just marginal covariances.
        extra_data : gpx.Dataset
            Extra data to condition on before prediction

        Returns
        -------
        gpx.distributions.GaussianDistribution
            The posterior distribution (**not** predictive -- does not include likelihood).
        """
        if extra_data is not None:
            alpha, B = self.compute_dual(extra_data)
            alpha += self.parameters.alpha
            B += self.parameters.B
        else:
            alpha, B = self.parameters.alpha, self.parameters.B
        
        m, S = dual_to_mean(alpha, B, self.prior.kernel, self.likelihood.obs_stddev)
        if full_cov:
            mu, V = predict(m, S, self.prior.kernel, test_inputs)
        else:
            mu, V = vpredict(m, S, self.prior.kernel, test_inputs)
            if mu.ndim > 1:
                V = jnp.concatenate([jnp.expand_dims(jnp.diag(Vi), -1) for Vi in V.T], -1)
            else:
                V = jnp.diag(V)
        # Add prior mean
        if mu.ndim > 1:
            mu += jnp.broadcast_to(self.prior.mean_function(test_inputs), mu.shape)
            return tuple([GaussianDistribution(jnp.atleast_1d(mi.squeeze()), 
                                cola.ops.Dense(Vi)) for mi, Vi in zip(mu.T, V.T)])[:]
        else:
            mu += self.prior.mean_function(test_inputs).squeeze()
            return GaussianDistribution(jnp.atleast_1d(mu), cola.ops.Dense(V))
        # return GaussianDistribution(loc=jnp.atleast_1d(mu.squeeze()),
        #                             scale=cola.ops.Dense(V))
    
    def compute_dual(self, data):
        """Computes the dual parameters of the given data.

        Parameters
        ----------
        data : gpx.Dataset
            Dataset to compute dual parameters for.

        Returns
        -------
        (Array, Array)
            (alpha, B) of the given data set
        """
        alpha, B = compute_dual(self.prior.kernel, data)
        return alpha, B

    def update_with_batch(self, data):
        """ Update the posterior with batch of data.

        Parameters
        ----------
        data : gpx.Dataset
            Dataset to update with.

        Returns
        -------
        BFGP
            An updated BFGP with new posterior parameters
        """
        alpha, B = self.compute_dual(data)

        new_parameters = self.parameters.replace(alpha=self.parameters.alpha + alpha,
                                                  B=self.parameters.B + B)
        return self.replace(parameters=new_parameters)
    
    @property
    def M(self):
        try:
            return jnp.sum(jnp.asarray([k.M for k in self.prior.kernel.kernels]))
        except:
            return self.prior.kernel.M

    @property
    def alpha(self):
        return self.parameters.alpha
    
    @property
    def B(self):
        return self.parameters.B
    
    @property
    def B_diag(self):
        return self.parameters.B.diagonal()
    
    @property
    def mean_parameters(self):
        return dual_to_mean(self.parameters.alpha, 
                            self.parameters.B, 
                            self.prior.kernel, 
                            self.likelihood.obs_stddev)
    
class PotentialBFGP(BFGP):
    """A basis function based GP that assumes a potential field kernel.
    """    
    def compute_dual(self, data):
        alpha, B = compute_dual(self.prior.kernel, data)
        # The dual parameters are computed **per** axes in the potential field kernel
        return alpha.sum(axis=-1), B.sum(axis=0)
        