import jax
import jax.numpy as jnp
import gpjax as gpx
from gpjax.base import param_field, static_field
from dataclasses import dataclass
from beartype.typing import Union, Callable
from jaxtyping import Float
from gpjax.typing import (
    Array,
    ScalarFloat,
)
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax.scipy.linalg import block_diag

class PDBFMixin:
    """Mixin class. Defines the Periodic PD basis functions and allows for the kernel to be represented with these basis functions instead.
    """ 
    @property
    def M(self):
        return self.mu.shape[0] * 2

    def _1d_call(self, x):
        op = 2 * jnp.pi * x * self.mu
        return jnp.prod(jnp.vstack([jnp.cos(op), jnp.sin(op)]), axis=1)
    
    phi = _1d_call # Alias
    
    def Phi(self, x):
        return jax.vmap(self.phi)(x)
    
    @property
    def Lambda(self):
        return jnp.diag(jnp.tile(jnp.exp(self.a), 2))*self.variance
        # return jnp.diag(jnp.tile(self.a, 2))*self.variance

class JacMixin:
    """Mixin class, crucial for the potential field modeling (i.e., it converts the basis functions to gradients of the potential field instead).
    """
    def phi(self, x, *args):
        return jax.jacfwd(self._1d_call)(x, *args)

@dataclass
class PD(gpx.kernels.AbstractKernel):
    """Standard (multi-dimensional) pattern detection kernel.

    Parameters
    ----------
    q : int
        Number of mixture components
    d : int
        Dimensionality of the data
    a : array_like, float
        Mixture weights
    mu : array_like, float
        Mixture periodicities
    nu : array_like, float
        Mixture (inverse) lengthscales
    """
    q: int = static_field(default=1)
    d: int = static_field(default=1)
    a: Union[ScalarFloat, Float[Array, "Q"]] = param_field(None, bijector=tfb.Softplus())
    mu: Union[ScalarFloat, Float[Array, "Q D"]] = param_field(None, bijector=tfb.Softplus())
    nu: Union[ScalarFloat, Float[Array, "Q D"]] = param_field(None, bijector=tfb.Softplus())
    name: str = "PD"

    def __post_init__(self):
        if self.a is None:
            self.a = jnp.ones((self.q,))/self.q
        if self.mu is None:
            self.mu = (jnp.arange(1, self.q+1)[:, None] * jnp.ones((self.d,)))/self.q
        if self.nu is None:
            self.nu = jnp.ones((self.q, self.d))
        self.mu = self.mu.reshape(self.q, -1)
        self.nu = self.nu.reshape(self.q, -1)
        
    def __call__(self, x, y):
        tau = x - y
        pd = jnp.cos(2 * jnp.pi * tau * self.mu)
        exp = jnp.exp(- 2 * jnp.pi * tau**2 * self.nu)
        K = jnp.sum(self.a * jnp.prod(pd * exp, axis=1))
        return K.squeeze() 

class PeriodicPDMixin:
    def __call__(self, x, y):
        tau = x - y
        # K = self.variance * jnp.sum(self.a * jnp.prod(jnp.cos(2 * jnp.pi * tau * self.mu), axis=1))
        K = self.variance * jnp.sum(jnp.exp(self.a) * jnp.prod(jnp.cos(2 * jnp.pi * tau * self.mu), axis=1))
        return K.squeeze()

@dataclass
class PeriodicPD(PeriodicPDMixin, PDBFMixin, gpx.kernels.AbstractKernel):
    """Purely periodic (multi-dimensional) pattern detection kernel.

    Representable as a basis function expansion as well, see PDBFMixin.

    Parameters
    ----------
    q : int
        Number of mixture components
    d : int
        Dimensionality of the data
    a : array_like, float
        Mixture weights
    mu : array_like, float
        Mixture periodicities
    """
    q: int = static_field(default=1)
    d: int = static_field(default=1)
    # a: Union[ScalarFloat, Float[Array, "Q"]] = param_field(None, bijector=tfb.Softplus())
    a: Union[ScalarFloat, Float[Array, "Q"]] = param_field(None)
    mu: Union[ScalarFloat, Float[Array, "Q D"]] = param_field(None, bijector=tfb.Softplus())
    variance: ScalarFloat = param_field(1., bijector=tfb.Softplus())
    name: str = "PeriodicPD"

    def __post_init__(self):
        if self.a is None:
            # self.a = jnp.ones((self.q,))/self.q
            self.a = jnp.log(jnp.ones((self.q,))/self.q)
        if self.mu is None:
            self.mu = (jnp.arange(1, self.q+1)[:, None] * jnp.ones((self.d,)))/self.q
        self.mu = self.mu.reshape(self.q, -1)
        if self.mu.shape[1] != self.d:
            self.mu = jnp.tile(self.mu, (1, self.d))
    
    def spectral_density(self, x):
        # return self.variance * jnp.tile(self.a, 2)
        return self.variance * jnp.tile(jnp.exp(self.a), 2)

@dataclass
class FourierPD(PeriodicPDMixin, PDBFMixin, gpx.kernels.AbstractKernel):
    """Purely periodic (multi-dimensional) pattern detection kernel.

    Particularly restricts the mixture periodicities to be integer multiples of one another -- this corresponds to a Fourier series of a purely periodic kernel.

    Parameters
    ----------
    q : int
        Number of mixture components
    d : int
        Dimensionality of the data
    a : array_like, float
        Mixture weights
    """
    q: int = static_field(default=1)
    d: int = static_field(default=1)
    a: Union[ScalarFloat, Float[Array, "Q"]] = param_field(None, bijector=tfb.Softplus())
    fundamental_frequency: Float = param_field(None, bijector=tfb.Softplus())
    variance: ScalarFloat = param_field(1., bijector=tfb.Softplus())
    name: str = "FourierPD"

    @property
    def mu(self):
        return jnp.arange(1, self.q+1)[:, None] * self.fundamental_frequency

    @mu.setter
    def mu(self, _mu):
        pass

    def __post_init__(self):
        if self.a is None:
            self.a = jnp.ones((self.q,))/self.q
        if self.fundamental_frequency is None:
            self.fundamental_frequency = jnp.ones((self.d,))

    def spectral_density(self, x):
        return self.variance * jnp.tile(self.a, 2)

@dataclass
class FourierPeriodic(PDBFMixin, gpx.kernels.Periodic):
    """Fourier representation of the periodic kernel.

    Parameters
    ----------
    q : int
        Number of Fourier components, for simplicity it's assumed equal components in each dimension
    d : int
        Dimensionality of the data
    """
    q: int = static_field(default=1)
    d: int = static_field(default=1)
    integrationN: int = static_field(default=1000)

    @property
    def a(self):
        P = self.period
        omega = 1/P
        w0 = jnp.pi * omega
        domains = [jnp.linspace(-pi/2, pi/2, self.integrationN) for pi in jnp.atleast_1d(P)]
        aq = jnp.ones((1,))
        Q = jnp.arange(0, self.q+1)

        for i, (l, p) in enumerate(zip(self.lengthscale, self.period)):
            init = dict(lengthscale=l, period=p, variance=self.variance)
            ki = type(self)(**init)
            Ki = jax.vmap(ki, (None, 0), 0)(jnp.zeros((1,)), domains[i])
            fs = jnp.cos(2 * w0[i] * domains[i] * Q[:,None])
            aq = jnp.kron(aq, 2 * fs @ Ki / self.integrationN)
        return aq
        # Q = jnp.array([x.flatten() for x in jnp.meshgrid(*jnp.tile(jnp.atleast_2d(self.q), (self.d, 1)), indexing='ij')]).T
        
        # domain = jnp.array([x.flatten() for x in jnp.meshgrid(*[jnp.linspace(-pi/2, pi/2, self.integrationN) for pi in jnp.atleast_1d(P)], indexing='ij')]).T

        # K = jax.vmap(self, (None, 0), 0)(jnp.zeros(P.shape), domain)

        # domain = jnp.arange(-P/2, P/2 + self.res, self.res)
        # domain = [jnp.linspace(-pi/2, pi/2, self.integrationN) for pi in jnp.atleast_1d(P)]
        # f = jax.vmap(self, (None, 0), 0)(0., domain)
        # fs = jnp.cos(2 * w0 * domain * q[:,None])
            
        # fs = jnp.array([jnp.prod(jnp.cos(2 * w0 * domain * qi), axis=1) for qi in Q])
        # # seper = jnp.exp(-2/l_per**2 * jnp.sin(w0[:, None] * domain)**2)
        # CQ = (fs @ K) / self.integrationN**self.d
        # Cd = jnp.ones_like(Q)*2
        # C = Cd.at[Q==0].set(1).prod(axis=1)
        # aq = C*CQ
        # return aq
        # fs = jnp.cos(2 * w0 * self.q[:,None] * domain)
        # # fs = jnp.cos(w0 * q[:,None] * domain)
        # cq = fs @ f * self.res / P # Complex Fourier series coefficients
        # a0 = f.sum() * self.res / P
        # aq = 2 * cq # Fourier series coefficients
        # ad = jnp.concatenate([a0, jnp.tile(aq, 2)])
        
        # return jnp.linalg.kron(ad,)

    @property
    def mu(self):
        Q = jnp.tile(jnp.arange(0, self.q+1)[:, None], (1, self.d))
        mud = (Q / jnp.atleast_1d(self.period)[None, :]).T

        # All covered frequencies
        return jnp.array([x.flatten() for x in jnp.meshgrid(*mud, indexing='ij')]).T

    @mu.setter
    def mu(self, _mu):
        pass
    
    def spectral_density(self, x):
        return jnp.tile(self.a, 2)
    
    @property
    def M(self):
        return self.mu.shape[0] * 2

    def _1d_call(self, x):
        op = 2 * jnp.pi * x * self.mu
        return jnp.prod(jnp.vstack([jnp.cos(op), jnp.sin(op)]), axis=1)
    
    phi = _1d_call # Alias
    
    def Phi(self, x):
        return jax.vmap(self.phi)(x)
    
    @property
    def Lambda(self):
        return jnp.diag(jnp.tile(self.a, 2))

@dataclass
class LaplaceBF(gpx.base.Module):
    """Laplace basis functions on a rectangular domain.

    Parameters
    ----------
    num_bfs : list, array_like
        Number of basis functions to use for each dimension.
    L : list, array_like
        Domain size in each dimension.
    center : list, array_like
        Center of the domain in each dimension (do not use)
    js : *do not specify*
    """
    num_bfs: Float[Array, "M"] = param_field(default=jnp.ones((1,)), trainable=False)
    L: Float[Array, "M"] = param_field(default=jnp.ones((1,)), trainable=False)
    center: Float[Array, "M"] = param_field(default=jnp.zeros((1,)), trainable=False)
    js: Float[Array, "M"] = param_field(default=jnp.zeros((1,)), trainable=False, repr=False)

    def __post_init__(self):
        """Initializes necessary parameters for the Laplace basis functions.
        """
        try:
            j = [jnp.arange(1, int(m) + 1) for m in self.num_bfs]
        except:
            j = [jnp.arange(1, self.num_bfs + 1)]
        # self.js = jnp.array(list(product(*[x.flatten() for x in j])), dtype=jnp.float64)
        self.js = jnp.vstack([x.ravel() for x in jnp.meshgrid(*j, indexing='ij')]).T.astype(jnp.float64)
        self.num_bfs = jnp.atleast_1d(jnp.array(self.num_bfs)).astype(jnp.float64)#.astype(int)
        D  = len(self.num_bfs)
        self.L = jnp.atleast_1d(jnp.array(self.L))
        self.center = jnp.atleast_1d(jnp.array(self.center))
        if len(self.L) != D:
            self.L = jnp.ones((D,)) * self.L[0]
        if len(self.center) != D:
            self.center = jnp.ones((D,)) * self.center[0]
    
    @property
    def M(self):
        return jnp.prod(self.num_bfs).astype(int)
        
    def _1d_call(self, x, j):
        z = x - self.center
        L = self.L
        op = jnp.pi * j * (z + L) / (2 * L)
        return jnp.prod(1 / jnp.sqrt(L) * jnp.sin(op) * (z >= -L) * (z <= L))
    
    phi = _1d_call

    def __call__(self, x):
        return jax.vmap(jax.vmap(self.phi, (None, 0), 0), (0, None), 0)(x, self.js)

    def eigenvalues(self):
        """Eigenvalues of the basis functions.

        Returns
        -------
        array_like
        """
        j = self.js
        L = self.L
        sq_lambda_j = ( jnp.pi * j / (2 * L) )#**2
        return sq_lambda_j

@dataclass
class RBF(gpx.kernels.RBF):
    """Redefines the RBF kernel from GPJax.

    Provides a basis function representation of the kernel (the Hilbert GP representation).
    Also implements the spectral density of the kernel (which is not valid in GPJax currently).

    Parameters
    ----------
    basis : LaplaceBF
        The Hilbert basis to use to represent the kernel.
    """ 
    basis: LaplaceBF = param_field(default=LaplaceBF(), trainable=False)

    @property
    def M(self):
        return self.basis.M

    def Phi(self, x):
        return self.basis(x)
    
    @property
    def Lambda(self):
        sq_lambda_j = self.basis.eigenvalues() 
        return jnp.diag(jax.vmap(self.spectral_density, 0, 0)(sq_lambda_j))
    
    def spectral_density(self, x):
        """The spectral density of a squared exponential kernel.

        Parameters  
        ---------- 
        lengthscale : ScalarFloat
            The lengthscale of the kernel. 
        variance : ScalarFloat 
            The variance of the kernel
        x : Float[Array, "N 1"]
            Input data

        Returns
        -------
        Float[Array, "N 1"]
            The spectral density of the kernel at specific inputs.
        """
        l, s2f = self.lengthscale, self.variance
        return jnp.prod(s2f * jnp.sqrt( 2 * jnp.pi * l**2) * jnp.exp( - 1 / 2 * x**2 * l**2)) 

@dataclass
class SumKernel(gpx.kernels.base.CombinationKernel):
    """Supersedes the SumKernel from GPJax .
    
    Allows for a SumKernel to be represented with a finite basis function expansion.
    """
    operator: Callable = static_field(jnp.sum)

    def Phi(self, x):
        return jnp.hstack([k.Phi(x) for k in self.kernels])
    
    @property
    def Lambda(self):
        return block_diag(*[k.Lambda for k in self.kernels])

class PotentialPeriodicPD(JacMixin, PeriodicPD):
    """ A periodic PD kernel particularly aimed at modeling a potential field. I.e., the basis functions are gradients of the standard periodic PD basis functions.
    """
    pass

class PotentialLaplaceBF(JacMixin, LaplaceBF):
    """ A Laplace BF particularly aimed at modeling a potential field. I.e., the basis functions are gradients of the standard Laplace basis functions.
    """
    pass