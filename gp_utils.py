from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp

@jax.jit
def compute_dual(kernel, data):
    """Compute dual representation of the given data.

    Parameters
    ----------
    bf : Callable
        The basis functions as a callable object.
    data : gpjax.Dataset
        A dataset to be processed.

    Returns
    -------
    (Float[Array, "M L"], Float[Array, "L M M"])
        First and second order dual parameters of the posterior given the data.
    """    
    X, y = data.X, data.y
    Phi = kernel.Phi(X)
    # Basically Phi.T @ Phi but enables multiple such products simultaneously
    B = jnp.einsum('ij...,ih...->...jh', Phi, Phi)
    # B = jnp.matmul(Phi.T, Phi)
    # Basically Phi.T @ y but enables multi-dimensional outputs
    alpha = jnp.einsum('ij...,i...->j...', Phi, y)
    # alpha = jnp.matmul(Phi.T, y)
    return (alpha.squeeze(), B)

@jax.jit
def mean_to_dual(m, 
                 S, 
                 kernel, 
                 noise_std):
    """Move from mean to dual representation of a posterior on bf weights.

    Go from mean parameters $(m, S)$ to dual parameters $(\alpha, B)$, i.e.,
    $(m, S) \to (\alpha, B)$ of the variational distribution on the basis 
    function weights.

    Parameters
    ----------
    m : Float[Array, "M"]
        Mean of the posterior.
    S : Float[Array, "M M"]
        Covariance of the posterior.
    kernel : Callable
    noise_std : Float
        Noise standard deviation.

    Returns
    -------
    GaussianDistribution
        Dual parametrization of the posterior
    """    
    jitter = 1e-6
    Lambda = kernel.Lambda
    Lambdainv = jnp.broadcast_to(jnp.diag(1/Lambda.diagonal()), S.shape)

    I = jnp.broadcast_to(jnp.identity(S.shape[-1]), S.shape)
    SR, _ = jsp.linalg.cho_factor(S + I * jitter, lower=True)
    Sigma = jsp.linalg.cho_solve((SR, True), I)
    
    # alpha = noise_std**2 * jnp.einsum('...jk,j...->k...', Sigma, m)
    alpha = noise_std**2 * Sigma @ m

    # SRinv = jsp.linalg.cho_solve((SR, True), jnp.identity(SR.shape[-1]))
    # lambda_j = bf.eigenvalues()
    # Lambdainv = jnp.diag(1/jax.vmap(spectral_density, 0, 0)(jnp.sqrt(lambda_j)))
    
    # Lambdainv = jnp.diag(1/jnp.prod(jnp.atleast_2d(spectral_density(jnp.sqrt(lambda_j))), axis=0))
    B = noise_std**2 * (Sigma - Lambdainv)
    return (alpha, B)

jax.jit
def dual_to_mean(alpha, 
                 B, 
                 kernel, 
                 noise_std):
    """Move from dual to mean representation of a posterior on bf weights.

    Go from dual parameters $(\alpha, B)$ to mean parameters $(m, S)$, i.e.,
    $(\alpha, B) \to (m, S)$ of the variational distribution on the basis 
    function weights.

    Parameters
    ----------
    alpha : Float[Array, "M L"]
        First order dual parameter
    B : Float[Array, "L M M"]
        Second order dual parameter
    bf : Callable
        The basis functions as a callable object.
    spectral_density : Callable
        Spectral density of the kernel being used
    noise_cov : Float
        Noise covariance.

    Returns
    -------
    GaussianDistribution
        Mean parametrization of the posterior
    """    
    Lambda = kernel.Lambda
    Lambdainv = jnp.broadcast_to(jnp.diag(1/Lambda.diagonal()), B.shape)

    # Lambdainv = jnp.diag(1/jnp.prod(jnp.atleast_2d(spectral_density(jnp.sqrt(lambda_j))), axis=0))
    Sigmainv = B + noise_std**2 * Lambdainv
    SR, _ = jsp.linalg.cho_factor(Sigmainv, lower=True)
    I = jnp.broadcast_to(jnp.identity(SR.shape[-1]), SR.shape)
    Sigma = jsp.linalg.cho_solve((SR, True), I)
    # Basically Sigma @ alpha
    m = jnp.einsum('...jk,j...->k...', Sigma, alpha)
    # m = Sigma @ alpha
    S = noise_std**2 * Sigma
    return (jnp.atleast_1d(m.squeeze()), S)

@jax.jit
def predict(mean,
            covariance,
            kernel, 
            test_inputs):
    """Predict at new test locations

    Parameters
    ----------
    mean : array_like
        Mean of the variational distribution on the basis function weights.
    covariance : array_like
        Covariance of the variational distribution on the basis function weights.
    kernel : Callable
        The kernel of the GP (must have a .Phi attribute)
    test_inputs : Float[Array, "N D"]
        Inputs at which to predict the function values.

    Returns
    -------
    GaussianDistribution
        Posterior on the test inputs.
    """    
    Phi = kernel.Phi(test_inputs)#.T
    # Basically Phi @ mean
    mean = jnp.einsum('ij...,j...->i...', Phi, mean).squeeze()
    # mean = Phi.T @ mean
    # Basically Phi @ covariance @ Phi.T
    cov = jnp.einsum('ij...,...jl,kl...->ik...', Phi, covariance, Phi).squeeze()
    # cov = Phi.T @ covariance @ Phi
    return mean, cov

vpredict = jax.jit(jax.vmap(lambda m, S, kernel, x: predict(m, S, kernel, x[None, :]), (None, None, None, 0), (0, 0)))