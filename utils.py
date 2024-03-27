import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
from beartype.typing import (
    Callable,
)
from gpjax.typing import (
    ScalarFloat
)
import pickle
import matplotlib.pyplot as plt

def save_model(model, filename):
    """Saves the model in two files. 
    
    The Pytree definition is saved in a pickle file
    The Pytree values are saved in an .npz format

    NB: Give the filename **without** extension.

    Parameters
    ----------
    model : Pytree to save
    filename : String
    """    
    values, tree = tree_flatten(model)
    jnp.savez(filename + "_values", *values)
    with open(filename + ".pickle", "wb") as file:
        pickle.dump(tree, file)

def load_model(filename):
    """Load the model from the given filename

    NB: Give the filename **without** extension.

    Parameters
    ----------
    filename : String

    Returns
    -------
        The model defined in the file given (if it exists)
    """    
    values = list(jnp.load(filename + "_values.npz").values())
    with open(filename + ".pickle", "rb") as file:
        tree = pickle.load(file)
    return tree_unflatten(tree, values)

def integrate(fun: Callable, 
              limits: list, 
              N: int=100, 
              args: list=[]) -> ScalarFloat:
    """Chebyshev-Gauss integration (second kind)

    Parameters
    ----------
    fun : Function to integrate, 0th argument is integrated w.r.t. to
    limits : Limits of the integration
    N : Number of polynomials to use
    args : Additional arguments to function.

    Returns
    -------
        Integration value
    """    
    i = jnp.arange(1, N+1)
    xi = jnp.cos(i/(N+1) * jnp.pi)[:, None]
    wi = jnp.pi/(N+1)*jnp.sin(i/(N+1) * jnp.pi)
    a, b = limits
    scale = (b - a)/2
    return jnp.prod(scale) * wi @ fun(scale * xi + (a + b)/2, *args)

def bitmappify(ax, dpi=None):
    """
    Convert vector axes content to raster (bitmap) images
    """
    fig = ax.figure
    # safe plot without axes
    ax.set_axis_off()
    fig.savefig('temp.png', dpi=dpi, transparent=True)
    ax.set_axis_on()

    # remember geometry
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    xb = ax.bbox._bbox.corners()[:, 0]
    xb = (min(xb), max(xb))
    yb = ax.bbox._bbox.corners()[:, 1]
    yb = (min(yb), max(yb))

    # compute coordinates to place bitmap image later
    xb = (- xb[0] / (xb[1] - xb[0]),
          (1 - xb[0]) / (xb[1] - xb[0]))
    xb = (xb[0] * (xl[1] - xl[0]) + xl[0],
          xb[1] * (xl[1] - xl[0]) + xl[0])
    yb = (- yb[0] / (yb[1] - yb[0]),
          (1 - yb[0]) / (yb[1] - yb[0]))
    yb = (yb[0] * (yl[1] - yl[0]) + yl[0],
          yb[1] * (yl[1] - yl[0]) + yl[0])

    ax.clear()
    ax.imshow(plt.imread('temp.png'), origin='upper',
              aspect='auto', extent=(xb[0], xb[1], yb[0], yb[1]), label='_nolegend_')

    # reset view
    ax.set_xlim(xl)
    ax.set_ylim(yl)