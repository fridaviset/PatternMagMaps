import jax 
import jax.numpy as jnp
import scipy.io as sio
import gpjax as gpx
import optax as ox
jax.config.update("jax_enable_x64", True)
from tqdm import tqdm
from itertools import product

data = sio.loadmat("data/basement.mat")
X = data['p']
Y = data['y']
Da = gpx.Dataset(X, Y)

west = - 50 - 4
east = 50 + 4
south = - 40 - 4
north = 40 + 4
Lx = (jnp.abs(east) + jnp.abs(west)).squeeze()/2
Ly = (jnp.abs(south) + jnp.abs(north)).squeeze()/2
Lz = 4


import kernels
import gps
import objectives
import utils
Q = 4
mu = jnp.arange(1, Q+1) * 1/12
fx, fy, fz = jnp.meshgrid(jnp.arange(0, Q+1) * 1/6, jnp.arange(0, Q+1) * 1/15, jnp.arange(0, Q+1) * 1/2, indexing='ij')
mu = jnp.array([fx.flatten(), fy.flatten(), fz.flatten()]).T
pd_kernel = kernels.PotentialPeriodicPD(q=mu.shape[0], d=3, mu=mu).replace_trainable(mu=False)

key = jax.random.PRNGKey(13)
optimizer = ox.adam(learning_rate=1e-1)
mean_function = gpx.mean_functions.Zero()
likelihood = gpx.likelihoods.Gaussian(num_datapoints=Da.n, obs_stddev=5.)

for n, m in tqdm(product(range(50, 60, 10), range(40, 60, 10)), desc="Learning models"):
    bf = kernels.PotentialLaplaceBF(num_bfs=[n, m, 3], L = [Lx, Ly, Lz])
    kernel = kernels.SumKernel(kernels=[pd_kernel, kernels.RBF(basis=bf, lengthscale=2., variance=5.)])
    # kernel = kernels.RBF(basis=bf, lengthscale=2., variance=5.)
    prior = gpx.gps.Prior(mean_function=mean_function, kernel=kernel)
    gp = gps.PotentialBFGP(likelihood=likelihood, prior=prior)
    pgp = gp.update_with_batch(Da)

    gp_full, history = gpx.fit(
                                model=pgp,
                                objective=objectives.BF_MLL(negative=True, compute_bf=False),
                                train_data=Da,
                                optim=optimizer,
                                num_iters=200,
                                key=key,
    )

    utils.save_model(gp_full, "warehouse_{}by{}".format(n, m))