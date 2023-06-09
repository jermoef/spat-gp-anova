import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

from tinygp import kernels, GaussianProcess


jax.config.update("jax_enable_x64", True)



def model(X, Y, X_pred):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    # noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    # jitter = numpyro.sample("jitter", dist.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))

    # kernel = var * kernels.Matern32(rho)
    kernel = var*kernels.ExpSquared(scale=length)
    gp = GaussianProcess(kernel, X, diag=1e-6 + noise, mean=0)
    numpyro.sample("gp", gp.numpyro_dist(), obs=Y)

    # numpyro.deterministic("pred2", cond_gp.mean)
    # numpyro.deterministic("pred", gp.predict(Y, X_pred))
    numpyro.deterministic("pred", gp.predict(Y, X_pred, return_var=True))
    

# helper function for doing hmc inference
def run_inference(model, args, rng_key, X, Y, X_pred):
    start = time.time()
    kernel = NUTS(model, target_accept_prob=0.9, init_strategy=init_to_median(num_samples=10))
    mcmc = MCMC(
        kernel,
        num_warmup=args['num_warmup'],
        num_samples=args['num_samples'],
        num_chains=args['num_chains'],
        thinning=args['thinning'],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y, X_pred)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()



# ANOVA model 

def anova_model(X, X_factor, Y, X_pred, factors):
    var_mu = numpyro.sample("var_mu", dist.Gamma(0.01, 0.01)) # change to unif if not working
    var_alpha = numpyro.sample("var_alpha", dist.Gamma(0.01, 0.01)) # change to unif if not working
    var_eps = numpyro.sample("var_eps", dist.Gamma(0.01, 0.01)) # change to unif if not working

    rho_mu = numpyro.sample("rho_mu", dist.Gamma(1, 1))
    Beta_0 = numpyro.sample("beta_0", dist.Normal(0,1)) # get rid of mean function if not working
    # Beta_1 = numpyro.sample("beta_1", dist.Normal(0,1))
    # Beta_2 = numpyro.sample("beta_2", dist.Normal(0,1))
    # mu_mean = Beta_0 * jnp.ones(X.shape[0]) + Beta_1 * X[:,0] + Beta_2 * X[:, 1] # assuming coordinates are first two cols
    mu_kernel = var_mu*kernels.Matern32(rho_mu)
    mu_gp = GaussianProcess(mu_kernel, X, diag=1e-5, mean=Beta_0)
    # mu_gp = GaussianProcess(mu_kernel, X, diag=1e-5, mean_value=mu_mean)
    mu = numpyro.sample("mu", mu_gp.numpyro_dist())

    rho_alpha = numpyro.sample("rho_alpha", dist.Gamma(1, 1))
    alpha_kernel = var_alpha*kernels.Matern32(rho_alpha)
    rho_eps = numpyro.sample("rho_eps", dist.Gamma(1, 1))
    eps_kernel = var_eps*kernels.Matern32(rho_eps)
    for l in factors:
        alpha_gp = GaussianProcess(alpha_kernel, X[X_factor==l,:], diag=1e-5, mean=0.0)
        alpha_l = numpyro.sample("alpha_"+str(l), alpha_gp.numpyro_dist())
        mu = mu.at[X_factor==l].set(mu[X_factor==l] + alpha_l) 

    Y_gp = GaussianProcess(eps_kernel, X, diag=1e-5, mean_value=mu)
    numpyro.sample("obs", Y_gp.numpyro_dist(), obs=Y)
    numpyro.deterministic("pred", Y_gp.predict(Y, X_pred, return_var=True))

# helper function for doing hmc inference
def run_inference_anova(anova_model, args, rng_key, X, X_factor, Y, X_pred, factors):
    start = time.time()
    kernel = NUTS(anova_model, target_accept_prob=0.9, init_strategy=init_to_median(num_samples=10))
    mcmc = MCMC(
        kernel,
        num_warmup=args['num_warmup'],
        num_samples=args['num_samples'],
        num_chains=args['num_chains'],
        thinning=args['thinning'],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, X_factor, Y, X_pred, factors)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()