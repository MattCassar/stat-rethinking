import numpy as np
import pandas as pd

import functools

from jax import random
import numpyro
from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.optim import Adam

seed = random.PRNGKey(0)


def model_total_effect(avgfood=None, weight=None):
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    a = numpyro.sample('a', dist.Normal(0, 0.2))
    b = numpyro.sample('b', dist.Normal(0, 0.5))
    mu = numpyro.deterministic('mu', a + b*avgfood)
    numpyro.sample('f', dist.Normal(mu, sigma), obs=weight)


def total_effect(df):
    kernel = NUTS(model_total_effect)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, num_chains=2, chain_method='sequential')
    mcmc.run(seed, avgfood=df['avgfood'].values, weight=df['weight'].values)
    mcmc.print_summary()

    return mcmc.get_samples()


def model_direct_effect(avgfood=None, groupsize=None, weight=None):
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    a = numpyro.sample('a', dist.Normal(0, 0.2))
    b_f = numpyro.sample('b_f', dist.Normal(0, 0.5))
    b_g = numpyro.sample('b_g', dist.Normal(0, 0.5))
    mu = numpyro.deterministic('mu', a + b_f*avgfood + b_g*groupsize)
    numpyro.sample('f', dist.Normal(mu, sigma), obs=weight)


def direct_effect(df):
    kernel = NUTS(model_direct_effect)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, num_chains=2, chain_method='sequential')
    mcmc.run(seed, avgfood=df['avgfood'].values, groupsize=df['groupsize'].values, weight=df['weight'].values)
    mcmc.print_summary()

    return mcmc.get_samples()


def main():
    df = pd.read_csv('foxes.csv', sep=';')
    normalized_df=(df-df.mean())/df.std()
    samples_tot = total_effect(normalized_df)
    samples_dir = direct_effect(normalized_df)


if __name__ == '__main__':
    main()
