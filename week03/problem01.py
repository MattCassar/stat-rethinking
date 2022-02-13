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


def model(area=None, avgfood=None):
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    a = numpyro.sample('a', dist.Normal(0, 0.2))
    b = numpyro.sample('b', dist.Normal(0, 0.5))
    mu = numpyro.deterministic('mu', a + b*area)
    numpyro.sample('f', dist.Normal(mu, sigma), obs=avgfood)


def quap_model(df):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, num_chains=2, chain_method='sequential')
    mcmc.run(seed, area=df['area'].values, avgfood=df['avgfood'].values)
    mcmc.print_summary()

    return mcmc.get_samples()


def main():
    df = pd.read_csv('foxes.csv', sep=';')
    normalized_df=(df-df.mean())/df.std()
    samples = quap_model(normalized_df)


if __name__ == '__main__':
    main()
