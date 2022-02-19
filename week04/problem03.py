import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import functools

import arviz as az
from jax import random
import numpyro
from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.optim import Adam

seed = random.PRNGKey(0)


def percentile_interval(data, alpha=.95):
    if 0 > alpha or alpha > 1:
        raise ValueError('Alpha must be between 0 and 1')

    alpha *= 100

    return np.percentile(data, [(100 - alpha) / 2, 50 + alpha/2])


def intercept_model(temp=None, doy=None):
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    a = numpyro.sample('a', dist.Normal(0, 0.2))
    mu = numpyro.deterministic('mu', a)
    numpyro.sample('doy_hat', dist.Normal(mu, sigma), obs=doy)


def linear_model(temp=None, doy=None):
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    a = numpyro.sample('a', dist.Normal(0, 0.2))
    b = numpyro.sample('b', dist.Normal(0, 0.5))
    mu = numpyro.deterministic('mu', a + b*temp)
    numpyro.sample('doy_hat', dist.Normal(mu, sigma), obs=doy)


def quadratic_model(temp=None, doy=None):
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    a = numpyro.sample('a', dist.Normal(0, 0.2))
    b = numpyro.sample('b', dist.Normal(0, 0.5))
    c = numpyro.sample('c', dist.Normal(0, 0.5))
    mu = numpyro.deterministic('mu', a + b*temp + c*temp**2)
    numpyro.sample('doy_hat', dist.Normal(mu, sigma), obs=doy)


def run_model(df, model):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=4, chain_method='sequential')
    mcmc.run(seed, temp=df['temp'].values, doy=df['doy'].values)

    return mcmc


def main():
    df = pd.read_csv('cherry_blossoms.csv', sep=';')
    df = df.dropna()
    normalized_df=(df-df.mean())/df.std()
    constant_samples = az.from_numpyro(run_model(normalized_df, intercept_model))
    mcmc = run_model(normalized_df, linear_model)
    linear_samples = az.from_numpyro(mcmc)
    quadratic_samples = az.from_numpyro(run_model(normalized_df, quadratic_model))
    print(az.compare({'constant': constant_samples, 'linear': linear_samples, 'quadratic': quadratic_samples}, scale='deviance'))

    predictive = Predictive(linear_model, mcmc.get_samples())
    prior_predictions = predictive(seed, temp=np.array([(9 - df['doy'].mean()) / df['doy'].std()]).reshape(1,))
    interval = prior_predictions['doy_hat'] * df['doy'].std() + df['doy'].mean()
    print(percentile_interval(interval))


if __name__ == '__main__':
    main()