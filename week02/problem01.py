import numpy as np
import pandas as pd

import functools

from jax import random
import numpyro
from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.optim import Adam

seed = random.PRNGKey(0)


def percentile_interval(data, alpha=.95):
    if 0 > alpha or alpha > 1:
        raise ValueError('Alpha must be between 0 and 1')

    alpha *= 100

    return np.percentile(data, [(100 - alpha) / 2, 50 + alpha/2])


def model(x_bar, height, weight=None):
    alpha = numpyro.sample("alpha", dist.Normal(178, 10))
    beta = numpyro.sample("beta", dist.LogNormal(0, 1))
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    mu = numpyro.deterministic("mu", alpha + beta * (height - x_bar))
    numpyro.sample("weight", dist.Normal(mu, sigma), obs=weight)


def quap_model(df):
    regression = functools.partial(model, df['height'].mean())
    posterior = AutoLaplaceApproximation(regression)
    svi = SVI(
        regression,
        posterior,
        Adam(1),
        Trace_ELBO(),
        height=df['height'].values,
        weight=df['weight'].values
    )
    svi_result = svi.run(seed, 2000)
    p4_1 = svi_result.params

    samples = posterior.sample_posterior(seed, p4_1, (1000,))
    samples.pop('mu')
    predictive = Predictive(regression, samples)
    predictions = predictive(seed, height=np.array([140, 160, 175]).reshape(3,))
    print(percentile_interval(predictions['weight'][:, 0], alpha=0.89))
    print(percentile_interval(predictions['weight'][:, 1], alpha=0.89))
    print(percentile_interval(predictions['weight'][:, 2], alpha=0.89))


def main():
    df = pd.read_csv('howell1.csv', sep=';')
    df = df[df['age'] >= 18]
    samples = quap_model(df)


if __name__ == '__main__':
    main()
