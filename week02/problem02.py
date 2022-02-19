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


def model(a_bar, age=None, weight=None):
    # Why don't we have to use the full causal inference described in the lecture?
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    a = numpyro.sample("a", dist.Normal(5, 1))
    b = numpyro.sample("b", dist.LogNormal(0, 1))

    # Why is this not a + b * (age - a_bar)?
    mu = numpyro.deterministic("mu", a + b * age)
    numpyro.sample("w", dist.Normal(mu, sigma), obs=weight)


def quap_model(df):
    regression = functools.partial(model, df["age"].mean())
    kernel = NUTS(regression)
    mcmc = MCMC(
        kernel,
        num_warmup=500,
        num_samples=2000,
        num_chains=2,
        chain_method="sequential",
    )
    mcmc.run(seed, age=df["age"].values, weight=df["weight"].values)
    mcmc.print_summary()


def main():
    df = pd.read_csv("howell1.csv", sep=";")
    df = df[df["age"] <= 13]
    samples = quap_model(df)


if __name__ == "__main__":
    main()
