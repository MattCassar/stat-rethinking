import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import arviz as az
from jax import random
import numpyro
from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.optim import Adam

seed = random.PRNGKey(0)


def m6_9(married, age, happiness=None):
    n = 2
    a = numpyro.sample("a", dist.Normal(0, 1), sample_shape=(n,))
    bA = numpyro.sample("bA", dist.Normal(0, 2))
    sigma = numpyro.sample("sigma", dist.Exponential(1))

    mu = a[married] + bA * age
    numpyro.sample("happy_hat", dist.Normal(mu, sigma), obs=happiness)


def m6_10(married, age, happiness=None):
    a = numpyro.sample("a", dist.Normal(0, 1))
    bA = numpyro.sample("bA", dist.Normal(0, 2))
    sigma = numpyro.sample("sigma", dist.Exponential(1))

    mu = a + bA * age
    numpyro.sample("happy_hat", dist.Normal(mu, sigma), obs=happiness)


def run_model(df, model):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=500,
        num_samples=2000,
        num_chains=2,
        chain_method="sequential",
    )
    mcmc.run(
        seed,
        married=pd.Categorical(df["married"], categories=[0, 1]).codes,
        age=df["age"].values,
        happiness=df["happiness"].values,
    )
    mcmc.print_summary()

    return mcmc


def main():
    df = pd.read_csv("happiness.csv", sep=",")
    df = df[df["age"] > 17]
    normalized_df = (df - df.mean()) / df.std()
    m6_9_samples = az.from_numpyro(run_model(df, m6_9))
    m6_10_samples = az.from_numpyro(run_model(df, m6_10))

    print(az.compare({"m6_9": m6_9_samples, "m6_10": m6_10_samples}, scale="deviance"))
    az.plot_trace(m6_9_samples)
    az.plot_trace(m6_10_samples)
    plt.show()


if __name__ == "__main__":
    main()
