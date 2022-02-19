import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import functools

from jax import random
import numpyro
from numpyro.diagnostics import print_summary
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.optim import Adam

seed = random.PRNGKey(0)


def model(a_bar, s, age=None, weight=None):
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    a = numpyro.sample("a", dist.Normal(5, 1), sample_shape=(2,))
    b = numpyro.sample("b", dist.LogNormal(0, 1), sample_shape=(2,))
    mu = numpyro.deterministic("mu", a[s] + b[s] * age)
    numpyro.sample("w", dist.Normal(mu, sigma), obs=weight)


def quap_model(df):
    regression = functools.partial(
        model, df["age"].mean(), pd.Categorical(df["male"], categories=[0, 1]).codes
    )
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

    return mcmc.get_samples()


def plot_contrast(samples):
    mu_contrast = samples["a"][:, 1] - samples["a"][:, 0]
    sns.set_style("whitegrid")
    sns.kdeplot(mu_contrast)
    plt.show()


def main():
    df = pd.read_csv("howell1.csv", sep=";")
    df = df[df["age"] <= 13]
    samples = quap_model(df)
    plot_contrast(samples)


if __name__ == "__main__":
    main()
