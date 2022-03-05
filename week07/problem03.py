import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import arviz as az
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

seed = random.PRNGKey(0)


def total_effect_model(C=None, U=None, D=None):
    """
    To get the total effect we only need to stratify by the district (based on the DAG from problem 2).
    """
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    b_district = numpyro.sample(
        "b_district", dist.Normal(a_bar, sigma), sample_shape=(61,)
    )
    b_urban = numpyro.sample("b_urban", dist.Normal(0, 1))
    logits = b_district[D] + b_urban * U
    numpyro.sample("C", dist.Bernoulli(logits=logits), obs=C)


def direct_effect_model(C=None, U=None, D=None):
    """
    To get the direct effect, we only need to stratify by the district (based on the DAG from problem 2).
    """
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    b_district = numpyro.sample(
        "b_district", dist.Normal(a_bar, sigma), sample_shape=(61,)
    )
    b_urban = numpyro.sample("b_urban", dist.Normal(0, 1))
    logits = b_district[D] + b_urban * U
    numpyro.sample("C", dist.Bernoulli(logits=logits), obs=C)


def run_model(df, model):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=500,
        num_samples=2_000,
        num_chains=2,
        chain_method="sequential",
    )
    mcmc.run(
        seed,
        C=df["use.contraception"].values,
        D=df["district"].values,
        U=df["urban"].values,
    )
    mcmc.print_summary()

    return mcmc


def main():
    df = pd.read_csv("bangladesh.csv", sep=";")
    mcmc = run_model(df, direct_effect_model)
    samples = az.from_numpyro(mcmc)
    az.plot_trace(samples, var_names=["b_urban"])

    plt.show()


if __name__ == "__main__":
    main()
