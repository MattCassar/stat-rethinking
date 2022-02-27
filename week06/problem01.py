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


def model(sigma_width=1):
    alpha_bar = numpyro.sample("alpha_bar", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(sigma_width))
    alpha_j = numpyro.sample("alpha_j", dist.Normal(alpha_bar, sigma))


def run_model():
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=500,
        num_samples=10_000,
        num_chains=2,
        chain_method="sequential",
    )
    mcmc.run(seed)
    mcmc.print_summary()

    return mcmc


def main():
    mcmc = run_model()
    samples = az.from_numpyro(mcmc)
    az.plot_trace(samples, var_names=["alpha_j"])
    plt.show()


if __name__ == "__main__":
    main()
