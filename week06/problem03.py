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


def model(S=None, D=None, D_log=None, T=None, pred=None, size=None):
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    alpha = numpyro.sample(
        "alpha", dist.Normal(0, sigma), sample_shape=(48,)
    )
    beta = numpyro.sample(
        "beta",
        dist.Normal(0, 1),
        sample_shape=(2, 2),  # len(pred.unique()), len(size.unique())
    )
    beta_d = numpyro.sample("beta_d", dist.Normal(0, 0.5), sample_shape=(2,))
    logits = alpha[T] + beta[pred, size] + beta_d[pred] * D_log
    numpyro.sample("S", dist.Binomial(D, logits=logits), obs=S)


def run_model(model, **kwargs):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=500,
        num_samples=2_000,
        num_chains=2,
        chain_method="sequential",
    )
    mcmc.run(seed, **kwargs)
    mcmc.print_summary()

    return mcmc


def main():
    df = pd.read_csv("reedfrogs.csv", sep=";")
    df["tank"] = np.arange(0, df.shape[0])
    df["d_log"] = df['density'].apply(np.log)
    df["d_log"] = (df["d_log"] - df["d_log"].mean()) / df["d_log"].std()
    mcmc = run_model(
        model,
        S=df["surv"].values,
        D=df['density'].values,
        D_log=df['d_log'].values,
        T=pd.Categorical(
            df["tank"], categories=list(range(len(df["tank"].unique())))
        ).codes,
        pred=pd.Categorical(
            df["pred"], categories=list(range(len(df["pred"].unique())))
        ).codes,
        size=pd.Categorical(
            df["size"], categories=list(range(len(df["size"].unique())))
        ).codes,
    )
    samples = az.from_numpyro(mcmc)
    az.plot_trace(samples, var_names=["beta_d"])

    plt.show()


if __name__ == "__main__":
    main()
