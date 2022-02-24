import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import arviz as az
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

from causalgraphicalmodels import CausalGraphicalModel

seed = random.PRNGKey(0)


def cgm():
    g = CausalGraphicalModel(
        nodes=["discipline", "gender", "awards"],
        edges=[
            ("gender", "discipline"),
            ("gender", "awards"),
            ("gender", "discipline"),
            ("discipline", "awards"),
        ],
        latent_edges=[
            ("discipline", "awards"),
        ],
    )
    g.draw().view()
    sets = g.get_all_backdoor_adjustment_sets("gender", "awards")

    if not sets:
        raise ValueError("No feasible adjustment set.")

    minimal_adjustment_set = sorted(sets, key=len)[0]
    print(set(sorted(minimal_adjustment_set)))


def model(gender, applications, awards=None):
    a = numpyro.sample("a", dist.Normal(-1, 1), sample_shape=(2,))
    logits = a[gender]
    numpyro.sample("awards", dist.Binomial(applications, logits=logits), obs=awards)


def run_model(df):
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
        gender=pd.Categorical(df["gender"], categories=[0, 1]).codes,
        applications=df["applications"].values,
        awards=df["awards"].values,
    )
    mcmc.print_summary()

    return mcmc


def main():
    df = pd.read_csv("NWOGrants.csv", sep=";")
    normalized_df = (df - df.mean()) / df.std()
    mcmc = run_model(df)
    samples = az.from_numpyro(mcmc)
    az.plot_trace(samples)
    plt.show()


if __name__ == "__main__":
    main()
