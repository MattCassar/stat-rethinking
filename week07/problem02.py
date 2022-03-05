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
        nodes=["A", "C", "D", "K", "U"],
        edges=[
            ("D", "C"),
            ("D", "K"),
            ("D", "U"),
            ("C", "K"),
            ("A", "K"),
            ("A", "C"),
            ("U", "K"),
            ("U", "C"),
        ],
        latent_edges=[
            ("C", "K"),
        ],
    )
    g.draw().view()
    sets = g.get_all_backdoor_adjustment_sets("U", "C")

    if not sets:
        raise ValueError("No feasible adjustment set.")

    minimal_adjustment_set = sorted(sets, key=len)[0]
    print(set(sorted(minimal_adjustment_set)))


def main():
    cgm()


if __name__ == "__main__":
    main()
