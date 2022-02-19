import functools
import itertools

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


def model(area=None, avgfood=None, group=None, groupsize=None, weight=None):
    sigma = numpyro.sample("sigma", dist.Exponential(1))

    a = numpyro.sample("a", dist.Normal(0, 0.2))
    regression = a

    if area is not None:
        b_area = numpyro.sample("b_area", dist.Normal(0, 0.5))
        regression += b_area * area

    if avgfood is not None:
        b_avgfood = numpyro.sample("b_avgfood", dist.Normal(0, 0.5))
        regression += b_avgfood * avgfood

    if group is not None:
        b_group = numpyro.sample("b_group", dist.Normal(0, 0.5))
        regression += b_group * group

    if groupsize is not None:
        b_groupsize = numpyro.sample("b_groupsize", dist.Normal(0, 0.5))
        regression += b_groupsize * groupsize

    mu = numpyro.deterministic("mu", regression)
    numpyro.sample("w", dist.Normal(mu, sigma), obs=weight)


def run_model(df):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_warmup=500, num_samples=500, num_chains=4, chain_method="sequential"
    )
    mcmc.run(seed, **{col: df[col].values for col in df.columns.tolist()})

    return mcmc


def main():
    df = pd.read_csv("foxes.csv", sep=";")
    normalized_df = (df - df.mean()) / df.std()
    df_ind = df[["group", "avgfood", "groupsize", "area"]]
    ind_vars = ["group", "avgfood", "groupsize", "area"]

    samples = {}
    combinations = itertools.chain.from_iterable(
        itertools.combinations(ind_vars, r) for r in range(1, len(ind_vars) + 1)
    )

    for combination in combinations:
        sample = az.from_numpyro(
            run_model(normalized_df[list(combination) + ["weight"]])
        )
        samples[str(combination)] = sample

    print(az.compare(samples, scale="deviance"))
    """
                                               rank         loo     p_loo      d_loo        weight         se       dse  warning loo_scale
    ('avgfood', 'groupsize', 'area')              0  322.733075  4.468362   0.000000  0.000000e+00  15.495549  0.000000    False  deviance
    ('group', 'avgfood', 'groupsize', 'area')     1  323.084721  5.175727   0.351645  1.372587e-14  15.587524  2.271146    False  deviance
    ('group', 'groupsize', 'area')                2  323.169701  4.068410   0.436625  1.008836e-02  15.091271  3.345115    False  deviance
    ('groupsize', 'area')                         3  323.636966  3.476349   0.903891  5.194706e-01  15.011549  2.799568    False  deviance
    ('avgfood', 'groupsize')                      4  323.963769  3.602039   1.230694  0.000000e+00  15.267709  3.363530    False  deviance
    ('group', 'avgfood', 'groupsize')             5  324.340062  4.218091   1.606987  4.704410e-01  15.604460  4.489121    False  deviance
    ('groupsize',)                                6  330.462231  2.456580   7.729156  0.000000e+00  14.141467  5.675025    False  deviance
    ('group',)                                    7  331.093548  2.664374   8.360473  4.269923e-14  14.049281  7.232363    False  deviance
    ('group', 'groupsize')                        8  331.622999  3.553362   8.889924  0.000000e+00  14.430400  6.444285    False  deviance
    ('group', 'area')                             9  332.322698  3.557321   9.589622  3.960391e-14  13.823895  7.276710    False  deviance
    ('group', 'avgfood')                         10  332.880910  3.427093  10.147835  2.815344e-14  13.916692  7.301940    False  deviance
    ('group', 'avgfood', 'area')                 11  332.998195  4.283165  10.265120  2.173284e-14  13.862014  6.951403    False  deviance
    ('avgfood',)                                 12  333.490931  2.372698  10.757855  1.156753e-14  13.319986  6.796360    False  deviance
    ('area',)                                    13  333.695170  2.563790  10.962095  1.609701e-14  13.293810  6.847149    False  deviance
    ('avgfood', 'area')                          14  334.505235  3.401935  11.772160  0.000000e+00  13.416799  6.623581    False  deviance
    """


if __name__ == "__main__":
    main()
