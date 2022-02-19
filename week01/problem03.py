import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import bayes_mvs
from scipy.stats import binom


def binomial(W, L, p):
    return binom(W + L, p).pmf(W)


def compute_posterior(W, L, grid_size=20):
    p_grid = np.linspace(0, 1, num=grid_size).reshape((grid_size,))
    prob_p = np.hstack((np.zeros((grid_size // 2,)), np.ones((grid_size // 2,))))
    prob_data = binomial(W, L, p_grid)
    posterior = np.multiply(prob_data, prob_p)
    posterior = posterior / np.sum(posterior)

    return p_grid, posterior


def percentile_interval(data, alpha=0.95):
    if 0 > alpha or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")

    alpha *= 100

    return np.percentile(data, [(100 - alpha) / 2, 50 + alpha / 2])


def compute_intervals(p_grid, posterior):
    samples = np.random.choice(
        p_grid, size=posterior.shape[0], p=posterior, replace=True
    )
    hpdi = az.hdi(samples.copy(), hdi_prob=0.89)
    pi = percentile_interval(samples.copy(), alpha=0.89)
    return hpdi, pi


def main():
    GRID_SIZE = 1000
    p_grid, posterior = compute_posterior(4, 2, grid_size=GRID_SIZE)
    hpdi, pi = compute_intervals(p_grid, posterior)
    print("hpdi", hpdi)
    print("pi  ", pi)


if __name__ == "__main__":
    main()
