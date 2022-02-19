import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


from scipy.stats import binom


def binomial(W, L, p):
    return binom(W + L, p).pmf(W)


def compute_posterior(W, L, grid_size=20):
    p_grid = np.linspace(0, 1, num=grid_size).reshape((grid_size,))
    prob_p = np.ones((grid_size,))
    prob_data = binomial(W, L, p_grid)
    posterior = np.multiply(prob_data, prob_p)
    posterior = posterior / np.sum(posterior)

    return p_grid, posterior


def plot_posterior(posterior, p_grid):
    samples = np.random.choice(
        p_grid, size=posterior.shape[0], p=posterior, replace=True
    )
    print("Sample mean", samples.mean())
    xs = np.linspace(0, 1, num=posterior.shape[0])
    plt.plot(xs, posterior)
    plt.show()


def main():
    GRID_SIZE = 1000
    p_grid, posterior = compute_posterior(4, 11, grid_size=GRID_SIZE)
    plot_posterior(posterior, p_grid)


if __name__ == "__main__":
    main()
