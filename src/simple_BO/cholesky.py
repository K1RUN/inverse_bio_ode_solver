import jax
import numpy as np
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt


from typing import Tuple
from numpy import typing as npt
from matplotlib.ticker import LinearLocator

from inverse_bio_ode_solver.src.try_bo.gaussian_process import update_posterior


@jax.jit
def get_sq_dist(vectors1: npt.ArrayLike, vectors2: npt.ArrayLike) -> npt.ArrayLike:
    """Calculate distance between each pair of given vectors"""
    sq_dist = (vectors1[:, np.newaxis] - vectors2) ** 2
    return np.sum(sq_dist, axis=-1)


def RBF_kernel(vectors1: npt.ArrayLike, vectors2: npt.ArrayLike, l=1.0, sigma_f=1.0):
    sq_dist = get_sq_dist(vectors1, vectors2)
    return sigma_f**2 * np.exp(-sq_dist / (2 * l**2))


def cholesky(matrix: npt.ArrayLike) -> npt.ArrayLike:
    # https://en.wikipedia.org/wiki/Cholesky_decomposition
    n_: int = len(matrix)
    # we store only lower triuangular side
    L_: np.array = np.zeros((n_ + 1) * n_ // 2)

    for i in range(n_):
        ith_row_ind: int = (1 + i) * i // 2  # compute index for lower triangular matrix
        for j in range(i + 1):
            jth_row_ind: int = (1 + j) * j // 2
            sum_tmp: np.float64 = sum(L_[ith_row_ind + k] * L_[jth_row_ind + k] for k in range(j))

            if i == j:
                L_[ith_row_ind + j] = jnp.sqrt(matrix[i][i] - sum_tmp)
            else:
                L_[ith_row_ind + j] = (matrix[i][j] - sum_tmp) / L_[jth_row_ind + j]
    return L_


def sample_multivariate(mean: npt.ArrayLike, cov: npt.ArrayLike, shape: Tuple[int, int]) -> np.array:
    """Apply affine transform to sample from multivariate gaussian"""
    L: np.ndarray = np.linalg.cholesky(cov + 1e-5 * np.eye(shape[0]))
    univariate_normal: np.ndarray = np.random.normal(size=shape)
    samples: np.ndarray = np.dot(L, univariate_normal) + mean[:, np.newaxis]
    return samples


@jax.jit
def plot_gp(
        mu: np.ndarray,
        cov: np.ndarray,
        X: np.ndarray,
        X_train: np.ndarray=None,
        Y_train: np.ndarray=None,
        samples_: np.array=None,
        y_truth: np.ndarray=None,
) -> None:
    samples_ = [] if None else samples_
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')

    for i, sample in enumerate(samples_):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    if y_truth is not None:
        plt.plot(X, y_truth, label='ground truth')
    plt.legend()


def plot_gp_2d(mu, cov, X1, X2, X1_train=None, X2_train=None, Y_train=None, samples_=None):
    samples_ = [] if None else samples_
    fig_, ax_ = plt.subplots(subplot_kw={"projection": "3d"})
    ax_.zaxis.set_major_locator(LinearLocator(10))
    ax_.zaxis.set_major_formatter('{x:.02f}')

    ax_.plot_surface(X1, X2, mu.T.reshape(X1.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False, label="mean")
    ax_.plot_surface(X1, X2, samples_[:,0].T.reshape(X1.shape), cmap=cm.coolwarm, linewidth=0, antialiased=False, label="Sample")
    if X1_train is not None:
        ax_.scatter(X1_train, X2_train, Y_train)
    plt.legend()


if __name__ == '__main__':
    noise = 0.1

    xd, yd = np.mgrid[-5:5:0.5, -5:5:0.5]
    x1, x2 = np.vstack([xd.ravel(), yd.ravel()])
    x1_train = x1[::11]
    x2_train = x2[::11]
    fun = lambda x1_, x2_: np.sin(np.sqrt(x1_**2 + x2_**2))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xd, yd, fun(xd, yd), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')


    y_train = fun(x1_train, x2_train)

    mu_s, cov_s = update_posterior(
        np.swapaxes(np.array([x1, x2]), 0, 1),
        np.swapaxes(np.array([x1_train, x2_train]), 0, 1),
        y_train,
        sigma_y=noise
    )
    s = sample_multivariate(mu_s, cov_s, (len(x1), 10))
    plot_gp_2d(mu_s, cov_s, xd, yd, samples_=s)
    plt.show()
    # print(timeit.timeit("get_sq_dist(x1_test, x1_test)", globals=globals(), number=100))
    # print("Sample ", timeit.timeit("sample_multivariate(mu_s, cov_s, (n, 10))", globals=globals(), number=100))
    # print(
    #     "Numpy cholesky ",
    #     timeit.timeit("np.linalg.cholesky(cov_s + 1e-5*np.eye(n))", globals=globals(), number=100)
    # )
    # print("My cholesky ", timeit.timeit("cholesky(cov_s + 1e-5*np.eye(n))", globals=globals(), number=100))
