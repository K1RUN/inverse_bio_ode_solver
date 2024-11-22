import jax
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from numpy import typing as npt
from typing import Optional, Union, Tuple
from matplotlib.ticker import LinearLocator


class GaussianProcess:
    def __init__(self, linspace_len: int):
        self.test_len: int = linspace_len
        self.mu: np.ndarray = np.zeros((linspace_len, 1))
        self.cov: np.ndarray = np.zeros((linspace_len, linspace_len))

        self.K_ss: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None
        self.samples: Optional[np.ndarray] = None


    @staticmethod
    @jax.jit
    def get_sq_dist(vectors1: npt.ArrayLike, vectors2: npt.ArrayLike) -> npt.ArrayLike:
        """Calculate distance between each pair of given vectors"""
        sq_dist: np.ndarray = (vectors1[:, np.newaxis] - vectors2) ** 2
        return np.sum(sq_dist, axis=-1)


    @staticmethod
    def RBF_kernel(
        vectors1: npt.ArrayLike,
        vectors2: npt.ArrayLike,
        l: float = 1.0,
        sigma_f: float = 1.0,
    ):
        sq_dist = GaussianProcess.get_sq_dist(vectors1, vectors2)
        return sigma_f**2 * np.exp(-sq_dist / (2 * l**2))


    def sample_multivariate(self, sample_count: int):
        """Apply affine transform to sample from multivariate gaussian"""
        L: np.ndarray = np.linalg.cholesky(self.cov + 1e-5 * np.eye(self.test_len))
        univariate_normal: np.ndarray = np.random.normal(size=(self.test_len, sample_count))
        samples: np.ndarray = np.dot(L, univariate_normal) + self.mu  # newaxis in multidim
        self.samples = samples if self.samples is None else np.hstack((self.samples, samples))
        return samples


    def calculate_covariance(self, X_test: npt.ArrayLike):
        self.X_test = X_test
        self.K_ss = GaussianProcess.RBF_kernel(X_test, X_test) + 1e-8 * np.eye(self.test_len)


    def fit(
        self,
        X_train: npt.ArrayLike,
        Y_train: npt.ArrayLike,
        l: float = 1.0,
        sigma_f: float = 1.0,
        sigma_y: float = 1e-8,
    ):
        self.X_train = X_train
        self.Y_train = Y_train
        K: np.ndarray = (
            GaussianProcess.RBF_kernel(self.X_train, self.X_train, l, sigma_f) +
            sigma_y ** 2 * np.eye(len(self.X_train))
        )
        K_s: np.ndarray = GaussianProcess.RBF_kernel(self.X_train, self.X_test, l, sigma_f)
        K_inv: np.ndarray = np.linalg.inv(K)
        self.mu = K_s.T.dot(K_inv).dot(self.Y_train)
        self.cov = self.K_ss - K_s.T.dot(K_inv).dot(K_s)

        return self.mu, self.cov


    def predict(
        self,
        x_sample: npt.ArrayLike,
        return_std: bool = False,
    ) -> Union[npt.ArrayLike, Tuple]:
        """Use line interpolation to connect points not in a linespace"""
        y = []
        indicies: np.ndarray = np.array([], dtype=np.int64)

        for x in x_sample:
            idx = np.abs(self.X_test - x).argmin()
            indicies = np.append(indicies, idx)

            y = np.append(y, self.samples.T[-1][idx])

        if return_std:
            sigmas: np.ndarray = np.array([])
            for idx in indicies:
                sigmas = np.append(sigmas, np.diag(self.cov)[idx])
            return np.asarray(np.array(y)), sigmas

        return np.asarray(np.array(y))


    def plot_gp(
        self,
        y_truth: np.ndarray = None,
    ) -> None:
        samples_: np.ndarray = self.samples
        X: np.ndarray = self.X_test.ravel()
        mu: np.ndarray = self.mu.ravel()
        uncertainty: np.ndarray = 1.96 * np.sqrt(np.diag(self.cov))
        plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
        plt.plot(X, mu, label='Mean')

        for i, sample in enumerate(samples_.T):
            plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
        if self.X_train is not None:
            plt.plot(self.X_train, self.Y_train, 'rx')
        if y_truth is not None:
            plt.plot(X, y_truth, label='ground truth')
        plt.legend()


    def plot_gp_2d(self,
        X1: np.ndarray,
        X2: np.ndarray,
    ):
        fig_, ax_ = plt.subplots(subplot_kw={"projection": "3d"})
        ax_.zaxis.set_major_locator(LinearLocator(10))
        ax_.zaxis.set_major_formatter('{x:.02f}')

        ax_.plot_surface(
            X1, X2,
            self.mu.T.reshape(X1.shape),
            linewidth=0,
            antialiased=False,
            label="mean",
            alpha = 0.6
        )
        for sample in self.samples.T:
            ax_.plot_surface(
                X1, X2,
                sample.reshape(X1.shape),
                linewidth = 0,
                antialiased = False,
                label = "sample",
                alpha = 0.5
            )

        ax_.scatter(
            self.X_train[:,0].flatten(),
            self.X_train[:,1].flatten(),
            self.Y_train.flatten(),
            color='r'
        )
        plt.legend()


if __name__ == '__main__':
    noise = 0.0
    points = 100
    x_test = np.linspace(-5, 5, points).reshape(-1, 1)
    # X_train = np.linspace(-5, 5, points // 10).reshape(-1, 1)
    x_train = np.linspace(-5, 5, points // 10).reshape(-1, 1)
    y_train = np.cos(x_train) + noise * np.random.randn(*x_train.shape)
    gp = GaussianProcess(x_test.shape[0])
    gp.calculate_covariance(x_test)
    gp.fit(x_train, y_train, sigma_y=noise)
    gp.sample_multivariate(sample_count=1)
    print(gp.predict(np.array([[0]]), return_std=True))
    gp.plot_gp()
    plt.show()

    x_train = np.vstack((x_train, np.array([-4, -2, -1.5]).reshape(-1, 1)))
    y_train = np.cos(x_train) + noise * np.random.randn(*x_train.shape)
    gp.fit(x_train, y_train, sigma_y=noise)
    gp.sample_multivariate(sample_count=1)
    gp.plot_gp()
    plt.show()

    x_train = np.vstack((x_train, np.array([0, 2.125, 3.23, 4.24]).reshape(-1, 1)))
    y_train = np.cos(x_train) + noise * np.random.randn(*x_train.shape)
    gp.fit(x_train, y_train, sigma_y=noise)
    gp.sample_multivariate(sample_count=1)
    gp.plot_gp()
    xs = np.linspace(-5, 5, 87).reshape(-1, 1)
    res = gp.predict(xs)
    plt.plot(xs, res, label='Interpolated')
    plt.legend(loc='upper right')
    plt.show()


    noise = 0.1
    xd, yd = np.mgrid[-5:5:0.5, -5:5:0.5]
    x1, x2 = np.vstack([xd.ravel(), yd.ravel()])
    x1_train = x1[::3]
    x2_train = x2[::3]
    fun = lambda x1_, x2_: np.sin(np.sqrt(x1_**2 + x2_**2))
    gp3d = GaussianProcess(len(x1))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xd, yd, fun(xd, yd), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')


    y_train = fun(x1_train, x2_train).reshape(-1, 1)

    gp3d.calculate_covariance(np.swapaxes(np.array([x1, x2]), 0, 1))
    gp3d.fit(
        np.swapaxes(np.array([x1_train, x2_train]), 0, 1),
        y_train,
        sigma_y=noise
    )
    s = gp3d.sample_multivariate(sample_count=1)
    gp3d.plot_gp_2d(xd, yd)
    plt.show()