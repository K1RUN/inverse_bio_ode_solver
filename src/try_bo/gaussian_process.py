import matplotlib.pyplot as plt
import jax
import numpy as np


@jax.jit
def get_sq_dist_jit(vector1, vector2):
    sq_diff = (vector1[:, np.newaxis] - vector2) ** 2
    return np.sum(sq_diff, axis=-1)


def ise_kernel(a, b, l=1.0, sigma_f=1.0):
    sq_dist = get_sq_dist_jit(a, b)
    return sigma_f**2 * np.exp(-sq_dist / (2 * l**2) )


def update_posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    K_ = ise_kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = ise_kernel(X_train, X_s, l, sigma_f)
    K_ss = ise_kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = np.linalg.inv(K_)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    return mu_s, cov_s


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples_=None):
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
    plt.legend()



if __name__ == '__main__':
    n = 1000
    noise = 0.1
    x1_test = np.linspace(-50, 50, n)
    x2_test = np.linspace(-50, 50, n)
    x3_test = np.linspace(-50, 50, n)
    x1_train = np.linspace(-50, 50, n // 10)
    x2_train = np.linspace(-50, 50, n // 10)
    x3_train = np.linspace(-50, 50, n // 10)
    y_train = np.cos(x1_train) + x2_train * x3_train * np.sin(x3_train)
    mu_s, cov_s = update_posterior(
        np.swapaxes(np.array([x1_test, x2_test, x3_test]), 0, 1),
        np.swapaxes(np.array([x1_train, x2_train, x3_train]), 0, 1),
        y_train,
        sigma_y=noise
    )
    L = np.linalg.cholesky(cov_s + 1e-5*np.eye(n))
    samples = np.dot(L, np.random.normal(size=(n,10))) + mu_s[:, np.newaxis]
    plot_gp(mu_s, cov_s, x1_test, X_train=x1_train, Y_train=y_train, samples_=samples.T)
    plt.show()
