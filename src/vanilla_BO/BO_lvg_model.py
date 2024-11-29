import numpy as np

from scipy.stats import norm
from typing import Dict, Union
from scipy.optimize import minimize

from inverse_bio_ode_solver.src.method.rk import rk
from inverse_bio_ode_solver.src.vanilla_BO.GP import GaussianProcess
from inverse_bio_ode_solver.src.utils.parse_tableau import parse_butcher_tableau
from inverse_bio_ode_solver.src.detected_values.specie_difference import detect_values
from inverse_bio_ode_solver.src.model.lotka_volterra_gause import LotkaVolterraGause


class TargetFunction:
    t_end = 20
    h = 0.01
    y0 = np.array([20, 5], dtype=float)
    table = parse_butcher_tableau('butcher_tables/rk2')
    target_function = LotkaVolterraGause()  # ground truth
    target_function.params = (0.4, 0.2, 11, 12, 12 / 11, 11 / 12)
    t, y = rk(0, t_end, y0, h, target_function.model, table)  # compute only once
    diff = detect_values(y)

    def set_initial_values(self, y0: np.array):
        self.y0 = y0

    @classmethod
    def set_gt_model_params(cls, b1_: float, b2_: float, K1_: int, K2_: int, alpha_: float, beta_: float):
        cls.target_function.params = (b1_, b2_, K1_, K2_, alpha_, beta_)

    @classmethod
    def set_table(cls, name):
        cls.table = parse_butcher_tableau('butcher_tables' + name)

    @classmethod
    def get_params(cls) -> Dict[str, Union[int, float]]:
        return {
            "b1": cls.target_function.b1,
            "b2": cls.target_function.b2,
            "K1": cls.target_function.K1,
            "K2": cls.target_function.K2,
            "alpha": cls.target_function.alpha,
            "beta": cls.target_function.beta,
        }

    @staticmethod
    def f(x, noise_lvl=0):
        evals: np.ndarray = np.array([])
        for x_i in x:
            LVG2: LotkaVolterraGause = LotkaVolterraGause()
            LVG2.params = (x_i[0], x_i[1], x_i[2], x_i[3], x_i[3] / x_i[2], x_i[2] / x_i[3])
            t, yp = rk(
                0,
                TargetFunction.t_end,
                TargetFunction.y0,
                TargetFunction.h,
                LVG2.model,
                TargetFunction.table,
            )
            search_vals = detect_values(yp)
            evals = np.append(
                evals, (
                    np.square(yp[0, :] - TargetFunction.y[0, :]).mean() +
                    np.square(yp[1, :] - TargetFunction.y[1, :]).mean() +
                    np.square(search_vals - TargetFunction.diff).mean()
                ) + np.random.randn() * noise_lvl
            )
        return evals


def expected_improvement(X, X_sample_, Y_sample_, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.max(Y_sample_)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[(sigma == 0.0).flatten()] = 0.0
    return ei


def propose_location(acquisition, X_sample_, Y_sample_, gpr, bounds_, n_restarts_=10):
    dim = X_sample_.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        return acquisition(X.reshape(-1, dim), X_sample_, Y_sample_, gpr)

    for x0 in np.random.uniform(bounds[:, 0], bounds_[:, 1], size=(n_restarts_, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x.reshape(-1, 1).T


bounds = np.array([[0.0, 1.0], [0.0, 1.0], [1, 20], [1, 20]])
noise = 0.0
X_init = np.array(
    [
        [0.56877043, 0.20439938, 8.19912888, 6.50589444],
        [0.3116978, 0.7898117, 2.84699641, 8.48214486],
        [0.51846433, 0.77328711, 6.66856083, 1.93513761],
        [0.85939717, 0.03306921, 10.63752992, 9.6237068],
        [0.94885231, 0.60550742, 7.80213811, 8.2485948],
        [0.54946039, 0.74096906, 12.77640884, 13.6627544],
        [0.90417461, 0.03475786, 6.55779809, 5.60149391],
        [0.42141947, 0.74197246, 9.9295268, 10.54709847],
        [0.38196472, 0.0288573, 6.98026887, 18.79080089],
        [0.62444852, 0.12723495, 11.34212829, 11.98793649], 
        [0.13198466, 0.65036622, 19.58171581, 4.74908577],
    ]
)

tf = TargetFunction()

b1d, b2d, K1d, K2d = np.mgrid[0.0:1.2:0.2, 0.0:1.2:0.2, 1:21:1, 1:21:1]
b1, b2, K1, K2 = np.vstack([b1d.ravel(), b2d.ravel(), K1d.ravel(), K2d.ravel()])

# b1d_init, b2d_init, K1d_init, K2d_init = np.mgrid[0.0:1.2:0.6, 0.0:1.2:0.6, 1:21:5, 1:21:5]
# b1_init, b2_init, K1_init, K2_init = np.array([b1d_init.ravel(), b2d_init.ravel(), K1d_init.ravel(), K2d_init.ravel()])
# X_init = np.array([b1_init, b2_init, K1_init, K2_init]).T
Y_init = tf.f(X_init).reshape(-1, 1)
gp = GaussianProcess(len(b1))
gp.calculate_covariance(np.swapaxes(np.array([b1, b2, K1, K2]), 0, 1))
X_sample = X_init
Y_sample = Y_init
n_iter = 20
for i in range(n_iter):
    gp.fit(X_sample, Y_sample, sigma_y=noise)
    gp.sample_multivariate(1)
    X_next = propose_location(expected_improvement, X_sample, Y_sample, gp, bounds)
    Y_next = tf.f(X_next)
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample.reshape(-1, 1), Y_next))
    print(f'{i} iteration - f({X_next[0]}) = {Y_next[0]}')

print()