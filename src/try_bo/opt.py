import time
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize
from typing import Union, Dict
from skopt.plots import plot_convergence

from inverse_bio_ode_solver.src.method.rk import rk
from inverse_bio_ode_solver.src.utils.parse_tableau import parse_butcher_tableau
from inverse_bio_ode_solver.src.model.lotka_volterra_gause import LotkaVolterraGause
from inverse_bio_ode_solver.src.detected_values.specie_difference import detect_values


class TargetFunction:
    y0 = np.array([20, 5], dtype=float)
    table = parse_butcher_tableau('butcher_tables/rk2')
    target_function = LotkaVolterraGause()  # ground truth
    target_function.params = (1.2, 1.4, 60, 40, 3, 1/3)
    t, y = rk(0, 70, y0, 0.01, target_function.model, table)  # compute only once
    diff = detect_values(y)

    @classmethod
    def set_initial_values(cls, y0: np.array):
        cls.y0 = y0

    @classmethod
    def set_gt_model_params(cls, b1: float, b2: float, K1: int, K2: int, alpha: float, beta: float):
        cls.target_function.params = (b1, b2, K1, K2, alpha, beta)

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

    @classmethod
    def f(cls, x, noise_lvl=0):

        LVG2: LotkaVolterraGause = LotkaVolterraGause()
        LVG2.params = (x[0], x[1], x[2], x[3], x[4], 1/x[4])
        t, yp = rk(0, 70, cls.y0, 0.01, LVG2.model, cls.table)
        search_vals = detect_values(yp)
        return (np.square(yp[0, :] - cls.y[0, :]) +
                np.square(yp[1, :] - cls.y[1, :]) +
                np.square(search_vals - cls.diff)).mean(axis=0) + np.random.randn() * noise_lvl


if __name__ == '__main__':
    acquisition_func = "PI"
    start = time.time()
    tf = TargetFunction()
    res = gp_minimize(tf.f,  # the function to minimize
                      [(0.0, 2.0), (0.0, 2.0),
                       (10, 70), (10, 70), (1, 5)],  # the bounds on each dimension of x
                      acq_func=acquisition_func,  # the acquisition function
                      n_calls=90,  # the number of evaluations of f
                      n_random_starts=30,  # the number of random initialization points
                      noise=0,  # the noise level (optional)
                      random_state=1234)  # the random seed
    end = time.time()
    print("Time to minimize in seconds: ", end - start)

    print(res.x)
    print(res.fun)

    ax = plot_convergence(res)
    plt.show()

    LVGpred: LotkaVolterraGause = LotkaVolterraGause()
    LVGpred.params = (res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], 1 / res.x[4])
    tpred, ypred = rk(0, 70, tf.y0, 0.01, LVGpred.model, tf.table)

    diffpred = detect_values(ypred)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(tf.t, tf.y[0, :], "r", label="Specie 1 (GROUND TRUTH)")
    axs[0, 0].plot(tf.t, tf.y[1, :], "b", label="Specie 2 (GROUND TRUTH)")
    axs[0, 0].set(xlabel="Time (t)", ylabel="Population (N)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(tf.t, tf.diff, "g", label="Specie Difference (GROUND TRUTH)")
    axs[0, 1].set(xlabel="Time (t)", ylabel="Population diff (N)")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 0].plot(tpred, ypred[0, :], "r", label="Specie 1 (PREDICTED)")
    axs[1, 0].plot(tpred, ypred[1, :], "b", label="Specie 2 (PREDICTED)")
    axs[1, 0].set(xlabel="Time (t)", ylabel="Population (N)")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[1, 1].plot(tpred, diffpred, "g", label="Specie Difference (PREDICTED)")
    axs[1, 1].set(xlabel="Time (t)", ylabel="Population diff (N)")
    axs[1, 1].legend()
    axs[1, 1].grid()

    fig.suptitle(
        f"Bayesian optimization with {acquisition_func} acq function\n"
        f"Ground truth params: {tf.get_params()}\n"
        f"Predicted: {tuple([*res.x, 1 / res.x[-1]])}\n"
    )

    # for spacing
    fig.tight_layout(pad=3)

    plt.show()

    plt.plot(tf.t, np.abs(tf.diff - diffpred), label="absolute difference ")
    plt.title("Absolute difference")
    plt.show()
