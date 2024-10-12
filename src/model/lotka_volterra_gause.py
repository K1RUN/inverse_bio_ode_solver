import numpy as np
import matplotlib.pyplot as plt

from bio_ode_solver.src.method.rk import rk
from bio_ode_solver.src.utils.parse_tableau import input_butcher_tableau


def lotka_volterra_gause(_, N):
    """
    Equilibrium is reached when alpha < K1/K2 and beta < K2/K1, e.g. alpha = 1, beta = 0.5, K1 = 60, K2 = 40.
    One specie displace another when beta = 1/alpha, e.g. alpha = 3, beta = 1/3, K1 = 60, K2 = 40
    """

    b1 = 1.2
    b2 = 1.4
    K1 = 60
    K2 = 40
    alpha = 1
    beta = 0.5

    xdot = np.array([b1 * N[0] * (1 - (N[0] + alpha * N[1]) / K1),
                     b2 * N[1] * (1 - (N[1] + beta * N[0]) / K2)])

    return xdot


if __name__ == "__main__":
    table = input_butcher_tableau()

    # SOLUTION
    y0 = np.array([20, 5], dtype=float)
    t, y = rk(0, 70, y0, 0.01, lotka_volterra_gause, table)

    fig, axs = plt.subplots(1, 2, figsize=(9, 5))

    # left graph
    axs[0].plot(t, y[0, :], "r", label="Specie 1")
    axs[0].plot(t, y[1, :], "b", label="Specie 2")
    axs[0].set(xlabel="Time (t)", ylabel="Population (N)")
    axs[0].legend()
    axs[0].grid()

    # right graph
    axs[1].plot(y[0, :], y[1, :])
    axs[1].set(xlabel="Specie 1", ylabel="Specie 2")
    axs[1].grid()

    # for spacing
    fig.tight_layout(pad=2.5)

    plt.show()
