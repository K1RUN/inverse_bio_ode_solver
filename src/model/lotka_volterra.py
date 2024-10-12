import numpy as np
import matplotlib.pyplot as plt

from bio_ode_solver.src.method.rk import rk
from bio_ode_solver.src.utils.parse_tableau import input_butcher_tableau


def lotka_volterra(_, N):
    alpha = 1.1
    beta = 0.4
    gamma = 0.4
    delta = 0.1

    xdot = np.array([alpha * N[0] - beta * N[0] * N[1], delta * N[0] * N[1] - gamma * N[1]])

    return xdot


if __name__ == "__main__":
    table = input_butcher_tableau()

    # SOLUTION
    y0 = np.array([20, 5], dtype=float)
    t, y = rk(0, 70, y0, 0.01, lotka_volterra, table)

    plt.subplot(1, 2, 1)
    plt.plot(t, y[0, :], "r", label="Preys")
    plt.plot(t, y[1, :], "b", label="Predators")
    plt.xlabel("Time (t)")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y[0, :], y[1, :])
    plt.xlabel("Preys")
    plt.ylabel("Predators")
    plt.grid()

    plt.show()
