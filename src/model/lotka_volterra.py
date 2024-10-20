import numpy as np
import matplotlib.pyplot as plt

from inverse_bio_ode_solver.src.method.rk import rk
from inverse_bio_ode_solver.src.utils.parse_tableau import input_butcher_tableau


class LotkaVolterra:
    alpha = 1.1
    beta = 0.4
    gamma = 0.4
    delta = 0.1

    @classmethod
    def set_params(cls, alpha: float, beta: float, gamma: float, delta: float):
        cls.alpha = alpha
        cls.beta = beta
        cls.gamma = gamma
        cls.delta = delta

    @classmethod
    def model(cls, _, N: np.array):
        """
        Default Lotka Volterra model

        :param _: t not used in this model (model should look like f(t, y))
        :param N: population score (predator, prey)
        :return: model values
        """

        Ndot = np.array([cls.alpha * N[0] - cls.beta * N[0] * N[1], cls.delta * N[0] * N[1] - cls.gamma * N[1]])

        return Ndot


if __name__ == "__main__":
    table = input_butcher_tableau()

    # SOLUTION
    y0 = np.array([20, 5], dtype=float)
    t, y = rk(0, 70, y0, 0.01, LotkaVolterra.model, table)

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
