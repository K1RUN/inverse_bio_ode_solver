import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

from inverse_bio_ode_solver.src.method.rk import rk
from inverse_bio_ode_solver.src.utils.parse_tableau import input_butcher_tableau


class LotkaVolterra:
    __slots__ = ('alpha', 'beta', 'gamma', 'delta')
    def __init__(self, alpha: float = 1.1, beta: float = 0.4, gamma: float = 0.4, delta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    @property
    def params(self):
        return self.alpha, self.beta, self.gamma, self.delta

    @params.setter
    def params(self, parameters: Tuple[float, float, float, float]):
        self.alpha, self.beta, self.gamma, self.delta = parameters

    def model(self, _, N: np.array):
        """
        Default Lotka Volterra model

        :param _: t not used in this model (model should look like f(t, y))
        :param N: population score (predator, prey)
        :return: model values
        """

        Ndot = np.array([self.alpha * N[0] - self.beta * N[0] * N[1], self.delta * N[0] * N[1] - self.gamma * N[1]])

        return Ndot


if __name__ == "__main__":
    table = input_butcher_tableau()

    # SOLUTION
    y0 = np.array([20, 5], dtype=float)
    lv = LotkaVolterra()
    t, y = rk(0, 70, y0, 0.01, lv.model, table)

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
