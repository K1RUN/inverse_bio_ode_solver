import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

from inverse_bio_ode_solver.src.method.rk import rk
from inverse_bio_ode_solver.src.utils.parse_tableau import input_butcher_tableau


class LotkaVolterraGause:
    __slots__ = ('b1', 'b2', 'K1', 'K2', 'alpha', 'beta')
    def __init__(
            self, b1: float = 1.2, b2: float = 1.4, K1: int = 60,
            K2: int = 40, alpha: float = 1, beta: float = 0.5
    ):
        self.b1 = b1
        self.b2 = b2
        self.K1 = K1
        self.K2 = K2
        self.alpha = alpha
        self.beta = beta

    @property
    def params(self):
        return self.b1, self.b2, self.K1, self.K2, self.alpha, self.beta

    @params.setter
    def params(self, parameters: Tuple[float, float, int, int, float, float]):
        self.b1, self.b2, self.K1, self.K2, self.alpha, self.beta = parameters

    def model(self, _, N: np.array):
        """
        Model is described in: README.md
        Equilibrium is reached when alpha < K1/K2 and beta < K2/K1, e.g. alpha = 1, beta = 0.5, K1 = 60, K2 = 40.
        One specie displace another when beta = 1/alpha, e.g. alpha = 3, beta = 1/3, K1 = 60, K2 = 40

        :param _: is not used in this model (supposed to be t)
        :param N: population score (for each specie)
        :return: model values
        """

        Ndot = np.array([self.b1 * N[0] * (1 - (N[0] + self.alpha * N[1]) / self.K1),
                         self.b2 * N[1] * (1 - (N[1] + self.beta * N[0]) / self.K2)])

        return Ndot


if __name__ == "__main__":
    table = input_butcher_tableau()

    # SOLUTION
    y0 = np.array([20, 5], dtype=float)
    lvg = LotkaVolterraGause()
    t, y = rk(0, 70, y0, 0.01, lvg.model, table)

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
