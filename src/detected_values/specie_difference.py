import numpy as np
import matplotlib.pyplot as plt

from inverse_bio_ode_solver.src.method.rk import rk
from inverse_bio_ode_solver.src.utils.parse_tableau import input_butcher_tableau
from inverse_bio_ode_solver.src.model.lotka_volterra_gause import lotka_volterra_gause


NOISE_AMPLITUDE = 0.2


def specie_difference(species: np.ndarray) -> np.ndarray:
    """
    :param species: 2d array, each row corresponds to a specie population
    :return: difference between species population
    """
    return np.abs(species[0] - species[1])


def detect_values(species: np.ndarray) -> np.ndarray:
    """
    Noise is created from Normal distribution
    Simulates detection process of difference between species population
    :param species: 2d array, each row corresponds to a specie population
    :return: noisy difference between species population
    """
    diff = specie_difference(species)
    rnd = np.random.normal(0, NOISE_AMPLITUDE, species.shape[1])
    return diff + rnd


if __name__ == "__main__":
    table = input_butcher_tableau()

    # SOLUTION
    y0 = np.array([20, 5], dtype=float)
    t, y = rk(0, 70, y0, 0.01, lotka_volterra_gause, table)

    y = detect_values(y)
    fig, axs = plt.subplots(1, 1, figsize=(9, 5))

    axs.plot(t, y, "g", label="Specie Difference")
    axs.set(xlabel="Time (t)", ylabel="Population diff (N)")
    axs.legend()
    axs.grid()

    # for spacing
    fig.tight_layout(pad=2.5)

    plt.show()
