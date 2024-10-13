import numpy as np
import matplotlib.pyplot as plt

from inverse_bio_ode_solver.src.method.rk import rk
from inverse_bio_ode_solver.src.utils.parse_tableau import input_butcher_tableau
from inverse_bio_ode_solver.src.model.lotka_volterra_gause import lotka_volterra_gause


NOISE_AMPLITUDE = 0.2


def specie_difference(species_diff: np.ndarray) -> np.ndarray:
    """
    :param species_diff: 2d array, each row corresponds to a specie population
    :return: difference between species population
    """
    return np.abs(species_diff[0] - species_diff[1])


def detect_values(species_population: np.ndarray) -> np.ndarray:
    """
    Noise is created from Normal distribution
    Simulates detection process of difference between species population
    :param species_population: 2d array, each row corresponds to a specie population
    :return: noisy difference between species population
    """
    population_diff = specie_difference(species_population)
    rnd = np.random.normal(0, NOISE_AMPLITUDE, species_population.shape[1])
    return population_diff + rnd


if __name__ == "__main__":
    table = input_butcher_tableau()

    # SOLUTION
    y0 = np.array([20, 5], dtype=float)
    t, y = rk(0, 70, y0, 0.01, lotka_volterra_gause, table)

    diff = detect_values(y)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # left graph
    axs[0].plot(t, y[0, :], "r", label="Specie 1")
    axs[0].plot(t, y[1, :], "b", label="Specie 2")
    axs[0].set(xlabel="Time (t)", ylabel="Population (N)")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t, diff, "g", label="Specie Difference")
    axs[1].set(xlabel="Time (t)", ylabel="Population diff (N)")
    axs[1].legend()
    axs[1].grid()

    # for spacing
    fig.tight_layout(pad=4.5)

    plt.show()
