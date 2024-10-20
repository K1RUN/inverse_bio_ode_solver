"""Demonstration of the `inverse_bio_ode_solver` package and module enterpoint."""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')

from inverse_bio_ode_solver.src.method.rk import rk
from inverse_bio_ode_solver.src.utils.parse_tableau import input_butcher_tableau
from inverse_bio_ode_solver.src.model.lotka_volterra_gause import LotkaVolterraGause
from inverse_bio_ode_solver.src.detected_values.specie_difference import detect_values

if __name__ == "__main__":
    table = input_butcher_tableau()

    # SOLUTION
    y0 = np.array([20, 5], dtype=float)
    LVG: LotkaVolterraGause = LotkaVolterraGause()
    LVG.set_params(b1=1.2, b2=1.4, K1=63, K2=41, alpha=1, beta=0.5)
    t, y = rk(0, 70, y0, 0.01, LVG.model, table)

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
