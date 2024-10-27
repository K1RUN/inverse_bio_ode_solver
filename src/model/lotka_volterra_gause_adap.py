import numpy as np
import matplotlib.pyplot as plt

from inverse_bio_ode_solver.src.method.rk_adaptive import rk_adaptive
from inverse_bio_ode_solver.src.utils.parse_tableau import input_butcher_tableau
from inverse_bio_ode_solver.src.model.lotka_volterra_gause import LotkaVolterraGause

if __name__ == "__main__":
    table = input_butcher_tableau()

    # SOLUTION
    y0 = np.array([20, 5], dtype=float)

    lvg = LotkaVolterraGause()
    t, y = rk_adaptive(
        0, 70, y0, 0.01, lvg.model, table, Atoli=1e-7, Rtoli=1e-6
    )

    fig, axs = plt.subplots(1, 2, figsize=(9, 5))

    # left graph
    axs[0].plot(t[:150], y[0, :150], "r", label="Specie 1")
    axs[0].plot(t[:150], y[1, :150], "b", label="Specie 2")
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
