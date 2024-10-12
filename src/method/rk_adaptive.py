import numpy as np
from typing import Callable

from bio_ode_solver.src.method.rk import get_k_coefficients


def rk_adaptive(t0: float, t_end: float, y0: np.ndarray[float], h_init: float,
                f: Callable[[..., ...], np.ndarray], tableau: dict,
                Atoli: float, Rtoli: float) -> tuple[np.ndarray, np.ndarray]:
    t_limit = int((t_end - t0) / h_init)
    t = np.zeros(t_limit)
    y = np.zeros((y0.size, t_limit))
    y_star = np.zeros((y0.size, t_limit))

    y[:, 0] = y0
    h = h_init
    for step in range(t_limit - 1):
        t[step + 1], y[:, step + 1], y_star[:, step + 1], h = rk_one_step_adaptive(float(t[step]), y[:, step],
                                                                                   y_star[:, step], h, f, tableau,
                                                                                   Atoli, Rtoli)

    return t, y


def rk_one_step_adaptive(t: float, y: np.ndarray, y_star: np.ndarray, h: float, f: Callable[[..., ...], np.ndarray],
                         tableau: dict, Atoli: float, Rtoli: float) -> tuple[float, np.ndarray, np.ndarray, float]:
    k = get_k_coefficients(t, y, h, f, tableau)
    y_n = y
    y_n_star = y_star
    b_ = tableau['b_']
    b_star = tableau['b_star']
    p, p_star = tableau['rank']

    for i in range(len(b_)):
        y_n += h * b_[i] * k[i]
        y_n_star += h * b_star[i] * k[i]
    t_n = t + h

    local_error = h * np.sum([(b_[i] - tableau['b_star'][i]) * k[i] for i in range(len(b_))])
    tol_first = Atoli + np.maximum(np.fabs(y_n[0]), np.fabs(y[0])) * Rtoli
    tol_second = Atoli + np.maximum(np.fabs(y_n[1]), np.fabs(y[1])) * Rtoli
    err = np.sqrt((1 / 2) * ((local_error / tol_first) ** 2 + (local_error / tol_second) ** 2))
    h_new = h * ((1.0 / err) ** (1.0 / (min(p, p_star) + 1)))

    return t_n, y_n, y_n_star, h_new
