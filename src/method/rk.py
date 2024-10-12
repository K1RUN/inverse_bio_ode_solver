import math
import numpy as np
from typing import Callable


def get_k_coefficients(t: float, y: np.ndarray, h: float, f: Callable[[..., ...], np.ndarray], tableau: dict) \
        -> list[np.ndarray]:
    a_ = tableau['a_']
    c_ = tableau['c_']
    k = []
    t_n = t

    # NEED TO COPY IT; BECAUSE NUMPY MODIFIES INITIAL ARRAYS AND SOLUTION DIVERGES
    y_n = np.copy(y)
    for i in range(len(a_)):
        for j in range(len(a_[0])):
            if not math.fabs(a_[i][j] - 0) < 1e-10:
                y_n += h * a_[i][j] * k[j - 1]
        k.append(f(t_n + c_[i] * h, y_n))
        y_n = np.copy(y)
    return k


def rk_one_step(t: float, y: np.ndarray, h: float, f: Callable[[..., ...], np.ndarray], tableau: dict) \
        -> tuple[float, np.ndarray]:
    k = get_k_coefficients(t, y, h, f, tableau)
    y_n = y
    b_ = tableau['b_']

    for i in range(len(b_)):
        y_n += h * b_[i] * k[i]
    t_n = t + h

    return t_n, y_n


def rk(t0: float, t_end: float, y0: np.ndarray[float], h: float,
       f: Callable[[..., ...], np.ndarray], tableau: dict) -> tuple[np.ndarray, np.ndarray]:
    t_limit = int((t_end - t0) / h)
    t = np.zeros(t_limit)

    y = np.zeros((y0.size, t_limit))
    y[:, 0] = y0

    for step in range(t_limit - 1):
        t[step + 1], y[:, step + 1] = rk_one_step(float(t[step]), y[:, step], h, f, tableau)

    return t, y
