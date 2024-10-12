import math
import numpy as np
from typing import Callable, List, Tuple, Any


def get_k_coefficients(t: float, y: np.ndarray, h: float, f: Callable[[Any, Any], np.ndarray], tableau: dict) \
        -> List[np.ndarray]:
    """
    Return k_i coefficients given a tableau.

    Notes:
        Explicit Runge-Kutta method uses following formulas
        y_(n+1) = y_n + h * sum (i = 1..s) b_i * k_i
        where
            k_1 = f(t_n, y_n)
            k_2 = f(t_n + c_2 * h, y_n + (a_21 * k_1) * h)
            k_3 = f(t_n + c_3 * h, y_n + (a_31 * k_1 + a_32 * k_2) * h)
            ...
            k_n = f(t_n + c_s * h, y_n + (a_s1 * k_1 + a_s2 * k_2 + ... + a_(s,s-1) * k_(s-1)) * h)
        s - the number of stages, a_ij, b_i, c_i is given by tableau.

    :param t: function t parameter
    :param y: modeling unit, we have two species and two y's
    :param h: step size
    :param f: our model
    :param tableau: tableau dictionary, table parsed from butcher_tables dir
    :return: k coefficient for each y value (for each specie)
    """

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


def rk_one_step(t: float, y: np.ndarray, h: float, f: Callable[[Any, Any], np.ndarray], tableau: dict) \
        -> Tuple[float, np.ndarray]:
    """
    Do one step of Runge Kutta method
    :param t: function t parameter
    :param y: modeling unit, we have two species and two y's
    :param h: step size
    :param f: our model
    :param tableau: tableau dictionary, table parsed from butcher_tables dir
    :return: t_i and y_i
    """
    k = get_k_coefficients(t, y, h, f, tableau)
    y_n = y
    b_ = tableau['b_']

    for i in range(len(b_)):
        y_n += h * b_[i] * k[i]
    t_n = t + h

    return t_n, y_n


def rk(t0: float, t_end: float, y0: np.ndarray, h: float,
       f: Callable[[Any, Any], np.ndarray], tableau: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    :param t0: Initial t parameter (time for lvg model)
    :param t_end: Final t parameter (time for lvg model)
    :param y0: Initial y parameter (population quantity for each specie)
    :param h: step size
    :param f: our model
    :param tableau: which butcher table to use
    :return: result of modelling, t and y values
    """
    t_limit = int((t_end - t0) / h)
    t = np.zeros(t_limit)

    y = np.zeros((y0.size, t_limit))
    y[:, 0] = y0

    for step in range(t_limit - 1):
        t[step + 1], y[:, step + 1] = rk_one_step(float(t[step]), y[:, step], h, f, tableau)

    return t, y
