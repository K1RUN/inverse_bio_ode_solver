import numpy as np

from typing import Callable, Tuple, Any
from inverse_bio_ode_solver.src.method.rk import get_k_coefficients


def rk_one_step_adaptive(t: float, y: np.ndarray, y_star: np.ndarray, h: float, f: Callable[[Any, Any], np.ndarray],
                         tableau: dict, Atoli: float, Rtoli: float) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Perform one step of the adaptive Runge-Kutta method, adjusting the step size based on error estimates.

    This function calculates two solutions: one using the primary coefficients (b_) and one using the error-estimate
    coefficients (b_star). The difference between these two solutions provides an estimate of the local truncation
    error, which is then used to adjust the step size for the next step.

    :param t: Current time value
    :param y: Current solution vector (state at time t)
    :param y_star: Current error-estimate solution vector (state at time t using lower-order coefficients)
    :param h: Current step size
    :param f: Function that defines the system of ODEs to solve, f(t, y) -> dy/dt
    :param tableau: Dictionary containing the Runge-Kutta coefficients ('a_', 'b_', 'b_star', etc.)
    :param Atoli: Absolute tolerance level for controlling error
    :param Rtoli: Relative tolerance level for controlling error
    :return: Tuple of (next time step, next solution, next error-estimate solution, updated step size)

    Notes:
        - The local error is calculated by comparing the solutions obtained with 'b_' and 'b_star' coefficients.
        - The new step size is computed using the error and the tolerances (Atoli and Rtoli), ensuring that
          the step size remains adaptive and suitable for the desired accuracy.
    """

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


def rk_adaptive(t0: float, t_end: float, y0: np.ndarray, h_init: float,
                f: Callable[[Any, Any], np.ndarray], tableau: dict,
                Atoli: float, Rtoli: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive Runge-Kutta method:
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods

    The method updates the step size dynamically based on the estimated local error and tolerance levels,
    adjusting the step size to ensure that the error remains within the prescribed absolute and relative tolerances.

    :param t0: Initial time (start of integration)
    :param t_end: Final time (end of integration)
    :param y0: Initial condition (population quantity for each specie)
    :param h_init: Initial step size for the method
    :param f: Function that defines the system of ODEs to solve, f(t, y) -> dy/dt
    :param tableau: Dictionary containing the Runge-Kutta coefficients ('a_', 'b_', 'b_star', 'c_', etc.)
                    'b_' and 'b_star' correspond to the coefficients for the primary and error-estimate solutions
    :param Atoli: Absolute tolerance level for controlling error
    :param Rtoli: Relative tolerance level for controlling error
    :return: Tuple of time steps (t values) and corresponding solutions (y values) at each step
    """

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
