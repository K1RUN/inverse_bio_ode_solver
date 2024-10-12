import numpy as np
import matplotlib.pyplot as plt

from bio_ode_solver.src.method.rk import rk
from bio_ode_solver.src.utils.parse_tableau import parse_butcher_tableau
from bio_ode_solver.src.model.lotka_volterra_gause import lotka_volterra_gause

prefix = 'butcher_tables/'
methods = ['rk2_ralston', 'rk_midpoint', 'rk2', 'rk4', 'rk5', 'dp8']
points = {method: {} for method in methods}

step = 1
steps = [0.001 * 2 ** (n - 1) for n in range(1, 12)]
y0 = np.array([20, 5], dtype=float)

for step in steps:
    for method in methods:
        table = parse_butcher_tableau(prefix + method)
        t_method, y_method = rk(0, 70, y0, step, lotka_volterra_gause, table)
        points[method][step] = {'t': t_method, 'y': y_method}

fig, axs = plt.subplots(2, 3)

step_colors = {}
color_index = 0

for step in points[methods[0]]:
    step_colors[step] = plt.cm.tab20(color_index)
    color_index += 1

step_labels = {}

for i, method in enumerate(methods):
    for step, data in points[method].items():
        t_points = data['t']
        y_points = data['y']

        if step not in step_labels:
            step_labels[step] = f'step={step}'

        axs[i // 3][i % 3].plot(t_points, y_points[0], color=step_colors[step])
        axs[i // 3][i % 3].plot(t_points, y_points[1], color=step_colors[step])
        axs[i // 3][i % 3].set(xlabel="Time (t)", ylabel="Population (N)")
        axs[i // 3][i % 3].grid(True)
        axs[i // 3][i % 3].set_title(f'{method}')

handles, labels = [], []
for step, color in step_colors.items():
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
    labels.append(step_labels[step])

fig.legend(handles, labels, loc='upper right')

plt.grid(True)
plt.show()
