# Lotka-Volterra model solving using Runge Kutta methods (predator-prey and competitive equations describing Gause’s “Struggle for Existence”) on Python
This project solves two Lotka-Volterra models: classic equations (predator-prey) 
```math
\dfrac{dN_1}{dt} = \alpha N_1 - \beta N_1 N_2 
```
```math
\dfrac{dN_2}{dt} = \delta N_1 N_2 - \gamma N_1
```
/bio_ode_solver/examples/lotka_volterra/lv.png:
<p align="center" width="50%">
  <img src="https://github.com/K1RUN/bio_ode_solver/blob/main/examples/lotka_volterra/lv.png" />
</p>


and competitive equations describing Gause’s “Struggle for Existence”. 

```math
\dfrac{dN_1}{dt} = b_1 N_1 \left(1 - \dfrac{N_1 + \alpha N_2} {K_1} \right)
```
```math
\dfrac{dN_2}{dt} = b_2 N_2 \left(1 - \dfrac{N_2 + \beta N_1} {K_2} \right)
```
/bio_ode_solver/examples/lotka_volterra/lv.png:
<p align="center" width="50%">
  <img src="https://github.com/K1RUN/bio_ode_solver/blob/main/examples/lotka_volterra_gause/displacing.png" />
</p>
No additional packages were used for computing ODE's.
Numpy is used to compute Runge-Kutta methods. 

## Configuring and running
Python>=3.9 is required for this project (Conda is suggested).

Using conda, run
```shell
conda env create -f environment.yml
```
This will configure venv for this project.
Working directory = inverse_bio_ode_solver.

To run model, run from python script in src/model and specify dp table name in stdin 
(they are stored inside butcher_table directory).