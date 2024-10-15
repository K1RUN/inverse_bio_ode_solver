# Lotka-Volterra model solving using Runge Kutta methods (predator-prey and competitive equations describing Gause’s “Struggle for Existence”) on Python

## General
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

## Work of the computational method
The main value under study is the difference between populations in the number of individuals at each point in time. Various Runge-Kutta methods are used to numerically solve the system of equations. The main idea of the method is to calculate the solution step by step using intermediate values that help improve the accuracy of calculations.

### Mathematical formulation of the method
The Runge-Kutta method can be written as:
```math
y_{n+1} = y_{n} + h \sum_{i=1}^{s} b_i k_i
```
where _k<sub>i</sub>_ are the intermediate values, which are calculated using the following formulas:
```math
k_{1} =f(y_n)
```
```math
k_{2} =f(y_n + a_{21} k_1 h)
```
```math
k_{3} =f(y_n + (a_{31} k_1 + a_{32} k_2) h)
```
```math
...
```
```math
k_{s} =f(y_n + h \sum_{j=1}^{s-1}a_{sj} k_j)
```

Here ℎ is the time step, 𝑠 is the number of stages, and the coefficients _a<sub>ij</sub>_, _b<sub>i</sub>_, _c<sub>i</sub>_ are given by the Butcher table, which defines a specific scheme of the method.

### Implementation of the method
The _get_k_coeffients()_ function calculates the intermediate coefficients for the Runge-Kutta method. The values from the Butcher table are used for correct calculation.
```python
for i in range(len(a_)):
    for j in range(len(a_[0])):
        if not math.fabs(a_[i][j] - 0) < 1e-10:
            y_n += h * a_[i][j] * k[j - 1]
    k.append(f(t_n + c_[i] * h, y_n))
    y_n = np.copy(y)
```
The intermediate values of _k<sub>i</sub>_ depend on the previous steps and the current state of the system, which makes the Runge-Kutta method more accurate than simple methods such as the Euler method.

The _rk_one_step()_ function implements one step of the Runge-Kutta method using intermediate coefficients _k<sub>i</sub>_.
```python
for i in range(len(b_)):
    y_n += h * b_[i] * k[i]
t_n = t + h
```
Here, the population values _y<sub>n+1</sub>_ are updated using intermediate coefficients _k<sub>i</sub>_ which are calculated in the _get_k_coeffients()_ function.

The _rk()_ function calculates the population size over the entire time interval by taking steps using the Runge-Kutta method.
```python
t_limit = int((t_end - t0) / h)
...
for step in range(t_limit - 1):
    t[step + 1], y[:, step + 1] = rk_one_step(float(t[step]), y[:, step], h, f, tableau)
```
Here, the cycle is organized in time, and at each time step, the _rk_one_step()_ function is called, which updates the values of the populations. As a result, we get arrays of population numbers for each time point.

To calculate the difference between populations, the specie_difference function is used, which returns the absolute difference in numbers |_N<sub>1</sub>-N<sub>2</sub>_|.
```python
def specie_difference(species_diff: np.ndarray) -> np.ndarray:
    return np.abs(species_diff[0] - species_diff[1])  # Абсолютная разница между популяциями
```
This function helps to understand how the numbers of species differ at each time step.

To simulate random fluctuations in the number of populations, noise is added, which is generated according to a normal distribution. This is implemented in the _detect_values()_ function.
```python
def detect_values(species_population: np.ndarray) -> np.ndarray:
    population_diff = specie_difference(species_population)
    rnd = np.random.normal(0, NOISE_AMPLITUDE, species_population.shape[1])  # Генерация шума
    return population_diff + rnd
```
This code adds random noise using a normal distribution with an amplitude specified by the NOISE_AMPLITUDE constant. Noise simulates external influences on populations that may be caused by environmental changes, random events, or other factors, such as miscalculations in calculations.

## Result
The graph below shows the difference in population size between species:
/bio_ode_solver/examples/detected_values/dv.png:
<p align="center" width="50%">
  <img src="https://github.com/K1RUN/inverse_bio_ode_solver/blob/main/examples/detected_values/dv.png" />
</p>

## Configuring and running
Python>=3.9 is required for this project (Conda is suggested).

Using conda, run
```shell
conda env create -f environment.yml
```
This will configure venv for this project.
Working directory = inverse_bio_ode_solver.

The butcher_tables directory contains files with coefficients a<sub>ij</sub>, b<sub>i</sub>, c<sub>i</sub>. The [a<sub>ij</sub>] matrix is called the Runge–Kutta matrix, while b<sub>i</sub> and c<sub>i</sub> are known as weights and nodes. These data are usually arranged in a mnemonic device, known as a Butcher tableau.

To run model, run from python script in src/model and specify dp table name in stdin 
(they are stored inside butcher_table directory). The data of the selected table will be parsed using src/utils/parse_tableau.py

To import PyCharm config copy all files from PyCharmConfig dir to projects .idea
