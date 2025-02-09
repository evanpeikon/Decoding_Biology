import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Set initial values for S, I, and R
initial_values = [990, 10, 0]

# Set parameter values
parameters = {"r_B": 3.0, "r_S": 0.01, "r_I": 0.0005, "r_R": 0.05, "r_D": 0.02}

# Create time grid
time_grid = np.linspace(0, 350, 350)  # Simulate over 350 days

# Define the infectious disease model
def infectious_disease_model(q, t, r_B, r_S, r_I, r_R, r_D):
    S, I, R = q

    dSdt = r_B + r_S * R - r_I * S * I
    dIdt = r_I * S * I - r_R * I - r_D * I
    dRdt = r_R * I - r_S * R

    return dSdt, dIdt, dRdt

# Solve the system of equations
solution = odeint(infectious_disease_model, initial_values, time_grid,
                  args=(parameters['r_B'], parameters['r_S'], parameters['r_I'], parameters['r_R'], parameters['r_D']))
S, I, R = solution.T

# Plot the results
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(time_grid, S, alpha=0.7, lw=2, label='Susceptible (S)')
ax.plot(time_grid, I, alpha=0.7, lw=2, label='Infected (I)')
ax.plot(time_grid, R, alpha=0.7, lw=2, label='Recovered (R)')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.grid(True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
plt.show()
