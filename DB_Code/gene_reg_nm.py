import numpy as np
import matplotlib.pyplot as plt

def gene_regulatory_network(t, state, params):
    A, B = state
    k1, k2, K1, K2, n, gamma = params
    dA_dt = k1/(1 + (B/K2)**n) - gamma*A  # Protein A production (repressed by B) and degradation
    dB_dt = k2*(A**n)/(K1**n + A**n) - gamma*B # Protein B production (activated by A) and degradation
    return np.array([dA_dt, dB_dt])

def euler_method(f, t_span, y0, h, params):
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / h)
    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0
    for i in range(n_steps):
        y[i + 1] = y[i] + h * f(t[i], y[i], params)
    return t, y

def runge_kutta4(f, t_span, y0, h, params):
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / h)
    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0
    for i in range(n_steps):
        k1 = h * f(t[i], y[i], params)
        k2 = h * f(t[i] + 0.5*h, y[i] + 0.5*k1, params)
        k3 = h * f(t[i] + 0.5*h, y[i] + 0.5*k2, params)
        k4 = h * f(t[i] + h, y[i] + k3, params)
        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, y

# Parameters representing realistic biological values
params = [1.0,    # k1: maximum production rate of A
          1.0,    # k2: maximum production rate of B
          0.5,    # K1: activation threshold for B
          0.5,    # K2: repression threshold for A
          4,      # n: Hill coefficient (cooperativity)
          0.2]     # gamma: protein degradation rate

# Initial protein concentrations
y0 = np.array([0.1, 0.1])

# Simulate using both methods with different step sizes
h_values = [0.5, 0.1, 0.01]  # Different step sizes to compare
t_span = (0, 50)  # Time span

# Create the comparison plot
plt.figure(figsize=(15, 6))

# Euler method plot
plt.subplot(1, 2, 1)
for h in h_values:
    t, sol = euler_method(gene_regulatory_network, t_span, y0, h, params)
    plt.plot(t, sol[:,0], '-', label=f'A (h={h})')
    plt.plot(t, sol[:,1], '--', label=f'B (h={h})')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Euler Method')
plt.legend()
plt.grid(True)

# RK4 method plot
plt.subplot(1, 2, 2)
for h in h_values:
    t, sol = runge_kutta4(gene_regulatory_network, t_span, y0, h, params)
    plt.plot(t, sol[:,0], '-', label=f'A (h={h})')
    plt.plot(t, sol[:,1], '--', label=f'B (h={h})')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Runge-Kutta 4')
plt.legend()
plt.grid(True)
plt.show()
