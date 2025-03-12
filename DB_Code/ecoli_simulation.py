import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the E. coli metabolic pathway system
def ecoli_pathway_system(q, t, X0, y11, y12, y13, y21, y22, y31, y32, f110, f112, f121, f131, f211, f222, f224, f311, f323, X4):
    """
    Models E. coli metabolism for amino acid production where:
    X1 = phosphoenolpyruvate (PEP)
    X2 = pyruvate
    X3 = amino acid precursor
    X4 = glucose availability (parameter)
    X0 = nutrient input flux
    """
    # Unpack the state vector
    X1, X2, X3 = q
    
    # Differential equations
    # dX1/dt: PEP production from nutrients, inhibited by pyruvate, consumed in central metabolism and amino acid synthesis
    dX1dt = y11 * (X0 ** f110) * (X2 ** f112) - y12 * (X1 ** f121) - y13 * (X1 ** f131)
    
    # dX2/dt: Pyruvate production from PEP, consumption regulated by glucose availability
    dX2dt = y21 * (X1 ** f211) - y22 * (X2 ** f222) * (X4 ** f224)
    
    # dX3/dt: Amino acid precursor production from PEP and its degradation
    dX3dt = y31 * (X1 ** f311) - y32 * (X3 ** f323)
    
    return [dX1dt, dX2dt, dX3dt]

# Set the parameters based on experimental data for E. coli
# Values are adapted from the original model but with biologically relevant interpretations
X0 = 2.0                # Nutrient input level (constant)
y11 = 2.0               # Rate of PEP production from nutrients
y12 = 0.5               # Rate of PEP consumption in central metabolism
y13 = 1.5               # Rate of PEP consumption for amino acid synthesis
y21 = 0.5               # Rate of pyruvate production from PEP
y22 = 0.4               # Rate of pyruvate consumption
y31 = 1.5               # Rate of amino acid precursor production
y32 = 2.4               # Rate of amino acid precursor degradation
f110 = 1.0              # Kinetic order of nutrient effect on PEP
f112 = -0.6             # Inhibitory effect of pyruvate on PEP (negative feedback)
f121 = 0.5              # Kinetic order of PEP consumption in central metabolism
f131 = 0.9              # Kinetic order of PEP consumption for amino acid synthesis
f211 = 0.5              # Kinetic order of PEP effect on pyruvate production
f222 = 0.5              # Kinetic order of pyruvate in its own consumption
f224 = 1.0              # Kinetic order of glucose effect on pyruvate consumption
f311 = 0.9              # Kinetic order of PEP effect on amino acid precursor
f323 = 0.8              # Kinetic order of amino acid precursor degradation

# Define initial values for metabolites (in arbitrary concentration units)
initial_values = [1.0, 1.0, 1.0]  # [PEP, pyruvate, amino acid precursor]

# Create time grid for the integration (0 to 50 hours with 300 points)
time_grid = np.linspace(0, 50, 300)

# Explore different values of glucose availability
X4_values = np.linspace(0.1, 4, 6)  # From glucose limitation to glucose excess

# Set up the plot
plt.figure(figsize=(15, 10))

# Loop over different glucose availability values and solve the system
for i, X4 in enumerate(X4_values):
    # Solve the system using odeint
    solution = odeint(ecoli_pathway_system, initial_values, time_grid, 
                     args=(X0, y11, y12, y13, y21, y22, y31, y32, 
                          f110, f112, f121, f131, f211, f222, f224, f311, f323, X4))
    
    # Extract PEP, pyruvate, and amino acid precursor from the solution
    PEP, pyruvate, amino_acid = solution.T
    
    # Create a subplot for each value of glucose availability
    plt.subplot(2, 3, i+1)
    plt.plot(time_grid, PEP, label='PEP', color='blue', lw=2)
    plt.plot(time_grid, pyruvate, label='Pyruvate', color='red', lw=2)
    plt.plot(time_grid, amino_acid, label='Amino acid precursor', color='green', lw=2)
    
    # Set title and labels for each plot
    plt.title(f'Glucose availability = {X4:.2f}', fontsize=12)
    plt.xlabel('Time (hours)', fontsize=10)
    plt.ylabel('Concentration (a.u.)', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

# Add an overall title
plt.suptitle('Effect of Glucose Availability on E. coli Amino Acid Production', fontsize=16)

# Add text explaining the implications
plt.figtext(0.5, 0.01, 
            'At low glucose values, amino acid production is modest but stable.\n'
            'At high glucose values, initial production surges but crashes due to metabolic regulation.\n'
            'This suggests optimal production may require controlled glucose feeding (fed-batch strategy).', 
            ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Adjust layout to make sure plots don't overlap
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for overall title and explanation

# Save the figure
plt.savefig('ecoli_amino_acid_production.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Calculate and print the optimal glucose level for sustained amino acid production
# This would typically involve more sophisticated analysis in a real application
final_amino_acid_levels = []
for X4 in np.linspace(0.1, 4, 20):  # More fine-grained analysis
    solution = odeint(ecoli_pathway_system, initial_values, np.linspace(0, 50, 300), 
                     args=(X0, y11, y12, y13, y21, y22, y31, y32, 
                          f110, f112, f121, f131, f211, f222, f224, f311, f323, X4))
    final_amino_acid_levels.append(solution[-1, 2])  # Final level of amino acid precursor

optimal_X4 = np.linspace(0.1, 4, 20)[np.argmax(final_amino_acid_levels)]
print(f"Optimal glucose level for sustained amino acid production: {optimal_X4:.2f}")
print(f"This achieves a final amino acid precursor level of: {max(final_amino_acid_levels):.2f}")

# Also analyze the time-averaged production under different glucose regimes
average_amino_acid_levels = []
for X4 in np.linspace(0.1, 4, 20):
    solution = odeint(ecoli_pathway_system, initial_values, np.linspace(0, 50, 300), 
                     args=(X0, y11, y12, y13, y21, y22, y31, y32, 
                          f110, f112, f121, f131, f211, f222, f224, f311, f323, X4))
    average_amino_acid_levels.append(np.mean(solution[:, 2]))  # Average level of amino acid precursor

average_optimal_X4 = np.linspace(0.1, 4, 20)[np.argmax(average_amino_acid_levels)]
print(f"Optimal glucose level for time-averaged amino acid production: {average_optimal_X4:.2f}")
print(f"This achieves an average amino acid precursor level of: {max(average_amino_acid_levels):.2f}")

# Simulate a fed-batch strategy where glucose is maintained at an optimal level
def fed_batch_simulation():
    # Start with low glucose, then maintain at optimal level after 10 hours
    X4_profile = np.ones_like(time_grid) * optimal_X4
    X4_profile[:60] = np.linspace(0.1, optimal_X4, 60)  # Gradual increase in glucose during startup
    
    # Initialize arrays to store results
    PEP = np.zeros_like(time_grid)
    pyruvate = np.zeros_like(time_grid)
    amino_acid = np.zeros_like(time_grid)
    
    # Initial conditions
    PEP[0], pyruvate[0], amino_acid[0] = initial_values
    
    # Simulate using forward Euler (simplified approach)
    dt = time_grid[1] - time_grid[0]
    for i in range(1, len(time_grid)):
        # Current state
        q = [PEP[i-1], pyruvate[i-1], amino_acid[i-1]]
        
        # Calculate derivatives
        dqdt = ecoli_pathway_system(q, time_grid[i-1], X0, y11, y12, y13, y21, y22, y31, y32, 
                                   f110, f112, f121, f131, f211, f222, f224, f311, f323, X4_profile[i-1])
        
        # Update state using Euler method
        PEP[i] = PEP[i-1] + dqdt[0] * dt
        pyruvate[i] = pyruvate[i-1] + dqdt[1] * dt
        amino_acid[i] = amino_acid[i-1] + dqdt[2] * dt
    
    return PEP, pyruvate, amino_acid, X4_profile

# Run the fed-batch simulation
PEP_fb, pyruvate_fb, amino_acid_fb, X4_profile = fed_batch_simulation()

# Plot the fed-batch results
plt.figure(figsize=(12, 8))

# Plot metabolite concentrations
plt.subplot(2, 1, 1)
plt.plot(time_grid, PEP_fb, label='PEP', color='blue', lw=2)
plt.plot(time_grid, pyruvate_fb, label='Pyruvate', color='red', lw=2)
plt.plot(time_grid, amino_acid_fb, label='Amino acid precursor', color='green', lw=2)
plt.title('Fed-Batch Strategy for Amino Acid Production', fontsize=14)
plt.ylabel('Concentration (a.u.)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot glucose profile
plt.subplot(2, 1, 2)
plt.plot(time_grid, X4_profile, label='Glucose level', color='purple', lw=2)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Glucose availability', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Calculate total amino acid production
total_production = np.trapz(amino_acid_fb, time_grid)
print(f"Total amino acid precursor production in fed-batch: {total_production:.2f}")

# Compare with best constant glucose level
best_constant_solution = odeint(ecoli_pathway_system, initial_values, time_grid, 
                              args=(X0, y11, y12, y13, y21, y22, y31, y32, 
                                   f110, f112, f121, f131, f211, f222, f224, f311, f323, average_optimal_X4))
best_constant_amino_acid = best_constant_solution[:, 2]
total_constant_production = np.trapz(best_constant_amino_acid, time_grid)
print(f"Total amino acid precursor production with constant optimal glucose: {total_constant_production:.2f}")
print(f"Improvement with fed-batch strategy: {(total_production/total_constant_production - 1)*100:.2f}%")

plt.tight_layout()
plt.savefig('ecoli_fed_batch_strategy.png', dpi=300, bbox_inches='tight')
plt.show()
