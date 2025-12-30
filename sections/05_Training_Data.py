"""
Training Data Generation and Visualization

This module generates noisy training data for the PINN and visualizes it
with two panels: full view and envelope plot.

Prerequisites:
    Run 04_Implementation_Setup.py first to define physical parameters
    and the analytical solution function.

Author: Physics-Informed Neural Networks for Physics
"""

import jax.numpy as np
from jax import random
import matplotlib.pyplot as plt

# Import from previous section (assumes 04_Implementation_Setup.py has been run)
# If running standalone, uncomment and run the setup file first:
# exec(open('04_Implementation_Setup.py').read())

# --- Generate Training Data ---

# Number of training points
N_data = 200

# Time points (no normalization - use actual time values)
t_train = np.linspace(0, t_max, N_data)

# Exact solution using analytical formula
x_exact = damped_vibration(m_true, k_true, c_true, x0, v0, t_train)

# Add Gaussian noise (2% noise level for realistic measurements)
key = random.PRNGKey(42)
noise_level = 0.02
noise = noise_level * random.normal(key, x_exact.shape)
x_train = x_exact + noise

print(f"\nTraining Data Generation:")
print(f"  Number of points: {N_data}")
print(f"  Noise level: {noise_level*100:.0f}%")
print(f"  Time range: [0, {t_max}] s (not normalized)")
print(f"  Noise variance: {np.var(noise):.6f}")

# --- Visualize Training Data ---

plt.figure(figsize=(12, 4))

# Plot 1: Full view with noisy measurements
plt.subplot(1, 2, 1)
plt.plot(t_train, x_exact, 'b-', linewidth=2, label='Exact solution')
plt.scatter(t_train[::3], x_train[::3], c='red', alpha=0.5, s=20, 
            label='Noisy measurements')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Displacement x(t)', fontsize=12)
plt.title('Damped Harmonic Oscillator', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Exponential decay envelope
plt.subplot(1, 2, 2)
envelope = x0 * np.exp(-zeta * omega_n * t_train.flatten())
plt.plot(t_train, x_exact, 'b-', linewidth=1, alpha=0.7, label='Oscillation')
plt.plot(t_train, envelope, 'r--', linewidth=2, 
         label=f'Envelope: $e^{{-\\zeta\\omega_n t}}$')
plt.plot(t_train, -envelope, 'r--', linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Displacement x(t)', fontsize=12)
plt.title('Exponential Decay Envelope', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_data_visualization.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'training_data_visualization.png'")
plt.show()
