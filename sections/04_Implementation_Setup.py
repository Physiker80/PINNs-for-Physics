"""
Implementation Setup for PINN Damped Oscillator

This module sets up the implementation environment including:
- JAX library imports with 64-bit precision
- Analytical solution function for damped harmonic oscillator
- Physical parameter definitions
- Initial conditions and derived quantities

Author: Physics-Informed Neural Networks for Physics
"""

import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
import itertools
from tqdm import trange
import matplotlib.pyplot as plt

# Enable 64-bit precision for better numerical accuracy
from jax import config
config.update("jax_enable_x64", True)

print("JAX imported successfully!")

# --- Analytical Solution Function ---

def damped_vibration(m, k, c, x_0, v_0, t):
    """
    Analytical solution for underdamped harmonic oscillator.
    
    This function computes the exact solution to the damped harmonic
    oscillator equation: m*x'' + c*x' + k*x = 0
    
    Parameters:
    -----------
    m : float
        Mass (kg)
    k : float
        Spring stiffness (N/m)
    c : float
        Damping coefficient (N·s/m)
    x_0 : float
        Initial displacement (m)
    v_0 : float
        Initial velocity (m/s)
    t : array-like
        Time points (s) at which to evaluate the solution
    
    Returns:
    --------
    x : array
        Displacement at each time point
        
    Notes:
    ------
    This solution is valid for the underdamped case (ζ < 1).
    The solution has the form:
        x(t) = X₀ * exp(-ζ*ωₙ*t) * sin(ωd*t + φ)
    where:
        - ωₙ = sqrt(k/m) is the natural frequency
        - ζ = c/(2*m*ωₙ) is the damping ratio
        - ωd = ωₙ*sqrt(1 - ζ²) is the damped frequency
    """
    # Natural frequency
    wn = np.sqrt(k / m)
    
    # Damping ratio
    zeta = c / (2 * m * wn)
    
    # Damped frequency
    wd = wn * np.sqrt(1 - zeta**2)
    
    # Amplitude from initial conditions
    X0 = np.sqrt(x_0**2 + (v_0 + zeta * wn * x_0)**2 / wd**2)
    
    # Phase angle
    phi = np.arctan2(wd * x_0, v_0 + zeta * wn * x_0)
    
    return X0 * np.exp(-zeta * wn * t) * np.sin(wd * t + phi)


# --- Physical Parameter Definitions ---

# True physical parameters (to be discovered by the PINN)
m_true = 1.0      # Mass (kg) - assumed known
k_true = 100.0    # Spring stiffness (N/m) - UNKNOWN (to be learned)
c_true = 2.0      # Damping coefficient (N·s/m) - UNKNOWN (to be learned)

# Initial conditions
x0 = 1.0          # Initial displacement (m)
v0 = 0.0          # Initial velocity (m/s)

# Time domain
t_max = 2.0       # Maximum time (s)

# --- Calculate Derived Quantities ---

# Natural frequency (undamped)
omega_n = np.sqrt(k_true / m_true)

# Damping ratio
zeta = c_true / (2 * m_true * omega_n)

# Damped natural frequency
omega_d = omega_n * np.sqrt(1 - zeta**2)

# Period of oscillation
period = 2 * np.pi / omega_d

# Classify damping regime
if zeta < 1:
    damping_type = 'Underdamped'
elif zeta > 1:
    damping_type = 'Overdamped'
else:
    damping_type = 'Critically damped'

# Display physical parameters
print(f"\nPhysical Parameters:")
print(f"  Mass m = {m_true} kg")
print(f"  Spring stiffness k = {k_true} N/m (UNKNOWN - to be discovered)")
print(f"  Damping coefficient c = {c_true} N·s/m (UNKNOWN - to be discovered)")
print(f"\nInitial Conditions:")
print(f"  Initial displacement x₀ = {x0} m")
print(f"  Initial velocity v₀ = {v0} m/s")
print(f"\nDerived Quantities:")
print(f"  Natural frequency ωₙ = {omega_n:.2f} rad/s")
print(f"  Damping ratio ζ = {zeta:.3f}")
print(f"  Damped frequency ωd = {omega_d:.2f} rad/s")
print(f"  Period T = {period:.3f} s")
print(f"  System type: {damping_type}")
