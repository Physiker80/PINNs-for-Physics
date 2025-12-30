"""
Physics-Informed Loss Function

This module defines the physics-informed loss function for the PINN,
including:
- Network forward pass with input scaling
- Time derivatives using JAX automatic differentiation
- Physics residual calculation
- Vectorized functions for batch processing

The physics loss enforces the governing equation:
    m*u'' + c*u' + k*u = 0

Prerequisites:
    Run previous sections (04, 05, 06) to set up environment and network

Author: Physics-Informed Neural Networks for Physics
"""

import jax.numpy as np
from jax import grad, vmap

# Import from previous sections (assumes they have been run)
# If running standalone, uncomment and run the setup files first:
# exec(open('04_Implementation_Setup.py').read())
# exec(open('06_Neural_Network_Architecture.py').read())

# --- Network Forward Pass ---

def net_forward(net_params, t):
    """
    Neural network forward pass for a single scalar time value.
    
    Parameters:
    -----------
    net_params : list
        Network parameters (weights and biases)
    t : float
        Time value (scalar)
        
    Returns:
    --------
    output : float
        Network prediction for displacement at time t
        
    Notes:
    ------
    Input scaling to [-1, 1] improves training stability and convergence.
    """
    # Scale input to [-1, 1] range for better training
    t_scaled = 2.0 * t / t_max - 1.0
    t_input = np.array([[t_scaled]])
    output = apply_net(net_params, t_input)
    return output[0, 0]


def u_pred_fn(params, t):
    """
    Neural network prediction at time t.
    
    Parameters:
    -----------
    params : list
        Combined parameters [net_params, inverse_params]
    t : float
        Time value
        
    Returns:
    --------
    u : float
        Predicted displacement at time t
    """
    net_p, _ = params
    return net_forward(net_p, t)


# --- Time Derivatives using Automatic Differentiation ---

def u_t_fn(params, t):
    """
    First time derivative of u: du/dt
    
    Uses JAX automatic differentiation to compute exact derivative.
    
    Parameters:
    -----------
    params : list
        Combined parameters
    t : float
        Time value
        
    Returns:
    --------
    u_t : float
        First derivative du/dt (velocity)
    """
    return grad(u_pred_fn, argnums=1)(params, t)


def u_tt_fn(params, t):
    """
    Second time derivative of u: d²u/dt²
    
    Uses JAX automatic differentiation to compute exact second derivative.
    
    Parameters:
    -----------
    params : list
        Combined parameters
    t : float
        Time value
        
    Returns:
    --------
    u_tt : float
        Second derivative d²u/dt² (acceleration)
    """
    return grad(u_t_fn, argnums=1)(params, t)


# --- Physics Residual ---

def residual_net(params, t):
    """
    Compute physics residual: m*u'' + c*u' + k*u
    
    For the damped harmonic oscillator, this residual should be zero
    when the network satisfies the governing equation.
    
    Parameters:
    -----------
    params : list
        Combined parameters [net_params, inverse_params]
        where inverse_params = [log_c, log_k]
    t : float
        Time value
        
    Returns:
    --------
    residual : float
        Physics residual (should be ~0 when physics is satisfied)
        
    Notes:
    ------
    We use exp() to recover k and c from their log values, ensuring
    they remain positive during optimization.
    """
    _, inv_p = params
    c_pred = np.exp(inv_p[0])  # exp(log_c) = c
    k_pred = np.exp(inv_p[1])  # exp(log_k) = k
    
    # Get network prediction and derivatives
    u = u_pred_fn(params, t)
    u_t = u_t_fn(params, t)
    u_tt = u_tt_fn(params, t)
    
    # Physics residual: m*u'' + c*u' + k*u = 0
    return m_true * u_tt + c_pred * u_t + k_pred * u


# --- Vectorized Functions for Batch Processing ---

# Vectorize over time dimension for efficient batch computation
v_residual = vmap(residual_net, (None, 0))
v_u_pred = vmap(u_pred_fn, (None, 0))
v_u_t = vmap(u_t_fn, (None, 0))

print("\nPhysics-Informed Loss Function:")
print("  Residual function defined: m*u'' + c*u' + k*u = 0")
print("  Using JAX automatic differentiation for exact gradients")
print("  Vectorized functions created for batch processing")
print("\nDerivative computation:")
print("  First derivative (velocity):     du/dt = grad(u)")
print("  Second derivative (acceleration): d²u/dt² = grad(grad(u))")
