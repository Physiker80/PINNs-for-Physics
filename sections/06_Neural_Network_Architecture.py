"""
Neural Network Architecture for PINN

This module defines the Multi-Layer Perceptron (MLP) architecture and
trainable physical parameters for the Physics-Informed Neural Network.

Architecture: 1 → 64 → 64 → 64 → 1
- Input: Time t
- Hidden layers: 3 layers with 64 neurons each
- Activation: tanh
- Output: Displacement x(t)

Trainable Parameters:
- Network weights and biases
- Physical parameters: log(k) and log(c)

Prerequisites:
    Run 04_Implementation_Setup.py first for JAX imports

Author: Physics-Informed Neural Networks for Physics
"""

import jax.numpy as np
from jax import random

# Import from previous section (assumes 04_Implementation_Setup.py has been run)
# If running standalone, uncomment and run the setup file first:
# exec(open('04_Implementation_Setup.py').read())

# --- Multi-Layer Perceptron Definition ---

def MLP(layers, activation=np.tanh):
    """
    Multi-Layer Perceptron with Glorot (Xavier) initialization.
    
    Parameters:
    -----------
    layers : list of int
        Number of neurons in each layer, including input and output.
        Example: [1, 64, 64, 64, 1] for a network with 3 hidden layers
    activation : function, optional
        Activation function to use. Default is tanh.
        
    Returns:
    --------
    init : function
        Initialization function that takes a PRNG key and returns
        initialized parameters
    apply : function
        Forward pass function that takes parameters and inputs and
        returns network output
        
    Notes:
    ------
    Glorot initialization sets weights with standard deviation:
        σ = sqrt(2 / (n_in + n_out))
    This helps prevent vanishing/exploding gradients.
    """
    def init(rng_key):
        """
        Initialize network parameters with Glorot initialization.
        
        Parameters:
        -----------
        rng_key : PRNGKey
            Random number generator key
            
        Returns:
        --------
        params : list of tuples
            List of (W, b) tuples for each layer
        """
        def init_layer(key, d_in, d_out):
            """Initialize a single layer."""
            k1, _ = random.split(key)
            glorot_stddev = np.sqrt(2.0 / (d_in + d_out))
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        
        keys = random.split(rng_key, len(layers))
        params = [init_layer(k, d_in, d_out)
                  for k, d_in, d_out in zip(keys, layers[:-1], layers[1:])]
        return params
    
    def apply(params, inputs):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        params : list of tuples
            Network parameters (weights and biases)
        inputs : array
            Input data
            
        Returns:
        --------
        outputs : array
            Network predictions
        """
        H = inputs
        # Hidden layers with activation
        for W, b in params[:-1]:
            outputs = np.dot(H, W) + b
            H = activation(outputs)
        # Output layer (no activation)
        W, b = params[-1]
        outputs = np.dot(H, W) + b
        return outputs
    
    return init, apply


# --- Network Architecture Setup ---

# Define network architecture: 1 → 64 → 64 → 64 → 1
layers = [1, 64, 64, 64, 1]
init_net, apply_net = MLP(layers)

# Initialize network parameters
net_params = init_net(random.PRNGKey(123))

# Count total parameters
total_params = sum(W.size + b.size for W, b in net_params)

print(f"\nNeural Network Architecture:")
print(f"  Architecture: {' → '.join(map(str, layers))}")
print(f"  Activation function: tanh")
print(f"  Total network parameters: {total_params}")
print(f"  Initialization: Glorot (Xavier)")

# --- Trainable Physical Parameters ---

# Initialize unknown physical parameters using log-scale
# We optimize log(k) and log(c) instead of k and c directly
# This ensures they remain positive and improves optimization stability

log_k_init = np.log(50.0)   # log(k), initial guess k=50 N/m
log_c_init = np.log(1.0)    # log(c), initial guess c=1 N·s/m

inverse_params = np.array([log_c_init, log_k_init])  # [log_c, log_k]

# Combine network weights and physical parameters
params = [net_params, inverse_params]

print(f"\nTrainable Physical Parameters:")
print(f"  Number of physical parameters: 2 (log_k and log_c)")
print(f"  Initial guess: k = {np.exp(log_k_init):.1f} N/m")
print(f"  Initial guess: c = {np.exp(log_c_init):.1f} N·s/m")
print(f"  True values:   k = {k_true:.1f} N/m (unknown to network)")
print(f"  True values:   c = {c_true:.1f} N·s/m (unknown to network)")
print(f"\nTotal trainable parameters: {total_params + 2}")
