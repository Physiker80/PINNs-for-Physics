"""
Training Phase 1: Data Fitting

This module implements Phase 1 of the training process, which focuses
exclusively on fitting the observed data without enforcing physics.

Phase 1 Strategy:
- Set physics weight to 0.0 (ignore physics loss initially)
- Train network to fit noisy measurements
- Goal: Reach noise floor (MSE ≈ noise variance)
- This provides a good initialization for Phase 2 (parameter discovery)

Prerequisites:
    Run previous sections (04-07) to set up environment, network, and loss functions

Author: Physics-Informed Neural Networks for Physics
"""

import jax.numpy as np
from jax import grad, jit
from jax.example_libraries import optimizers
from tqdm import trange

# Import from previous sections (assumes they have been run)
# If running standalone, uncomment and run the setup files first:
# exec(open('04_Implementation_Setup.py').read())
# exec(open('05_Training_Data.py').read())
# exec(open('06_Neural_Network_Architecture.py').read())
# exec(open('07_Physics_Informed_Loss.py').read())

# --- Training Configuration ---

# Hyperparameters
learning_rate = 1e-4      # Adam learning rate
n_epochs_phase1 = 60000   # Number of training epochs

# Loss weights for Phase 1
lambda_data = 1.0         # Weight for data loss
lambda_physics_phase1 = 0.0  # Weight for physics loss (set to 0 for Phase 1)
lambda_ic = 1.0           # Weight for initial conditions

# Collocation points for physics (used in Phase 2, prepared here)
N_colloc = 1000
t_colloc = np.linspace(0, t_max, N_colloc)

print("\nPhase 1 Training Configuration:")
print(f"  Epochs: {n_epochs_phase1}")
print(f"  Learning rate: {learning_rate}")
print(f"  Data loss weight: {lambda_data}")
print(f"  Physics loss weight: {lambda_physics_phase1} (DISABLED for Phase 1)")
print(f"  Initial condition weight: {lambda_ic}")
print(f"  Target: MSE ≈ {noise_level**2:.6f} (noise variance)")

# --- Re-initialize Parameters ---

# Start fresh to ensure no bad history affects training
print("\nRe-initializing network parameters...")
net_params = init_net(random.PRNGKey(42))
inverse_params = np.array([log_c_init, log_k_init])
params = [net_params, inverse_params]

# --- Setup Optimizer ---

# Adam optimizer with constant learning rate
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(params)

print("Optimizer: Adam")
print("Status: Ready to train")

# --- Define Phase 1 Loss Function ---

@jit
def loss_phase1(params):
    """
    Phase 1 loss function: focuses on data fitting only.
    
    Components:
    1. Data loss: MSE between predictions and noisy measurements
    2. Physics loss: Disabled (weight = 0)
    3. Initial condition loss: Ensures u(0) = x0, u'(0) = v0
    
    Parameters:
    -----------
    params : list
        Combined network and physical parameters
        
    Returns:
    --------
    loss : float
        Total weighted loss
    """
    # Data loss: fit the noisy measurements
    u_pred = v_u_pred(params, t_train)
    loss_data = np.mean((u_pred - x_train)**2)
    
    # Physics loss: compute but don't enforce (weight = 0)
    res = v_residual(params, t_colloc)
    loss_physics = np.mean(res**2)
    
    # Initial condition loss: enforce u(0) = x0, u'(0) = v0
    u_0 = u_pred_fn(params, 0.0)
    u_t_0 = u_t_fn(params, 0.0)
    loss_ic = (u_0 - x0)**2 + (u_t_0 - v0)**2
    
    # Combined loss (physics weight is 0 for Phase 1)
    return lambda_data * loss_data + lambda_physics_phase1 * loss_physics + lambda_ic * loss_ic


@jit
def step_phase1(i, opt_state):
    """
    Single optimization step.
    
    Parameters:
    -----------
    i : int
        Iteration number (for optimizer state)
    opt_state : 
        Optimizer state
        
    Returns:
    --------
    opt_state :
        Updated optimizer state
    """
    p = get_params(opt_state)
    g = grad(loss_phase1)(p)
    return opt_update(i, g, opt_state)


# --- Training Loop ---

# Storage for training history
k_history = []
c_history = []
loss_history = []

print("\n" + "="*60)
print("STARTING PHASE 1 TRAINING")
print("="*60)
print("Objective: Fit noisy data to noise floor")
print("Target MSE: ~0.0004")
print("="*60 + "\n")

# Progress bar
pbar = trange(n_epochs_phase1, desc="Phase 1: Deep Data Fitting")

for i in pbar:
    # Perform optimization step
    opt_state = step_phase1(i, opt_state)
    
    # Log progress every 1000 iterations
    if i % 1000 == 0:
        p = get_params(opt_state)
        current_loss = loss_phase1(p)
        
        # Calculate MSE specifically
        u_pred_curr = v_u_pred(p, t_train)
        mse = np.mean((u_pred_curr - x_train)**2)
        
        # Track parameters (even though they may not be accurate in Phase 1)
        c_curr = float(np.exp(p[1][0]))
        k_curr = float(np.exp(p[1][1]))
        
        k_history.append(k_curr)
        c_history.append(c_curr)
        loss_history.append(float(current_loss))
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{current_loss:.6f}',
            'MSE': f'{mse:.6f}',
            'Target': '0.0004'
        })

# Get final parameters
params_phase1 = get_params(opt_state)

# Final evaluation
u_pred_final = v_u_pred(params_phase1, t_train)
final_mse = np.mean((u_pred_final - x_train)**2)
target_mse = np.var(noise)

print("\n" + "="*60)
print("PHASE 1 COMPLETE")
print("="*60)
print(f"Final Data MSE: {final_mse:.6f}")
print(f"Noise variance: {target_mse:.6f}")

if final_mse < 0.001:
    print("\n✓ SUCCESS: Network has converged to the noise floor!")
    print("  The network can accurately represent the training data.")
else:
    print("\n✗ WARNING: Network did not reach noise floor.")
    print("  Consider training longer or checking network capacity.")

print("\nPhase 1 Result:")
print(f"  k estimate: {np.exp(params_phase1[1][1]):.2f} N/m (may not be accurate yet)")
print(f"  c estimate: {np.exp(params_phase1[1][0]):.2f} N·s/m (may not be accurate yet)")
print("\nNote: Physical parameters are not expected to be accurate in Phase 1.")
print("Phase 2 will discover the correct parameters using physics constraints.")
