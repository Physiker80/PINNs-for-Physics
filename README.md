# PINNs-for-Physics
JAX-PINN: Inverse Parameter Discovery for Mechanical Systems

## ✨ Key Features

*   ⚙️ **Systems Modeled:**
    *   **Damped Harmonic Oscillator:** Discovering spring stiffness ($k$) and damping coefficient ($c$).
*   ⚡ **Powered by JAX:** Utilizes JAX for high-performance automatic differentiation (`grad`, `vmap`, `jit`) to compute exact physics residuals. 
*   ⚓ **Robust Training Strategy:** Implements a **Two-Phase Training** approach (Pre-training → Frozen Network) to solve **Gradient Pathology**, preventing the physics loss from overpowering data fitting.
*   ✅ **Automated Optimization:** Integrates **Optuna** for hyperparameter tuning, achieving parameter estimation errors as low as **0.1%**.
*   📊 **Physical Validation:** Includes in-depth analysis using Phase Portraits, Energy Conservation/Dissipation plots, and FFT frequency analysis. 

## 🔧 Optimization Strategy

This project uses a **two-level optimization approach**: 

| Level | Tool | Purpose |
|-------|------|---------|
| **Inner Loop** | **Adam Optimizer** | Trains neural network weights by minimizing the combined data + physics loss via gradient descent |
| **Outer Loop** | **Optuna** | Searches for optimal hyperparameters (learning rate, hidden layers, neurons, loss weights, etc.) |

### Why Both? 
- **Adam** efficiently navigates the high-dimensional weight space using adaptive learning rates
- **Optuna** automates the tedious process of hyperparameter tuning using Bayesian optimization (TPE sampler), eliminating manual trial-and-error

## 🚀 Methodology

The solver treats physical parameters as trainable variables within the computational graph. To ensure convergence, the training is split into two phases:
1.  **Deep Data Fitting:** The network learns the underlying trajectory from noisy data without physics constraints.
2.  **Parameter Discovery:** The network weights are frozen, and the optimizer focuses solely on finding the physical parameters that satisfy the governing ODEs.
3.  

## 🛠️ Tech Stack

*   **Core:** Python, JAX
*   **Network Training:** Adam (via Optax)
*   **Hyperparameter Optimization:** Optuna
*   **Simulation & Vis:** SciPy (`odeint`), Matplotlib, NumPy
