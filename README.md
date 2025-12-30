# PINNs-for-Physics
JAX-PINN: Inverse Parameter Discovery for Mechanical Systems
# JAX-PINN: Inverse Parameter Discovery for Mechanical Systems

This repository contains a robust implementation of **Physics-Informed Neural Networks (PINNs)** using **JAX** to solve data-driven inverse problems. The project demonstrates how to accurately discover unknown physical parameters from noisy measurement data for both linear and nonlinear mechanical systems.

## ✨ Key Features

*   ⚙️ **Systems Modeled:**
    *   **Damped Harmonic Oscillator:** Discovering spring stiffness ($k$) and damping coefficient ($c$).
*   ⚡ **Powered by JAX:** Utilizes JAX for high-performance automatic differentiation (`grad`, `vmap`, `jit`) to compute exact physics residuals.
*   ⚓ **Robust Training Strategy:** Implements a **Two-Phase Training** approach (Pre-training $\rightarrow$ Frozen Network) to solve **Gradient Pathology**, preventing the physics loss from overpowering the data fit.
*   ✅ **Automated Optimization:** Integrates **Optuna** for hyperparameter tuning, achieving parameter estimation errors as low as **0.1%**.
*   📊 **Physical Validation:** Includes in-depth analysis using Phase Portraits, Energy Conservation/Dissipation plots, and FFT frequency analysis.

## 🚀 Methodology

The solver treats physical parameters as trainable variables within the computational graph. To ensure convergence, the training is split into two phases:
1.  **Deep Data Fitting:** The network learns the underlying trajectory from noisy data without physics constraints.
2.  **Parameter Discovery:** The network weights are frozen, and the optimizer focuses solely on finding the physical parameters that satisfy the governing ODEs.

## 🛠️ Tech Stack

*   **Core:** Python, JAX
*   **Optimization:** Optuna
*   **Simulation & Vis:** SciPy (`odeint`), Matplotlib, NumPy
