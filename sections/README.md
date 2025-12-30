# PINN Damped Oscillator - Modular Sections

This directory contains the separated components from the `PINN_Damped_Oscillator.ipynb` notebook for better organization and modularity.

## Section Files

### Theory and Background

1. **`01_Theoretical_Background.md`**
   - Governing equation of motion for damped harmonic oscillators
   - Natural frequency and damping ratio formulas
   - Classification of damping regimes (underdamped, critically damped, overdamped)
   - Analytical solution for the underdamped case

2. **`03_PINNs_Theory.md`**
   - Physics-Informed Neural Networks (PINNs) concept
   - Loss function composition (data loss + physics loss)
   - Inverse problem formulation
   - Automatic differentiation advantages

### Visualization

3. **`02_Physical_Simulation_Visualization.py`**
   - Physics engine with Euler integration
   - Dark theme visualization using GridSpec layout
   - Main simulation view with spring, damper, and force vectors
   - Phase portrait (velocity vs position)
   - Displacement history time series
   - Animation function and GIF export capability

### Implementation

4. **`04_Implementation_Setup.py`**
   - JAX library imports with 64-bit precision enabled
   - Analytical solution function for damped vibration
   - Physical parameter definitions (mass, spring constant, damping coefficient)
   - Initial conditions setup
   - Derived quantities calculation (ωₙ, ζ, ωd)

5. **`05_Training_Data.py`**
   - Training data generation (200 points)
   - Addition of 2% Gaussian noise
   - Two-panel visualization: full view and envelope plot

6. **`06_Neural_Network_Architecture.py`**
   - Multi-Layer Perceptron (MLP) with Glorot initialization
   - Network architecture: 1 → 64 → 64 → 64 → 1
   - Trainable physical parameters (log_k, log_c)

7. **`07_Physics_Informed_Loss.py`**
   - Network forward pass with input scaling
   - First and second time derivatives using JAX automatic differentiation
   - Physics residual calculation (m*u'' + c*u' + k*u = 0)
   - Vectorized functions for efficient batch processing

8. **`08_Loss_Analysis.py`**
   - Individual loss component calculations
   - Data loss vs physics loss comparison
   - Fit quality assessment relative to noise variance

9. **`09_Training_Phase1.py`**
   - Parameter re-initialization
   - Adam optimizer setup
   - Phase 1 loss function (data fitting only, physics weight = 0)
   - Training loop with progress tracking using tqdm

## Usage

These files are designed to be:
- **Educational**: Each file focuses on a specific concept or implementation detail
- **Modular**: Files can be studied independently or combined as needed
- **Reusable**: Code can be imported and adapted for similar problems

## Original Notebook

The original complete notebook `PINN_Damped_Oscillator.ipynb` remains unchanged in the repository root.
