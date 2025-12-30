# 1. 📚 Theoretical Background

## 1.1 The Damped Harmonic Oscillator

The damped harmonic oscillator is one of the most fundamental systems in physics, describing phenomena from mechanical vibrations to electrical circuits.

**The governing equation of motion:**

$$m\ddot{x} + c\dot{x} + kx = 0$$

Where:
- $m$ = mass (kg)
- $c$ = damping coefficient (N·s/m)
- $k$ = spring stiffness (N/m)
- $x$ = displacement (m)
- $\dot{x} = \frac{dx}{dt}$ = velocity
- $\ddot{x} = \frac{d^2x}{dt^2}$ = acceleration

## 1.2 Natural Frequency and Damping Ratio

**Natural frequency** (undamped):
$$\omega_n = \sqrt{\frac{k}{m}}$$

**Damping ratio**:
$$\zeta = \frac{c}{2m\omega_n} = \frac{c}{2\sqrt{km}}$$

**Damped natural frequency**:
$$\omega_d = \omega_n\sqrt{1 - \zeta^2}$$

## 1.3 Classification of Damping

| Condition | Type | Behavior |
|-----------|------|----------|
| $\zeta < 1$ | Underdamped | Oscillatory decay |
| $\zeta = 1$ | Critically damped | Fastest non-oscillatory decay |
| $\zeta > 1$ | Overdamped | Slow exponential decay |

## 1.4 Analytical Solution (Underdamped Case: $\zeta < 1$)

For the underdamped case, the general solution is:

$$x(t) = X_0 e^{-\zeta\omega_n t} \sin(\omega_d t + \phi)$$

Where the amplitude $X_0$ and phase $\phi$ are determined by initial conditions:

$$X_0 = \sqrt{x_0^2 + \frac{(v_0 + \zeta\omega_n x_0)^2}{\omega_d^2}}$$

$$\phi = \arctan\left(\frac{\omega_d x_0}{v_0 + \zeta\omega_n x_0}\right)$$

With $x_0 = x(0)$ (initial displacement) and $v_0 = \dot{x}(0)$ (initial velocity).
