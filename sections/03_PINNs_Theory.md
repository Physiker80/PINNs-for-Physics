# 2. 🧠 Physics-Informed Neural Networks (PINNs)

## 2.1 The Concept

PINNs embed physical laws (PDEs/ODEs) directly into the neural network's loss function:

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \mathcal{L}_{physics}$$

**Data Loss** (fitting observed data):
$$\mathcal{L}_{data} = \frac{1}{N}\sum_{i=1}^{N}\left(u_{NN}(t_i) - x_{obs}(t_i)\right)^2$$

**Physics Loss** (satisfying the ODE):
$$\mathcal{L}_{physics} = \frac{1}{N}\sum_{i=1}^{N}\left(m\ddot{u}_{NN} + c\dot{u}_{NN} + ku_{NN}\right)^2$$

## 2.2 Inverse Problem

In inverse problems, we don't know the physical parameters ($k$, $c$). We treat them as **trainable variables** that the network learns along with its weights!

## 2.3 Automatic Differentiation

Unlike numerical differentiation, automatic differentiation (AD) provides:
- **Exact** derivatives (no approximation errors)
- **Efficient** computation through computational graph traversal
- **Stable** gradients even for complex nested functions

JAX uses AD to compute $\dot{u}_{NN}$ and $\ddot{u}_{NN}$ exactly from the neural network's output, enabling precise enforcement of the physics constraint.
