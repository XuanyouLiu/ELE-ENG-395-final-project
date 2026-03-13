# Math Development

This document summarizes the mathematical formulation used in the project and maps each equation to implementation choices in `code/scripts/run_experiments.py`.

## 1) Van der Pol oscillator

State:

$$\mathbf{h}(t) = \begin{bmatrix} z(t) \\ v(t) \end{bmatrix}$$

Dynamics:

$$\dot{z} = v, \qquad \dot{v} = \mu(1-z^2)v - z, \qquad \mu=1$$

### Numerical reference

Trajectory generation uses RK4 with fixed step $\Delta t$:

$$\mathbf{h}_{n+1} = \mathbf{h}_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

with standard RK4 stage terms from $\dot{\mathbf{h}} = f(\mathbf{h})$.

### Learning objectives

1. **Time-regression MLP**:

$$\hat{z}(t) = g_\phi(t), \qquad \mathcal{L}_{\text{MLP}} = \frac{1}{N}\sum_i \|g_\phi(t_i) - z_i^{\text{noisy}}\|_2^2$$

2. **Neural ODE / ResNet rollout**:

$$\hat{\mathbf{h}}_{n+1} = \hat{\mathbf{h}}_n + \Delta t \, F_\theta(\hat{\mathbf{h}}_n)$$

$$\mathcal{L}_{\text{NODE}} = \frac{1}{N}\sum_{n=0}^{N-1}\|\hat{\mathbf{h}}_n - \mathbf{h}_n^{\text{noisy}}\|_2^2$$

## 2) Simple pendulum

Equation of motion:

$$\ddot{\theta} + \frac{g}{\ell}\sin\theta = 0$$

Converted to first-order form:

$$\dot{\theta} = \omega, \qquad \dot{\omega} = -\frac{g}{\ell}\sin\theta$$

## 3) Structured Lagrangian model

For $m = \ell = 1$, the Lagrangian is:

$$L(\theta, \omega) = \frac{1}{2}\omega^2 - V(\theta)$$

A neural potential $V_W(\theta)$ is learned. Using Euler-Lagrange:

$$\ddot{\theta}_{\text{pred}} = -\frac{\mathrm{d}V_W}{\mathrm{d}\theta}$$

Training target uses finite-difference acceleration from simulated trajectories:

$$\mathcal{L}_{\text{LNN}} = \frac{1}{N}\sum_i \left(\ddot{\theta}_{\text{pred}}(\theta_i) - \ddot{\theta}^{\text{FD}}_i\right)^2$$

## 4) Metrics

Primary metrics are trajectory RMSE and MAE on held-out test trajectories:

$$\mathrm{RMSE}(a, b) = \sqrt{\frac{1}{N}\sum_i (a_i - b_i)^2}, \qquad \mathrm{MAE}(a, b) = \frac{1}{N}\sum_i |a_i - b_i|$$

For each model we report:
- Van der Pol: RMSE for $z$, $v$
- Pendulum: RMSE for $\theta$, $\omega$
- Parameter count
- Training wall-clock time
- Energy drift standard deviation (pendulum only)
