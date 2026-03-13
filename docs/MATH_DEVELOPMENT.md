# Math Development

This document summarizes the mathematical formulation used in the project and maps each equation to implementation choices in `code/scripts/run_experiments.py`.

## 1) Van der Pol oscillator

State:
\[
h(t) = egin{bmatrix} z(t) \ v(t) \end{bmatrix}
\]
Dynamics:
\[
\dot{z} = v, \qquad \dot{v} = \mu(1-z^2)v - z, \qquad \mu=1.
\]

### Numerical reference
Trajectory generation uses RK4 with fixed step \(\Delta t\):
\[
h_{n+1} = h_n + rac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\]
with standard RK4 stage terms from \(\dot{h}=f(h)\).

### Learning objectives

1. **Time-regression MLP**:
\[
\hat{z}(t) = g_\phi(t), \qquad
\mathcal{L}_{	ext{MLP}} = rac{1}{N}\sum_i \|g_\phi(t_i)-z_i^{	ext{noisy}}\|_2^2.
\]

2. **Neural ODE / ResNet rollout**:
\[
\hat{h}_{n+1} = \hat{h}_n + \Delta t\,F_	heta(\hat{h}_n),
\]
\[
\mathcal{L}_{	ext{NODE}} = rac{1}{N}\sum_{n=0}^{N-1}\|\hat{h}_n-h_n^{	ext{noisy}}\|_2^2.
\]

## 2) Simple pendulum

Equation of motion:
\[
\ddot{	heta} + rac{g}{\ell}\sin	heta = 0,
\]
converted to first-order form:
\[
\dot{	heta}=\omega, \qquad \dot{\omega}= -rac{g}{\ell}\sin	heta.
\]

## 3) Structured Lagrangian model

For \(m=\ell=1\), use
\[
L(	heta,\omega)=rac{1}{2}\omega^2 - V(	heta).
\]
A neural potential \(V_W(	heta)\) is learned. Using Euler-Lagrange,
\[
\ddot{	heta}_{	ext{pred}} = -rac{\mathrm{d}V_W}{\mathrm{d}	heta}.
\]

Training target uses finite-difference acceleration from simulated trajectories:
\[
\mathcal{L}_{	ext{LNN}} = rac{1}{N}\sum_i \left(\ddot{	heta}_{	ext{pred}}(	heta_i)-\ddot{	heta}^{	ext{FD}}_iight)^2.
\]

## 4) Metrics

Primary metrics are trajectory RMSE and MAE on held-out test trajectories:
\[
\mathrm{RMSE}(a,b)=\sqrt{rac{1}{N}\sum_i (a_i-b_i)^2},
\qquad
\mathrm{MAE}(a,b)=rac{1}{N}\sum_i |a_i-b_i|.
\]

For each model we report:
- Van der Pol: RMSE for \(z\), \(v\)
- Pendulum: RMSE for \(	heta\), \(\omega\)
- Training wall-clock time
