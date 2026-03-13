# Final Project Plan: Comparing Scientific ML Methods for Nonlinear ODE Dynamics

## 1. What You Learned This Quarter (Summary)

- **Ch1–2:** Optimization (gradients, contours), ML basics (supervised/unsupervised, metrics).
- **Ch3:** ODEs, analytical solutions, Forward/Backward Euler and RK2 stability.
- **Ch4:** ResNets and backprop through residual blocks.
- **Ch5:** Van der Pol oscillator; RK4; noisy observations; MLP regression (t → z); capacity (depth/width); Neural ODE (ResNet-style rollout, vector field F_θ(h), MSE on trajectories).
- **Ch6:** Numerical integration (improper integrals, quadrature).
- **Ch7:** Simple pendulum; Lagrangian T−V; Euler–Lagrange; Lyapunov; structured Lagrangian NN (learn V_W(θ), acceleration = −dV_W/dθ); train on FD acceleration, integrate for trajectories.
- **Ch9:** 1D wave equation; FDM benchmark; PINN (u_θ(x,t), PDE residual + IC/BC losses).

## 2. Project Idea (Direction a)

**Extend** the problems from class by:

- Using **two dynamical systems**: Van der Pol (Ch5) and simple pendulum (Ch7).
- **Comparing** at least three approaches:
  - Baseline MLP (mapping time → state or one-step state → state).
  - Neural ODE / ResNet (shared F_θ, rollout, trajectory MSE).
  - Structured Lagrangian NN (pendulum only); for Van der Pol you can either skip LNN or use a simple black-box ODE model.
- Optionally: add **PINN for ODE** (learn u_θ(t) with residual u'_θ − f(u_θ)) on one system, or compare **numerical solvers** (e.g., Euler vs RK4) for data generation and stability.

This matches the instructions: *"compare different numerical solvers, and compare different machine learning approaches such as baseline neural networks, ResNets, Neural ODEs, Lagrangian/Hamiltonian Networks, PINNs whenever applicable."*

## 3. Problem Statement (for report and presentation)

- **Problem:** Learn unknown or partially known dynamics from noisy trajectory data.
- **Why it matters:** Many real systems (biology, circuits, mechanics) are modeled by ODEs; data are often noisy and sparse; physics-inspired models can improve generalization and interpretability.
- **What we do:** Generate data with RK4 for Van der Pol and pendulum; add noise; train baseline MLP, Neural ODE, and (for pendulum) Lagrangian NN; compare trajectory error, sample efficiency, and training cost.

## 4. Methods (short)

- **Data:** RK4 with fixed step; multiple initial conditions for pendulum; one or a few trajectories for Van der Pol; Gaussian noise on state.
- **Baseline MLP:** Input t (or state); output state (or next state); MSE on observed states.
- **Neural ODE:** F_θ(state) in R^2; rollout h_{n+1} = h_n + dt*F_θ(h_n); loss = MSE(predicted trajectory, target).
- **Structured LNN (pendulum):** L = (1/2)ω^2 − V_W(θ); α_pred = −dV_W/dθ; train on FD acceleration; at test time integrate θ̇=ω, ω̇=α_pred.
- **Metrics:** Trajectory RMSE (and optionally MAE, max error); training time; number of parameters.

## 5. Report Outline (2-page LaTeX, conference style)

1. **Introduction (≈0.5 page)**  
   - Learning dynamics from noisy data; scientific ML (Neural ODEs, Lagrangian NNs).  
   - Goal: compare baseline MLP, Neural ODE, and Lagrangian NN on Van der Pol and pendulum.

2. **Method (≈0.5 page)**  
   - Data generation (RK4, noise level).  
   - Model architectures (one short paragraph each).  
   - Losses and training (MSE on trajectories or on accelerations for LNN).

3. **Results (≈0.5 page)**  
   - Table: trajectory RMSE (and optionally MAE) per model and per system.  
   - One or two figures: e.g., predicted vs true trajectories; or phase portraits.  
   - Short comment on stability/sample efficiency.

4. **Discussion (≈0.5 page)**  
   - Inductive bias: Lagrangian preserves structure; Neural ODE is flexible.  
   - Limitations (noise, extrapolation, single system size).  
   - Possible extensions (PINN for ODE, more systems).

## 6. Presentation Outline (12 min)

- **0–2 min:** Title, problem (learning dynamics from data), why it matters.
- **2–4 min:** Systems: Van der Pol and pendulum; data (RK4 + noise).
- **4–7 min:** Methods: baseline MLP, Neural ODE (ResNet rollout), Lagrangian NN (pendulum).
- **7–10 min:** Results: table and plots; which method wins where and why.
- **10–12 min:** Takeaways, limitations, and future work; Q&A reminder.

Keep slides minimal: bullet points and one key equation per method; details in GitHub.

## 7. GitHub Repo Checklist (40%)

- [ ] Code: data generation, model definitions, training/eval scripts.
- [ ] Math: short derivation of dynamics and of each model (in README or a `math/` or in report).
- [ ] Clear structure (e.g., `data/`, `models/`, `scripts/`, `figures/`).
- [ ] Reproducibility: `requirements.txt`, fixed seeds, instructions to run from scratch.

## 8. Suggested Order of Work

1. Copy and adapt Ch5/Ch7 code into `code/` (data gen, Neural ODE, LNN).
2. Add a small baseline MLP that predicts state from time (or next state).
3. Run experiments; save metrics and figures.
4. Write 2-page report from outline above.
5. Build slides and record 12-min presentation.
