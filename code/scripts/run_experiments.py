"""
Refined experiment runner for ELE_ENG 395 final project.

Features:
- Configurable quick/full modes via CLI
- Reproducible seeds
- Van der Pol: MLP capacity study + Neural ODE + vector-field visualization
- Pendulum: baseline one-step model, Neural ODE, structured Lagrangian NN
- Exports figures and results JSON for report generation
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


CODE_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = CODE_ROOT / "figures"
RESULTS_PATH = CODE_ROOT / "results_summary.json"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def rk4_step_autonomous(rhs: Callable[[np.ndarray], np.ndarray], y: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * dt * k1)
    k3 = rhs(y + 0.5 * dt * k2)
    k4 = rhs(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_rk4(rhs: Callable[[np.ndarray], np.ndarray], y0: np.ndarray, steps: int, dt: float) -> np.ndarray:
    out = np.zeros((steps, len(y0)), dtype=np.float64)
    out[0] = y0
    for i in range(1, steps):
        out[i] = rk4_step_autonomous(rhs, out[i - 1], dt)
    return out


class TimeMLP(nn.Module):
    def __init__(self, output_dim: int = 1, hidden: int = 32, layers: int = 2):
        super().__init__()
        modules = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateMLP(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2):
        super().__init__()
        modules = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ODEFunc(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PotentialMLP(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        modules = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta)


def rollout_euler_single(func: nn.Module, x0: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
    xs = [x0]
    x = x0
    for _ in range(steps - 1):
        x = x + dt * func(x)
        xs.append(x)
    return torch.stack(xs, dim=0)


def rollout_euler_batch(func: nn.Module, x0_batch: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
    xs = [x0_batch]
    x = x0_batch
    for _ in range(steps - 1):
        x = x + dt * func(x)
        xs.append(x)
    return torch.stack(xs, dim=0)


def predict_accel_lnn(model: nn.Module, theta_batch: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    theta_batch = theta_batch.requires_grad_(True)
    potential = model(theta_batch)
    grad = torch.autograd.grad(
        outputs=potential,
        inputs=theta_batch,
        grad_outputs=torch.ones_like(potential),
        create_graph=create_graph,
    )[0]
    return -grad


@dataclass
class RunConfig:
    mode: str
    seed: int
    device: str
    save_plots: bool

    @property
    def quick(self) -> bool:
        return self.mode == "quick"


def run_vanderpol_experiments(results: dict, config: RunConfig, device: torch.device) -> None:
    print("\n[Van der Pol] Running experiments...")

    mu = 1.0
    dt = 0.1
    t0, t1 = 0.0, 20.0
    noise_std = 0.1
    y0 = np.array([2.0, 0.0], dtype=np.float64)

    cap_epochs = 400 if config.quick else 1500
    ode_epochs = 500 if config.quick else 1800

    def rhs(state: np.ndarray) -> np.ndarray:
        z, v = state
        return np.array([v, mu * (1.0 - z * z) * v - z], dtype=np.float64)

    t = np.arange(t0, t1 + dt, dt, dtype=np.float64)
    true_traj = rollout_rk4(rhs, y0, len(t), dt)
    z_true = true_traj[:, 0]
    v_true = true_traj[:, 1]

    z_noisy = z_true + np.random.normal(0.0, noise_std, size=z_true.shape)
    v_noisy = v_true + np.random.normal(0.0, noise_std, size=v_true.shape)

    t_t = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(1)
    z_noisy_t = torch.tensor(z_noisy, dtype=torch.float32, device=device).unsqueeze(1)

    cap_cfgs = {
        "base_2x32": {"hidden": 32, "layers": 2},
        "deep_6x32": {"hidden": 32, "layers": 6},
        "wide_2x128": {"hidden": 128, "layers": 2},
    }

    cap_preds: dict[str, np.ndarray] = {}
    cap_metrics: dict[str, dict] = {}

    for name, cfg in cap_cfgs.items():
        model = TimeMLP(output_dim=1, hidden=cfg["hidden"], layers=cfg["layers"]).to(device)
        opt = optim.Adam(model.parameters(), lr=0.01)
        start = time.time()
        for _ in range(cap_epochs):
            opt.zero_grad()
            pred = model(t_t)
            loss = nn.functional.mse_loss(pred, z_noisy_t)
            loss.backward()
            opt.step()
        elapsed = time.time() - start

        with torch.no_grad():
            pred_np = model(t_t).cpu().numpy().reshape(-1)
        cap_preds[name] = pred_np
        cap_metrics[name] = {
            "z_rmse": rmse(pred_np, z_true),
            "z_mae": mae(pred_np, z_true),
            "train_seconds": elapsed,
        }

    state_noisy_t = torch.tensor(np.column_stack((z_noisy, v_noisy)), dtype=torch.float32, device=device)
    ode_func = ODEFunc(hidden=64).to(device)
    opt_ode = optim.Adam(ode_func.parameters(), lr=0.005)
    x0_t = state_noisy_t[0]

    start = time.time()
    for _ in range(ode_epochs):
        opt_ode.zero_grad()
        pred_traj = rollout_euler_single(ode_func, x0_t, len(t), dt)
        loss = nn.functional.mse_loss(pred_traj, state_noisy_t)
        loss.backward()
        opt_ode.step()
    ode_seconds = time.time() - start

    with torch.no_grad():
        ode_pred = rollout_euler_single(ode_func, x0_t, len(t), dt).cpu().numpy()

    results["vanderpol"] = {
        "dt": dt,
        "time_horizon": [t0, t1],
        "noise_std": noise_std,
        "capacity_metrics": cap_metrics,
        "model_metrics": {
            "baseline_mlp": {
                "z_rmse": cap_metrics["base_2x32"]["z_rmse"],
                "v_rmse": rmse(np.gradient(cap_preds["base_2x32"], dt), v_true),
                "train_seconds": cap_metrics["base_2x32"]["train_seconds"],
            },
            "neural_ode": {
                "z_rmse": rmse(ode_pred[:, 0], z_true),
                "v_rmse": rmse(ode_pred[:, 1], v_true),
                "train_seconds": ode_seconds,
            },
        },
    }

    if not config.save_plots:
        return

    plt.figure(figsize=(11, 4.5))
    plt.plot(t, z_true, "k--", linewidth=2.0, label="True z(t)")
    plt.scatter(t, z_noisy, s=8, c="gray", alpha=0.35, label="Noisy observations")
    plt.plot(t, cap_preds["base_2x32"], label="Base MLP (2x32)")
    plt.plot(t, cap_preds["deep_6x32"], label="Deep MLP (6x32)")
    plt.plot(t, cap_preds["wide_2x128"], label="Wide MLP (2x128)")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.title("Van der Pol capacity study")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vdp_capacity.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(t, z_true, "k--", linewidth=2.0, label="True")
    axes[0].plot(t, cap_preds["base_2x32"], label="Baseline MLP")
    axes[0].plot(t, ode_pred[:, 0], label="Neural ODE")
    axes[0].scatter(t, z_noisy, s=8, c="gray", alpha=0.3)
    axes[0].set_title("Time series")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("z")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(z_true, v_true, "k--", linewidth=2.0, label="True")
    axes[1].plot(ode_pred[:, 0], ode_pred[:, 1], label="Neural ODE")
    axes[1].set_title("Phase space")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("v")
    axes[1].grid(True, alpha=0.25)
    axes[1].axis("equal")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vdp_model_comparison.png", dpi=180)
    plt.close(fig)

    z_grid = np.linspace(-3.0, 3.0, 20)
    v_grid = np.linspace(-3.0, 3.0, 20)
    Z, V = np.meshgrid(z_grid, v_grid)
    grid_t = torch.tensor(np.column_stack((Z.ravel(), V.ravel())), dtype=torch.float32, device=device)

    with torch.no_grad():
        vec = ode_func(grid_t).cpu().numpy()
    U = vec[:, 0].reshape(Z.shape)
    W = vec[:, 1].reshape(Z.shape)

    plt.figure(figsize=(7, 7))
    plt.quiver(Z, V, U, W, color="gray", alpha=0.65)
    sampled_ics = np.random.multivariate_normal(mean=y0, cov=[[0.1, 0.0], [0.0, 0.1]], size=3)
    for i, ic in enumerate(sampled_ics):
        ic_t = torch.tensor(ic, dtype=torch.float32, device=device)
        with torch.no_grad():
            traj_i = rollout_euler_single(ode_func, ic_t, len(t), dt).cpu().numpy()
        plt.plot(traj_i[:, 0], traj_i[:, 1], linewidth=2.0, label=f"Sampled IC {i+1}")
        plt.scatter(ic[0], ic[1], s=32)

    plt.title("Van der Pol learned vector field and nearby trajectories")
    plt.xlabel("z")
    plt.ylabel("v")
    plt.grid(True, alpha=0.25)
    plt.axis("equal")
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vdp_vector_field.png", dpi=180)
    plt.close()


def run_pendulum_experiments(results: dict, config: RunConfig, device: torch.device) -> None:
    print("\n[Pendulum] Running experiments...")

    g = 9.81
    ell = 1.0
    t_final = 10.0
    num_steps = 250 if config.quick else 400
    num_traj = 30 if config.quick else 80
    noise_std = 0.02
    dt = t_final / (num_steps - 1)
    time_grid = np.linspace(0.0, t_final, num_steps)

    baseline_epochs = 200 if config.quick else 600
    ode_epochs = 250 if config.quick else 700
    lnn_epochs = 350 if config.quick else 1200

    def rhs(state: np.ndarray) -> np.ndarray:
        theta, omega = state
        return np.array([omega, -(g / ell) * np.sin(theta)], dtype=np.float64)

    theta0 = np.deg2rad(np.random.uniform(-80.0, 80.0, size=num_traj))
    omega0 = np.deg2rad(np.random.uniform(-100.0, 100.0, size=num_traj))

    true_states = np.zeros((num_traj, num_steps, 2), dtype=np.float64)
    noisy_states = np.zeros_like(true_states)
    alpha_fd = np.zeros((num_traj, num_steps), dtype=np.float64)

    for i in range(num_traj):
        traj = rollout_rk4(rhs, np.array([theta0[i], omega0[i]], dtype=np.float64), num_steps, dt)
        true_states[i] = traj
        noisy_states[i] = traj + np.random.normal(0.0, noise_std, size=traj.shape)
        alpha_fd[i] = np.gradient(traj[:, 1], dt, edge_order=2)

    perm = np.random.permutation(num_traj)
    n_train = int(0.7 * num_traj)
    n_val = int(0.15 * num_traj)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    x_train_t = torch.tensor(noisy_states[idx_train, :-1, :].reshape(-1, 2), dtype=torch.float32, device=device)
    y_train_t = torch.tensor(noisy_states[idx_train, 1:, :].reshape(-1, 2), dtype=torch.float32, device=device)

    baseline = StateMLP(hidden=64, layers=2).to(device)
    opt_base = optim.Adam(baseline.parameters(), lr=1e-3)
    start = time.time()
    for _ in range(baseline_epochs):
        opt_base.zero_grad()
        pred = baseline(x_train_t)
        loss = nn.functional.mse_loss(pred, y_train_t)
        loss.backward()
        opt_base.step()
    base_seconds = time.time() - start

    def rollout_baseline(model: nn.Module, x0: np.ndarray, steps: int) -> np.ndarray:
        x = x0.copy()
        out = np.zeros((steps, 2), dtype=np.float64)
        out[0] = x
        for k in range(1, steps):
            xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                x = model(xt).cpu().numpy()[0]
            out[k] = x
        return out

    true_test = true_states[idx_test]
    base_pred_test = np.zeros_like(true_test)
    for j, tidx in enumerate(idx_test):
        base_pred_test[j] = rollout_baseline(baseline, true_states[tidx, 0], num_steps)

    ode_func = ODEFunc(hidden=64).to(device)
    opt_ode = optim.Adam(ode_func.parameters(), lr=2e-3)
    target_train = torch.tensor(noisy_states[idx_train], dtype=torch.float32, device=device).transpose(0, 1)
    x0_train = target_train[0]

    start = time.time()
    for _ in range(ode_epochs):
        opt_ode.zero_grad()
        pred = rollout_euler_batch(ode_func, x0_train, num_steps, dt)
        loss = nn.functional.mse_loss(pred, target_train)
        loss.backward()
        opt_ode.step()
    ode_seconds = time.time() - start

    ode_pred_test = np.zeros_like(true_test)
    for j, tidx in enumerate(idx_test):
        x0 = torch.tensor(true_states[tidx, 0], dtype=torch.float32, device=device)
        with torch.no_grad():
            ode_pred_test[j] = rollout_euler_single(ode_func, x0, num_steps, dt).cpu().numpy()

    theta_train_t = torch.tensor(true_states[idx_train, :, 0].reshape(-1, 1), dtype=torch.float32, device=device)
    alpha_train_t = torch.tensor(alpha_fd[idx_train].reshape(-1, 1), dtype=torch.float32, device=device)
    theta_val_t = torch.tensor(true_states[idx_val, :, 0].reshape(-1, 1), dtype=torch.float32, device=device)
    alpha_val_t = torch.tensor(alpha_fd[idx_val].reshape(-1, 1), dtype=torch.float32, device=device)

    lnn = PotentialMLP(hidden=64, layers=3).to(device)
    opt_lnn = optim.Adam(lnn.parameters(), lr=1e-3)

    best_state = None
    best_val = float("inf")
    wait = 0
    patience = 40 if config.quick else 80

    start = time.time()
    for _ in range(lnn_epochs):
        opt_lnn.zero_grad()
        pred_alpha = predict_accel_lnn(lnn, theta_train_t, create_graph=True)
        loss = nn.functional.mse_loss(pred_alpha, alpha_train_t)
        loss.backward()
        opt_lnn.step()

        with torch.enable_grad():
            val_pred = predict_accel_lnn(lnn, theta_val_t, create_graph=False)
            val_loss = float(nn.functional.mse_loss(val_pred, alpha_val_t).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in lnn.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        lnn.load_state_dict(best_state)
    lnn_seconds = time.time() - start

    def lnn_rhs_np(state_np: np.ndarray) -> np.ndarray:
        theta_tensor = torch.tensor([[state_np[0]]], dtype=torch.float32, device=device)
        with torch.enable_grad():
            alpha = predict_accel_lnn(lnn, theta_tensor, create_graph=False).detach().cpu().numpy()[0, 0]
        return np.array([state_np[1], alpha], dtype=np.float64)

    lnn_pred_test = np.zeros_like(true_test)
    for j, tidx in enumerate(idx_test):
        lnn_pred_test[j] = rollout_rk4(lnn_rhs_np, true_states[tidx, 0], num_steps, dt)

    def two_state_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
        return {
            "theta_rmse": rmse(pred[:, :, 0], truth[:, :, 0]),
            "omega_rmse": rmse(pred[:, :, 1], truth[:, :, 1]),
            "theta_mae": mae(pred[:, :, 0], truth[:, :, 0]),
            "omega_mae": mae(pred[:, :, 1], truth[:, :, 1]),
        }

    results["pendulum"] = {
        "dt": dt,
        "num_steps": num_steps,
        "num_trajectories": num_traj,
        "noise_std": noise_std,
        "split": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "model_metrics": {
            "baseline_mlp": {**two_state_metrics(base_pred_test, true_test), "train_seconds": base_seconds},
            "neural_ode": {**two_state_metrics(ode_pred_test, true_test), "train_seconds": ode_seconds},
            "structured_lnn": {**two_state_metrics(lnn_pred_test, true_test), "train_seconds": lnn_seconds},
        },
        "best_val_lnn_mse": best_val,
    }

    if not config.save_plots:
        return

    rep = 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(time_grid, true_test[rep, :, 0], "k--", linewidth=2.0, label="True")
    axes[0].plot(time_grid, base_pred_test[rep, :, 0], label="Baseline")
    axes[0].plot(time_grid, ode_pred_test[rep, :, 0], label="Neural ODE")
    axes[0].plot(time_grid, lnn_pred_test[rep, :, 0], label="Structured LNN")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("theta")
    axes[0].set_title("Pendulum trajectory")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(true_test[rep, :, 0], true_test[rep, :, 1], "k--", linewidth=2.0, label="True")
    axes[1].plot(base_pred_test[rep, :, 0], base_pred_test[rep, :, 1], label="Baseline")
    axes[1].plot(ode_pred_test[rep, :, 0], ode_pred_test[rep, :, 1], label="Neural ODE")
    axes[1].plot(lnn_pred_test[rep, :, 0], lnn_pred_test[rep, :, 1], label="Structured LNN")
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel("omega")
    axes[1].set_title("Pendulum phase portrait")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pendulum_model_comparison.png", dpi=180)
    plt.close(fig)


def create_overall_bar_figure(results: dict, save_plots: bool) -> None:
    if not save_plots:
        return

    labels = ["VDP Baseline", "VDP Neural ODE", "Pend Baseline", "Pend Neural ODE", "Pend LNN"]
    values = [
        results["vanderpol"]["model_metrics"]["baseline_mlp"]["z_rmse"],
        results["vanderpol"]["model_metrics"]["neural_ode"]["z_rmse"],
        results["pendulum"]["model_metrics"]["baseline_mlp"]["theta_rmse"],
        results["pendulum"]["model_metrics"]["neural_ode"]["theta_rmse"],
        results["pendulum"]["model_metrics"]["structured_lnn"]["theta_rmse"],
    ]

    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(labels, values, color=["#888888", "#4c78a8", "#888888", "#4c78a8", "#f58518"])
    plt.ylabel("RMSE on primary state")
    plt.title("Model comparison across systems")
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "overall_rmse_bar.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final project experiments")
    parser.add_argument("--mode", choices=["quick", "full"], default="full", help="quick for fast sanity run")
    parser.add_argument("--seed", type=int, default=395, help="random seed")
    parser.add_argument("--device", choices=["cpu"], default="cpu", help="compute device")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(mode=args.mode, seed=args.seed, device=args.device, save_plots=not args.no_plots)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device(config.device)

    print(f"Running experiments in {config.mode} mode...")
    results = {
        "seed": config.seed,
        "device": str(device),
        "mode": config.mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    run_vanderpol_experiments(results, config, device)
    run_pendulum_experiments(results, config, device)
    create_overall_bar_figure(results, config.save_plots)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nAll experiments completed.")
    print(f"Results JSON: {RESULTS_PATH}")
    if config.save_plots:
        print(f"Figures folder: {FIG_DIR}")


if __name__ == "__main__":
    main()
