"""
Experiment runner for ELE_ENG 395 final project.

Compares data-driven and physics-structured models on two nonlinear ODE systems:
  1. Van der Pol oscillator  (MLP capacity study + Neural ODE)
  2. Simple pendulum         (baseline MLP + Neural ODE + structured LNN)

New in this version:
  - Modular imports from code.models, code.solvers, code.plotting
  - Training loss curves logged and plotted
  - Parameter count per model
  - Energy conservation analysis for pendulum
  - Learned potential vs true potential comparison
  - RK4 evaluation rollout for Neural ODE (fairer comparison)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from models import (
    TimeMLP,
    StateMLP,
    ODEFunc,
    PotentialMLP,
    count_parameters,
    predict_accel_lnn,
)
from solvers import (
    rmse,
    mae,
    rollout_rk4,
    rollout_euler_single,
    rollout_euler_batch,
    rollout_rk4_torch,
    pendulum_energy,
)
from plotting import (
    plot_vdp_capacity,
    plot_vdp_model_comparison,
    plot_vdp_vector_field,
    plot_vdp_loss_curves,
    plot_pendulum_model_comparison,
    plot_pendulum_energy,
    plot_learned_potential,
    plot_pendulum_loss_curves,
    plot_overall_bar,
)


FIG_DIR = CODE_ROOT / "figures"
RESULTS_PATH = CODE_ROOT / "results_summary.json"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class RunConfig:
    mode: str
    seed: int
    device: str
    save_plots: bool

    @property
    def quick(self) -> bool:
        return self.mode == "quick"


def _log_epoch(epoch: int, total: int, loss: float, interval: int = 200) -> None:
    if (epoch + 1) % interval == 0 or epoch == 0:
        print(f"    epoch {epoch+1:>5d}/{total}  loss={loss:.6f}")


# -----------------------------------------------------------------------
# Van der Pol experiments
# -----------------------------------------------------------------------

def run_vanderpol_experiments(
    results: dict, config: RunConfig, device: torch.device,
) -> None:
    print("\n" + "=" * 60)
    print("[Van der Pol] Running experiments")
    print("=" * 60)

    mu = 1.0
    dt = 0.1
    t0, t1 = 0.0, 20.0
    noise_std = 0.1
    y0 = np.array([2.0, 0.0], dtype=np.float64)

    cap_epochs = 400 if config.quick else 1500
    ode_epochs = 500 if config.quick else 2000

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
    loss_histories: dict[str, list[float]] = {}

    for name, cfg in cap_cfgs.items():
        model = TimeMLP(output_dim=1, hidden=cfg["hidden"], layers=cfg["layers"]).to(device)
        n_params = count_parameters(model)
        print(f"\n  Training {name} ({n_params} params, {cap_epochs} epochs)")
        opt = optim.Adam(model.parameters(), lr=0.01)
        losses: list[float] = []
        start = time.time()
        for ep in range(cap_epochs):
            opt.zero_grad()
            pred = model(t_t)
            loss = nn.functional.mse_loss(pred, z_noisy_t)
            loss.backward()
            opt.step()
            lv = loss.item()
            losses.append(lv)
            _log_epoch(ep, cap_epochs, lv)
        elapsed = time.time() - start

        with torch.no_grad():
            pred_np = model(t_t).cpu().numpy().reshape(-1)
        cap_preds[name] = pred_np
        cap_metrics[name] = {
            "z_rmse": rmse(pred_np, z_true),
            "z_mae": mae(pred_np, z_true),
            "train_seconds": round(elapsed, 2),
            "param_count": n_params,
        }
        loss_histories[name] = losses
        print(f"    -> z_rmse={cap_metrics[name]['z_rmse']:.4f}  time={elapsed:.1f}s")

    # Neural ODE
    state_noisy_t = torch.tensor(
        np.column_stack((z_noisy, v_noisy)), dtype=torch.float32, device=device,
    )
    ode_func = ODEFunc(hidden=64, layers=2).to(device)
    n_params_ode = count_parameters(ode_func)
    print(f"\n  Training Neural ODE ({n_params_ode} params, {ode_epochs} epochs)")
    opt_ode = optim.Adam(ode_func.parameters(), lr=0.005)
    x0_t = state_noisy_t[0]
    ode_losses: list[float] = []

    start = time.time()
    for ep in range(ode_epochs):
        opt_ode.zero_grad()
        pred_traj = rollout_euler_single(ode_func, x0_t, len(t), dt)
        loss = nn.functional.mse_loss(pred_traj, state_noisy_t)
        loss.backward()
        opt_ode.step()
        lv = loss.item()
        ode_losses.append(lv)
        _log_epoch(ep, ode_epochs, lv)
    ode_seconds = time.time() - start
    loss_histories["Neural ODE"] = ode_losses

    with torch.no_grad():
        ode_pred_euler = rollout_euler_single(ode_func, x0_t, len(t), dt).cpu().numpy()
        ode_pred_rk4 = rollout_rk4_torch(ode_func, x0_t, len(t), dt).cpu().numpy()

    print(f"    -> z_rmse(euler)={rmse(ode_pred_euler[:, 0], z_true):.4f}  "
          f"z_rmse(rk4)={rmse(ode_pred_rk4[:, 0], z_true):.4f}  time={ode_seconds:.1f}s")

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
                "param_count": cap_metrics["base_2x32"]["param_count"],
            },
            "neural_ode": {
                "z_rmse": rmse(ode_pred_euler[:, 0], z_true),
                "v_rmse": rmse(ode_pred_euler[:, 1], v_true),
                "z_rmse_rk4": rmse(ode_pred_rk4[:, 0], z_true),
                "v_rmse_rk4": rmse(ode_pred_rk4[:, 1], v_true),
                "train_seconds": round(ode_seconds, 2),
                "param_count": n_params_ode,
            },
        },
    }

    if not config.save_plots:
        return

    plot_vdp_capacity(t, z_true, z_noisy, cap_preds, FIG_DIR)
    plot_vdp_model_comparison(t, z_true, v_true, z_noisy, cap_preds["base_2x32"],
                              ode_pred_euler, ode_pred_rk4, FIG_DIR)
    plot_vdp_vector_field(ode_func, y0, len(t), dt, device, FIG_DIR)
    plot_vdp_loss_curves(loss_histories, FIG_DIR)


# -----------------------------------------------------------------------
# Pendulum experiments
# -----------------------------------------------------------------------

def run_pendulum_experiments(
    results: dict, config: RunConfig, device: torch.device,
) -> None:
    print("\n" + "=" * 60)
    print("[Pendulum] Running experiments")
    print("=" * 60)

    g = 9.81
    ell = 1.0
    g_over_l = g / ell
    t_final = 10.0
    num_steps = 250 if config.quick else 400
    num_traj = 30 if config.quick else 80
    noise_std = 0.02
    dt = t_final / (num_steps - 1)
    time_grid = np.linspace(0.0, t_final, num_steps)

    baseline_epochs = 200 if config.quick else 800
    ode_epochs = 250 if config.quick else 900
    lnn_epochs = 350 if config.quick else 1500

    def rhs(state: np.ndarray) -> np.ndarray:
        theta, omega = state
        return np.array([omega, -g_over_l * np.sin(theta)], dtype=np.float64)

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

    # --- Baseline MLP ---
    x_train_t = torch.tensor(
        noisy_states[idx_train, :-1, :].reshape(-1, 2), dtype=torch.float32, device=device,
    )
    y_train_t = torch.tensor(
        noisy_states[idx_train, 1:, :].reshape(-1, 2), dtype=torch.float32, device=device,
    )

    baseline = StateMLP(hidden=64, layers=2).to(device)
    n_params_base = count_parameters(baseline)
    print(f"\n  Training baseline MLP ({n_params_base} params, {baseline_epochs} epochs)")
    opt_base = optim.Adam(baseline.parameters(), lr=1e-3)
    base_losses: list[float] = []

    start = time.time()
    for ep in range(baseline_epochs):
        opt_base.zero_grad()
        pred = baseline(x_train_t)
        loss = nn.functional.mse_loss(pred, y_train_t)
        loss.backward()
        opt_base.step()
        lv = loss.item()
        base_losses.append(lv)
        _log_epoch(ep, baseline_epochs, lv)
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

    # --- Neural ODE ---
    ode_func = ODEFunc(hidden=64, layers=2).to(device)
    n_params_ode = count_parameters(ode_func)
    print(f"\n  Training Neural ODE ({n_params_ode} params, {ode_epochs} epochs)")
    opt_ode = optim.Adam(ode_func.parameters(), lr=2e-3)
    target_train = torch.tensor(noisy_states[idx_train], dtype=torch.float32, device=device).transpose(0, 1)
    x0_train = target_train[0]
    ode_losses: list[float] = []

    start = time.time()
    for ep in range(ode_epochs):
        opt_ode.zero_grad()
        pred = rollout_euler_batch(ode_func, x0_train, num_steps, dt)
        loss = nn.functional.mse_loss(pred, target_train)
        loss.backward()
        opt_ode.step()
        lv = loss.item()
        ode_losses.append(lv)
        _log_epoch(ep, ode_epochs, lv)
    ode_seconds = time.time() - start

    ode_pred_test = np.zeros_like(true_test)
    for j, tidx in enumerate(idx_test):
        x0 = torch.tensor(true_states[tidx, 0], dtype=torch.float32, device=device)
        with torch.no_grad():
            ode_pred_test[j] = rollout_euler_single(ode_func, x0, num_steps, dt).cpu().numpy()

    # --- Structured LNN ---
    theta_train_t = torch.tensor(
        true_states[idx_train, :, 0].reshape(-1, 1), dtype=torch.float32, device=device,
    )
    alpha_train_t = torch.tensor(
        alpha_fd[idx_train].reshape(-1, 1), dtype=torch.float32, device=device,
    )
    theta_val_t = torch.tensor(
        true_states[idx_val, :, 0].reshape(-1, 1), dtype=torch.float32, device=device,
    )
    alpha_val_t = torch.tensor(
        alpha_fd[idx_val].reshape(-1, 1), dtype=torch.float32, device=device,
    )

    lnn = PotentialMLP(hidden=64, layers=3).to(device)
    n_params_lnn = count_parameters(lnn)
    print(f"\n  Training structured LNN ({n_params_lnn} params, {lnn_epochs} epochs)")
    opt_lnn = optim.Adam(lnn.parameters(), lr=1e-3)

    best_state = None
    best_val = float("inf")
    wait = 0
    patience = 40 if config.quick else 100
    lnn_losses: list[float] = []

    start = time.time()
    for ep in range(lnn_epochs):
        opt_lnn.zero_grad()
        pred_alpha = predict_accel_lnn(lnn, theta_train_t, create_graph=True)
        loss = nn.functional.mse_loss(pred_alpha, alpha_train_t)
        loss.backward()
        opt_lnn.step()
        lv = loss.item()
        lnn_losses.append(lv)
        _log_epoch(ep, lnn_epochs, lv)

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
                print(f"    early stop at epoch {ep+1}")
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

    # Energy conservation metric
    rep = 0
    E_true = pendulum_energy(true_test[rep, :, 0], true_test[rep, :, 1], g_over_l)
    E_lnn = pendulum_energy(lnn_pred_test[rep, :, 0], lnn_pred_test[rep, :, 1], g_over_l)
    E_ode = pendulum_energy(ode_pred_test[rep, :, 0], ode_pred_test[rep, :, 1], g_over_l)
    energy_drift_lnn = float(np.std(E_lnn - E_true))
    energy_drift_ode = float(np.std(E_ode - E_true))

    results["pendulum"] = {
        "dt": round(dt, 6),
        "num_steps": num_steps,
        "num_trajectories": num_traj,
        "noise_std": noise_std,
        "split": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "model_metrics": {
            "baseline_mlp": {
                **two_state_metrics(base_pred_test, true_test),
                "train_seconds": round(base_seconds, 2),
                "param_count": n_params_base,
            },
            "neural_ode": {
                **two_state_metrics(ode_pred_test, true_test),
                "train_seconds": round(ode_seconds, 2),
                "param_count": n_params_ode,
                "energy_drift_std": round(energy_drift_ode, 6),
            },
            "structured_lnn": {
                **two_state_metrics(lnn_pred_test, true_test),
                "train_seconds": round(lnn_seconds, 2),
                "param_count": n_params_lnn,
                "energy_drift_std": round(energy_drift_lnn, 6),
            },
        },
        "best_val_lnn_mse": best_val,
    }

    print(f"\n  Baseline MLP  theta_rmse={results['pendulum']['model_metrics']['baseline_mlp']['theta_rmse']:.4f}")
    print(f"  Neural ODE    theta_rmse={results['pendulum']['model_metrics']['neural_ode']['theta_rmse']:.4f}")
    print(f"  Struct. LNN   theta_rmse={results['pendulum']['model_metrics']['structured_lnn']['theta_rmse']:.4f}")
    print(f"  Energy drift  LNN={energy_drift_lnn:.6f}  NODE={energy_drift_ode:.6f}")

    if not config.save_plots:
        return

    loss_hists = {"Baseline MLP": base_losses, "Neural ODE": ode_losses, "Struct. LNN": lnn_losses}
    plot_pendulum_model_comparison(time_grid, true_test, base_pred_test, ode_pred_test, lnn_pred_test, rep, FIG_DIR)
    plot_pendulum_energy(time_grid, true_test, lnn_pred_test, ode_pred_test, g_over_l, rep, FIG_DIR)
    plot_learned_potential(lnn, g_over_l, device, FIG_DIR)
    plot_pendulum_loss_curves(loss_hists, FIG_DIR)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final project experiments")
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--seed", type=int, default=395)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        mode=args.mode,
        seed=args.seed,
        device=args.device,
        save_plots=not args.no_plots,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device(config.device)

    print(f"Mode: {config.mode}  |  Seed: {config.seed}  |  Device: {config.device}")
    results: dict = {
        "seed": config.seed,
        "device": str(device),
        "mode": config.mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    run_vanderpol_experiments(results, config, device)
    run_pendulum_experiments(results, config, device)

    if config.save_plots:
        plot_overall_bar(results, FIG_DIR)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("All experiments completed.")
    print(f"  Results JSON : {RESULTS_PATH}")
    if config.save_plots:
        print(f"  Figures      : {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
