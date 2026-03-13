"""Publication-quality figure generation for all experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Van der Pol figures
# ---------------------------------------------------------------------------

def plot_vdp_capacity(
    t: np.ndarray,
    z_true: np.ndarray,
    z_noisy: np.ndarray,
    cap_preds: dict[str, np.ndarray],
    save_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, z_true, "k--", linewidth=2.0, label="True z(t)")
    ax.scatter(t, z_noisy, s=8, c="gray", alpha=0.35, label="Noisy data")
    style = {"base_2x32": ("C0", "-"), "deep_6x32": ("C1", "-."), "wide_2x128": ("C2", ":")}
    for name, pred in cap_preds.items():
        c, ls = style.get(name, ("C3", "-"))
        ax.plot(t, pred, color=c, linestyle=ls, label=name.replace("_", " "))
    ax.set(xlabel="t", ylabel="z")
    ax.set_title("Van der Pol: MLP capacity study")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "vdp_capacity.png")
    plt.close(fig)


def plot_vdp_model_comparison(
    t: np.ndarray,
    z_true: np.ndarray,
    v_true: np.ndarray,
    z_noisy: np.ndarray,
    mlp_z: np.ndarray,
    ode_pred: np.ndarray,
    ode_rk4_pred: np.ndarray,
    save_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(t, z_true, "k--", linewidth=2.0, label="True")
    axes[0].scatter(t, z_noisy, s=8, c="gray", alpha=0.3)
    axes[0].plot(t, mlp_z, label="Baseline MLP")
    axes[0].plot(t, ode_pred[:, 0], label="Neural ODE (Euler)")
    axes[0].plot(t, ode_rk4_pred[:, 0], label="Neural ODE (RK4)")
    axes[0].set(xlabel="t", ylabel="z", title="Time series")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(z_true, v_true, "k--", linewidth=2.0, label="True")
    axes[1].plot(ode_pred[:, 0], ode_pred[:, 1], label="Neural ODE (Euler)")
    axes[1].plot(ode_rk4_pred[:, 0], ode_rk4_pred[:, 1], label="Neural ODE (RK4)")
    axes[1].set(xlabel="z", ylabel="v", title="Phase space")
    axes[1].grid(True, alpha=0.25)
    axes[1].axis("equal")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, v_true, "k--", linewidth=2.0, label="True v(t)")
    axes[2].plot(t, ode_pred[:, 1], label="Neural ODE (Euler)")
    axes[2].plot(t, ode_rk4_pred[:, 1], label="Neural ODE (RK4)")
    axes[2].set(xlabel="t", ylabel="v", title="Velocity comparison")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_dir / "vdp_model_comparison.png")
    plt.close(fig)


def plot_vdp_vector_field(
    ode_func,
    y0: np.ndarray,
    t_len: int,
    dt: float,
    device,
    save_dir: Path,
) -> None:
    import torch
    from solvers import rollout_euler_single

    z_grid = np.linspace(-3.0, 3.0, 20)
    v_grid = np.linspace(-3.0, 3.0, 20)
    Z, V = np.meshgrid(z_grid, v_grid)
    grid_t = torch.tensor(
        np.column_stack((Z.ravel(), V.ravel())), dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        vec = ode_func(grid_t).cpu().numpy()
    U = vec[:, 0].reshape(Z.shape)
    W = vec[:, 1].reshape(Z.shape)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.quiver(Z, V, U, W, color="gray", alpha=0.65)

    rng = np.random.default_rng(42)
    sampled_ics = rng.multivariate_normal(mean=y0, cov=np.diag([0.1, 0.1]), size=3)
    for i, ic in enumerate(sampled_ics):
        ic_t = torch.tensor(ic, dtype=torch.float32, device=device)
        with torch.no_grad():
            traj_i = rollout_euler_single(ode_func, ic_t, t_len, dt).cpu().numpy()
        ax.plot(traj_i[:, 0], traj_i[:, 1], linewidth=2.0, label=f"IC {i+1}")
        ax.scatter(ic[0], ic[1], s=32, zorder=5)

    ax.set(xlabel="z", ylabel="v", title="Learned vector field with sampled trajectories")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "vdp_vector_field.png")
    plt.close(fig)


def plot_vdp_loss_curves(
    loss_histories: dict[str, list[float]],
    save_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, losses in loss_histories.items():
        ax.semilogy(losses, label=name)
    ax.set(xlabel="Epoch", ylabel="MSE loss (log scale)", title="Van der Pol training loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "vdp_loss_curves.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pendulum figures
# ---------------------------------------------------------------------------

def plot_pendulum_model_comparison(
    time_grid: np.ndarray,
    true_test: np.ndarray,
    base_pred: np.ndarray,
    ode_pred: np.ndarray,
    lnn_pred: np.ndarray,
    rep_idx: int,
    save_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    r = rep_idx
    axes[0].plot(time_grid, true_test[r, :, 0], "k--", linewidth=2.0, label="True")
    axes[0].plot(time_grid, base_pred[r, :, 0], label="Baseline MLP")
    axes[0].plot(time_grid, ode_pred[r, :, 0], label="Neural ODE")
    axes[0].plot(time_grid, lnn_pred[r, :, 0], label="Struct. LNN")
    axes[0].set(xlabel="t", ylabel=r"$\theta$", title="Pendulum trajectory (test)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(true_test[r, :, 0], true_test[r, :, 1], "k--", linewidth=2.0, label="True")
    axes[1].plot(base_pred[r, :, 0], base_pred[r, :, 1], label="Baseline MLP")
    axes[1].plot(ode_pred[r, :, 0], ode_pred[r, :, 1], label="Neural ODE")
    axes[1].plot(lnn_pred[r, :, 0], lnn_pred[r, :, 1], label="Struct. LNN")
    axes[1].set(xlabel=r"$\theta$", ylabel=r"$\omega$", title="Phase portrait (test)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_dir / "pendulum_model_comparison.png")
    plt.close(fig)


def plot_pendulum_energy(
    time_grid: np.ndarray,
    true_test: np.ndarray,
    lnn_pred: np.ndarray,
    ode_pred: np.ndarray,
    g_over_l: float,
    rep_idx: int,
    save_dir: Path,
) -> None:
    from solvers import pendulum_energy

    r = rep_idx
    E_true = pendulum_energy(true_test[r, :, 0], true_test[r, :, 1], g_over_l)
    E_lnn = pendulum_energy(lnn_pred[r, :, 0], lnn_pred[r, :, 1], g_over_l)
    E_ode = pendulum_energy(ode_pred[r, :, 0], ode_pred[r, :, 1], g_over_l)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time_grid, E_true, "k--", linewidth=2.0, label="True (conserved)")
    ax.plot(time_grid, E_lnn, label="Struct. LNN")
    ax.plot(time_grid, E_ode, label="Neural ODE", alpha=0.8)
    ax.set(xlabel="t", ylabel="Total energy E", title="Energy conservation comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "pendulum_energy.png")
    plt.close(fig)


def plot_learned_potential(
    lnn_model,
    g_over_l: float,
    device,
    save_dir: Path,
) -> None:
    import torch
    theta_range = np.linspace(-np.pi, np.pi, 300)
    V_true = -g_over_l * np.cos(theta_range)

    theta_t = torch.tensor(theta_range.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        V_learned = lnn_model(theta_t).cpu().numpy().reshape(-1)

    V_learned_shifted = V_learned - V_learned.mean() + V_true.mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.degrees(theta_range), V_true, "k--", linewidth=2.0, label=r"True: $-\frac{g}{\ell}\cos\theta$")
    ax.plot(np.degrees(theta_range), V_learned_shifted, "C3", linewidth=2.0, label=r"Learned $V_W(\theta)$ (shifted)")
    ax.set(xlabel=r"$\theta$ (degrees)", ylabel="Potential V", title="Learned vs. true potential energy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "pendulum_learned_potential.png")
    plt.close(fig)


def plot_pendulum_loss_curves(
    loss_histories: dict[str, list[float]],
    save_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, losses in loss_histories.items():
        ax.semilogy(losses, label=name)
    ax.set(xlabel="Epoch", ylabel="MSE loss (log scale)", title="Pendulum training loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "pendulum_loss_curves.png")
    plt.close(fig)


def plot_overall_bar(results: dict, save_dir: Path) -> None:
    labels = ["VDP Baseline", "VDP NODE", "Pend Baseline", "Pend NODE", "Pend LNN"]
    values = [
        results["vanderpol"]["model_metrics"]["baseline_mlp"]["z_rmse"],
        results["vanderpol"]["model_metrics"]["neural_ode"]["z_rmse"],
        results["pendulum"]["model_metrics"]["baseline_mlp"]["theta_rmse"],
        results["pendulum"]["model_metrics"]["neural_ode"]["theta_rmse"],
        results["pendulum"]["model_metrics"]["structured_lnn"]["theta_rmse"],
    ]
    colors = ["#666", "#4c78a8", "#666", "#4c78a8", "#f58518"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_ylabel("RMSE (primary state)")
    ax.set_title("Cross-system model comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_yscale("log")
    plt.xticks(rotation=20, ha="right")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, val * 1.15,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(save_dir / "overall_rmse_bar.png")
    plt.close(fig)
