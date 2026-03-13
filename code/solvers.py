"""Numerical solvers, rollout utilities, and evaluation metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


# ---------------------------------------------------------------------------
# RK4 integrator (numpy, for ground-truth generation and LNN evaluation)
# ---------------------------------------------------------------------------

def rk4_step(rhs: Callable[[np.ndarray], np.ndarray], y: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * dt * k1)
    k3 = rhs(y + 0.5 * dt * k2)
    k4 = rhs(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_rk4(
    rhs: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    steps: int,
    dt: float,
) -> np.ndarray:
    out = np.zeros((steps, len(y0)), dtype=np.float64)
    out[0] = y0
    for i in range(1, steps):
        out[i] = rk4_step(rhs, out[i - 1], dt)
    return out


# ---------------------------------------------------------------------------
# Euler rollouts (torch, for Neural ODE training)
# ---------------------------------------------------------------------------

def rollout_euler_single(
    func: nn.Module, x0: torch.Tensor, steps: int, dt: float,
) -> torch.Tensor:
    xs = [x0]
    x = x0
    for _ in range(steps - 1):
        x = x + dt * func(x)
        xs.append(x)
    return torch.stack(xs, dim=0)


def rollout_euler_batch(
    func: nn.Module, x0_batch: torch.Tensor, steps: int, dt: float,
) -> torch.Tensor:
    xs = [x0_batch]
    x = x0_batch
    for _ in range(steps - 1):
        x = x + dt * func(x)
        xs.append(x)
    return torch.stack(xs, dim=0)


# ---------------------------------------------------------------------------
# RK4 rollout (torch, for fair Neural ODE evaluation)
# ---------------------------------------------------------------------------

def rk4_step_torch(func: nn.Module, x: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = func(x)
    k2 = func(x + 0.5 * dt * k1)
    k3 = func(x + 0.5 * dt * k2)
    k4 = func(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_rk4_torch(
    func: nn.Module, x0: torch.Tensor, steps: int, dt: float,
) -> torch.Tensor:
    """RK4 rollout using learned vector field (single trajectory)."""
    xs = [x0]
    x = x0
    for _ in range(steps - 1):
        x = rk4_step_torch(func, x, dt)
        xs.append(x)
    return torch.stack(xs, dim=0)


# ---------------------------------------------------------------------------
# Pendulum energy helper
# ---------------------------------------------------------------------------

def pendulum_energy(theta: np.ndarray, omega: np.ndarray, g_over_l: float = 9.81) -> np.ndarray:
    """Total mechanical energy E = 0.5 * omega^2 - g/l * cos(theta)."""
    return 0.5 * omega ** 2 - g_over_l * np.cos(theta)
