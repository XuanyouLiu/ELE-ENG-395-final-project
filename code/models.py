"""Neural network architectures for learning ODE dynamics."""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeMLP(nn.Module):
    """Maps scalar time to output state via fully connected layers."""

    def __init__(self, output_dim: int = 1, hidden: int = 32, layers: int = 2):
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateMLP(nn.Module):
    """One-step state map h_n -> h_{n+1} for autoregressive rollout."""

    def __init__(self, state_dim: int = 2, hidden: int = 64, layers: int = 2):
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(state_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, state_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ODEFunc(nn.Module):
    """Learned vector field F_theta: R^d -> R^d for Neural ODE rollout."""

    def __init__(self, state_dim: int = 2, hidden: int = 64, layers: int = 2):
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(state_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, state_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PotentialMLP(nn.Module):
    """Learns scalar potential V(theta) for structured Lagrangian dynamics."""

    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            modules += [nn.Linear(hidden, hidden), nn.Tanh()]
        modules.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_accel_lnn(
    model: PotentialMLP,
    theta_batch: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Compute predicted angular acceleration from learned potential: a = -dV/dtheta."""
    theta_batch = theta_batch.requires_grad_(True)
    potential = model(theta_batch)
    grad = torch.autograd.grad(
        outputs=potential,
        inputs=theta_batch,
        grad_outputs=torch.ones_like(potential),
        create_graph=create_graph,
    )[0]
    return -grad
