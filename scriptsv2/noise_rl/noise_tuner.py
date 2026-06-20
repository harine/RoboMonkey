"""SAC tuner for the noise-conditioned search policy's ``tcont_context``.

Mirrors the timestep head in tmrl's SAC actor (``tmrl/tmrl/models/sac_agent.py``):
one extra "action" dimension that selects the obs-noise level fed into the
frozen flow / diffusion policy. The base policy is never updated; only the
tuner's actor + critics are trained.

Pieces:
  * ``NoiseTunerActor``        — π(tcont | obs)
  * ``NoiseTunerCritic``       — Q(obs, tcont) (twin heads)
  * ``NoiseTunerAgent``        — SAC update: actor / critic / α losses + Polyak
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
ACTION_EPS = 1e-6


def _mlp(sizes, last_activation=False, activation=nn.Mish):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


class NoiseTunerActor(nn.Module):
    """Squashed Gaussian over tcont ∈ [0, 1].

    Sampling pipeline:
        u ~ Normal(mu, exp(log_std))
        z = tanh(u)                              ∈ [-1, 1]
        tcont = 0.5 * (z + 1)                    ∈ [0,  1]

    The standard SAC tanh log-prob correction is applied; the 0.5 rescale
    adds a constant log(2) we drop from the gradient.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 3) -> None:
        super().__init__()
        sizes = [obs_dim] + [hidden_dim] * n_layers
        self.trunk = _mlp(sizes, last_activation=True)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = torch.tanh(self.log_std_head(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (tcont in [0,1], log_prob, mean_tcont) all of shape (B, 1)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()
        z = torch.tanh(u)
        log_prob = normal.log_prob(u) - torch.log(1 - z.pow(2) + ACTION_EPS)
        tcont = 0.5 * (z + 1.0)
        mean_tcont = 0.5 * (torch.tanh(mu) + 1.0)
        return tcont, log_prob, mean_tcont


class NoiseTunerCritic(nn.Module):
    """Twin Q-heads operating on ``cat(obs, tcont)``."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 3) -> None:
        super().__init__()
        in_dim = obs_dim + 1
        sizes = [in_dim] + [hidden_dim] * n_layers + [1]
        self.q1 = _mlp(sizes, last_activation=False)
        self.q2 = _mlp(sizes, last_activation=False)

    def forward(self, obs: torch.Tensor, tcont: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, tcont], dim=-1)
        return self.q1(x), self.q2(x)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float = -1.0    # = -|A| for a 1-D action
    init_alpha: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 256
    n_layers: int = 3


@dataclass
class SACMetrics:
    critic_loss: float = 0.0
    actor_loss: float = 0.0
    alpha_loss: float = 0.0
    alpha: float = 0.0
    q_mean: float = 0.0
    target_q_mean: float = 0.0
    tcont_mean: float = 0.0
    extras: dict = field(default_factory=dict)


class NoiseTunerAgent:
    """SAC agent that controls one scalar action (the noise level)."""

    def __init__(self, obs_dim: int, device: torch.device, cfg: SACConfig | None = None) -> None:
        self.device = device
        self.cfg = cfg or SACConfig()

        self.actor = NoiseTunerActor(obs_dim, self.cfg.hidden_dim, self.cfg.n_layers).to(device)
        self.critic = NoiseTunerCritic(obs_dim, self.cfg.hidden_dim, self.cfg.n_layers).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(self.cfg.init_alpha))),
            device=device, requires_grad=True,
        )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """obs: (B, obs_dim) on any device. Returns (B, 1) tcont in [0,1]."""
        obs = obs.to(self.device)
        tcont, _, mean_tcont = self.actor.sample(obs)
        return mean_tcont if deterministic else tcont

    def soft_update_target(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)

    def update(self, batch) -> SACMetrics:
        """One SAC gradient step.

        ``batch`` is a tuple ``(obs, tcont, reward, next_obs, done)`` of tensors,
        each leading-dim ``batch_size``.
        """
        obs, tcont, reward, next_obs, done = (t.to(self.device) for t in batch)

        # --- Critic ---
        with torch.no_grad():
            next_tcont, next_logp, _ = self.actor.sample(next_obs)
            next_q1, next_q2 = self.critic_target(next_obs, next_tcont)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_logp
            target_q = reward + self.cfg.gamma * (1 - done) * next_q

        q1, q2 = self.critic(obs, tcont)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor ---
        new_tcont, new_logp, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_tcont)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * new_logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # --- Alpha ---
        with torch.no_grad():
            logp_for_alpha = new_logp.detach()
        alpha_loss = -(self.log_alpha * (logp_for_alpha + self.cfg.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update_target()

        return SACMetrics(
            critic_loss=float(critic_loss.detach().item()),
            actor_loss=float(actor_loss.detach().item()),
            alpha_loss=float(alpha_loss.detach().item()),
            alpha=float(self.alpha.detach().item()),
            q_mean=float(q1.detach().mean().item()),
            target_q_mean=float(target_q.detach().mean().item()),
            tcont_mean=float(new_tcont.detach().mean().item()),
        )

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.log_alpha.data.copy_(torch.as_tensor(sd["log_alpha"]).to(self.device))


class ReplayBuffer:
    """Simple FIFO replay storing per-chunk SMDP transitions."""

    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, dtype=torch.float32)
        self.tcont = torch.zeros(capacity, 1, dtype=torch.float32)
        self.reward = torch.zeros(capacity, 1, dtype=torch.float32)
        self.next_obs = torch.zeros(capacity, obs_dim, dtype=torch.float32)
        self.done = torch.zeros(capacity, 1, dtype=torch.float32)
        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def push(self, obs, tcont, reward, next_obs, done) -> None:
        i = self.idx
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32).reshape(-1)
        self.tcont[i] = float(tcont)
        self.reward[i] = float(reward)
        self.next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32).reshape(-1)
        self.done[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int):
        n = len(self)
        idx = torch.randint(0, n, (batch_size,))
        return (
            self.obs[idx],
            self.tcont[idx],
            self.reward[idx],
            self.next_obs[idx],
            self.done[idx],
        )
