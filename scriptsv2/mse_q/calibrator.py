"""MLP calibrator: RoboMonkey verifier reward -> MSEVerifier-scale score.

The state-based search policy is trained with ``MSEVerifier``, whose score is
the *negative* MSE between a candidate action chunk and the dataset expert
action (higher = closer to expert).  At eval time there is no ground-truth
action, so ``MSEVerifier`` cannot run — only the RoboMonkey reward model is
available, and its scores live on a different, unknown scale.

``MLPCalibrator`` is a small 1-D MLP that learns the transformation

    g(robomonkey_reward) ~= -mse(candidate, expert)

so the RoboMonkey verifier can stand in for ``MSEVerifier`` when evaluating
the MSE-trained search policy.

Input/output standardization stats are stored as buffers, so a saved
checkpoint is fully self-contained: ``MLPCalibrator.load(path)`` restores the
architecture and the stats, and ``calibrator(reward_tensor)`` returns scores
already on the ``-mse`` scale.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn


class MLPCalibrator(nn.Module):
    """1-D MLP mapping a scalar verifier reward to an MSE-scale score.

    Args:
        hidden: Hidden layer widths. ``(64, 64)`` -> ``1->64->64->1``.
        reward_mean / reward_std: Standardization stats for the input reward.
        target_mean / target_std: Standardization stats for the ``-mse`` target.

    The standardization stats are registered as buffers so they travel with
    ``state_dict()`` and are restored by :meth:`load`.
    """

    def __init__(
        self,
        hidden: Sequence[int] = (64, 64),
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden: Tuple[int, ...] = tuple(int(h) for h in hidden)

        dims = [1, *self.hidden, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        for name, value in (
            ("reward_mean", reward_mean),
            ("reward_std", reward_std),
            ("target_mean", target_mean),
            ("target_std", target_std),
        ):
            self.register_buffer(name, torch.tensor(float(value)))

    # -- internal (standardized) space ------------------------------------
    def standardize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Raw verifier reward -> standardized input."""
        return (reward - self.reward_mean) / self.reward_std.clamp_min(1e-8)

    def net_std(self, reward_standardized: torch.Tensor) -> torch.Tensor:
        """Run the MLP in standardized space. In/out shape: ``(...,)``."""
        return self.net(reward_standardized.unsqueeze(-1)).squeeze(-1)

    # -- public (mse-scale) space -----------------------------------------
    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        """Raw verifier reward -> calibrated score on the ``-mse`` scale."""
        std_out = self.net_std(self.standardize_reward(reward))
        return std_out * self.target_std + self.target_mean

    # -- (de)serialization -------------------------------------------------
    def save(self, path: str, metadata: Dict[str, Any] | None = None) -> None:
        """Write a self-contained checkpoint (weights + arch + stats + meta)."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden": list(self.hidden),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str, map_location: str = "cpu"
    ) -> Tuple["MLPCalibrator", Dict[str, Any]]:
        """Load a checkpoint. Returns ``(calibrator, metadata)``; model in eval mode."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(hidden=tuple(ckpt["hidden"]))
        model.load_state_dict(ckpt["state_dict"])  # restores stat buffers too
        model.eval()
        return model, ckpt.get("metadata", {})

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden}, "
            f"reward~({self.reward_mean.item():.4g}, {self.reward_std.item():.4g}), "
            f"target~({self.target_mean.item():.4g}, {self.target_std.item():.4g})"
        )
