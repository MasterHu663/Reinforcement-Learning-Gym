from __future__ import annotations

import torch
import torch.nn as nn

from models.mlpModel import MLPModel


class CircularPressureEncoder(nn.Module):
    """Encode ordered pressure probes without creating a 0/360-degree edge."""

    def __init__(
        self,
        *,
        probe_count: int,
        output_dim: int = 32,
        channels: tuple[int, int] = (8, 16),
    ) -> None:
        super().__init__()
        if probe_count < 5:
            raise ValueError("probe_count must be at least 5 for the pressure CNN")
        if output_dim < 1:
            raise ValueError("output_dim must be positive")

        self.probe_count = probe_count
        self.output_dim = output_dim
        self.convolutions = nn.Sequential(
            nn.Conv1d(
                1, channels[0], kernel_size=5, padding=2, padding_mode="circular"
            ),
            nn.SiLU(),
            nn.Conv1d(
                channels[0],
                channels[1],
                kernel_size=5,
                padding=2,
                padding_mode="circular",
            ),
            nn.SiLU(),
        )
        # Flattening retains where a pressure feature occurs relative to the
        # body-fixed head. Global pooling would discard that useful phase.
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(channels[1] * probe_count, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        if pressure.shape[-1] != self.probe_count:
            raise ValueError(
                f"Expected {self.probe_count} pressure probes, "
                f"got {pressure.shape[-1]}"
            )
        features = self.convolutions(pressure.unsqueeze(1))
        return self.projection(features)


class CNNModel(MLPModel):
    """Encode circular pressure samples, then fuse them with scalar state."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        pressure_dim: int,
        base_state_dim: int = 7,
        pressure_feature_dim: int = 32,
        hidden_dims: list[int] | tuple[int, ...] = (128, 128),
        **kwargs,
    ) -> None:
        if state_dim != base_state_dim + pressure_dim:
            raise ValueError(
                "state_dim must equal base_state_dim + pressure_dim "
                f"({base_state_dim} + {pressure_dim}), got {state_dim}"
            )

        encoded_state_dim = base_state_dim + pressure_feature_dim
        super().__init__(
            state_dim=encoded_state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            **kwargs,
        )
        self.input_state_dim = state_dim
        self.state_dim = state_dim
        self.base_state_dim = base_state_dim
        self.pressure_dim = pressure_dim
        self.pressure_encoder = CircularPressureEncoder(
            probe_count=pressure_dim,
            output_dim=pressure_feature_dim,
        )

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.input_state_dim:
            raise ValueError(
                f"Expected input feature size {self.input_state_dim}, "
                f"got {obs.shape[-1]}"
            )
        base_state = obs[..., : self.base_state_dim]
        pressure = obs[..., self.base_state_dim :]
        pressure_features = self.pressure_encoder(pressure)
        return torch.cat((base_state, pressure_features), dim=-1)
