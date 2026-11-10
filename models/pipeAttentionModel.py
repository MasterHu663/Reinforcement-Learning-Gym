"""Small categorical actor/critic for Flappy Bird's pipe and bird features."""

from __future__ import annotations

import torch
from torch import nn

from models.mlpModel import MLPModel


class PipeAttentionModel(MLPModel):
    """Let the bird query shared pipe features before the two-layer MLP.

    Input: normalized (x, gap_top, gap_bottom) per pipe, followed by (y, vy,
    rotation). Attention is over pipes within each environment, never over
    environments or time. Geometry must match the environment: pipes whose
    right edges have passed the bird's left edge receive zero attention.
    """

    def __init__(
        self, *, pipe_count: int, attention_dim: int = 16,
        hidden_dims: list[int] | None = None, num_actions: int = 2,
        screen_width: int = 128, player_x: int = 24, pipe_width: int = 10,
    ) -> None:
        if pipe_count < 1 or attention_dim < 1:
            raise ValueError("pipe_count and attention_dim must be positive")
        if screen_width <= 0 or pipe_width <= 0 or player_x < 0:
            raise ValueError("Invalid screen or pipe geometry")
        # The MLP receives the pooled pipe features and the original bird state.
        super().__init__(
            state_dim=attention_dim + 3, action_dim=1,
            hidden_dims=[64, 64] if hidden_dims is None else hidden_dims,
            action_mode="discrete", num_actions=num_actions,
        )
        self.state_dim = 3 * pipe_count + 3  # Public width of the raw observation.
        self.pipe_count = pipe_count
        self.attention_scale = attention_dim ** -0.5
        # A plain scalar keeps existing checkpoint keys and shapes compatible.
        self.passed_pipe_x = (player_x - pipe_width) / screen_width
        self.pipe_encoder = nn.Sequential(nn.Linear(3, attention_dim), nn.ReLU())
        self.bird_query = nn.Linear(3, attention_dim)
        self.pipe_key = nn.Linear(attention_dim, attention_dim, bias=False)
        nn.init.orthogonal_(self.pipe_encoder[0].weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.pipe_encoder[0].bias)
        nn.init.orthogonal_(self.bird_query.weight)
        nn.init.zeros_(self.bird_query.bias)
        nn.init.orthogonal_(self.pipe_key.weight)

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.state_dim:
            raise ValueError(f"Expected {self.state_dim} observation features, got {obs.shape[-1]}")
        pipes = obs[..., :-3].reshape(*obs.shape[:-1], self.pipe_count, 3)
        bird = obs[..., -3:]
        values = self.pipe_encoder(pipes)
        keys = self.pipe_key(values)
        query = self.bird_query(bird).unsqueeze(-2)
        relevant = pipes[..., 0] > self.passed_pipe_x
        logits = (keys * query).sum(dim=-1) * self.attention_scale
        # Use a finite minimum so even an all-masked row has a finite softmax.
        weights = logits.masked_fill(~relevant, torch.finfo(logits.dtype).min).softmax(dim=-1)
        weights = weights.masked_fill(~relevant, 0.0)
        context = (weights.unsqueeze(-1) * values).sum(dim=-2)
        return torch.cat((context, bird), dim=-1)
