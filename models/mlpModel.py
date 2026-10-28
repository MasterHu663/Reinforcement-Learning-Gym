from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from models.gaussian import GaussianMixin
import gymnasium as gym


class MLPModel(nn.Module, GaussianMixin):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [128, 128, 128],
        reference_init: bool = False,
        **kwargs,
    ) -> None:
        
        nn.Module.__init__(self)
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))

        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
            
        self.trunk = nn.Sequential(*layers)

        self.action_mean_head = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.value_head = nn.Linear(in_dim, 1)
        
        self._init_weights(reference_init=reference_init)
        
        GaussianMixin.__init__(self, **kwargs)
        
    def _init_weights(self, *, reference_init: bool = False) -> None:
        if reference_init:
            # The reference ActorCritic applies orthogonal_ with its default
            # gain to every Linear weight and leaves PyTorch's biases intact.
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight)
            return
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.constant_(layer.bias, 0.0)
                
        nn.init.orthogonal_(self.action_mean_head.weight, gain=0.01)
        nn.init.constant_(self.action_mean_head.bias, 0.0)
        
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:

        x = inputs.get("states", inputs.get("observations"))
        
        processed_states = self.preprocess(x)
        features = self.trunk(processed_states)
        
        raw_mean_actions = self.action_mean_head(features)
        # A squashed Gaussian applies tanh inside the distribution so its
        # Jacobian can be included in log_prob. Keep the legacy bounded mean
        # for non-squashed policies.
        mean_actions = raw_mean_actions if self._g_squash_actions else torch.tanh(raw_mean_actions)
        
        outputs = {"log_std": self.log_std.expand_as(mean_actions)}
        
        return mean_actions, outputs

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:

        if role == "value":
            x = inputs.get("states", inputs.get("observations"))
            processed_states = self.preprocess(x)
            features = self.trunk(processed_states)
            values = self.value_head(features)
            return values, {}
        else:
            return GaussianMixin.act(self, inputs, role=role)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        processed_states = self.preprocess(states)
        features = self.trunk(processed_states)
        values = self.value_head(features)
        return values.squeeze(-1)
