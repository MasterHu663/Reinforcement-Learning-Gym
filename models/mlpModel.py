from __future__ import annotations
from typing import Any, Literal
import torch
import torch.nn as nn
from models.gaussian import GaussianMixin
from models.categorical import CategoricalMixin
import gymnasium as gym


class MLPModel(nn.Module, GaussianMixin, CategoricalMixin):
    """Actor/critic MLP: continuous Gaussian (Tanh) or discrete Categorical (ReLU).

    Discrete policies use action_dim=1 and num_actions logits. Deploy the trunk
    and action_logits_head, then take argmax; the value head is training-only.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [128, 128, 128],
        reference_init: bool = False,
        action_mode: Literal["continuous", "discrete"] = "continuous",
        num_actions: int | None = None,
        **kwargs,
    ) -> None:
        
        nn.Module.__init__(self)
        
        if action_mode not in ("continuous", "discrete"):
            raise ValueError("action_mode must be 'continuous' or 'discrete'")
        self.action_mode = action_mode
        if action_mode == "discrete":
            if action_dim != 1 or num_actions is None or num_actions < 2:
                raise ValueError("Discrete policies require action_dim=1 and num_actions >= 2")
            self.action_space = gym.spaces.Discrete(num_actions)
            self._policy_mixin = CategoricalMixin
        else:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,))
            self._policy_mixin = GaussianMixin

        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU() if action_mode == "discrete" else nn.Tanh())
            in_dim = h_dim
            
        self.trunk = nn.Sequential(*layers)

        if action_mode == "discrete":
            self.action_logits_head = nn.Linear(in_dim, num_actions)
        else:
            self.action_mean_head = nn.Linear(in_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.value_head = nn.Linear(in_dim, 1)
        
        self._init_weights(reference_init=reference_init)
        
        self._policy_mixin.__init__(self, **kwargs)
        
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
                activation = "relu" if self.action_mode == "discrete" else "tanh"
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(activation))
                nn.init.constant_(layer.bias, 0.0)
                
        action_head = self.action_logits_head if self.action_mode == "discrete" else self.action_mean_head
        nn.init.orthogonal_(action_head.weight, gain=0.01)
        nn.init.constant_(action_head.bias, 0.0)
        
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:

        x = inputs.get("states")
        if x is None:
            x = inputs["observations"]
        
        processed_states = self.preprocess(x)
        features = self.trunk(processed_states)

        if self.action_mode == "discrete":
            logits = self.action_logits_head(features)
            if self._c_unnormalized_log_prob:
                return logits, {}
            return logits.softmax(dim=-1), {}
        
        raw_mean_actions = self.action_mean_head(features)
        # A squashed Gaussian applies tanh inside the distribution so its
        # Jacobian can be included in log_prob. Keep the legacy bounded mean
        # for non-squashed policies.
        mean_actions = raw_mean_actions if self._g_squash_actions else torch.tanh(raw_mean_actions)
        
        outputs = {"log_std": self.log_std.expand_as(mean_actions)}
        
        return mean_actions, outputs

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:

        if role == "value":
            x = inputs.get("states")
            if x is None:
                x = inputs["observations"]
            processed_states = self.preprocess(x)
            features = self.trunk(processed_states)
            values = self.value_head(features)
            return values, {}
        else:
            return self._policy_mixin.act(self, inputs, role=role)

    def random_act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        return self._policy_mixin.random_act(self, inputs, role=role)

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        return self._policy_mixin.get_entropy(self, role=role)

    def distribution(self, *, role: str = "") -> torch.distributions.Distribution | None:
        return self._policy_mixin.distribution(self, role=role)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        processed_states = self.preprocess(states)
        features = self.trunk(processed_states)
        values = self.value_head(features)
        return values.squeeze(-1)
