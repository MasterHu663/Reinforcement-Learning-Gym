from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from models.categorical import CategoricalMixin
import gymnasium as gym


class DiscreteMLPModel(nn.Module, CategoricalMixin):
    """Actor/critic MLP: discrete Categorical (ReLU).

    Discrete policies use action_dim=1 and num_actions logits. Deploy the trunk
    and action_logits_head, then take argmax; the value head is training-only.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dims: list[int] = [128, 128, 128],
        reference_init: bool = False,
        num_actions: int | None = None,
    ) -> None:

        nn.Module.__init__(self)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)

        if num_actions is None or num_actions < 2:
            raise ValueError("Discrete policies require action_dim=1 and num_actions >= 2")
        self.action_space = gym.spaces.Discrete(num_actions)
        self.state_dim = state_dim


        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.action_logits_head = nn.Linear(in_dim, num_actions)
        self.value_head = nn.Linear(in_dim, 1)

        self._init_weights(reference_init=reference_init)

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
                activation = "relu"
                nn.init.orthogonal_(
                    layer.weight,
                    gain=nn.init.calculate_gain(activation)
                )
                nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.action_logits_head.weight, gain=0.01)
        nn.init.constant_(self.action_logits_head.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:

        x = inputs.get("observations")
        features = self.trunk(x)

        logits = self.action_logits_head(features)

        return logits, {}


    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:

        if role == "value":
            x = inputs.get("states")
            features = self.trunk(x)
            values = self.value_head(features)
            return values, {}
        else:
            return CategoricalMixin.act(self, inputs, role=role)
