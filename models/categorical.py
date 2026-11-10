from __future__ import annotations

from typing import Any
import math

import torch
from torch.distributions import Categorical


# speed up distribution construction by disabling checking
Categorical.set_default_validate_args(False)


class CategoricalMixin:
    def __init__(self, *, unnormalized_log_prob: bool = True, role: str = "") -> None:
        """Categorical mixin model (stochastic model).

        :param unnormalized_log_prob: Flag to indicate how to the model's output will be interpreted.
            If True, the model's output is interpreted as unnormalized log probabilities (it can be any real number),
            otherwise as normalized probabilities (the output must be non-negative, finite and have a non-zero sum).
        :param role: Role played by the model.
        """
        self._c_unnormalized_log_prob = unnormalized_log_prob
        self._c_distribution = None

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """Act stochastically in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing the following extra output values:

            - ``"log_prob"``: log of the probability density function.
            - ``"net_output"``: network output.
            - ``"mean_actions"``: most likely action (PPO's deterministic-action key).
        """
        # map from observations/states to normalized probabilities or unnormalized log probabilities
        net_output, outputs = self.compute(inputs, role)

        # unnormalized log probabilities
        if self._c_unnormalized_log_prob:
            self._c_distribution = Categorical(logits=net_output)
        # normalized probabilities
        else:
            self._c_distribution = Categorical(probs=net_output)

        # actions and log of the probability density function
        mode_actions = net_output.argmax(dim=-1)
        taken_actions = inputs.get("taken_actions")
        if taken_actions is not None:
            actions = taken_actions.reshape(net_output.shape[:-1]).long()
        elif inputs.get("deterministic", False):
            actions = mode_actions
        else:
            actions = self._c_distribution.sample()
        log_prob = self._c_distribution.log_prob(actions)

        outputs["log_prob"] = log_prob.unsqueeze(-1)
        outputs["net_output"] = net_output
        outputs["mean_actions"] = mode_actions.unsqueeze(-1)
        return actions.unsqueeze(-1), outputs

    def random_act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample uniformly and return the behavior policy's log probability."""
        states = inputs.get("states")
        if states is None:
            states = inputs["observations"]
        num_actions = self.action_space.n
        actions = torch.randint(num_actions, (*states.shape[:-1], 1), device=self.device)
        log_prob = torch.full(actions.shape, -math.log(num_actions), device=self.device)
        return actions, {"log_prob": log_prob}

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model.

        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        if self._c_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._c_distribution.entropy().unsqueeze(-1)

    def distribution(self, *, role: str = "") -> torch.distributions.Categorical:
        """Get the current distribution of the model.

        :param role: Role played by the model.

        :return: Distribution of the model.
        """
        return self._c_distribution

    def enable_training_mode(self, enabled: bool = True) -> None:
        self.train(enabled)

    def save(self, path: str, *, state_dict: dict[str, Any] | None = None) -> None:
        torch.save(self.state_dict() if state_dict is None else state_dict, path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()
