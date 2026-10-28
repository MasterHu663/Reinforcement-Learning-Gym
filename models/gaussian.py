from __future__ import annotations

from typing import Any, Literal

import torch
from torch.distributions import Normal

from resources.spaces import compute_space_limits

# speed up distribution construction by disabling checking
Normal.set_default_validate_args(False)


class GaussianMixin:
    def __init__(
        self,
        *,
        clip_actions: bool = False,
        squash_actions: bool = False,
        clip_mean_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: Literal["mean", "sum", "prod", "none"] = "sum",
        role: str = "",
    ) -> None:
        """Gaussian mixin model (stochastic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param squash_actions: Apply a tanh transform and its log-probability
            correction so sampled actions are naturally bounded to ``[-1, 1]``.
        :param clip_mean_actions: Flag to indicate whether the mean actions should be clipped to the action space.
            If ``True``, the mean actions will be clipped before sampling the actions.
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped.
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True.
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True.
        :param reduction: Reduction method for returning the log probability density function.
            If ``"none"``, the log probability density function is returned as a tensor of shape
            ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
        :param role: Role played by the model.

        :raises ValueError: If the reduction method is not valid.
        """
        self._g_clip_actions = clip_actions
        self._g_squash_actions = squash_actions
        self._g_clip_mean_actions = clip_mean_actions
        min_actions, max_actions = compute_space_limits(
            self.action_space, device=self.device, none_if_unbounded="any"
        )
        # These tensors are created before callers usually move the model to
        # CUDA. Registering them as non-persistent buffers makes ``.to(...)``
        # move them with the model without changing checkpoint compatibility.
        self.register_buffer("_g_min_actions", min_actions, persistent=False)
        self.register_buffer("_g_max_actions", max_actions, persistent=False)

        self._g_clip_log_std = clip_log_std
        self._g_min_log_std = min_log_std
        self._g_max_log_std = max_log_std

        self._g_distribution = None
        self._g_entropy = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        """Act stochastically in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing the following extra output values:

            - ``"log_std"``: log of the standard deviation.
            - ``"log_prob"``: log of the probability density function.
            - ``"mean_actions"``: mean actions (network output after optional clipping).
        """
        # map from observations/states to mean actions and log standard deviations
        mean_actions, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]

        # clip mean actions
        if self._g_clip_mean_actions:
            mean_actions = torch.clamp(mean_actions, min=self._g_min_actions, max=self._g_max_actions)

        # clip log standard deviations
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        # distribution
        self._g_distribution = Normal(mean_actions, log_std.exp())

        # sample using the reparameterization trick
        raw_actions = self._g_distribution.rsample()
        taken_actions = inputs.get("taken_actions")

        if self._g_squash_actions:
            actions = torch.tanh(raw_actions)
            actions_for_log_prob = actions if taken_actions is None else taken_actions
            # atanh is undefined at exactly +/-1. Floating-point tanh can
            # reach those endpoints for large latent samples.
            actions_for_log_prob = torch.clamp(actions_for_log_prob, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
            latent_actions = torch.atanh(actions_for_log_prob)
            log_prob = self._g_distribution.log_prob(latent_actions)
            log_prob -= torch.log(1.0 - actions_for_log_prob.square() + 1.0e-6)
            deterministic_actions = torch.tanh(mean_actions)
            sampled_log_prob = self._g_distribution.log_prob(raw_actions)
            sampled_log_prob -= torch.log(1.0 - actions.square() + 1.0e-6)
            self._g_entropy = -sampled_log_prob.sum(dim=-1, keepdim=True)
        else:
            actions = raw_actions
            if self._g_clip_actions:
                actions = torch.clamp(actions, min=self._g_min_actions, max=self._g_max_actions)
            log_prob = self._g_distribution.log_prob(inputs.get("taken_actions", actions))
            deterministic_actions = mean_actions
            self._g_entropy = self._g_distribution.entropy().sum(dim=-1, keepdim=True)

        # log of the probability density function
        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = deterministic_actions
        return actions, outputs

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model.

        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        if self._g_entropy is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_entropy

    def distribution(self, *, role: str = "") -> torch.distributions.Normal:
        """Get the current distribution of the model.

        :param role: Role played by the model.

        :return: Distribution of the model.
        """
        return self._g_distribution

    def enable_training_mode(self, enabled: bool = True) -> None:

        self.train(enabled)

    def save(self, path: str, *, state_dict: dict[str, Any] | None = None) -> None:
        """Save the model to the specified path.

        :param path: Path to save the model to.
        :param state_dict: State dictionary to save. If ``None``, the model's ``state_dict`` will be saved.

        Example::

            # save the current model to the specified path
            >>> model.save("/tmp/model.pt")

            # save an older version of the model to the specified path
            >>> old_state_dict = copy.deepcopy(model.state_dict())
            >>> # ...
            >>> model.save("/tmp/model.pt", old_state_dict)
        """
        torch.save(self.state_dict() if state_dict is None else state_dict, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path.

        .. note::

            The final storage device is determined by the constructor of the model.

        :param path: Path to load the model from.

        Example::

            # load the model onto the CPU
            >>> model = Model(device="cpu")
            >>> model.load("model.pt")

            # load the model onto the GPU 1
            >>> model = Model(device="cuda:1")
            >>> model.load("model.pt")
        """

        state_dict = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning

        self.load_state_dict(state_dict)
        self.eval()
