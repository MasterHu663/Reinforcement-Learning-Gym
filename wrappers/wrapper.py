from __future__ import annotations

from typing import Any

import gymnasium

import torch

from resources.spaces import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space


class Wrapper():
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper.

        :param env: The environment instance to wrap.
        """
        self._env = env
        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        self._device = torch.device("cuda:0")
        
        # TODO: maybe have problem with this
        self._seed = 42

        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}
        
    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment.

        :param key: The attribute name.

        :return: The attribute value.

        :raises AttributeError: If the attribute does not exist.
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self._unwrapped, key):
            return getattr(self._unwrapped, key)
        raise AttributeError(
            f"Wrapped environment ({self._unwrapped.__class__.__name__}) does not have attribute '{key}'"
        )

    @property
    def state_space(self) -> gymnasium.Space | None:
        """State space."""
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space."""
        try:
            return self._unwrapped.single_observation_space["policy"]
        except:
            return self._unwrapped.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space."""
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment.

        :param actions: The actions to perform.

        :return: Observation, reward, terminated, truncated, info.
        """
        actions = unflatten_tensorized_space(self.action_space, actions)
        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))
        states = observations.get("critic", None)
        if states is not None:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def state(self) -> torch.Tensor | None:
        """Get the environment state.

        :return: State.
        """
        return self._states

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        :return: Observation, info.
        """
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            self._observations = flatten_tensorized_space(
                tensorize_space(self.observation_space, observations["policy"])
            )
            states = observations.get("critic", None)
            if states is not None:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
