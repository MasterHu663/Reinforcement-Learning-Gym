from __future__ import annotations
import dataclasses
import os
import itertools
import datetime
import collections

from typing import Any

import torch
import torch.nn as nn

import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from resources.kl_adaptive import KLAdaptiveLR

import gymnasium

from agents.ppo_cfg import PPOConfig
from memories.memory import Memory


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
    time_limit_bootstrap: bool = False,
) -> torch.Tensor:

    advantage = 0
    advantages = torch.zeros_like(rewards)
    # The vector environment auto-resets and returns the next episode's first
    # observation.  Both termination types must therefore cut the GAE trace;
    # otherwise advantages leak across unrelated episodes.
    not_done = (terminated | truncated).logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


def compute_value_loss(
    *,
    predicted_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip: float,
    scale: float,
) -> torch.Tensor:
    """Compute PPO's clipped value objective.

    The maximum is essential: clipping may prevent an overly large helpful
    update, but it must never hide an update that moves the critic farther
    away from the return target.
    """
    losses = (predicted_values - returns).square()
    if clip > 0.0:
        clipped_values = old_values + torch.clamp(
            predicted_values - old_values, min=-clip, max=clip
        )
        clipped_losses = (clipped_values - returns).square()
        losses = torch.maximum(losses, clipped_losses)
    return scale * losses.mean()

class PPO:
    def __init__(
        self,
        *,
        cfg: PPOConfig,
        models: dict[str, nn.Module], 
        memory: Memory,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device
    ) -> None:

        self.training = True # False
        
        self.models = models
        self.memory = memory
        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space

        self.device = torch.device(device) if isinstance(device, str) else device

        self.cfg = cfg
        
        # data tracking
        self.tracking_data = collections.defaultdict(list)
        self.write_interval = self.cfg.experiment.write_interval

        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

        # checkpoint
        self.checkpoint_modules = {}
        self.checkpoint_interval = self.cfg.experiment.checkpoint_interval
        self.checkpoint_best_modules = {"timestep": 0, "reward": -(2**31), "saved": False, "modules": {}}
        
        # experiment directory
        directory = (
            self.cfg.experiment.directory if self.cfg.experiment.directory else os.path.join(os.getcwd(), "runs")
        )
        
        # experiment directory
        experiment_name = (
            self.cfg.experiment.experiment_name
            if self.cfg.experiment.experiment_name
            else f"{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}_{self.__class__.__name__}"
        )
        self.experiment_dir = os.path.join(directory, experiment_name)
        
        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)
        
        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        
        # set up optimizers and learning rate scheduler
        lr=self.cfg.learning_rate if isinstance(self.cfg.learning_rate, float) else self.cfg.learning_rate[0]
        trainable_parameters = (
            parameter
            for parameter in itertools.chain(
                self.policy.parameters(), self.value.parameters()
            )
            if parameter.requires_grad
        )
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
        self.checkpoint_modules["optimizer"] = self.optimizer
        self.scheduler = KLAdaptiveLR(
            self.optimizer,
            min_lr=self.cfg.min_learning_rate,
            max_lr=self.cfg.max_learning_rate,
        )
        
        # set up preprocessors
        # - observations
        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        # - values
        if self.cfg.value_preprocessor:
            self._value_preprocessor = self.cfg.value_preprocessor(**self.cfg.value_preprocessor_kwargs)
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:

        # trainer_cfg = {} if trainer_cfg is None else dataclasses.asdict(trainer_cfg)
        if trainer_cfg is None:
            trainer_cfg = {}
        elif isinstance(trainer_cfg, dict):
            trainer_cfg = trainer_cfg
        elif dataclasses.is_dataclass(trainer_cfg):
            trainer_cfg = dataclasses.asdict(trainer_cfg)
        else:
            trainer_cfg = {k: v for k, v in trainer_cfg.__dict__.items() if not k.startswith('_')}
        
        # main entry to log data for consumption and visualization by TensorBoard
        if self.write_interval == "auto":
            self.write_interval = int(trainer_cfg.get("timesteps", 0) / 100)
        if self.write_interval > 0:
            self.writer = SummaryWriter(log_dir=self.experiment_dir)
        
        # checkpoint directory creation
        if self.checkpoint_interval == "auto":
            self.checkpoint_interval = int(trainer_cfg.get("timesteps", 0) / 10)
        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
        
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None
        self._current_values = None
        self._rollout = 0
               
    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")
        
        states = states.to(self.device)
        
        # sample stochastic actions
        actions, outputs = self.policy.act(inputs, role="policy")
        if deterministic:
            actions = outputs["mean_actions"]
        self._current_log_prob = outputs["log_prob"]

        # compute values
        if self.training:
            values, _ = self.value.act(inputs, role="value")
            self._current_values = self._value_preprocessor(values, inverse=True)

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        
        if self.write_interval > 0:
            # compute the cumulative sum of the rewards and timesteps
            if self._cumulative_rewards is None:
                self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
                self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

            self._cumulative_rewards.add_(rewards)
            self._cumulative_timesteps.add_(1)

            # check ended episodes
            finished_episodes = (terminated + truncated).nonzero(as_tuple=True)[0]
            if finished_episodes.numel():

                # storage cumulative rewards and timesteps
                self._track_rewards.extend(self._cumulative_rewards[finished_episodes].tolist())
                self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes].tolist())

                # reset the cumulative rewards and timesteps
                self._cumulative_rewards[finished_episodes] = 0
                self._cumulative_timesteps[finished_episodes] = 0

            # record data
            self.tracking_data["Reward / Instantaneous reward (max)"].append(torch.max(rewards).item())
            self.tracking_data["Reward / Instantaneous reward (min)"].append(torch.min(rewards).item())
            self.tracking_data["Reward / Instantaneous reward (mean)"].append(torch.mean(rewards).item())

            if len(self._track_rewards):
                track_rewards = np.array(self._track_rewards)
                track_timesteps = np.array(self._track_timesteps)

                self.tracking_data["Reward / Total reward (max)"].append(np.max(track_rewards))
                self.tracking_data["Reward / Total reward (min)"].append(np.min(track_rewards))
                self.tracking_data["Reward / Total reward (mean)"].append(np.mean(track_rewards))

                self.tracking_data["Episode / Total timesteps (max)"].append(np.max(track_timesteps))
                self.tracking_data["Episode / Total timesteps (min)"].append(np.min(track_timesteps))
                self.tracking_data["Episode / Total timesteps (mean)"].append(np.mean(track_timesteps))
        
        if self.training:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            # # reward shaping
            # if self.cfg.rewards_shaper is not None:
            #     rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # TODO time-limit (truncation) bootstrapping
            if self.cfg.time_limit_bootstrap and truncated.any():
                with torch.no_grad():
                    inputs = {
                        "observations": self._observation_preprocessor(next_observations),
                        "states": self._state_preprocessor(next_states),
                    }
                    next_values, _ = self.value.act(inputs, role="value")
                    next_values = self._value_preprocessor(next_values, inverse=True)

                rewards += self.cfg.discount_factor * next_values * truncated

            # storage transition in memory
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=self._current_values,
            )
            
    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass
    
    def post_interaction(self, timestep: int, timesteps: int) -> None:
        
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                
        timestep += 1

        # update best models and write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # update best models
            reward = np.mean(self.tracking_data.get("Reward / Total reward (mean)", -(2**31)))
            if reward > self.checkpoint_best_modules["reward"]:
                self.checkpoint_best_modules["timestep"] = timestep
                self.checkpoint_best_modules["reward"] = reward
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {
                    k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules.items()
                }
            # write checkpoints
            self.write_checkpoint(timestep=timestep, timesteps=timesteps)

        # write to tensorboard
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep=timestep, timesteps=timesteps)

    def update(self, timestep: int, timesteps: int) -> None:

        # compute returns and advantages
        with torch.no_grad():
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        update_kl_divergences = []
        optimization_steps = 0
        stop_early = False
        for epoch in range(self.cfg.learning_epochs):
            # mini-batches loop
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in self.memory.sample(
                names=self._tensors_names, batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
            ):

                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                    "states": self._state_preprocessor(sampled_states, train=not epoch),
                }

                _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                next_log_prob = outputs["log_prob"]

                # compute approximate KL divergence
                with torch.no_grad():
                    log_ratio = next_log_prob - sampled_log_prob
                    # The policy ratio is mathematically exp(log_ratio), but
                    # an unconstrained exponent can overflow during a PPO
                    # epoch and permanently turn model weights into NaN.
                    bounded_log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                    kl_divergence = ((torch.exp(bounded_log_ratio) - 1) - bounded_log_ratio).mean()
                    update_kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                    stop_early = True
                    break

                # compute entropy loss
                if self.cfg.entropy_loss_scale:
                    entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                log_ratio = next_log_prob - sampled_log_prob
                # Only the upper bound is needed as an overflow guard. Very
                # negative log-ratios safely underflow toward a zero ratio.
                ratio = torch.exp(torch.clamp_max(log_ratio, 20.0))
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(
                    ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                )

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                predicted_values, _ = self.value.act(inputs, role="value")
                value_loss = compute_value_loss(
                    predicted_values=predicted_values,
                    old_values=sampled_values,
                    returns=sampled_returns,
                    clip=self.cfg.value_clip,
                    scale=self.cfg.value_loss_scale,
                )

                # optimization step
                self.optimizer.zero_grad()
                
                loss = policy_loss + entropy_loss + value_loss
                loss.backward()

                if self.cfg.grad_norm_clip > 0:
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                    else:
                        # A large critic error must not consume the shared
                        # global norm budget and suppress the policy update.
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                        nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.grad_norm_clip)
                        
                self.optimizer.step()
                optimization_steps += 1

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            if stop_early:
                break

        # Adapt once per rollout update. Updating once per learning epoch made
        # a 12-epoch PPO update multiply the learning rate up to 1.5^12 times.
        mean_kl = torch.stack(update_kl_divergences).mean() if update_kl_divergences else None
        if self.scheduler and mean_kl is not None:
            self.scheduler.step(mean_kl.item())

        # record data
        self.track_data(
            "Loss / Policy loss", cumulative_policy_loss / max(1, optimization_steps)
        )
        self.track_data("Loss / Value loss", cumulative_value_loss / max(1, optimization_steps))
        if self.cfg.entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / max(1, optimization_steps)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
            
    def _empty_preprocessor(self, _input: Any, *args, **kwargs) -> Any:

        return _input

    def _get_internal_value(self, _module: Any) -> Any:

        return _module.state_dict() if hasattr(_module, "state_dict") else _module
 
    def enable_training_mode(self, enabled: bool = True, *, apply_to_models: bool = False) -> None:
        """Set the training mode of the agent: enabled (training) or disabled (evaluation).

        The training mode can be queried by the ``training`` property.

        :param enabled: True to enable the training mode, False to enable the evaluation mode.
        :param apply_to_models: Whether to apply the training mode to all the agent's models.
        """
        self.training = enabled
        if apply_to_models:
            self.enable_models_training_mode(enabled)

    def enable_models_training_mode(self, enabled: bool = True) -> None:
        """Set the training mode of all the agent's models: enabled (training) or disabled (evaluation).

        :param enabled: True to enable the training mode, False to enable the evaluation mode.
        """
        for model in self.models.values():
            if model is not None:
                model.enable_training_mode(enabled)

    def track_data(self, tag: str, value: float) -> None:
        """Track data to TensorBoard.

        .. note::

            Currently only scalar data is supported.

        :param tag: Data identifier (e.g. 'Loss/Policy loss').
        :param value: Value to track.
        """
        self.tracking_data[tag].append(value)

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        """Write tracking data to TensorBoard.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        for k, v in self.tracking_data.items():
            if k.endswith("(min)"):
                self.writer.add_scalar(tag=k, scalar_value=np.min(v), global_step=timestep)
            elif k.endswith("(max)"):
                self.writer.add_scalar(tag=k, scalar_value=np.max(v), global_step=timestep)
            else:
                self.writer.add_scalar(tag=k, scalar_value=np.mean(v), global_step=timestep)
        # reset data containers
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()

    def write_checkpoint(self, *, timestep: int, timesteps: int) -> None:
        """Write checkpoint (modules) to persistent storage.

        .. note::

            The checkpoints are stored in the subdirectory ``checkpoints`` within the experiment directory.
            The checkpoint name is the ``timestep`` argument value (if it is not ``None``),
            or the current system date-time otherwise.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        tag = str(timestep if timestep is not None else datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        # separated modules
        if self.cfg.experiment.store_separately:
            for name, module in self.checkpoint_modules.items():
                torch.save(
                    self._get_internal_value(module),
                    os.path.join(self.experiment_dir, "checkpoints", f"{name}_{tag}.pt"),
                )
        # whole agent
        else:
            modules = {name: self._get_internal_value(module) for name, module in self.checkpoint_modules.items()}
            torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", f"agent_{tag}.pt"))

        # best modules
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            # separated modules
            if self.cfg.experiment.store_separately:
                for name in self.checkpoint_modules:
                    torch.save(
                        self.checkpoint_best_modules["modules"][name],
                        os.path.join(self.experiment_dir, "checkpoints", f"best_{name}.pt"),
                    )
            # whole agent
            else:
                modules = {name: self.checkpoint_best_modules["modules"][name] for name in self.checkpoint_modules}
                torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", "best_agent.pt"))
            self.checkpoint_best_modules["saved"] = True
            
    def save(self, path: str) -> None:
        """Save the agent to the specified path.

        :param path: Path to save the agent to.
        """
        modules = {name: self._get_internal_value(module) for name, module in self.checkpoint_modules.items()}
        torch.save(modules, path)

    def load(self, path: str) -> None:
        """Load the agent from the specified path.

        .. note::

            The final storage device is determined by the constructor of the agent.

        :param path: Path to load the agent from.
        """
        modules = torch.load(path, map_location=self.device, weights_only=False)  # prevent torch:FutureWarning

        if type(modules) is dict:
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name, None)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                    else:
                        raise NotImplementedError
                else:
                    pass
            
    