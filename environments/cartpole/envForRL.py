"""Batched CartPole dynamics following the FluidEnv three-file interface.

All environments are advanced with one set of torch tensor expressions.  No
Python loop over environment instances is used in dynamics/reward/reset.

This class deliberately follows the user's FluidEnv convention: *same-step*
autoreset and {'policy', 'critic'} observations.  This differs from Gymnasium's
stock CartPoleVectorEnv (NEXT_STEP autoreset), but preserves the CartPole
equations, rewards, termination conditions, and time limit.

Run as a namespace package from the parent directory:

    import torch
    from cartpole_parallel.envConfig import CartPoleEnvConfig
    from cartpole_parallel.envForRL import CartPoleEnv

    env = CartPoleEnv(CartPoleEnvConfig(num_envs=1024, device='cuda:0'))
    obs, info = env.reset(seed=42)
    actions = torch.randint(2, (env.num_envs,), device=env.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    env.close()
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from .envConfig import CartPoleEnvConfig


class CartPoleEnv(gym.Env):
    """Parallel CartPole, with Torch states/rewards on the chosen device.

    Default action space: ``Discrete(2)``, batched actions ``(num_envs,)``.
    Optional continuous space: ``Box(-1, 1, shape=(1,))`` and batched actions
    ``(num_envs, 1)``.  Only the discrete variant matches CartPole-v1.

    All observations have shape ``(num_envs, 4)`` and are wrapped as
    ``{'policy': obs, 'critic': obs}``, matching the supplied FluidEnv.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    is_vector_env = True

    def __init__(self, cfg: CartPoleEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; use device='cpu'")
            torch.cuda.set_device(self.device)
        self._dtype = torch.float32 if cfg.physics_dtype == "float32" else torch.float64
        self.max_episode_length = cfg.max_episode_length
        self.render_mode = cfg.render_mode

        if cfg.action_mode == "discrete":
            self.single_action_space = gym.spaces.Discrete(2)
        else:
            self.single_action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        high = np.array([
            cfg.x_threshold * 2.0, np.inf,
            cfg.theta_threshold_radians * 2.0, np.inf,
        ], dtype=np.float32)
        raw_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.single_observation_space = gym.spaces.Dict(policy=raw_space, critic=raw_space)
        self.single_state_space = raw_space
        batch_space = gym.vector.utils.batch_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.state_space = batch_space(self.single_state_space, self.num_envs)

        self.state_buf = torch.zeros((self.num_envs, 4), dtype=self._dtype, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_truncated = torch.zeros_like(self.reset_terminated)
        self._rng = torch.Generator(device=self.device)
        if cfg.seed is not None:
            self._rng.manual_seed(cfg.seed)

        self._reset_low = cfg.reset_low
        self._reset_high = cfg.reset_high
        self._has_reset = False
        self._renderer = None
        self._render_during_step = False
        self._render_interval = 1
        self._steps_since_render = 0

    @property
    def state(self) -> torch.Tensor:
        """Read-only-by-convention view of current (num_envs, 4) state."""
        return self.state_buf

    def _reset_indices(self, ids: torch.Tensor) -> None:
        """Reset only the selected parallel environments; keep others intact."""
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1)
        random_state = torch.rand(
            (ids.numel(), 4), generator=self._rng,
            device=self.device, dtype=self._dtype,
        )
        self.state_buf[ids] = self._reset_low + (
            self._reset_high - self._reset_low
        ) * random_state
        self.episode_length_buf[ids] = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.manual_seed(seed)
        options = options or {}
        self._reset_low = float(options.get("low", self.cfg.reset_low))
        self._reset_high = float(options.get("high", self.cfg.reset_high))
        if not np.isfinite(self._reset_low) or not np.isfinite(self._reset_high):
            raise ValueError("reset bounds must be finite")
        if self._reset_low >= self._reset_high:
            raise ValueError("reset low must be smaller than high")

        self._reset_indices(torch.arange(self.num_envs, device=self.device))
        self.reward_buf.zero_()
        self.reset_terminated.zero_()
        self.reset_truncated.zero_()
        self._has_reset = True
        if self.render_mode == "human":
            self.render()
        return self._pack_observation(), {}

    def _force_from_action(self, action: torch.Tensor | np.ndarray) -> torch.Tensor:
        action = torch.as_tensor(action, device=self.device)
        if self.cfg.action_mode == "discrete":
            if action.shape == (self.num_envs, 1):
                action = action[:, 0]
            if tuple(action.shape) != (self.num_envs,):
                raise ValueError(
                    f"Expected discrete actions shape ({self.num_envs},), got {tuple(action.shape)}"
                )
            if self.cfg.validate_actions and not bool(torch.all((action == 0) | (action == 1))):
                raise ValueError("Discrete CartPole actions must be 0 or 1")
            # Conventional values from a Discrete(2) policy are exactly 0/1.
            return torch.where(
                action == 1,
                torch.full_like(action, self.cfg.force_mag, dtype=self._dtype),
                torch.full_like(action, -self.cfg.force_mag, dtype=self._dtype),
            )

        if action.shape == (self.num_envs,):
            action = action[:, None]
        if tuple(action.shape) != (self.num_envs, 1):
            raise ValueError(
                f"Expected continuous actions shape ({self.num_envs}, 1), got {tuple(action.shape)}"
            )
        if self.cfg.validate_actions and not bool(torch.all(torch.isfinite(action))):
            raise ValueError("Continuous CartPole actions must be finite")
        # Normalize [-1, 1] policy actions to physical horizontal force.
        return torch.clamp(
            torch.nan_to_num(action[:, 0].to(self._dtype)), -1.0, 1.0
        ) * self.cfg.force_mag

    def step(self, action: torch.Tensor | np.ndarray):
        if not self._has_reset:
            raise RuntimeError("Call reset() before step()")
        force = self._force_from_action(action)
        x, x_dot, theta, theta_dot = self.state_buf.unbind(dim=1)

        c = self.cfg
        total_mass = c.masspole + c.masscart
        polemass_length = c.masspole * c.length
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (force + polemass_length * theta_dot.square() * sintheta) / total_mass
        thetaacc = (c.gravity * sintheta - costheta * temp) / (
            c.length * (4.0 / 3.0 - c.masspole * costheta.square() / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        if c.kinematics_integrator == "euler":
            next_x = x + c.tau * x_dot
            next_x_dot = x_dot + c.tau * xacc
            next_theta = theta + c.tau * theta_dot
            next_theta_dot = theta_dot + c.tau * thetaacc
        else:
            next_x_dot = x_dot + c.tau * xacc
            next_x = x + c.tau * next_x_dot
            next_theta_dot = theta_dot + c.tau * thetaacc
            next_theta = theta + c.tau * next_theta_dot

        self.state_buf.copy_(torch.stack((
            next_x, next_x_dot, next_theta, next_theta_dot,
        ), dim=1))
        self.episode_length_buf.add_(1)
        terminated = (
            (torch.abs(next_x) > c.x_threshold)
            | (torch.abs(next_theta) > c.theta_threshold_radians)
        )
        truncated = self.episode_length_buf >= self.max_episode_length
        if c.sutton_barto_reward:
            self.reward_buf.copy_(-terminated.to(dtype=torch.float32))
        else:
            self.reward_buf.fill_(1.0)

        self.reset_terminated.copy_(terminated)
        self.reset_truncated.copy_(truncated)
        done = terminated | truncated
        # Preserve pre-reset terminal states for logging/bootstrapping.
        final_observation = self.state_buf.to(dtype=torch.float32).clone()
        final_episode_length = self.episode_length_buf.clone()
        reset_ids = torch.nonzero(done, as_tuple=True)[0]
        self._reset_indices(reset_ids)

        if self._render_during_step:
            self._steps_since_render += 1
            if self._steps_since_render >= self._render_interval:
                self.render()
                self._steps_since_render = 0
        elif self.render_mode == "human":
            self.render()

        info = {
            "reset_mask": done.clone(),
            "final_observation": final_observation,
            "_final_observation": done.clone(),
            "episode_length": final_episode_length,
            "time_outs": (truncated & ~terminated).clone(),
        }
        return (
            self._pack_observation(),
            self.reward_buf.clone(),
            self.reset_terminated.clone(),
            self.reset_truncated.clone(),
            info,
        )

    def _pack_observation(self) -> dict[str, torch.Tensor]:
        observation = self.state_buf.to(dtype=torch.float32).clone()
        return {"policy": observation, "critic": observation}

    def configure_rendering(self, enabled: bool, interval: int = 1) -> None:
        if interval < 1:
            raise ValueError("render interval must be at least one")
        self._render_during_step = bool(enabled)
        self._render_interval = int(interval)
        self._steps_since_render = 0

    def render(self):
        if not self._has_reset:
            return None
        if self._renderer is None:
            from .renderer import CartPoleRenderer
            self._renderer = CartPoleRenderer(self.cfg, watched_envs=self.cfg.render_envs)
        return self._renderer.update(self.state_buf, render_mode=self.render_mode)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
