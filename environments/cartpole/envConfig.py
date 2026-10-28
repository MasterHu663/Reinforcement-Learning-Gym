"""Configuration for a batched CartPole environment.

The default parameters reproduce Gymnasium's discrete CartPole-v1 task.
Only ``action_mode='continuous'`` defines an optional, nonstandard variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal


@dataclass
class CartPoleEnvConfig:
    num_envs: int = 1024
    device: str = "cuda:0"  # use "cpu" to debug without CUDA
    max_episode_length: int = 500  # CartPole-v0 uses 200
    seed: int | None = None

    # Gymnasium CartPole-v1 defaults to two discrete actions, 0 and 1.
    # The optional continuous mode uses a normalized force in [-1, 1].
    action_mode: Literal["discrete", "continuous"] = "discrete"
    sutton_barto_reward: bool = False
    kinematics_integrator: Literal["euler", "semi-implicit"] = "euler"
    physics_dtype: Literal["float32", "float64"] = "float32"
    validate_actions: bool = False  # True costs a GPU-to-CPU synchronization

    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5  # half the pole length (m)
    force_mag: float = 10.0
    tau: float = 0.02  # integration timestep (s)
    x_threshold: float = 2.4
    theta_threshold_radians: float = 12.0 * math.pi / 180.0
    reset_low: float = -0.05
    reset_high: float = 0.05

    # Follow FluidEnv: return post-reset observations on the same terminal step.
    # Terminal states are available in info["final_observation"].
    render_mode: Literal["human", "rgb_array"] | None = None
    render_envs: tuple[int, ...] = (0,)
    render_fps: int = 50

    action_dim: int = field(init=False, default=1)
    observation_dim: int = field(init=False, default=4)
    state_dim: int = field(init=False, default=4)

    def __post_init__(self) -> None:
        if self.num_envs <= 0 or self.max_episode_length <= 0:
            raise ValueError("num_envs and max_episode_length must be positive")
        if self.action_mode not in ("discrete", "continuous"):
            raise ValueError("action_mode must be 'discrete' or 'continuous'")
        if self.kinematics_integrator not in ("euler", "semi-implicit"):
            raise ValueError("unsupported kinematics_integrator")
        if self.physics_dtype not in ("float32", "float64"):
            raise ValueError("physics_dtype must be 'float32' or 'float64'")
        if self.render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if self.render_fps <= 0:
            raise ValueError("render_fps must be positive")
        positive = (
            self.gravity, self.masscart, self.masspole, self.length,
            self.force_mag, self.tau, self.x_threshold, self.theta_threshold_radians,
        )
        if not all(math.isfinite(v) and v > 0.0 for v in positive):
            raise ValueError("physical parameters must be finite and positive")
        if not all(math.isfinite(v) for v in (self.reset_low, self.reset_high)):
            raise ValueError("reset bounds must be finite")
        if self.reset_low >= self.reset_high:
            raise ValueError("reset_low must be smaller than reset_high")
        if any(i < 0 or i >= self.num_envs for i in self.render_envs):
            raise ValueError("render_envs contains an invalid environment index")
