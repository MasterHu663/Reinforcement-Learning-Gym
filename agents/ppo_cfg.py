from __future__ import annotations
import dataclasses

from typing import Callable

import datetime

class ExperimentConfig:
    def __init__(self):
        self.experiment_name: str = "PPO"
        self.write_interval: str | int = "auto"
        self.checkpoint_interval: str | int = "auto"
        self.directory: str = "runs/torch/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.store_separately: bool = False

@dataclasses.dataclass(kw_only=True)
class PPOConfig:
    
    experiment: ExperimentConfig = dataclasses.field(default_factory=ExperimentConfig)

    # Low-level PPO settings reproduced for the hover controller.
    rollouts: int = 1000
    learning_epochs: int = 40
    mini_batches: int = 4
    discount_factor: float = 0.99
    # lambda=1 gives the Monte-Carlo return target used by the reference
    # implementation while retaining the shared rollout code path.
    gae_lambda: float = 1.0
    # Actor / critic learning rates.
    learning_rate: float | tuple[float, float] = (1.0e-4, 2.0e-4)
    min_learning_rate: float = 1.0e-6
    max_learning_rate: float = 1.0e-3
    learning_rate_scheduler: type | tuple[type | None, type | None] | None = None
    learning_rate_scheduler_kwargs: dict | tuple[dict, dict] = dataclasses.field(default_factory=dict)
    
    # TODO How to define the preprocessor
    observation_preprocessor: type | None = None
    observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    state_preprocessor: type | None = None
    state_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    value_preprocessor: type | None = None
    value_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    
    random_timesteps: int = 0
    learning_starts: int = 0
    grad_norm_clip: float = 0.5
    ratio_clip: float = 0.2
    # Values are trained against unnormalized returns containing +/-200
    # terminal rewards, so a 0.2-wide value update window is far too narrow.
    value_clip: float = 10.0
    entropy_loss_scale: float = 0.01
    value_loss_scale: float = 0.5
    kl_threshold: float = 0.02
    time_limit_bootstrap: bool = False
    rewards_shaper: Callable | None = None
    mixed_precision: bool = False
    # Hierarchical-navigation compatibility with the reference Classic_PPO.
    reference_mode: bool = False
    # Record asynchronous high-level decisions as one event per memory row.
    # stream_ids preserve each environment's independent return trajectory.
    stream_rollouts: bool = False
    auto_update: bool = True
    
    def expand(self) -> None:
        pass
