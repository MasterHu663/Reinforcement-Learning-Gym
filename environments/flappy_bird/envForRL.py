#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Implementation of a Flappy Bird OpenAI gymnasium environment that yields simple
numerical information about the game's state as observations.

Some of the code in this module is an adaption of the code in the `FlapPyBird`
GitHub repository by `sourahbhv` (https://github.com/sourabhv/FlapPyBird),
released under the MIT license.
"""


from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from .envConfig import FlappyBirdEnvConfig
from gymnasium.vector.utils import batch_space



class FlappyBirdEnv(gym.Env):
    """Batched integer physics with same-step autoreset and Torch observations."""

    is_vector_env = True

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
            self, 
            cfg: FlappyBirdEnvConfig
    ):

        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; use device='cpu'")
            torch.cuda.set_device(self.device)
        self._dtype = torch.int32 if cfg.physics_dtype == "int32" else torch.int16
        self.render_mode = cfg.render_mode
        self.metadata = {**self.metadata, "render_fps": cfg.tick_hz}

        self.single_action_space = gym.spaces.Discrete(2)

        if cfg.state_mode != "Lidar-off":
            raise NotImplementedError(
                "Batched Lidar observations are not implemented; use state_mode='Lidar-off'"
            )
        raw_space = gym.spaces.Box(
            -1.0, 1.0, shape=(cfg.observation_dim,), dtype=np.float32,
        )
        self.single_observation_space = gym.spaces.Dict(policy=raw_space, critic=raw_space)
        self.single_state_space = raw_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.state_space = batch_space(self.single_state_space, self.num_envs)



        #Runtime return
        
        # Three dynamic state variables for the bird and two per pipe pair.
        physics_state_dim = 3 + 2 * cfg.pipe_count
        '''
        [
            bird_y,
            bird_vy,
            bird_rot,

            pipe_0_x,
            pipe_0_gap_y,

            pipe_1_x,
            pipe_1_gap_y,

            pipe_2_x,
            pipe_2_gap_y,
        ]
        '''
        self.state_buf = torch.zeros(
            (self.num_envs, physics_state_dim),
            dtype=self._dtype,
            device=self.device,
        )
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        #game score
        self.score_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_truncated = torch.zeros_like(self.reset_terminated)
        
        #Seed reset
        self._rng = torch.Generator(device=self.device)
        if cfg.seed is not None:
            self._rng.manual_seed(cfg.seed)

        self._reset_low = cfg.reset_low
        self._reset_high = cfg.reset_high
        self._has_reset = False
        self._renderer = None
        self._render_during_step = None  # None follows render_mode; the trainer can override it.
        self._render_interval = 1
        self._steps_since_render = 0

    @property
    def state(self) -> torch.Tensor:

        return self.state_buf

    def _sample_pipe_gap_y(self, size: tuple[int, ...]) -> torch.Tensor:
        """Sample pipe opening top rows in pixels, including both reset bounds."""
        return torch.randint(
            low=self._reset_low,
            high=self._reset_high + 1,
            size=size,
            generator=self._rng,
            device=self.device,
            dtype=self._dtype,
        )

    def _reset_indices(self, ids: torch.Tensor) -> None:
        """Reset selected rows: bird (y, vy, rotation), then pipe (x, gap_y) pairs.
        
        [y vy rot x1 y1 x2 y2 x3 y3]
        0  1  2   3  4  5  6   7  8
        """
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return

        cfg = self.cfg

        # Bird: y in whole pixels, vy in pixels/tick, rotation in degrees.
        self.state_buf[ids, 0] = cfg.player_start_y_px
        self.state_buf[ids, 1] = cfg.flap_velocity
        self.state_buf[ids, 2] = 45

        # Pipe x positions occupy columns 3, 5, 7, ... (whole pixels).
        pipe_x = cfg.pipe_first_x + cfg.pipe_spacing * torch.arange(
            cfg.pipe_count, device=self.device, dtype=self._dtype,
        )
        self.state_buf[ids, 3::2] = pipe_x

        self.state_buf[ids, 4::2] = self._sample_pipe_gap_y(
            (ids.numel(), cfg.pipe_count)
        )
        self.episode_length_buf[ids] = 0
        self.score_buf[ids] = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.manual_seed(seed)
        options = options or {}
        low = options.get("low", self.cfg.reset_low)
        high = options.get("high", self.cfg.reset_high)
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (low, high)):
            raise ValueError("reset bounds must be integer pixel rows")
        if not 1 <= low <= high <= self.cfg.ground_y - self.cfg.pipe_gap - 1:
            raise ValueError("reset bounds must leave at least one pipe row above and below")
        self._reset_low, self._reset_high = int(low), int(high)

        self._reset_indices(torch.arange(self.num_envs, device=self.device))
        self.reward_buf.zero_()
        self.reset_terminated.zero_()
        self.reset_truncated.zero_()
        self._has_reset = True
        self._steps_since_render = 0
        if self._render_during_step is True or (
            self._render_during_step is None and self.render_mode == "human"
        ):
            self.render()
        return self._pack_observation(), {"score": self.score_buf.clone()}

    def _move_pipes_and_update_score(self) -> torch.Tensor:
        """Move, score, and recycle pipes; return counts of shape (num_envs,).

        Score before recycling offscreen pipes. A pipe scores when its center
        crosses from strictly right of the bird center to at or left of it.
        Pipe storage order does not matter.
        """
        cfg = self.cfg
        pipe_x = self.state_buf[:, 3::2]  # (num_envs, pipe_count), a view
        gap_y = self.state_buf[:, 4::2]

        # Double center coordinates to preserve half-pixel centers for odd widths without floating-point arithmetic.
        # This creates a new tensor, so moving pipe_x in place below preserves the previous positions.
        player_mid2 = 2 * cfg.player_x + cfg.player_width
        previous_mid2 = 2 * pipe_x.to(dtype=torch.int64) + cfg.pipe_width

        # pipe_speed is positive; subtract it to move left, once per step.
        pipe_x.sub_(cfg.pipe_speed)
        current_mid2 = 2 * pipe_x.to(dtype=torch.int64) + cfg.pipe_width

        passed = (previous_mid2 > player_mid2) & (current_mid2 <= player_mid2)
        passed_count = passed.sum(dim=1)  # Number of pipes passed this frame in each environment.
        self.score_buf.add_(passed_count)

        # Once a pipe's right edge leaves the screen, place it after the rightmost pipe in that environment.
        offscreen = pipe_x <= -cfg.pipe_width
        rightmost_x = pipe_x.amax(dim=1, keepdim=True).to(dtype=torch.int32)
        # Space out pipes recycled in the same frame to avoid overlap.
        recycle_rank = offscreen.cumsum(dim=1, dtype=torch.int32)
        new_x = rightmost_x + recycle_rank * cfg.pipe_spacing
        # pipe_x is a view of state_buf; write back in place instead of rebinding the variable.

        pipe_x.copy_(torch.where(offscreen, new_x, pipe_x))
        gap_y.copy_(torch.where(
            offscreen, self._sample_pipe_gap_y(gap_y.shape), gap_y,
        ))
        return passed_count

    def step(
        self,
        action: torch.Tensor | np.ndarray,
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Advance every row once, then reset only finished environments.

        Rewards and done flags have shape (num_envs,). Returned observations
        contain reset states for finished rows; info['final_observation'] holds
        the pre-reset features, selected by info['_final_observation'].
        """

        if not self._has_reset:
            raise RuntimeError("Call reset() before step()")
        cfg = self.cfg

        action = torch.as_tensor(action, device=self.device)
        # 1. Convert (num_envs, 1) to (num_envs,).
        if action.shape == (self.num_envs, 1):
            action = action[:, 0]

        # 2. Ensure there is exactly one action per environment.
        if action.shape != (self.num_envs,):
            raise ValueError(
                f"Expected actions shape ({self.num_envs},) "
                f"or ({self.num_envs}, 1), got {tuple(action.shape)}"
            )

        # 3. Validate action values as configured, regardless of the original input shape.
        if cfg.validate_actions:
            valid = (action == 0) | (action == 1)
            if not bool(torch.all(valid)):
                raise ValueError("Discrete FlappyBird actions must be 0 or 1")

        by, byv, rot = self.state_buf[:, :3].unbind(dim=1)

        # Determine whether flapping is allowed in each environment.
        flap = (action == 1) & (by > cfg.flap_min_y_px)

        # 1. Update velocity.
        # Flapping: set the upward velocity directly.
        # Otherwise: apply gravity and cap the falling speed.
        next_byv = torch.where(
            flap,
            torch.full_like(byv, cfg.flap_velocity),
            (byv + cfg.gravity).clamp(max=cfg.max_fall_velocity),
        )

        # 2. Update the internal angle.
        # Flapping: reset to 45 degrees.
        # Otherwise: decrease by 3 degrees per step, down to -90 degrees.
        next_rot = torch.where(
            flap,
            torch.full_like(rot, 45),
            (rot - 3).clamp(min=-90),
        )

        # 3. Clamp at the ceiling, retain the hit flag for rewards, and zero the upward velocity.
        next_by = by + next_byv
        hit_ceiling = next_by < 0
        next_by.clamp_(min=0, max=cfg.player_max_y_px)
        next_byv.masked_fill_(hit_ceiling, 0)


        # 4. Write the updated bird values back to the environment state.
        self.state_buf[:, :3] = torch.stack(
            (next_by, next_byv, next_rot),
            dim=1,
        )



        # 5. Move pipes, update scores, then recycle offscreen pipes.

        passed_count = self._move_pipes_and_update_score()
        # Scores accumulate; rewards apply to this frame. Passing a pipe overrides the survival reward.
        self.reward_buf.fill_(cfg.reward_alive)
        self.reward_buf.masked_fill_(passed_count > 0, cfg.reward_pipe)
        self.reward_buf.masked_fill_(hit_ceiling, cfg.reward_ceiling)

        # 6. Check termination and truncation per environment; the crash reward takes priority.
        self.episode_length_buf.add_(1)
        self.reset_terminated.copy_(self._check_crash())
        self.reward_buf.masked_fill_(self.reset_terminated, cfg.reward_crash)
        self.reset_truncated.zero_()
        if cfg.score_limit is not None:
            self.reset_truncated.logical_or_(self.score_buf >= cfg.score_limit)
        if cfg.max_episode_length is not None:
            self.reset_truncated.logical_or_(self.episode_length_buf >= cfg.max_episode_length)
        done = self.reset_terminated | self.reset_truncated

        # 7. Save results before reset for logging and value estimates at time-limit truncations.
        final_observation, _ = self._get_observation_features()
        info = {
            "score": self.score_buf.clone(),
            "episode_length": self.episode_length_buf.clone(),
            "reset_mask": done,
            "final_observation": final_observation,
            "_final_observation": done,
            "time_outs": self.reset_truncated & ~self.reset_terminated,
        }

        # Collisions and rewards are finalized; render this frame before resetting finished environments.
        if self._render_during_step is True or (
            self._render_during_step is None and self.render_mode == "human"
        ):
            self._steps_since_render += 1
            if self._steps_since_render >= self._render_interval:
                self.render()
                self._steps_since_render = 0

        reset_ids = torch.nonzero(done, as_tuple=True)[0]
        self._reset_indices(reset_ids)
        if reset_ids.numel():
            observation = self._pack_observation()
        else:
            observation = {"policy": final_observation, "critic": final_observation}

        return (
            observation,
            self.reward_buf.clone(),
            self.reset_terminated.clone(),
            self.reset_truncated.clone(),
            info,
        )

    def configure_rendering(self, enabled: bool, interval: int = 1) -> None:
        """Let the trainer disable rendering or display every N physics steps."""
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("render interval must be a positive integer")
        self._render_during_step = bool(enabled)
        self._render_interval = interval
        self._steps_since_render = 0

    @property
    def render_closed(self) -> bool:
        """Whether the user closed the preview window (physics may continue)."""
        return self._renderer is not None and self._renderer.closed

    def render(self):
        """Draw selected rows; rgb_array returns one native RGB image per row."""
        if not self._has_reset:
            return None
        if self._renderer is None:
            from .renderer import FlappyBirdRenderer
            self._renderer = FlappyBirdRenderer(self.cfg)
        return self._renderer.update(
            self.state_buf, self.score_buf,
            episode_lengths=self.episode_length_buf, render_mode=self.render_mode,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

    def _check_crash(self) -> torch.Tensor:
        """Return a bool tensor of shape (num_envs,) from real pixel coordinates.

        Use the axis-aligned rectangles defined in envConfig: upper pipes span
        [0, gap_y), lower pipes span [gap_y + pipe_gap, ground_y). Merely
        touching a pipe edge is allowed; touching the ground ends the episode.
        """
        cfg = self.cfg
        # Promote int16 before adding dimensions, to avoid coordinate overflow.
        state = self.state_buf.to(dtype=torch.int32)
        bird_top = state[:, 0:1]  # (num_envs, 1), broadcast over all pipes
        bird_bottom = bird_top + cfg.player_height
        pipe_x = state[:, 3::2]  # (num_envs, pipe_count)
        gap_y = state[:, 4::2]

        # A pipe collision requires horizontal overlap and vertical overlap with an upper or lower pipe.
        horizontal_overlap = (
            (cfg.player_x < pipe_x + cfg.pipe_width)
            & (cfg.player_x + cfg.player_width > pipe_x)
        )
        upper_overlap = (bird_top < gap_y) & (bird_bottom > 0)
        lower_overlap = (
            (bird_bottom > gap_y + cfg.pipe_gap)
            & (bird_top < cfg.ground_y)
        )
        # Reduce only over the pipe dimension to get a separate collision result per environment.
        hit_pipe = (horizontal_overlap & (upper_overlap | lower_overlap)).any(dim=1)
        hit_ground = bird_bottom[:, 0] >= cfg.ground_y
        return hit_pipe | hit_ground

    def _get_observation_features(
        self, *, normalize: bool = True,
    ) -> tuple[torch.Tensor, None]:
        """Build (num_envs, 3 * pipe_count + 3) float32 observations on-device.

        Each row contains pipe (x, gap_top, gap_bottom) triples sorted by x,
        followed by bird (y, vy, rotation). Raw observations keep offscreen
        coordinates; normalized pipe x coordinates are clipped to [0, 1].
        The second return value retains the existing extra-reward interface.
        """
        cfg = self.cfg
        state = self.state_buf.to(dtype=torch.float32)
        pipe_x = state[:, 3::2]  # (num_envs, pipe_count)
        gap_y = state[:, 4::2]

        # Sort by x within each environment, keeping each gap aligned with its pipe.
        # gather creates observation data without changing the pipe order in state_buf.
        order = pipe_x.argsort(dim=1)
        pipe_x = pipe_x.gather(dim=1, index=order)
        gap_y = gap_y.gather(dim=1, index=order)
        pipes = torch.stack(
            (pipe_x, gap_y, gap_y + cfg.pipe_gap), dim=-1,
        )  # (num_envs, pipe_count, 3)
        bird = state[:, :3]  # (num_envs, 3)

        if normalize:
            pipes = pipes / pipes.new_tensor(
                [cfg.screen_width, cfg.screen_height, cfg.screen_height]
            )
            # pipes.shape = (num_envs, pipe_count, 3)
            # Last dimension: [x, gap_top, gap_bottom].
            # Clip only the observation's x coordinates to [0, 1].
            pipes[..., 0] = pipes[..., 0].clamp(min=0.0, max=1.0)

            bird = bird / bird.new_tensor([
                cfg.screen_height,
                max(abs(cfg.flap_velocity), cfg.max_fall_velocity),
                90.0,
            ])

        observation = torch.cat((pipes.flatten(start_dim=1), bird), dim=1)
        return observation, None

    def _pack_observation(self) -> dict[str, torch.Tensor]:
        observation, _ = self._get_observation_features()
        return {"policy": observation, "critic": observation}


"""
Assume the following configuration:

```python
num_envs = 2
pipe_count = 3

screen_width = 128
screen_height = 64
pipe_gap = 20

flap_velocity = -3
max_fall_velocity = 2
```

The values below were verified by running the example.

**1. The initial `state_buf`**

Each row stores values in this order:

```text
[bird_y, bird_vy, bird_rotation, pipe0_x, pipe0_gap_y, pipe1_x, pipe1_gap_y, pipe2_x, pipe2_gap_y]
```

Assume the current state is:

```python
self.state_buf = tensor([
    [24, -3,  45, 192,  8,  64, 16, 128, 24],  # Environment 0
    [32,  0, -90,  96, 28, 160, 12,  32, 20],  # Environment 1
], dtype=torch.int32)

# shape: (2, 9)
```

For example, in environment 0 the bird has `y=24` and velocity `vy=-3`. The three pipe pairs have x coordinates `192, 64, 128`, which are not yet ordered from left to right.

Run:

```python
cfg = self.cfg
state = self.state_buf.to(dtype=torch.float32)
```

`cfg` is a convenient reference to the configuration. `state` keeps the same shape and values, but its dtype becomes floating point:

```python
state = tensor([
    [24., -3.,  45., 192.,  8.,  64., 16., 128., 24.],
    [32.,  0., -90.,  96., 28., 160., 12.,  32., 20.],
])

# shape: (2, 9)
```

The original integer `state_buf` is not converted to floating point in place.

**2. Extract the pipe x coordinates and gap tops**

```python
pipe_x = state[:, 3::2]
gap_y = state[:, 4::2]
```

This gives:

```python
pipe_x = tensor([
    [192.,  64., 128.],  # The three x coordinates in environment 0
    [ 96., 160.,  32.],  # The three x coordinates in environment 1
])

gap_y = tensor([
    [ 8., 16., 24.],    # The three gap tops in environment 0
    [28., 12., 20.],    # The three gap tops in environment 1
])

# Both shapes are (2, 3)
```

**Values at the same row and column belong to the same pipe pair.** For example, pipe pair 0 in environment 0 has:

```python
x = 192
gap_y = 8
```

**3. Use `argsort()` to find the order from left to right**

```python
order = pipe_x.argsort(dim=1)
```

`dim=1` sorts within each row, keeping different environments separate.

The result is:

```python
order = tensor([
    [1, 2, 0],
    [2, 0, 1],
])

# shape: (2, 3)
```

**The returned values are the original column indices, not the sorted coordinates.**

For environment 0, the mapping is:

```text
Original coordinates: [192, 64, 128]
Original indices:     [  0,  1,   2]

Ascending order:      [ 64, 128, 192]
Corresponding indices:[  1,   2,   0]
```

Therefore, `order` for environment 0 is `[1, 2, 0]`.

**4. Use `gather()` to retrieve values in this order**

```python
pipe_x = pipe_x.gather(dim=1, index=order)
gap_y = gap_y.gather(dim=1, index=order)
```

Result:

```python
pipe_x = tensor([
    [64., 128., 192.],
    [32.,  96., 160.],
])

gap_y = tensor([
    [16., 24.,  8.],
    [20., 28., 12.],
])

# Both shapes remain (2, 3)
```

`gap_y` must use the same `order` to keep pipe positions aligned with their gap positions:

```text
Environment 0 before sorting:
(x=192, gap=8), (x=64, gap=16), (x=128, gap=24)

Environment 0 after sorting:
(x=64, gap=16), (x=128, gap=24), (x=192, gap=8)
```

This changes only the observation order; **it does not reorder `state_buf`**.

**5. Use `stack()` to group the three features of each pipe pair**

First, compute the gap bottoms:

```python
gap_y + cfg.pipe_gap
```

Since the gap height is `20`, this gives:

```python
tensor([
    [36., 44., 28.],
    [40., 48., 32.],
])
```

Then run:

```python
pipes = torch.stack(
    (pipe_x, gap_y, gap_y + cfg.pipe_gap),
    dim=-1,
)
```

This gives:

```python
pipes = tensor([
    # Environment 0
    [
        [ 64., 16., 36.],
        [128., 24., 44.],
        [192.,  8., 28.],
    ],

    # Environment 1
    [
        [ 32., 20., 40.],
        [ 96., 28., 48.],
        [160., 12., 32.],
    ],
])

# shape: (2, 3, 3)
# Dimensions: environments, pipes, features
```

`dim=-1` places the new feature dimension last, so:

```python
pipes[0, 1]
# tensor([128., 24., 44.])
```

represents **`[x, gap_top, gap_bottom]` for pipe pair 1 after sorting in environment 0**.

**6. Extract the three bird state variables**

```python
bird = state[:, :3]
```

Result:

```python
bird = tensor([
    [24., -3.,  45.],
    [32.,  0., -90.],
])

# shape: (2, 3)
# Each row: [y, vy, rotation]
```

**7. Scale the values when `normalize=True`**

The divisors for the pipe features are:

```python
pipes.new_tensor([
    cfg.screen_width,
    cfg.screen_height,
    cfg.screen_height,
])

# tensor([128., 64., 64.])
# shape: (3,)
```

These divisors apply to each pipe pair in every environment:

```text
[x,  gap_top, gap_bottom]
 ↓      ↓         ↓
÷128   ÷64       ÷64
```

For example, the first pipe pair:

```python
[64, 16, 36]
# becomes
[64 / 128, 16 / 64, 36 / 64]
# equivalent to
[0.5, 0.25, 0.5625]
```

The full `pipes` tensor becomes:

```python
pipes = tensor([
    [
        [0.5000, 0.2500, 0.5625],
        [1.0000, 0.3750, 0.6875],
        [1.5000, 0.1250, 0.4375],
    ],
    [
        [0.2500, 0.3125, 0.6250],
        [0.7500, 0.4375, 0.7500],
        [1.2500, 0.1875, 0.5000],
    ],
])

# shape remains (2, 3, 3)
```

Note that `192 / 128 = 1.5`: this scaling preserves the actual positions of offscreen pipes, so **scaled values can exceed 1**.

The divisors for the bird features are:

```python
[
    64,
    max(abs(-3), 2),
    90,
]

# equivalent to [64, 3, 90]
```

This gives:

```python
bird = tensor([
    [0.375, -1.0,  0.5],  # [24/64, -3/3,  45/90]
    [0.500,  0.0, -1.0],  # [32/64,  0/3, -90/90]
])

# shape remains (2, 3)
```

**8. Use `flatten(start_dim=1)` to flatten pipe data for each environment**

```python
pipes.flatten(start_dim=1)
```

Shape change:

```text
(2, 3, 3) → (2, 9)
```

**Keep dimension 0 for environments and merge the remaining pipe and feature dimensions.**

Result:

```python
tensor([
    [0.5,  0.25,   0.5625,  1.0,  0.375,  0.6875,  1.5,  0.125,  0.4375],
    [0.25, 0.3125, 0.625,   0.75, 0.4375, 0.75,    1.25, 0.1875, 0.5],
])
```

Each row is arranged as:

```text
[three features of pipe 0, three features of pipe 1, three features of pipe 2]
```

**9. Use `cat()` to append bird features to each row**

```python
observation = torch.cat(
    (pipes.flatten(start_dim=1), bird),
    dim=1,
)
```

Shape change:

```text
Pipes:       (2, 9)
Bird:        (2, 3)
              ↓ Concatenate along columns
Observation: (2, 12)
```

The final result is:

```python
observation = tensor([
    [
        0.5,  0.25,   0.5625,   # Pipe 0
        1.0,  0.375,  0.6875,   # Pipe 1
        1.5,  0.125,  0.4375,   # Pipe 2
        0.375, -1.0,  0.5,      # Bird y, vy, rotation
    ],
    [
        0.25, 0.3125, 0.625,    # Pipe 0
        0.75, 0.4375, 0.75,     # Pipe 1
        1.25, 0.1875, 0.5,      # Pipe 2
        0.5,  0.0,   -1.0,      # Bird y, vy, rotation
    ],
])
```

Finally:

```python
return observation, None
```

The function returns two values, which can be unpacked as follows:

```python
obs, extra_reward = self._get_observation_features()

obs.shape
# torch.Size([2, 12])

extra_reward
# None indicates that this function does not compute an additional reward
```

The physical state has **9 dimensions**, while the observation has **12 dimensions** because each pipe pair contributes an additional gap bottom computed as `gap_y + pipe_gap`. This derived value does not need to be stored in `state_buf`.



"""
