# Flappy Bird Pixel Environment

Run from the project root in the `rl-gym` environment with Torch, Gymnasium, and Pygame installed:

```bash
python -m environments.flappy_bird.demo
```

The demo uses a simple controller that follows pipe gaps rather than a trained policy. Close the window or press Esc to exit.
By default, it runs one episode on the CPU and displays the native 128×64 image at 6× scale. Use `--scale 8` to change the window size.

Export a preview at native resolution without opening a window (requires Pillow):

```bash
python -m environments.flappy_bird.demo --output /tmp/flappy.gif --steps 160
python -m environments.flappy_bird.demo --output /tmp/flappy.png --steps 90
```

## Manual Play

Run from the project root:

```bash
python -m environments.flappy_bird.play
```

- Space / Up arrow: start the game or flap once per key press. Holding the key does not trigger repeated flaps.
- P: pause / resume. R: reset and wait for the next start.
- Esc / close window: exit.
- After a crash, the final frame, score, and step count remain visible. Press Space or Up arrow to start the next episode.

By default, manual play uses one CPU environment and a 9×6 bird, matching the current training entry point. The native image is 128×64,
displayed at 6× scale with controls shown below it. There are no step or score limits by default, so play can continue indefinitely.
Use command-line options to change the bird dimensions and random seed:

```bash
python -m environments.flappy_bird.play --player-width 9 --player-height 6 --seed 42
```

Manual play, demos, training, and evaluation share the default physics and geometry settings in `envConfig.py`: 20 Hz (50 ms per step),
flap velocity -4, gravity 1, and maximum falling speed 3. Flaps are ignored when the bird's top is at y <= 3. There are 4 pipe pairs, each 10 pixels wide,
with a gap of 19 and spacing of 40. Pipes move left by 1 pixel per step; the first starts at x=96, and gap tops are sampled as integers from 5 through 34.
Edit this configuration file to adjust these parameters. Training still truncates episodes at 1000 steps or 30 points; manual play has no limits.

## Rendering Interface

```python
import torch
from environments.flappy_bird.envConfig import FlappyBirdEnvConfig
from environments.flappy_bird.envForRL import FlappyBirdEnv

cfg = FlappyBirdEnvConfig(
    num_envs=2, device="cpu", render_mode="human", render_envs=(0, 1),
)
env = FlappyBirdEnv(cfg)
try:
    observation, info = env.reset(seed=42)
    for _ in range(100):
        observation, reward, terminated, truncated, info = env.step(
            torch.randint(2, (cfg.num_envs,), device=env.device),
        )
        if env.render_closed:
            break
finally:
    env.close()
```

- `render_mode="human"` displays each frame after collision detection and before autoreset. Multiple `render_envs` are displayed side by side.
- With `render_mode="rgb_array"`, `env.render()` returns a list of native `(64, 128, 3)` RGB images with dtype `uint8`. `render_scale` affects only the window, not the array dimensions.
- The bird uses three PNG frames from `assets`, preserving transparency. Flapping animation and ground scrolling are driven by `episode_length_buf`; rendering the same state repeatedly does not advance the animation.
- Pipes, the bird, and the ground are drawn at the environment's pixel coordinates. The score and the episode's `STEPS` counter in the lower left use 3×5 pixel glyphs.
  Each environment displays its own step count, which returns to zero on reset. Both window and RGB output include the counter. Images use nearest-neighbor scaling only.
- `configure_rendering(False)` disables automatic rendering within `reset/step`; `configure_rendering(True, interval=4)` displays one frame every four physics steps. This is compatible with the current `normalTrainer` training and evaluation calls.
- Until rendering is requested, Pygame is not imported, images are not loaded, and state is not copied to the CPU.

## PPO Training and Evaluation

Run from the project root:

```bash
python train_eval.py --env flappy_bird --headless --num-envs 256 --rollouts 128 --timesteps 32000
python train_eval.py --env flappy_bird --eval --num-envs 1 --checkpoint checkpoints/<timestamp>/flappy_bird_final_model.pt
```

A full PPO checkpoint is saved after training. Evaluation selects the action with the largest logit and displays the trained policy in the window.
Statistics are collected over the number of episodes specified by `--eval-episodes`. Press Ctrl+C to exit. Use `--device cpu` to select the CPU.
`--timesteps` is the number of steps per parallel environment; total interactions equal this value multiplied by `--num-envs`.
These settings provide a starting point for training and do not imply that a high-scoring policy has already been obtained.

To display the environments selected by `render_envs` during training, omit `--headless`. The entry point displays the first two by default.
For example, refresh the display every 4 steps:

```bash
python train_eval.py --env flappy_bird --num-envs 256 --rollouts 128 --timesteps 32000 --render-interval 4
```

The window shows the actions actually sampled during training. Rendering slows training down; training continues after the Flappy Bird window is closed.

- The default actor is `15 → 64 → 64 → 2`, with ReLU hidden layers and two raw output logits.
  Training samples from a Categorical distribution: action 0 means no flap, and action 1 means flap.
- `action_dim=1` means each output contains one action index; `num_actions=2` means there are two available actions.
- The critic uses a separate network with a scalar output. PPO logs entropy for discrete policies and does not read the Categorical distribution's standard deviation.
- The current state already includes vertical velocity, so additional frame stacking is unnecessary.
- MLP inference on an MCU needs only the actor's `trunk`, `action_logits_head`, and argmax. This includes 5314 parameters,
  or about 20.8 KiB of FP32 weights. MCU export is not implemented yet, and deployment must reproduce the observation sorting and normalization used during training.
- The previous model with 3 pipe pairs had 12 input dimensions and cannot be loaded directly into the current network with 15 input dimensions. Train from scratch with the new configuration.
- The original CartPole entry point remains available through `--env cartpole`; CartPole is also the default.

### Trying Pipe Attention

`--flappy-model attention` enables a separate single-head attention module for the actor and critic. Each pipe's
`(x, gap_top, gap_bottom)` is mapped to a shared 16-dimensional encoding, while the bird's `(y, vy, rotation)` generates the query.
The weighted sum of pipe features is combined with the bird state and fed into a `[64, 64]` MLP. Attention is computed only among pipes
within the same environment. `--flappy-model mlp` is the default; both options use the same environment and PPO settings.

Start a comparison with the same seed, number of sampling steps, and number of parallel environments:

```bash
python train_eval.py --env flappy_bird --flappy-model mlp --headless --seed 42 --timesteps 32000
python train_eval.py --env flappy_bird --flappy-model attention --headless --seed 42 --timesteps 32000
```

Attention may help the model focus on relevant pipes, but it does not guarantee better sample efficiency or higher training throughput.
Both models use an MLP with two hidden layers of 64 units. The attention actor has 5954 inference parameters (excluding the value head),
compared with 5314 for the MLP, so the comparison also reflects the effects of extra parameters and a different feature aggregation method.
Compare evaluation scores at the same interaction count and repeat across multiple random seeds. Training rewards include survival rewards and are not equivalent to pipe scores.
The current entry point uses 256 parallel Flappy Bird environments, giving `timesteps × 256` interactions.

Logs are stored under `PPO_mlp` and `PPO_attention` in the run directory. The final attention model is saved as
`flappy_bird_attention_final_model.pt`; pass `--flappy-model attention` when evaluating or resuming training:

```bash
python train_eval.py --env flappy_bird --flappy-model attention --eval \
  --checkpoint checkpoints/<timestamp>/flappy_bird_attention_final_model.pt
```

The two architectures have incompatible weights, so train from scratch when first trying the attention model. Deploying the attention model also requires its pipe encoder,
query and key projections, and softmax weighting; exporting only the final MLP is insufficient.

Initial short experiments (training seeds 42 and 43; 4096 steps × 256 environments per run, or 1,048,576 interactions):

| Model | Mean evaluation score, seed 42 | Mean evaluation score, seed 43 | Mean training time across both runs |
| --- | ---: | ---: | ---: |
| MLP `[64, 64]` | 1.15625 | 0.6875 | 29.1 s |
| Attention + `[64, 64]` | 0.28125 | 0.6875 | 30.8 s |

Each evaluation uses 32 environments with seed 1701 and deterministic actions, recording only the first episode in each environment, capped at 1000 steps or 30 points.
PPO uses rollout=128, 4 learning epochs, 4 minibatches, an initial learning rate of 3e-4, and GAE lambda=0.95,
with value bootstrapping at truncation enabled. Runs use an RTX 4060 Laptop GPU with rendering, logging, and periodic saving disabled. Timings exclude evaluation and initialization.
This early comparison did not show a sample-efficiency advantage for attention. Two seeds and short training runs are insufficient to assess final convergence.

## Current Environment Status

Completed fixes:

- `cfg.observation_dim/state_dim` matches the returned dimensions: `3 * pipe_count + 3`, or 15 by default. Observations contain sorted pipe `(x, gap_top, gap_bottom)` features and bird `(y, vy, rotation)` features.
- `configure_rendering()` is implemented for the trainer. Both training and evaluation control rendering through `--headless` and `--render-interval`.
- `metadata["render_fps"]` uses `cfg.tick_hz`, which defaults to 20.
- RGB output and the window share the same native pixel canvas, including the score and bird sprites.
- Categorical policies, the discrete MLP, and the PPO training entry point are integrated. Action buffers use integers.
- `PPOConfig(time_limit_bootstrap=True)` enables value bootstrapping at truncation. PPO's `record_transition()` reads final observations directly from
  `infos["final_observation"]` for rows selected by `truncated & ~terminated`, then adds
  `gamma * V(final_observation)`. The trainer passes the original `next_observations/next_states`; the next interaction uses observations after reset.
  This option is disabled by default. Truncation does not occur when both `max_episode_length` and `score_limit` are `None`.

Remaining training and hardware features:

1. **Lidar**: batched scanning is not implemented yet. The supported mode is currently `state_mode="Lidar-off"`.
2. **Lookup tables and MCU export**: discrete state ranges, table size estimates, and C constant export are available. Discrete state indexing, trained policy table export, and SSD1309 monochrome framebuffer conversion are not implemented yet. `obs_include_next_gap` currently affects only the planned table dimensions, not the current MLP observations.

The environment uses same-step autoreset: completed rows in the returned `observation` already belong to a new episode. The completed step's scores, step counts, and final observations are stored in `info`. `time_outs` excludes environments that also terminated on that step.

Run the relevant checks:

```bash
python -m unittest discover -s tests -p 'test_flappy_bird_*.py'
python -m unittest discover -s tests -p 'test_categorical_ppo.py'
python -m unittest discover -s tests -p 'test_pipe_attention.py'
```
