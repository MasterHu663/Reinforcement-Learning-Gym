"""Train or evaluate the parallel CartPole environment with the project's PPO.

Run from the project root (which also contains agents/, models/, etc.):

    python -m cartpole_parallel.train_eval --headless
    python -m cartpole_parallel.train_eval --eval --checkpoint checkpoints/.../final_model.pt

This uses the *continuous-force* CartPole variant because the project's PPO
and MLPModel are configured for Gaussian, continuous actions.  Standard
Gymnasium CartPole-v1 has Discrete(2) actions and requires a categorical PPO
policy, which is not configured in this entrypoint.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

from loguru import logger
import torch

from agents.ppo import PPO
from agents.ppo_cfg import PPOConfig
from memories.memory import Memory
from models.mlpModel import MLPModel
from trainers.normalTrainer import normalTrainer
from wrappers.wrapper import Wrapper

from environments.cartpole.envConfig import CartPoleEnvConfig
from environments.cartpole.envForRL import CartPoleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained policy")
    parser.add_argument("--headless", action="store_true", help="Disable rendering")
    parser.add_argument(
        "--render-interval", type=int, default=1,
        help="Render every N physics steps (CartPole has one physics step per action)",
    )
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel CartPole instances")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--max-episode-length", type=int, default=500)
    parser.add_argument("--rollouts", type=int, default=None, help="Override default rollout length")
    parser.add_argument("--timesteps", type=int, default=None, help="Override trainer time steps")
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    # Keep parse_known_args() behavior from the supplied training entrypoint.
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.max_episode_length <= 0:
        raise ValueError("--num-envs and --max-episode-length must be positive")
    if args.render_interval <= 0 or args.eval_episodes <= 0:
        raise ValueError("--render-interval and --eval-episodes must be positive")
    if args.rollouts is not None and args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    if args.timesteps is not None and args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg = CartPoleEnvConfig(
        num_envs=args.num_envs,
        device=device,
        max_episode_length=args.max_episode_length,
        action_mode="continuous",  # Matches this project's Gaussian PPO.
        render_envs=(0,1),
        seed=args.seed,
        # The trainer controls when render() is called, as in the source script.
        render_mode=None,
    )

    rollouts = args.rollouts or max(
        1, int(256 * cfg.max_episode_length / cfg.num_envs)
    )
    ppo_cfg = PPOConfig(rollouts=rollouts)

    env = Wrapper(CartPoleEnv(cfg))
    device = env.device

    model_kwargs = {
        "state_dim": cfg.observation_dim,
        "action_dim": cfg.action_dim,
        "hidden_dims": [128, 128],
    }
    policy_model = MLPModel(**model_kwargs, squash_actions=True).to(device)
    value_model = MLPModel(**model_kwargs).to(device)
    # Keep the critic neutral initially, matching the supplied training script.
    torch.nn.init.zeros_(value_model.value_head.weight)
    torch.nn.init.zeros_(value_model.value_head.bias)
    models = {"policy": policy_model, "value": value_model}

    memory = Memory(
        memory_size=rollouts,
        num_envs=env.num_envs,
        device=device,
    )
    agent = PPO(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )

    total_steps = args.timesteps or int(250 * rollouts)
    trainer_cfg = {
        "timesteps": total_steps,
        "headless": args.headless,
        "disable_progressbar": False,
        "environment_info": "env_info",
        "render_interval": args.render_interval,
    }
    trainer = normalTrainer(
        env=env,
        cfg=trainer_cfg,
        agents=agent,
        device=device,
    )

    try:
        if args.checkpoint:
            if not os.path.isfile(args.checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
            agent.load(args.checkpoint)
            logger.info(f"Loaded checkpoint: {args.checkpoint}")

        if args.eval:
            logger.info("Running CartPole in evaluation mode...")
            trainer.eval(num_eval_episodes=args.eval_episodes)
        else:
            logger.info(
                f"Training CartPole | num_envs={cfg.num_envs} | "
                f"rollouts={rollouts} | timesteps={total_steps} | device={device}"
            )
            trainer.train()
            checkpoint_dir = Path("checkpoints") / datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            final_model_path = checkpoint_dir / "cartpole_final_model.pt"
            agent.save(str(final_model_path))
            logger.info(f"Saved final trained agent to: {final_model_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
