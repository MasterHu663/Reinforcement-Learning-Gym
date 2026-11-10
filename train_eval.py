"""Train or evaluate parallel CartPole or Flappy Bird with the project's PPO.

Run from the project root (which also contains agents/, models/, etc.):

    python train_eval.py --env cartpole --headless
    python train_eval.py --env flappy_bird --headless --num-envs 256
    python train_eval.py --env flappy_bird --eval --checkpoint checkpoints/.../flappy_bird_final_model.pt

CartPole uses continuous Gaussian actions. Flappy Bird uses a small categorical
policy with two discrete actions and ReLU hidden layers.
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
from models.pipeAttentionModel import PipeAttentionModel
from trainers.normalTrainer import normalTrainer
from wrappers.wrapper import Wrapper

from environments.cartpole.envConfig import CartPoleEnvConfig
from environments.cartpole.envForRL import CartPoleEnv
from environments.flappy_bird.envConfig import FlappyBirdEnvConfig
from environments.flappy_bird.envForRL import FlappyBirdEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("cartpole", "flappy_bird"), default="cartpole")
    parser.add_argument(
        "--flappy-model", choices=("mlp", "attention"), default="mlp",
        help="Flappy Bird actor/critic architecture; use the same choice when loading a checkpoint",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained policy")
    parser.add_argument("--headless", action="store_true", help="Disable rendering during training or evaluation")
    parser.add_argument(
        "--render-interval", type=int, default=1,
        help="Render every N physics steps",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help="Parallel environment instances (default: Flappy Bird 256, CartPole 32)",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--max-episode-length", type=int, default=500)
    parser.add_argument("--rollouts", type=int, default=None, help="Override default rollout length")
    parser.add_argument("--timesteps", type=int, default=None, help="Override trainer time steps")
    parser.add_argument(
        "--eval-episodes", type=int, default=500,
        help="Episodes per evaluation round; repeats until Ctrl+C",
    )
    parser.add_argument("--seed", type=int, default=42)
    # Keep parse_known_args() behavior from the supplied training entrypoint.
    args, _ = parser.parse_known_args()
    if args.num_envs is None:
        args.num_envs = 256 if args.env == "flappy_bird" else 32
    return args


def main() -> None:
    args = parse_args()
    if args.env != "flappy_bird" and args.flappy_model != "mlp":
        raise ValueError("--flappy-model attention requires --env flappy_bird")
    if args.num_envs <= 0 or args.max_episode_length <= 0:
        raise ValueError("--num-envs and --max-episode-length must be positive")
    if args.render_interval <= 0 or args.eval_episodes <= 0:
        raise ValueError("--render-interval and --eval-episodes must be positive")
    if args.rollouts is not None and args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    if args.timesteps is not None and args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    env_kwargs = {
        "num_envs": args.num_envs,
        "device": device,
        "max_episode_length": args.max_episode_length,
        "render_envs": tuple(range(min(2, args.num_envs))),
        "seed": args.seed,
        # The trainer controls when render() is called.
        "render_mode": None,
    }
    if args.env == "flappy_bird":
        env_kwargs.update(
            max_episode_length=5000,
            score_limit=50,
            render_mode="human" if not args.headless else None,
        )

        # Share the bird, physics, and pipe parameters in envConfig.py with manual play.
        cfg = FlappyBirdEnvConfig(**env_kwargs)
        env = Wrapper(FlappyBirdEnv(cfg))
        rollouts = args.rollouts or 128
        ppo_cfg = PPOConfig(
            rollouts=rollouts,
            learning_epochs=4,
            learning_rate=3.0e-4,
            gae_lambda=0.95,
            time_limit_bootstrap=True
        )
        ppo_cfg.experiment.experiment_name = f"PPO_{args.flappy_model}"

        model_kwargs = {
            "hidden_dims": [64, 64],
            "num_actions": cfg.num_actions
        }

        if args.flappy_model == "attention":
            model_class = PipeAttentionModel
            model_kwargs.update(
                pipe_count=cfg.pipe_count, attention_dim=16,
                screen_width=cfg.screen_width, player_x=cfg.player_x,
                pipe_width=cfg.pipe_width,
            )
        else:
            model_class = MLPModel
            model_kwargs.update(
                state_dim=cfg.observation_dim, action_dim=cfg.action_dim,
                action_mode="discrete",
            )

        logger.info(f"Flappy Bird model: {args.flappy_model}")
    else:
        model_class = MLPModel
        cfg = CartPoleEnvConfig(**env_kwargs, action_mode="continuous")
        env = Wrapper(CartPoleEnv(cfg))
        rollouts = args.rollouts or 128
        ppo_cfg = PPOConfig(
            rollouts=rollouts, time_limit_bootstrap=True)
        model_kwargs = {
            "state_dim": cfg.observation_dim,
            "action_dim": cfg.action_dim,
            "hidden_dims": [128, 128],
            "squash_actions": True,
        }

    device = env.device
    policy_model = model_class(**model_kwargs).to(device)
    value_model = model_class(**model_kwargs).to(device)

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

            
            logger.info(f"Running {args.env} in evaluation mode...")
            trainer.eval(num_eval_episodes=args.eval_episodes)



        else:
            logger.info(
                f"Training {args.env} | num_envs={cfg.num_envs} | "
                f"rollouts={rollouts} | timesteps={total_steps} | device={device}"
            )
            trainer.train()
            checkpoint_dir = Path("checkpoints") / datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model_tag = "_attention" if args.env == "flappy_bird" and args.flappy_model == "attention" else ""
            final_model_path = checkpoint_dir / f"{args.env}{model_tag}_final_model.pt"
            agent.save(str(final_model_path))
            logger.info(f"Saved final trained agent to: {final_model_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
