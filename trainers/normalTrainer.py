import sys

import torch
import tqdm
from loguru import logger

class normalTrainer:
    def __init__(self, env, agents, device: str | torch.device, cfg: dict):
        
        self.env = env
        
        if not isinstance(agents, (list, tuple)):
            self.agents = [agents]
        else:
            self.agents = agents
        self.scopes = [[0, self.env.num_envs]]
            
        self.device = torch.device(device)
        self.num_envs = env.num_envs
        self.cfg = cfg
        
        # init agents
        for agent in self.agents:
            agent.init(trainer_cfg=self.cfg)
        # if self.num_simultaneous_agents > 1:
        #     for agent in self.agents:
        #         agent.init(trainer_cfg=self.cfg)
        # else:
        #     self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        
        # set running mode
        for agent in self.agents:
            agent.enable_training_mode(True)
        unwrapped_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(unwrapped_env, "configure_rendering"):
            # Training is always render-free; visualization is an evaluation
            # concern and must not slow down data collection.
            unwrapped_env.configure_rendering(enabled=False)
        # if self.num_simultaneous_agents > 1:
        #     for agent in self.agents:
        #         agent.enable_training_mode(False)
        # else:
        #     self.agents.enable_training_mode(False)

        # # non-simultaneous agents
        # if self.num_simultaneous_agents == 1:
        #     super().eval()
        #     return
        
        # print(f"[Trainer] | num_envs: {self.num_envs}")

        observations, infos = self.env.reset()
        states = self.env.state()
        
        for timestep in tqdm.tqdm(range(self.cfg["timesteps"]), disable=self.cfg["disable_progressbar"], file=sys.stdout):
            
            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.cfg["timesteps"])

            with torch.no_grad():
                # compute actions
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    actions, outputs = agent.act(
                        observations[scope[0] : scope[1]],
                        states[scope[0] : scope[1]] if states is not None else None,
                        timestep=timestep,
                        timesteps=self.cfg["timesteps"],
                    )
                    _actions.append(actions)
                    _outputs.append(outputs)
                actions = torch.vstack(_actions)
                
                # step the environments
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.state()

                # record the environments' transitions
                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]] if states is not None else None,
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.cfg["timesteps"],
                    )

                # log environment info
                if self.cfg["environment_info"] in infos:
                    for k, v in infos[self.cfg["environment_info"]].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(k if "/" in k else f"Info / {k}", v.item())

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.cfg["timesteps"])

            # FluidEnv is vectorized even when ``num_envs == 1`` and performs
            # per-environment auto-reset inside ``step``.  The returned next
            # observation is therefore always the correct input for the next
            # interaction, for both single- and multi-environment runs.
            observations = next_observations
            states = next_states
                   
    def eval(self, num_eval_episodes=100):

        if num_eval_episodes < 1:
            raise ValueError("num_eval_episodes must be positive")

        logger.info("Starting evaluation...")
        
        for agent in self.agents:
            agent.enable_training_mode(False)

        unwrapped_env = getattr(self.env, "unwrapped", self.env)
        if hasattr(unwrapped_env, "configure_rendering"):
            unwrapped_env.configure_rendering(
                enabled=not self.cfg.get("headless", False),
                interval=self.cfg.get("render_interval", 1),
            )

        observations, infos = self.env.reset()
        states = self.env.state()
        if not torch.isfinite(observations).all().item():
            raise RuntimeError(
                "Evaluation reset produced non-finite policy observations; "
                "inspect the initial-flow phase bank and flow state"
            )
        if not self.cfg.get("headless", False):
            # Show the reset pose before the first physics substep.
            self.env.render()

        num_envs = self.env.num_envs
        completed_rounds = 0
        round_episodes = 0
        round_successes = 0
        round_rewards = 0.0
        success_rate = 0.0
        avg_reward = 0.0
        policy_steps = 0
        current_episode_rewards = torch.zeros(num_envs, device=self.device)
        progress = tqdm.tqdm(
            total=num_eval_episodes,
            desc="Evaluation round 1",
            disable=self.cfg.get("disable_progressbar", False),
            file=sys.stdout,
            mininterval=1.0,
        )

        try:
            with torch.no_grad():
                while True:
                    for agent in self.agents:
                        agent.pre_interaction(timestep=0, timesteps=1)

                    _actions = []
                    for agent, scope in zip(self.agents, self.scopes):
                        try:
                            actions, outputs = agent.act(
                                observations[scope[0] : scope[1]],
                                states[scope[0] : scope[1]] if states is not None else None,
                                timestep=0,
                                timesteps=1,
                                deterministic=True,
                            )
                        except TypeError:
                            actions, outputs = agent.act(
                                observations[scope[0] : scope[1]],
                                states[scope[0] : scope[1]] if states is not None else None,
                                timestep=0,
                                timesteps=1,
                            )
                        _actions.append(actions)

                    actions = torch.vstack(_actions)

                    # --- step environment ---
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    policy_steps += 1
                    # Episode counts can stay at zero for many physics steps.
                    # Keep the progress display alive while rendering too.
                    if policy_steps % 25 == 0:
                        progress.set_postfix_str(f"policy steps {policy_steps}")

                    current_episode_rewards += rewards.flatten()

                    if hasattr(unwrapped_env, "reset_terminated"):
                        success_mask = unwrapped_env.reset_terminated
                    elif hasattr(self.env, "reset_terminated"):
                        success_mask = self.env.reset_terminated
                    else:
                        success_mask = terminated

                    done_indices = torch.where((terminated | truncated).flatten())[0]
                    finished_rewards = current_episode_rewards[done_indices].tolist()
                    finished_successes = success_mask.flatten()[done_indices].tolist()
                    current_episode_rewards[done_indices] = 0.0

                    # Several environments may finish on the same vector step.
                    # Carry completed episodes past a round boundary into the
                    # next round, and keep unfinished returns in the buffer.
                    for episode_reward, succeeded in zip(
                        finished_rewards, finished_successes
                    ):
                        round_episodes += 1
                        round_rewards += episode_reward
                        round_successes += int(succeeded)
                        progress.update(1)

                        if round_episodes == num_eval_episodes:
                            completed_rounds += 1
                            success_rate = round_successes / num_eval_episodes
                            avg_reward = round_rewards / num_eval_episodes
                            progress.refresh()
                            logger.info(
                                f"Evaluation round {completed_rounds}: "
                                f"{num_eval_episodes} episodes | "
                                f"success {success_rate * 100:.2f}% "
                                f"({round_successes}/{num_eval_episodes}) | "
                                f"average reward {avg_reward:.4f}"
                            )
                            round_episodes = 0
                            round_successes = 0
                            round_rewards = 0.0
                            progress.reset(total=num_eval_episodes)
                            progress.set_description(
                                f"Evaluation round {completed_rounds + 1}"
                            )

                    observations = next_observations
                    states = next_states

                    for agent in self.agents:
                        agent.post_interaction(timestep=0, timesteps=1)
        except KeyboardInterrupt:
            logger.info(f"Evaluation stopped after {completed_rounds} full rounds")
        finally:
            progress.close()

        return success_rate, avg_reward
                
