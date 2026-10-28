"""High-level-only PPO training over a frozen low-level fluid controller."""

from __future__ import annotations

import math
from collections import deque

import torch
from tqdm.auto import tqdm


class HierarchicalFluidTrainer:
    """Train planner directions while reusing an immutable local stabilizer.

    A planner action is a two-dimensional vector, normalized and mapped to a
    fixed-radius world-frame waypoint. Each parallel environment
    independently closes its high-level transition after reaching that point,
    exhausting the decision interval, or ending its navigation episode.
    """

    def __init__(
        self,
        env,
        planner_agent,
        controller_policy,
        device: str | torch.device,
        decision_interval: int = 8,
        joint_stage: bool = False,
    ) -> None:
        if decision_interval < 1:
            raise ValueError("decision_interval must be positive")
        self.env = env
        self.planner = planner_agent
        self.controller = controller_policy
        self.device = torch.device(device)
        self.decision_interval = decision_interval
        self.joint_stage = joint_stage

    @staticmethod
    def _unit_direction(action: torch.Tensor) -> torch.Tensor:
        """Match the source's clipped two-component action and unit vector."""
        direction = action.clamp(-1.0, 1.0)
        magnitude = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        fallback = torch.zeros_like(direction)
        fallback[:, 0] = 1.0
        return torch.where(magnitude > 1.0e-6, direction / magnitude.clamp_min(1.0e-6), fallback)

    @staticmethod
    def _turn_distance(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Use the shortest angular distance across the +/-pi seam."""
        difference = current - previous
        return torch.abs(torch.atan2(torch.sin(difference), torch.cos(difference)))

    def _set_reference_std(self, recorded: int, *, joint: bool) -> None:
        """Match the source's 20 decays over its 200k-decision budget."""
        if not self.env.cfg.navigation_reference_rules:
            return
        decays = min(20, recorded // 10_000)
        log_std = math.log(max(0.1, 0.5 - 0.02 * decays))
        with torch.no_grad():
            self.planner.policy.log_std.fill_(log_std)
            if joint:
                self.controller.policy.log_std.fill_(log_std)

    def train(self, total_decisions: int) -> None:
        if total_decisions < 1:
            raise ValueError("total_decisions must be positive")
        self.planner.enable_training_mode(True, apply_to_models=False)
        self.controller.eval()
        for parameter in self.controller.parameters():
            parameter.requires_grad_(False)

        low_observation, _ = self.env.reset()
        num_envs = self.env.num_envs
        observation_dim = self.env.high_observation_dim
        batch_steps = total_decisions
        current_observation = torch.zeros((num_envs, observation_dim), device=self.device)
        current_action = torch.zeros((num_envs, 2), device=self.device)
        current_log_prob = torch.zeros((num_envs, 1), device=self.device)
        current_value = torch.zeros((num_envs, 1), device=self.device)
        current_direction = torch.zeros((num_envs, 2), device=self.device)
        previous_angle = torch.zeros(num_envs, device=self.device)
        accumulated_reward = torch.zeros(num_envs, device=self.device)
        duration = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        recent_successes = deque(maxlen=100)
        recorded = 0
        record_step = 0

        def select_actions(ids: torch.Tensor) -> None:
            if not ids.numel():
                return
            observation = self.env.get_high_level_observation()[ids]
            self.planner.pre_interaction(timestep=record_step, timesteps=batch_steps)
            with torch.no_grad():
                action, outputs = self.planner.act(
                    observation, observation,
                    timestep=record_step, timesteps=batch_steps,
                )
                direction = self._unit_direction(action)
            current_observation[ids] = observation
            current_action[ids] = action
            current_log_prob[ids] = outputs["log_prob"]
            current_value[ids] = outputs["values"]
            current_direction[ids] = direction
            angle = torch.atan2(direction[:, 1], direction[:, 0])
            accumulated_reward[ids] = -self.env.cfg.navigation_high_smoothness_scale * (
                self._turn_distance(angle, previous_angle[ids]) / math.pi
            )
            duration[ids] = 0
            local_target = (
                self.env.sim_handle.agent_pos[ids]
                + self.env.cfg.navigation_local_target_radius * direction
            )
            self.env.set_local_targets(local_target, env_ids=ids)

        all_ids = torch.arange(num_envs, device=self.device)
        goal_delta = self.env.global_target_pos - self.env.sim_handle.agent_pos
        previous_angle.copy_(torch.atan2(goal_delta[:, 1], goal_delta[:, 0]))
        select_actions(all_ids)
        low_observation = self.env.get_low_level_observation()
        progress = tqdm(total=total_decisions, desc="High-level decisions", dynamic_ncols=True)
        last_info = {}
        try:
            while recorded < total_decisions:
                with torch.no_grad():
                    low_action, controller_outputs = self.controller.act(
                        {"observations": low_observation["policy"]}, role="policy"
                    )
                    if not self.env.cfg.navigation_reference_rules:
                        low_action = controller_outputs["mean_actions"]
                low_observation, reward, terminated, truncated, last_info = self.env.step(low_action)
                accumulated_reward += reward
                duration += 1
                done = terminated | truncated
                ready = done | last_info["waypoint_reached"] | (duration >= self.decision_interval)
                ready_ids = torch.where(ready)[0]
                if not ready_ids.numel():
                    continue

                next_observation = self.env.get_high_level_observation()
                previous_angle[ready_ids] = torch.atan2(
                    current_direction[ready_ids, 1], current_direction[ready_ids, 0]
                )
                done_ids = torch.where(done)[0]
                if done_ids.numel():
                    recent_successes.extend(last_info["global_success"][done_ids].int().tolist())
                    new_goal_delta = self.env.global_target_pos[done_ids] - self.env.sim_handle.agent_pos[done_ids]
                    previous_angle[done_ids] = torch.atan2(new_goal_delta[:, 1], new_goal_delta[:, 0])

                for env_id in ready_ids.tolist():
                    if recorded >= total_decisions:
                        break
                    self.planner.record_transition(
                        observations=current_observation[env_id:env_id + 1],
                        states=current_observation[env_id:env_id + 1],
                        actions=current_action[env_id:env_id + 1],
                        rewards=accumulated_reward[env_id:env_id + 1, None],
                        next_observations=next_observation[env_id:env_id + 1],
                        next_states=next_observation[env_id:env_id + 1],
                        terminated=terminated[env_id:env_id + 1, None],
                        truncated=truncated[env_id:env_id + 1, None],
                        infos={}, timestep=record_step, timesteps=batch_steps,
                        log_prob=current_log_prob[env_id:env_id + 1],
                        values=current_value[env_id:env_id + 1],
                        stream_ids=torch.tensor([[env_id]], device=self.device),
                    )
                    self.planner.post_interaction(
                        timestep=record_step, timesteps=batch_steps
                    )
                    record_step += 1
                    recorded += 1
                    progress.update(1)
                    self._set_reference_std(recorded, joint=False)

                if recorded < total_decisions:
                    select_actions(ready_ids)
                    low_observation = self.env.get_low_level_observation()
                progress.set_postfix(
                    low_steps=int(duration.float().mean().item()),
                    success=int(last_info["global_success"].sum().item()),
                    success_100=f"{sum(recent_successes) / len(recent_successes):.1%}" if recent_successes else "n/a",
                )
        finally:
            progress.close()
        self.planner.enable_training_mode(False, apply_to_models=False)

    def train_joint(self, total_decisions: int) -> None:
        """Joint open-water Stage 1: update planner and local controller PPOs."""
        if total_decisions < 1:
            raise ValueError("total_decisions must be positive")
        if not hasattr(self.controller, "record_transition"):
            raise TypeError("joint training requires the low-level PPO, not a frozen policy module")
        self.planner.enable_training_mode(True, apply_to_models=False)
        self.controller.enable_training_mode(True, apply_to_models=False)
        low_obs, _ = self.env.reset()
        n = self.env.num_envs
        horizon = total_decisions
        obs_dim = self.env.high_observation_dim
        current_obs = torch.zeros((n, obs_dim), device=self.device)
        current_action = torch.zeros((n, 2), device=self.device)
        current_logp = torch.zeros((n, 1), device=self.device)
        current_value = torch.zeros((n, 1), device=self.device)
        direction = torch.zeros((n, 2), device=self.device)
        previous_angle = torch.zeros(n, device=self.device)
        reward_sum = torch.zeros(n, device=self.device)
        duration = torch.zeros(n, device=self.device, dtype=torch.long)
        recent_successes = deque(maxlen=100)
        recorded = high_step = low_step = 0

        def decide(ids: torch.Tensor) -> None:
            if not ids.numel():
                return
            obs = self.env.get_high_level_observation()[ids]
            self.planner.pre_interaction(timestep=high_step, timesteps=horizon)
            with torch.no_grad():
                action, out = self.planner.act(obs, obs, timestep=high_step, timesteps=horizon)
            unit = self._unit_direction(action)
            current_obs[ids], current_action[ids] = obs, action
            current_logp[ids], current_value[ids], direction[ids] = out["log_prob"], out["values"], unit
            angle = torch.atan2(unit[:, 1], unit[:, 0])
            reward_sum[ids] = -self.env.cfg.navigation_high_smoothness_scale * (
                self._turn_distance(angle, previous_angle[ids]) / math.pi
            )
            duration[ids] = 0
            self.env.set_local_targets(self.env.sim_handle.agent_pos[ids] + self.env.cfg.navigation_local_target_radius * unit, env_ids=ids)

        goal_delta = self.env.global_target_pos - self.env.sim_handle.agent_pos
        previous_angle.copy_(torch.atan2(goal_delta[:, 1], goal_delta[:, 0]))
        decide(torch.arange(n, device=self.device))
        low_obs = self.env.get_low_level_observation()
        progress = tqdm(total=total_decisions, desc="Joint high-level decisions", dynamic_ncols=True)
        try:
            while recorded < total_decisions:
                self.controller.pre_interaction(timestep=low_step, timesteps=horizon * self.decision_interval)
                with torch.no_grad():
                    low_action, _ = self.controller.act(low_obs["policy"], low_obs["policy"], timestep=low_step, timesteps=horizon * self.decision_interval)
                next_low_obs, high_reward, terminated, truncated, info = self.env.step(low_action)
                done = terminated | truncated
                duration += 1
                ready = done | info["waypoint_reached"] | (duration >= self.decision_interval)
                # One low-level transition for every CFD-control step.
                self.controller.record_transition(
                    observations=low_obs["policy"], states=low_obs["policy"], actions=low_action,
                    rewards=info["low_reward"][:, None], next_observations=next_low_obs["policy"], next_states=next_low_obs["policy"],
                    terminated=ready[:, None], truncated=torch.zeros_like(ready[:, None]), infos={}, timestep=low_step, timesteps=horizon * self.decision_interval,
                )
                self.controller.post_interaction(timestep=low_step, timesteps=horizon * self.decision_interval)
                low_step += 1
                reward_sum += high_reward
                ids = torch.where(ready)[0]
                if ids.numel():
                    # The joint script alone adds local progress, capped at
                    # -2 cells when a waypoint becomes less reachable.
                    reward_sum[ids] += 10.0 * torch.clamp(
                        self.env.cfg.navigation_local_target_radius - info["local_distance"][ids], min=-2.0
                    )
                    next_obs = self.env.get_high_level_observation()
                    for env_id in ids.tolist():
                        if recorded >= total_decisions:
                            break
                        self.planner.record_transition(
                            observations=current_obs[env_id:env_id + 1], states=current_obs[env_id:env_id + 1],
                            actions=current_action[env_id:env_id + 1], rewards=reward_sum[env_id:env_id + 1, None],
                            next_observations=next_obs[env_id:env_id + 1], next_states=next_obs[env_id:env_id + 1],
                            terminated=terminated[env_id:env_id + 1, None], truncated=truncated[env_id:env_id + 1, None],
                            infos={}, timestep=high_step, timesteps=horizon,
                            log_prob=current_logp[env_id:env_id + 1], values=current_value[env_id:env_id + 1],
                            stream_ids=torch.tensor([[env_id]], device=self.device),
                        )
                        self.planner.post_interaction(timestep=high_step, timesteps=horizon)
                        high_step += 1
                        if (self.controller.cfg.reference_mode
                                and high_step % self.planner.cfg.rollouts == 0):
                            self.controller.enable_models_training_mode(True)
                            self.controller.update(timestep=low_step, timesteps=horizon * self.decision_interval)
                            self.controller.enable_models_training_mode(False)
                            self.controller.memory.reset()
                        recorded += 1
                        progress.update(1)
                        self._set_reference_std(recorded, joint=True)
                    previous_angle[ids] = torch.atan2(direction[ids, 1], direction[ids, 0])
                    done_ids = torch.where(done)[0]
                    if done_ids.numel():
                        recent_successes.extend(info["global_success"][done_ids].int().tolist())
                        new_goal_delta = self.env.global_target_pos[done_ids] - self.env.sim_handle.agent_pos[done_ids]
                        previous_angle[done_ids] = torch.atan2(new_goal_delta[:, 1], new_goal_delta[:, 0])
                    if recorded < total_decisions:
                        decide(ids)
                        low_obs = self.env.get_low_level_observation()
                    else:
                        low_obs = next_low_obs
                else:
                    low_obs = next_low_obs
                progress.set_postfix(success_100=f"{sum(recent_successes) / len(recent_successes):.1%}" if recent_successes else "n/a")
        finally:
            progress.close()
        self.planner.enable_training_mode(False, apply_to_models=False)
        self.controller.enable_training_mode(False, apply_to_models=False)

    def eval(
        self, num_episodes: int = 100, *, headless: bool = False, render_interval: int = 1
    ) -> dict[str, object]:
        """Evaluate loaded policies with the reference's small action noise."""
        if num_episodes < 1:
            raise ValueError("num_episodes must be positive")
        self.planner.enable_training_mode(False, apply_to_models=True)
        self.controller.eval()
        self.env.configure_rendering(enabled=not headless, interval=render_interval)

        low_observation, _ = self.env.reset()
        if not headless:
            self.env.render()

        num_envs = self.env.num_envs
        completed = successes = 0
        outcomes = {"collision": 0, "out_of_bounds": 0, "over_roll": 0, "timeout": 0}
        layout_episodes: dict[int, int] = {}
        layout_successes: dict[int, int] = {}
        total_reward = 0.0
        total_policy_reward = 0.0
        total_low_steps = total_decisions = 0
        episode_reward = torch.zeros(num_envs, device=self.device)
        episode_policy_reward = torch.zeros(num_envs, device=self.device)
        episode_low_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        episode_decisions = torch.zeros_like(episode_low_steps)
        duration = torch.zeros_like(episode_low_steps)
        current_direction = torch.zeros((num_envs, 2), device=self.device)
        previous_angle = torch.zeros(num_envs, device=self.device)

        def select_actions(ids: torch.Tensor) -> None:
            if not ids.numel():
                return
            observation = self.env.get_high_level_observation()[ids]
            high_action, _ = self.planner.act(
                observation, observation,
                timestep=0, timesteps=1,
                deterministic=not self.env.cfg.navigation_reference_rules,
            )
            direction = self._unit_direction(high_action)
            angle = torch.atan2(direction[:, 1], direction[:, 0])
            episode_policy_reward[ids] -= self.env.cfg.navigation_high_smoothness_scale * (
                self._turn_distance(angle, previous_angle[ids]) / math.pi
            )
            previous_angle[ids] = angle
            current_direction[ids] = direction
            episode_decisions[ids] += 1
            duration[ids] = 0
            local_target = (
                self.env.sim_handle.agent_pos[ids]
                + self.env.cfg.navigation_local_target_radius * direction
            )
            self.env.set_local_targets(local_target, env_ids=ids)

        all_ids = torch.arange(num_envs, device=self.device)
        goal_delta = self.env.global_target_pos - self.env.sim_handle.agent_pos
        previous_angle.copy_(torch.atan2(goal_delta[:, 1], goal_delta[:, 0]))
        progress = tqdm(total=num_episodes, desc="Hierarchical evaluation", dynamic_ncols=True)
        try:
            with torch.no_grad():
                select_actions(all_ids)
                low_observation = self.env.get_low_level_observation()
                while completed < num_episodes:
                    low_action, controller_outputs = self.controller.act(
                        {"observations": low_observation["policy"]}, role="policy"
                    )
                    if not self.env.cfg.navigation_reference_rules:
                        low_action = controller_outputs["mean_actions"]
                    low_observation, reward, terminated, truncated, info = self.env.step(low_action)
                    episode_reward += reward
                    episode_policy_reward += reward
                    episode_low_steps += 1
                    duration += 1
                    done = terminated | truncated
                    ready = done | info["waypoint_reached"] | (duration >= self.decision_interval)
                    if self.joint_stage:
                        episode_policy_reward[ready] += 10.0 * torch.clamp(
                            self.env.cfg.navigation_local_target_radius - info["local_distance"][ready], min=-2.0
                        )
                    done_indices = torch.where(done)[0]
                    for env_id in done_indices.tolist():
                        if completed >= num_episodes:
                            break
                        successes += int(info["global_success"][env_id].item())
                        layout_id = int(info["flow_init"][env_id].item())
                        layout_episodes[layout_id] = layout_episodes.get(layout_id, 0) + 1
                        if info["global_success"][env_id].item():
                            layout_successes[layout_id] = layout_successes.get(layout_id, 0) + 1
                        else:
                            # A terminal reason is deliberately exclusive here so rates sum
                            # cleanly even when an invalid state also crosses a boundary.
                            for reason in ("collision", "out_of_bounds", "over_roll", "timeout"):
                                if info[reason][env_id].item():
                                    outcomes[reason] += 1
                                    break
                        reward_value = float(episode_reward[env_id].item())
                        low_steps_value = int(episode_low_steps[env_id].item())
                        decisions_value = int(episode_decisions[env_id].item())
                        total_reward += reward_value
                        total_policy_reward += float(episode_policy_reward[env_id].item())
                        total_low_steps += low_steps_value
                        total_decisions += decisions_value
                        completed += 1
                        progress.update(1)
                        progress.set_postfix(
                            success=f"{successes}/{completed}",
                            reward=f"{reward_value:.1f}",
                            low_steps=low_steps_value,
                        )
                    episode_reward[done] = 0.0
                    episode_policy_reward[done] = 0.0
                    episode_low_steps[done] = 0
                    episode_decisions[done] = 0
                    if done_indices.numel():
                        new_goal_delta = self.env.global_target_pos[done_indices] - self.env.sim_handle.agent_pos[done_indices]
                        previous_angle[done_indices] = torch.atan2(new_goal_delta[:, 1], new_goal_delta[:, 0])
                    ready_ids = torch.where(ready)[0]
                    if completed < num_episodes:
                        select_actions(ready_ids)
                        low_observation = self.env.get_low_level_observation()
        finally:
            progress.close()

        layout_success_rate = {
            layout_id: layout_successes.get(layout_id, 0) / count
            for layout_id, count in sorted(layout_episodes.items())
        }
        return {
            "episodes": float(completed),
            "success_rate": successes / completed,
            "average_reward": total_reward / completed,
            "average_policy_reward": total_policy_reward / completed,
            "average_low_steps": total_low_steps / completed,
            "average_decisions": total_decisions / completed,
            "collision_rate": outcomes["collision"] / completed,
            "out_of_bounds_rate": outcomes["out_of_bounds"] / completed,
            "over_roll_rate": outcomes["over_roll"] / completed,
            "timeout_rate": outcomes["timeout"] / completed,
            "layout_episodes": layout_episodes,
            "layout_success_rate": layout_success_rate,
        }
