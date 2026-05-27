"""Isaac Lab compatible position setpoint task for aerial robots.

This module provides a DirectRLEnv wrapper that connects the aerial_gym
task system with Isaac Lab's RL training infrastructure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.utils.configclass import configclass

from aerial_gym.utils.math import (
    quat_apply_inverse,
    quat_axis,
    quat_rotate_inverse,
    get_euler_xyz_tensor,
    vehicle_frame_quat_from_quat,
    ssa,
)
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("PositionSetpointTaskIsaacLab")


@configclass
class AerialGymPositionSetpointCfg(DirectRLEnvCfg):
    """Configuration for the aerial gym position setpoint task."""

    # Environment
    decimation = 1
    episode_length_s = 10.0
    action_space = 4
    observation_space = 13
    state_space = 0

    # Task parameters
    sim_name: str = "base_sim_config"
    env_name: str = "empty_env"
    robot_name: str = "quadrotor"
    controller_name: str = "lee_controller"
    use_warp: bool = True
    return_state_before_reset: bool = False
    episode_len_steps: int = 500

    # Reward parameters
    reward_parameters: dict = None


class AerialGymPositionSetpointEnv(DirectRLEnv):
    """Isaac Lab compatible position setpoint task for aerial robots.

    This class wraps the aerial_gym task system in Isaac Lab's DirectRLEnv interface,
    enabling use with Isaac Lab's RL training infrastructure (rl_games, rsl_rl, etc.).
    """

    cfg: AerialGymPositionSetpointCfg

    def __init__(self, cfg: AerialGymPositionSetpointCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize task-specific tensors
        self._actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self._prev_actions = torch.zeros_like(self._actions)
        self._target_position = torch.zeros(
            (self.num_envs, 3), device=self.device, requires_grad=False
        )
        self._rewards = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        """Set up the scene using the existing aerial_gym environment manager."""
        from aerial_gym.sim.sim_builder import SimBuilder

        # Build the environment using the existing sim builder
        self._sim_env = SimBuilder().build_env(
            sim_name=self.cfg.sim_name,
            env_name=self.cfg.env_name,
            robot_name=self.cfg.robot_name,
            controller_name=self.cfg.controller_name,
            device=self.device,
            num_envs=self.num_envs,
            use_warp=self.cfg.use_warp,
            headless=True,
        )

        # Get the observation dictionary
        self._obs_dict = self._sim_env.get_obs()
        self._obs_dict["num_obstacles_in_env"] = 1

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        self._prev_actions[:] = self._actions
        self._actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply actions to the simulation."""
        self._sim_env.step(actions=self._actions)

    def _get_observations(self) -> dict:
        """Compute observations for the RL agent."""
        obs = torch.zeros(
            (self.num_envs, self.cfg.observation_space),
            device=self.device,
            requires_grad=False,
        )
        obs[:, 0:3] = self._target_position - self._obs_dict["robot_position"]
        obs[:, 3:7] = self._obs_dict["robot_orientation"]
        obs[:, 7:10] = self._obs_dict["robot_body_linvel"]
        obs[:, 10:13] = self._obs_dict["robot_body_angvel"]
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        robot_position = self._obs_dict["robot_position"]
        robot_linvel = self._obs_dict["robot_linvel"]
        robot_vehicle_orientation = self._obs_dict["robot_vehicle_orientation"]
        robot_orientation = self._obs_dict["robot_orientation"]
        angular_velocity = self._obs_dict["robot_body_angvel"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (self._target_position - robot_position)
        )

        reward, crashes = _compute_reward(
            pos_error_vehicle_frame,
            robot_linvel,
            robot_orientation,
            angular_velocity,
            self._obs_dict["crashes"],
            1.0,
            self._actions,
            self._prev_actions,
        )

        self._obs_dict["crashes"][:] = crashes
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dones (terminated and truncated)."""
        terminated = self._obs_dict["crashes"].clone()
        truncated = self._sim_env.sim_steps > self.cfg.episode_len_steps
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environments at given indices."""
        if env_ids is None:
            env_ids = self._sim_env.num_envs

        self._target_position[env_ids, :3] = 0.0
        self._sim_env.reset_idx(env_ids)
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0


@torch.jit.script
def _exp_func(x: torch.Tensor, gain: float, exp: float) -> torch.Tensor:
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def _compute_reward(
    pos_error: torch.Tensor,
    lin_vels: torch.Tensor,
    robot_quats: torch.Tensor,
    robot_angvels: torch.Tensor,
    crashes: torch.Tensor,
    curriculum_level_multiplier: float,
    current_action: torch.Tensor,
    prev_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.norm(pos_error, dim=1)
    pos_reward = _exp_func(dist, 3.0, 8.0) + _exp_func(dist, 2.0, 4.0)
    dist_reward = (20 - dist) / 40.0

    ups = quat_axis(robot_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 0.2 / (0.1 + tiltage * tiltage)

    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3

    total_reward = pos_reward + dist_reward + pos_reward * (up_reward + ang_vel_reward)
    total_reward[:] = curriculum_level_multiplier * total_reward

    crashes = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)
    total_reward[:] = torch.where(crashes > 0.0, -20 * torch.ones_like(total_reward), total_reward)

    return total_reward, crashes
