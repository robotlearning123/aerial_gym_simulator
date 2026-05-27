"""Isaac Lab based environment orchestrator for aerial_gym.

This module replaces the old env_manager.py with Isaac Lab 3 compatible code.
It orchestrates the robot, assets, sensors, and simulation using Isaac Lab APIs
while preserving the global_tensor_dict pattern.
"""

from __future__ import annotations

import math
import random

import torch

from aerial_gym.env_manager.isaaclab_env_manager import IsaacLabEnvManager
from aerial_gym.env_manager.asset_manager import AssetManager
from aerial_gym.env_manager.obstacle_manager import ObstacleManager
from aerial_gym.robots.robot_manager import RobotManagerIGE
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.sim_registry import sim_config_registry
from aerial_gym.registry.robot_registry import robot_registry

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("IsaacLabEnvOrchestrator")


class IsaacLabEnvOrchestrator:
    """Isaac Lab based environment orchestrator.

    Replaces the old EnvManager class with Isaac Lab compatible code.
    Manages the lifecycle of the environment including robot, obstacles,
    sensors, and simulation.
    """

    def __init__(
        self,
        sim_name,
        env_name,
        robot_name,
        controller_name,
        device,
        args=None,
        num_envs=None,
        use_warp=None,
        headless=None,
    ):
        self.robot_name = robot_name
        self.controller_name = controller_name
        self.sim_config = sim_config_registry.make_sim(sim_name)
        self.cfg = env_config_registry.make_env(env_name)
        self.device = device

        if num_envs is not None:
            self.cfg.env.num_envs = num_envs
        if use_warp is not None:
            self.cfg.env.use_warp = use_warp
        if headless is not None:
            self.sim_config.viewer.headless = headless

        self.num_envs = self.cfg.env.num_envs
        self.use_warp = self.cfg.env.use_warp

        self.asset_manager = None
        self.env_args = args
        self.keep_in_env = None
        self.global_tensor_dict = {}

        logger.info("Creating Isaac Lab environment.")
        self.env_manager = IsaacLabEnvManager(self.cfg, self.sim_config, self.device)
        self.env_manager.create_sim()

        if self.cfg.env.create_ground_plane:
            self.env_manager.create_ground_plane()

        logger.info("Populating environment.")
        self.populate_env()
        logger.info("[DONE] Populating environment.")

        logger.info("Preparing simulation.")
        self.prepare_sim()
        logger.info("[DONE] Preparing simulation.")

        self.sim_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, requires_grad=False, device=self.device
        )

    def populate_env(self):
        """Populate the environment with robot and obstacles."""
        # Create robot manager
        self.global_sim_dict = {
            "env_cfg": self.cfg,
            "use_warp": self.cfg.env.use_warp,
            "num_envs": self.num_envs,
            "sim_cfg": self.sim_config,
            "sim": self.env_manager.sim,
        }

        # Initialize robot
        self.robot_manager = RobotManagerIGE(
            self.global_sim_dict, self.robot_name, self.controller_name, self.device
        )
        self.global_sim_dict["robot_config"] = self.robot_manager.cfg

        # Create crash/truncation tensors
        self.global_tensor_dict["crashes"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )
        self.global_tensor_dict["truncations"] = torch.zeros(
            (self.num_envs), device=self.device, requires_grad=False, dtype=torch.bool
        )

        self.num_env_actions = self.cfg.env.num_env_actions
        self.global_tensor_dict["num_env_actions"] = self.num_env_actions
        self.global_tensor_dict["env_actions"] = None
        self.global_tensor_dict["prev_env_actions"] = None

        self.collision_tensor = self.global_tensor_dict["crashes"]
        self.truncation_tensor = self.global_tensor_dict["truncations"]

    def prepare_sim(self):
        """Prepare the simulation."""
        self.env_manager.prepare_for_simulation(self, self.global_tensor_dict)
        self.robot_manager.prepare_for_sim(self.global_tensor_dict)
        self.num_robot_actions = self.global_tensor_dict["num_robot_actions"]

    def reset_idx(self, env_ids=None):
        """Reset environments at given indices."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.env_manager.reset_idx(env_ids)
        self.robot_manager.reset_idx(env_ids)
        self.env_manager.write_to_sim()
        self.sim_steps[env_ids] = 0

    def reset(self):
        self.reset_idx(env_ids=torch.arange(self.num_envs, device=self.device))

    def pre_physics_step(self, actions, env_actions):
        """Apply actions before physics step."""
        self.robot_manager.pre_physics_step(actions)
        self.env_manager.pre_physics_step(actions)

    def reset_tensors(self):
        self.collision_tensor[:] = 0
        self.truncation_tensor[:] = 0

    def simulate(self, actions, env_actions):
        """Run one simulation step."""
        self.pre_physics_step(actions, env_actions)
        self.env_manager.physics_step()
        self.post_physics_step(actions, env_actions)

    def post_physics_step(self, actions, env_actions):
        """Update state after physics step."""
        self.env_manager.post_physics_step()
        self.robot_manager.post_physics_step()

    def compute_observations(self):
        """Compute contact-based observations."""
        if "robot_contact_force_tensor" in self.global_tensor_dict:
            self.collision_tensor[:] += (
                torch.norm(self.global_tensor_dict["robot_contact_force_tensor"], dim=1)
                > self.cfg.env.collision_force_threshold
            )

    def reset_terminated_and_truncated_envs(self):
        """Reset environments that have terminated or been truncated."""
        envs_to_reset = (
            (self.collision_tensor * int(self.cfg.env.reset_on_collision) + self.truncation_tensor)
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        if len(envs_to_reset) > 0:
            self.reset_idx(envs_to_reset)
        return envs_to_reset

    def render(self, render_components="sensors"):
        """Render sensors or viewer."""
        if render_components == "viewer":
            self.env_manager.render_viewer()
        elif render_components == "sensors":
            self.robot_manager.capture_sensors()

    def post_reward_calculation_step(self):
        """Reset terminated/truncated envs and render sensors."""
        envs_to_reset = self.reset_terminated_and_truncated_envs()
        self.render(render_components="sensors")
        return envs_to_reset

    def step(self, actions, env_actions=None):
        """Step the simulation."""
        self.reset_tensors()
        num_physics_step_per_env_step = max(
            math.floor(
                random.gauss(
                    self.cfg.env.num_physics_steps_per_env_step_mean,
                    self.cfg.env.num_physics_steps_per_env_step_std,
                )
            ),
            0,
        )
        for timestep in range(num_physics_step_per_env_step):
            self.simulate(actions, env_actions)
            self.compute_observations()
        self.sim_steps[:] = self.sim_steps[:] + 1

    def get_obs(self):
        """Return the global tensor dictionary."""
        return self.global_tensor_dict
