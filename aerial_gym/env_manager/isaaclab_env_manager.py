"""Isaac Lab based environment manager for aerial_gym.

This module replaces the Isaac Gym based IGE_env_manager.py with Isaac Lab 3 APIs.
It preserves the global_tensor_dict pattern for backward compatibility with existing
robot, controller, and sensor code.
"""

from __future__ import annotations

import math
import random

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.configclass import configclass

from aerial_gym.utils.math import torch_rand_float_tensor, torch_interpolate_ratio
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("IsaacLabEnvManager")


class IsaacLabEnvManager:
    """Isaac Lab based environment manager.

    Replaces the Isaac Gym based IsaacGymEnv class. Uses Isaac Lab's SimulationContext
    and InteractiveScene for environment management while preserving the global_tensor_dict
    pattern for compatibility with existing robot/controller/sensor code.
    """

    def __init__(self, config, sim_config, device):
        self.cfg = config
        self.sim_config = sim_config
        self.device = device
        self.num_envs = self.cfg.env.num_envs

        # env bounds
        self.env_lower_bound_min = torch.tensor(
            self.cfg.env.lower_bound_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.env_lower_bound_max = torch.tensor(
            self.cfg.env.lower_bound_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.env_upper_bound_min = torch.tensor(
            self.cfg.env.upper_bound_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.env_upper_bound_max = torch.tensor(
            self.cfg.env.upper_bound_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)

        self.env_lower_bound = torch_rand_float_tensor(
            self.env_lower_bound_min, self.env_lower_bound_max
        )
        self.env_upper_bound = torch_rand_float_tensor(
            self.env_upper_bound_min, self.env_upper_bound_max
        )

        self.sim_has_dof = False
        self.dof_control_mode = "none"

        # Isaac Lab objects (set during setup)
        self.sim: SimulationContext | None = None
        self.scene: InteractiveScene | None = None
        self.robot_articulation: Articulation | None = None
        self.obstacle_objects: list[RigidObject] = []

    def create_sim(self):
        """Create Isaac Lab simulation context."""
        logger.info("Creating Isaac Lab SimulationContext")

        sim_cfg = SimulationCfg(
            device=self.device,
            dt=self.sim_config.sim.dt,
            render_interval=2,
            gravity=self.sim_config.sim.gravity,
        )
        self.sim = SimulationContext(sim_cfg)
        logger.info("Created Isaac Lab SimulationContext")
        return self.sim

    def create_ground_plane(self):
        """Create ground plane using Isaac Lab."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        logger.info("Created ground plane")

    def prepare_for_simulation(self, env_manager, global_tensor_dict):
        """Prepare tensors for simulation using Isaac Lab API.

        This replaces the Isaac Gym tensor acquisition pattern with Isaac Lab's
        direct tensor access via Articulation.data.
        """
        self.global_tensor_dict = global_tensor_dict
        self.env_manager = env_manager

        # Get robot articulation from scene
        if self.robot_articulation is not None:
            self.num_rigid_bodies_robot = len(self.robot_articulation.data.body_names)

            # Root state tensors
            root_pos = self.robot_articulation.data.root_pos_w.torch
            root_quat = self.robot_articulation.data.root_quat_w.torch
            root_linvel = self.robot_articulation.data.root_lin_vel_w.torch
            root_angvel = self.robot_articulation.data.root_ang_vel_w.torch

            # Build robot state tensor: [pos(3), quat(4), linvel(3), angvel(3)]
            robot_state = torch.cat([root_pos, root_quat, root_linvel, root_angvel], dim=-1)

            # Build vec_root_tensor for all actors (robot + obstacles)
            num_obstacles = len(self.obstacle_objects)
            self.num_assets_per_env = 1 + num_obstacles

            if num_obstacles > 0:
                obstacle_states = []
                for obj in self.obstacle_objects:
                    o_pos = obj.data.root_pos_w.torch
                    o_quat = obj.data.root_quat_w.torch
                    o_linvel = obj.data.root_lin_vel_w.torch
                    o_angvel = obj.data.root_ang_vel_w.torch
                    obstacle_states.append(torch.cat([o_pos, o_quat, o_linvel, o_angvel], dim=-1))
                obstacle_state_tensor = torch.stack(obstacle_states, dim=1)
                vec_root_tensor = torch.cat(
                    [robot_state.unsqueeze(1), obstacle_state_tensor], dim=1
                )
            else:
                vec_root_tensor = robot_state.unsqueeze(1)

            self.global_tensor_dict["vec_root_tensor"] = vec_root_tensor
            self.global_tensor_dict["robot_state_tensor"] = vec_root_tensor[:, 0, :]
            self.global_tensor_dict["env_asset_state_tensor"] = vec_root_tensor[:, 1:, :]
            self.global_tensor_dict["unfolded_env_asset_state_tensor"] = vec_root_tensor.reshape(
                -1, 13
            )
            self.global_tensor_dict["unfolded_env_asset_state_tensor_const"] = (
                self.global_tensor_dict["unfolded_env_asset_state_tensor"].clone()
            )

            # Rigid body state tensor from robot articulation
            body_pos_w = self.robot_articulation.data.body_pos_w.torch
            body_quat_w = self.robot_articulation.data.body_quat_w.torch
            body_linvel_w = self.robot_articulation.data.body_lin_vel_w.torch
            body_angvel_w = self.robot_articulation.data.body_ang_vel_w.torch
            rigid_body_state = torch.cat(
                [body_pos_w, body_quat_w, body_linvel_w, body_angvel_w], dim=-1
            )
            self.global_tensor_dict["rigid_body_state_tensor"] = rigid_body_state.reshape(-1, 13)

            # Force/torque tensors
            self.global_tensor_dict["global_force_tensor"] = torch.zeros(
                (self.num_envs * self.num_rigid_bodies_per_env, 3),
                device=self.device,
                requires_grad=False,
            )
            self.global_tensor_dict["global_torque_tensor"] = torch.zeros(
                (self.num_envs * self.num_rigid_bodies_per_env, 3),
                device=self.device,
                requires_grad=False,
            )

            # DOF state tensors
            if self.robot_articulation.data.joint_pos is not None:
                self.sim_has_dof = True
                joint_pos = self.robot_articulation.data.joint_pos.torch
                joint_vel = self.robot_articulation.data.joint_vel.torch
                dof_state = torch.stack([joint_pos, joint_vel], dim=-1)
                self.global_tensor_dict["unfolded_dof_state_tensor"] = dof_state.reshape(-1, 2)
                self.global_tensor_dict["dof_state_tensor"] = dof_state

            # Contact force tensor (zeros initially - populated during simulation)
            self.global_tensor_dict["global_contact_force_tensor"] = torch.zeros(
                (self.num_envs, self.num_rigid_bodies_per_env, 3),
                device=self.device,
                requires_grad=False,
            )
            self.global_tensor_dict["robot_contact_force_tensor"] = torch.zeros(
                (self.num_envs, 3), device=self.device, requires_grad=False
            )

            # Robot state slices
            self.global_tensor_dict["robot_position"] = self.global_tensor_dict[
                "robot_state_tensor"
            ][:, :3]
            self.global_tensor_dict["robot_orientation"] = self.global_tensor_dict[
                "robot_state_tensor"
            ][:, 3:7]
            self.global_tensor_dict["robot_linvel"] = self.global_tensor_dict[
                "robot_state_tensor"
            ][:, 7:10]
            self.global_tensor_dict["robot_angvel"] = self.global_tensor_dict[
                "robot_state_tensor"
            ][:, 10:]
            self.global_tensor_dict["robot_body_angvel"] = torch.zeros_like(
                self.global_tensor_dict["robot_state_tensor"][:, 10:13]
            )
            self.global_tensor_dict["robot_body_linvel"] = torch.zeros_like(
                self.global_tensor_dict["robot_state_tensor"][:, 7:10]
            )
            self.global_tensor_dict["robot_euler_angles"] = torch.zeros_like(
                self.global_tensor_dict["robot_state_tensor"][:, 7:10]
            )

            idx = self.num_rigid_bodies_robot
            self.global_tensor_dict["robot_force_tensor"] = self.global_tensor_dict[
                "global_force_tensor"
            ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, :idx, :]
            self.global_tensor_dict["robot_torque_tensor"] = self.global_tensor_dict[
                "global_torque_tensor"
            ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, :idx, :]

            # Obstacle tensors
            if num_obstacles > 0:
                self.global_tensor_dict["obstacle_position"] = self.global_tensor_dict[
                    "env_asset_state_tensor"
                ][:, :, 0:3]
                self.global_tensor_dict["obstacle_orientation"] = self.global_tensor_dict[
                    "env_asset_state_tensor"
                ][:, :, 3:7]
                self.global_tensor_dict["obstacle_linvel"] = self.global_tensor_dict[
                    "env_asset_state_tensor"
                ][:, :, 7:10]
                self.global_tensor_dict["obstacle_angvel"] = self.global_tensor_dict[
                    "env_asset_state_tensor"
                ][:, :, 10:]
                self.global_tensor_dict["obstacle_body_angvel"] = torch.zeros_like(
                    self.global_tensor_dict["env_asset_state_tensor"][:, :, 10:13]
                )
                self.global_tensor_dict["obstacle_body_linvel"] = torch.zeros_like(
                    self.global_tensor_dict["env_asset_state_tensor"][:, :, 7:10]
                )
                self.global_tensor_dict["obstacle_euler_angles"] = torch.zeros_like(
                    self.global_tensor_dict["env_asset_state_tensor"][:, :, 7:10]
                )
                self.global_tensor_dict["obstacle_force_tensor"] = self.global_tensor_dict[
                    "global_force_tensor"
                ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, idx:, :]
                self.global_tensor_dict["obstacle_torque_tensor"] = self.global_tensor_dict[
                    "global_torque_tensor"
                ].view(self.num_envs, self.num_rigid_bodies_per_env, 3)[:, idx:, :]

            # Environment bounds and constants
            self.global_tensor_dict["env_bounds_max"] = self.env_upper_bound
            self.global_tensor_dict["env_bounds_min"] = self.env_lower_bound
            self.global_tensor_dict["gravity"] = torch.tensor(
                self.sim_config.sim.gravity, device=self.device, requires_grad=False
            ).expand(self.num_envs, -1)
            self.global_tensor_dict["dt"] = self.sim_config.sim.dt

        logger.info("Prepared Isaac Lab environment for simulation")
        return True

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def reset_idx(self, env_ids):
        """Reset environment bounds for given environment indices."""
        self.env_lower_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_lower_bound_min, self.env_lower_bound_max
        )[env_ids]
        self.env_upper_bound[env_ids, :] = torch_rand_float_tensor(
            self.env_upper_bound_min, self.env_upper_bound_max
        )[env_ids]

    def write_to_sim(self):
        """Write state tensors to simulation using Isaac Lab API."""
        if self.robot_articulation is not None:
            robot_state = self.global_tensor_dict["robot_state_tensor"]
            root_pose = robot_state[:, :7]  # pos + quat
            root_vel = robot_state[:, 7:]  # linvel + angvel
            self.robot_articulation.write_root_pose_to_sim(root_pose)
            self.robot_articulation.write_root_velocity_to_sim(root_vel)

            if self.sim_has_dof and "dof_state_tensor" in self.global_tensor_dict:
                dof_state = self.global_tensor_dict["dof_state_tensor"]
                joint_pos = dof_state[:, :, 0]
                joint_vel = dof_state[:, :, 1]
                self.robot_articulation.write_joint_state_to_sim(joint_pos, joint_vel)

        # Write obstacle states
        for i, obj in enumerate(self.obstacle_objects):
            if "env_asset_state_tensor" in self.global_tensor_dict:
                obstacle_state = self.global_tensor_dict["env_asset_state_tensor"][:, i, :]
                root_pose = obstacle_state[:, :7]
                root_vel = obstacle_state[:, 7:]
                obj.write_root_pose_to_sim(root_pose)
                obj.write_root_velocity_to_sim(root_vel)

    def pre_physics_step(self, actions):
        """Apply forces and torques before physics step using Isaac Lab API."""
        if self.cfg.env.write_to_sim_at_every_timestep:
            self.write_to_sim()

        # Apply forces and torques to robot rigid bodies
        if self.robot_articulation is not None:
            forces = self.global_tensor_dict["robot_force_tensor"]
            torques = self.global_tensor_dict["robot_torque_tensor"]
            self.robot_articulation.set_external_force_and_torque(
                forces, torques, body_ids=torch.arange(self.num_rigid_bodies_robot, device=self.device)
            )
            self.robot_articulation.write_data_to_sim()

        # Apply DOF targets
        if self.sim_has_dof:
            self.dof_control_mode = self.global_tensor_dict.get("dof_control_mode", "none")
            if self.dof_control_mode == "position":
                target = self.global_tensor_dict["dof_position_setpoint_tensor"]
                self.robot_articulation.set_joint_position_target(target)
            elif self.dof_control_mode == "velocity":
                target = self.global_tensor_dict["dof_velocity_setpoint_tensor"]
                self.robot_articulation.set_joint_velocity_target(target)
            elif self.dof_control_mode == "effort":
                target = self.global_tensor_dict["dof_effort_tensor"]
                self.robot_articulation.set_joint_effort_target(target)
            self.robot_articulation.write_data_to_sim()

    def physics_step(self):
        """Step the simulation using Isaac Lab."""
        self.sim.step()

    def post_physics_step(self):
        """Update state tensors after physics step using Isaac Lab."""
        # Update robot state tensor from Isaac Lab data
        if self.robot_articulation is not None:
            root_pos = self.robot_articulation.data.root_pos_w.torch
            root_quat = self.robot_articulation.data.root_quat_w.torch
            root_linvel = self.robot_articulation.data.root_lin_vel_w.torch
            root_angvel = self.robot_articulation.data.root_ang_vel_w.torch

            robot_state = torch.cat([root_pos, root_quat, root_linvel, root_angvel], dim=-1)
            self.global_tensor_dict["robot_state_tensor"][:] = robot_state

            # Update rigid body states
            body_pos_w = self.robot_articulation.data.body_pos_w.torch
            body_quat_w = self.robot_articulation.data.body_quat_w.torch
            body_linvel_w = self.robot_articulation.data.body_lin_vel_w.torch
            body_angvel_w = self.robot_articulation.data.body_ang_vel_w.torch
            rigid_body_state = torch.cat(
                [body_pos_w, body_quat_w, body_linvel_w, body_angvel_w], dim=-1
            )
            self.global_tensor_dict["rigid_body_state_tensor"][:] = rigid_body_state.reshape(-1, 13)

            # Update DOF states
            if self.sim_has_dof:
                joint_pos = self.robot_articulation.data.joint_pos.torch
                joint_vel = self.robot_articulation.data.joint_vel.torch
                dof_state = torch.stack([joint_pos, joint_vel], dim=-1)
                self.global_tensor_dict["dof_state_tensor"][:] = dof_state

            # Update obstacle states
            for i, obj in enumerate(self.obstacle_objects):
                o_pos = obj.data.root_pos_w.torch
                o_quat = obj.data.root_quat_w.torch
                o_linvel = obj.data.root_lin_vel_w.torch
                o_angvel = obj.data.root_ang_vel_w.torch
                self.global_tensor_dict["env_asset_state_tensor"][:, i, :] = torch.cat(
                    [o_pos, o_quat, o_linvel, o_angvel], dim=-1
                )

            # Update vec_root_tensor
            if len(self.obstacle_objects) > 0:
                self.global_tensor_dict["vec_root_tensor"][:, 0, :] = robot_state
                for i, obj in enumerate(self.obstacle_objects):
                    o_state = self.global_tensor_dict["env_asset_state_tensor"][:, i, :]
                    self.global_tensor_dict["vec_root_tensor"][:, i + 1, :] = o_state
            else:
                self.global_tensor_dict["vec_root_tensor"][:, 0, :] = robot_state

    def step_graphics(self):
        """Step graphics (no-op in Isaac Lab, handled by simulation context)."""
        pass

    def render_viewer(self):
        """Render viewer (handled by Isaac Lab's viewport)."""
        pass

    @property
    def num_rigid_bodies_per_env(self):
        """Get number of rigid bodies per environment."""
        if self.robot_articulation is not None:
            n_robot = len(self.robot_articulation.data.body_names)
            n_obstacles = len(self.obstacle_objects)
            return n_robot + n_obstacles
        return 0
