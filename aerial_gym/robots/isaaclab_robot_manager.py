"""Isaac Lab compatible robot manager for aerial_gym.

This module provides a robot manager that works with Isaac Lab's APIs
instead of Isaac Gym's gymapi/gymtorch. It preserves the global_tensor_dict
pattern for backward compatibility with existing controller and sensor code.
"""

import torch

from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("IsaacLabRobotManager")


class IsaacLabRobotManager:
    """Isaac Lab compatible robot manager.

    Manages robot creation, tensor initialization, and control without
    depending on isaacgym. Uses the global_tensor_dict pattern for
    compatibility with existing controller code.
    """

    def __init__(self, global_sim_dict, robot_name, controller_name, device):
        logger.debug("Initializing IsaacLabRobotManager")
        self.env_config = global_sim_dict["env_cfg"]
        self.use_warp = global_sim_dict.get("use_warp", False)
        self.num_envs = global_sim_dict["num_envs"]
        self.device = device

        # Create robot from registry
        self.robot, robot_config = robot_registry.make_robot(
            robot_name, controller_name, self.env_config, device
        )

        self.cfg = robot_config
        self.robot_inertia = None
        self.robot_mass = None
        self.robot_masses = torch.zeros(self.num_envs, device=self.device)
        self.robot_inertias = torch.zeros((self.num_envs, 3, 3), device=self.device)

        self.dof_control_mode = "none"

        # Sensors
        self.camera_sensor = None
        self.warp_sensor = None
        self.imu_sensor = None

        logger.debug("[DONE] Initializing IsaacLabRobotManager")

    def prepare_for_sim(self, global_tensor_dict):
        """Initialize tensors for simulation."""
        self.global_tensor_dict = global_tensor_dict

        self.global_tensor_dict["robot_mass"] = self.robot_masses
        self.global_tensor_dict["robot_inertia"] = self.robot_inertias

        self.global_tensor_dict["robot_actions"] = torch.zeros(
            (self.num_envs, self.robot.num_actions), device=self.device
        )
        self.global_tensor_dict["robot_prev_actions"] = torch.zeros_like(
            self.global_tensor_dict["robot_actions"]
        )

        self.actions = self.global_tensor_dict["robot_actions"]
        self.prev_actions = self.global_tensor_dict["robot_prev_actions"]

        self.global_tensor_dict["dof_control_mode"] = self.dof_control_mode

        # Let the robot initialize its tensors
        self.robot.init_tensors(self.global_tensor_dict)

        # Initialize warp sensor if needed
        if self.use_warp and self.cfg.sensor_config.enable_camera:
            from aerial_gym.sensors.warp.warp_sensor import WarpSensor

            self.warp_sensor_config = self.cfg.sensor_config.camera_config
            if self.global_tensor_dict.get("CONST_WARP_MESH_ID_LIST") is not None:
                self.image_tensor = torch.zeros(
                    (
                        self.num_envs,
                        self.warp_sensor_config.num_sensors,
                        self.warp_sensor_config.height,
                        self.warp_sensor_config.width,
                    ),
                    device=self.device,
                    requires_grad=False,
                )
                self.global_tensor_dict["depth_range_pixels"] = self.image_tensor
                self.warp_sensor = WarpSensor(
                    self.warp_sensor_config,
                    self.num_envs,
                    self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"],
                    self.device,
                )
                self.warp_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)
                self.warp_sensor.update()

        logger.debug("Prepared IsaacLabRobotManager for simulation")

    def reset_idx(self, env_ids):
        """Reset robot state at given indices."""
        self.robot.reset_idx(env_ids)
        if self.warp_sensor is not None:
            self.warp_sensor.reset_idx(env_ids)
        if self.imu_sensor is not None:
            self.imu_sensor.reset_idx(env_ids)

    def pre_physics_step(self, actions):
        """Apply actions before physics step."""
        self.prev_actions[:] = self.actions[:]
        self.actions[:] = actions
        self.robot.step(self.actions)

    def post_physics_step(self):
        """Update state after physics step."""
        if self.imu_sensor is not None:
            self.imu_sensor.update()

    def capture_sensors(self):
        """Capture sensor data."""
        if self.warp_sensor is not None:
            self.warp_sensor.update()
        if self.camera_sensor is not None:
            self.camera_sensor.update()
