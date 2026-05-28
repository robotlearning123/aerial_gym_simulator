"""Isaac Lab compatible asset wrapper for aerial_gym.

This module provides an Isaac Lab compatible asset wrapper that replaces
the Isaac Gym based isaacgym_asset.py.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("IsaacLabAsset")


class IsaacLabAsset:
    """Isaac Lab compatible asset wrapper.

    Wraps URDF/MJCF/USD asset loading for use with Isaac Lab's ArticulationCfg
    and RigidObjectCfg.
    """

    def __init__(self, asset_root, asset_file, asset_options=None, device="cuda:0"):
        self.asset_root = asset_root
        self.asset_file = asset_file
        self.device = device
        self.asset_options = asset_options or {}

        # Determine file type
        self.file_ext = os.path.splitext(asset_file)[1].lower()
        self.full_path = os.path.join(asset_root, asset_file)

        logger.debug(f"Created IsaacLabAsset: {self.full_path}")

    def to_articulation_cfg(
        self,
        prim_path: str = "{ENV_REGEX_NS}/Robot",
        fix_base: bool = False,
        init_pos: tuple = (0.0, 0.0, 0.0),
        init_quat: tuple = (0.0, 0.0, 0.0, 1.0),  # (x, y, z, w) identity
    ) -> ArticulationCfg:
        """Convert to Isaac Lab ArticulationCfg."""

        # Build spawn config based on file type
        if self.file_ext == ".urdf":
            spawn_cfg = sim_utils.UrdfFileCfg(
                asset_path=self.full_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=self.asset_options.get("max_linear_velocity", 1000.0),
                    max_angular_velocity=self.asset_options.get("max_angular_velocity", 64.0),
                    max_depenetration_velocity=10.0,
                    disable_gravity=self.asset_options.get("disable_gravity", False),
                    linear_damping=self.asset_options.get("linear_damping", 0.0),
                    angular_damping=self.asset_options.get("angular_damping", 0.05),
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.001,
                    fix_root_link=fix_base,
                ),
            )
        elif self.file_ext == ".usd":
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=self.full_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=self.asset_options.get("max_linear_velocity", 1000.0),
                    max_angular_velocity=self.asset_options.get("max_angular_velocity", 64.0),
                    max_depenetration_velocity=10.0,
                    disable_gravity=self.asset_options.get("disable_gravity", False),
                    linear_damping=self.asset_options.get("linear_damping", 0.0),
                    angular_damping=self.asset_options.get("angular_damping", 0.05),
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.001,
                    fix_root_link=fix_base,
                ),
            )
        else:
            raise ValueError(f"Unsupported asset file type: {self.file_ext}")

        cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=spawn_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_quat,
            ),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0,
                    damping=0.0,
                ),
            },
        )

        return cfg

    def to_rigid_object_cfg(
        self,
        prim_path: str = "{ENV_REGEX_NS}/Object",
        init_pos: tuple = (0.0, 0.0, 0.0),
        init_quat: tuple = (0.0, 0.0, 0.0, 1.0),  # (x, y, z, w) identity
    ) -> RigidObjectCfg:
        """Convert to Isaac Lab RigidObjectCfg."""

        if self.file_ext == ".urdf":
            spawn_cfg = sim_utils.UrdfFileCfg(
                asset_path=self.full_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=self.asset_options.get("max_linear_velocity", 1000.0),
                    max_angular_velocity=self.asset_options.get("max_angular_velocity", 64.0),
                    max_depenetration_velocity=10.0,
                    disable_gravity=self.asset_options.get("disable_gravity", False),
                    linear_damping=self.asset_options.get("linear_damping", 0.0),
                    angular_damping=self.asset_options.get("angular_damping", 0.05),
                ),
            )
        elif self.file_ext == ".usd":
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=self.full_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=self.asset_options.get("max_linear_velocity", 1000.0),
                    max_angular_velocity=self.asset_options.get("max_angular_velocity", 64.0),
                    max_depenetration_velocity=10.0,
                    disable_gravity=self.asset_options.get("disable_gravity", False),
                    linear_damping=self.asset_options.get("linear_damping", 0.0),
                    angular_damping=self.asset_options.get("angular_damping", 0.05),
                ),
            )
        else:
            raise ValueError(f"Unsupported asset file type: {self.file_ext}")

        cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_quat,
            ),
        )

        return cfg
