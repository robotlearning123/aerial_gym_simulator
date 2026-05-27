"""Position setpoint task registration for Isaac Lab."""

import gymnasium as gym

from .position_setpoint_task_isaaclab import (
    AerialGymPositionSetpointEnv,
    AerialGymPositionSetpointCfg,
)

# Register the environment with gymnasium for Isaac Lab compatibility
gym.register(
    id="AerialGym-PositionSetpoint-Direct-v0",
    entry_point="aerial_gym.task.position_setpoint_task:AerialGymPositionSetpointEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialGymPositionSetpointCfg,
    },
)
