"""Test script to verify aerial_gym works with Isaac Sim 6.0 / Isaac Lab 3.0.

Must be run with the Isaac Sim venv:
  /mnt/storage/isaacsim-6.0-official/venv/bin/python test_isaaclab_sim.py

All progress is written to stderr so it's visible even when Isaac Sim
redirects stdout.
"""

import sys

def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

# Step 1: Initialize Isaac Sim runtime BEFORE any other imports
log("[1/8] Creating SimulationApp...")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
log("[1/8] SimulationApp created")

# Step 2: Import aerial_gym (now that omni modules are available)
log("[2/8] Importing aerial_gym...")
import aerial_gym
log("[2/8] aerial_gym imported")

# Step 3: Import and verify pytorch3d_compat
log("[3/8] Verifying pytorch3d_compat...")
from aerial_gym.utils.pytorch3d_compat import (
    matrix_to_quaternion, quaternion_to_matrix,
    euler_angles_to_matrix, matrix_to_euler_angles,
    rotation_6d_to_matrix, matrix_to_rotation_6d,
)
import torch
m = torch.eye(3).unsqueeze(0).to("cuda:0")
q = matrix_to_quaternion(m)
assert torch.allclose(q, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda:0")), f"quat failed: {q}"
log("[3/8] pytorch3d_compat verified")

# Step 4: Import controllers
log("[4/8] Importing controllers...")
from aerial_gym.control.controllers.position_control import LeePositionController
from aerial_gym.control.controllers.attitude_control import LeeAttitudeController
from aerial_gym.control.controllers.rates_control import LeeRatesController
from aerial_gym.control.controllers.acceleration_control import LeeAccelerationController
log("[4/8] All controllers imported")

# Step 5: Import Isaac Lab components
log("[5/8] Importing Isaac Lab components...")
from aerial_gym.env_manager.isaaclab_env_manager import IsaacLabEnvManager
from aerial_gym.env_manager.isaaclab_env_orchestrator import IsaacLabEnvOrchestrator
from aerial_gym.assets.isaaclab_asset import IsaacLabAsset
log("[5/8] Isaac Lab components imported")

# Step 6: Build environment
log("[6/8] Building environment (2 envs)...")
from aerial_gym.sim.sim_builder import SimBuilder
env_manager = SimBuilder().build_env(
    sim_name="base_sim",
    env_name="empty_env",
    robot_name="base_quadrotor",
    controller_name="lee_position_control",
    args=None,
    device="cuda:0",
    num_envs=2,
    headless=True,
    use_warp=False,
)
log("[6/8] Environment built successfully")

# Step 7: Run simulation steps
log("[7/8] Resetting and running 10 simulation steps...")
actions = torch.zeros((env_manager.num_envs, 4), device="cuda:0")
env_manager.reset()
for i in range(10):
    env_manager.step(actions=actions)
    if (i + 1) % 5 == 0:
        log(f"  Step {i+1}/10 done")
log("[7/8] Ran 10 simulation steps successfully")

# Step 8: Verify global_tensor_dict has expected keys
log("[8/8] Verifying observation tensors...")
obs = env_manager.get_obs()
expected_keys = ["robot_position", "robot_orientation", "robot_linvel", "robot_angvel"]
for key in expected_keys:
    assert key in obs, f"Missing key: {key}"
    log(f"  {key}: shape={obs[key].shape}, dtype={obs[key].dtype}")

# Verify robot is at expected height (z ~ 1.0m for quadrotor spawn)
pos = obs["robot_position"]
log(f"  robot_position values:\n{pos}")
assert pos.shape == (2, 3), f"Expected (2,3), got {pos.shape}"
assert pos[0, 2] > 0.5, f"Robot z should be > 0.5, got {pos[0, 2]}"

log("")
log("=== ALL TESTS PASSED ===")

app.close()
