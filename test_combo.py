"""Test a single robot+controller+environment combination.

Called by test_full_evidence.py as a subprocess for each combo.
Exit code 0 = pass, non-zero = fail.
"""

import sys
import json

sim_name = sys.argv[1]
env_name = sys.argv[2]
robot_name = sys.argv[3]
ctrl_name = sys.argv[4]
num_envs = int(sys.argv[5])

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import torch
from aerial_gym.sim.sim_builder import SimBuilder

env_mgr = SimBuilder().build_env(
    sim_name=sim_name,
    env_name=env_name,
    robot_name=robot_name,
    controller_name=ctrl_name,
    args=None,
    device="cuda:0",
    num_envs=num_envs,
    headless=True,
    use_warp=False,
)
actions = torch.zeros((num_envs, 4), device="cuda:0")
env_mgr.reset()
for _ in range(5):
    env_mgr.step(actions=actions)
obs = env_mgr.get_obs()
pos = obs["robot_position"]
z = pos[0, 2].item()

result = {"status": "PASS", "z": z, "shape": list(pos.shape)}
sys.stderr.write(json.dumps(result) + "\n")

app.close()
sys.exit(0)
