"""Comprehensive evidence test for aerial_gym on Isaac Sim 6.0 / Isaac Lab 3.0.

Tests all robot types, controllers, and environments using subprocesses
to avoid USD stage conflicts.

Run with:
  /mnt/storage/isaacsim-6.0-official/venv/bin/python test_full_evidence.py
"""

import sys
import os
import subprocess
import json
import time

PYTHON = "/mnt/storage/isaacsim-6.0-official/venv/bin/python"
COMBO_SCRIPT = "/mnt/storage/isaacsim-6.0-official/aerial_gym_simulator/test_combo.py"
AERIAL_GYM_DIR = "/mnt/storage/isaacsim-6.0-official/aerial_gym_simulator"

def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def run_combo(sim_name, env_name, robot_name, ctrl_name, num_envs=2, timeout=120):
    """Run a single combo in a subprocess. Returns (ok, detail)."""
    try:
        result = subprocess.run(
            [PYTHON, COMBO_SCRIPT, sim_name, env_name, robot_name, ctrl_name, str(num_envs)],
            capture_output=True, text=True, timeout=timeout,
            cwd=AERIAL_GYM_DIR,
        )
        if result.returncode == 0:
            # Parse the JSON result from stderr
            for line in result.stderr.strip().split("\n"):
                try:
                    data = json.loads(line)
                    if data.get("status") == "PASS":
                        return True, f"z={data['z']:.4f}, shape={data['shape']}"
                except json.JSONDecodeError:
                    continue
            return True, "passed (no detail)"
        else:
            # Extract error from stderr
            err_lines = result.stderr.strip().split("\n")
            err = err_lines[-1] if err_lines else "unknown error"
            return False, err[:100]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:100]

# ── Test matrix ─────────────────────────────────────────────────────────
ROBOTS = [
    "base_quadrotor",
    "base_octarotor",
    "base_random",
    "morphy_stiff",
    "lmf1",
    "lmf2",
    "x500",
    "tinyprop",
]

CONTROLLERS = [
    "lee_position_control",
    "lee_velocity_control",
    "lee_attitude_control",
    "lee_rates_control",
    "lee_acceleration_control",
    "no_control",
]

ENVIRONMENTS = [
    "empty_env",
    "env_with_obstacles",
    "forest_env",
]

# ── Run tests ───────────────────────────────────────────────────────────
log("=" * 60)
log("  AERIAL GYM SIMULATOR - FULL EVIDENCE TEST")
log("  Isaac Sim 6.0 / Isaac Lab 3.0")
log("=" * 60)

test_results = []
total = 0
passed = 0
failed = 0

# Test each robot with position controller
log("\n[1/3] Testing all robot types...")
for robot_name in ROBOTS:
    total += 1
    tag = f"{robot_name} + lee_position_control"
    ok, detail = run_combo("base_sim", "empty_env", robot_name, "lee_position_control")
    test_results.append((tag, "PASS" if ok else "FAIL", detail))
    if ok:
        passed += 1
        log(f"  PASS: {tag}  ({detail})")
    else:
        failed += 1
        log(f"  FAIL: {tag}  ({detail})")

# Test each environment
log("\n[2/3] Testing all environments...")
for env_name in ENVIRONMENTS:
    total += 1
    tag = f"base_quadrotor + {env_name}"
    ok, detail = run_combo("base_sim", env_name, "base_quadrotor", "lee_position_control", num_envs=4)
    test_results.append((tag, "PASS" if ok else "FAIL", detail))
    if ok:
        passed += 1
        log(f"  PASS: {tag}  ({detail})")
    else:
        failed += 1
        log(f"  FAIL: {tag}  ({detail})")

# Test each controller
log("\n[3/3] Testing all controllers...")
for ctrl_name in CONTROLLERS:
    total += 1
    tag = f"base_quadrotor + {ctrl_name}"
    ok, detail = run_combo("base_sim", "empty_env", "base_quadrotor", ctrl_name)
    test_results.append((tag, "PASS" if ok else "FAIL", detail))
    if ok:
        passed += 1
        log(f"  PASS: {tag}  ({detail})")
    else:
        failed += 1
        log(f"  FAIL: {tag}  ({detail})")

# ── Summary ─────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log(f"  EVIDENCE SUMMARY")
log(f"  Total combos tested: {total}")
log(f"  Passed: {passed}")
log(f"  Failed: {failed}")
log("=" * 60)

# Save results to file
output_dir = "/mnt/storage/isaacsim-6.0-official/aerial_gym_simulator/evidence"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
    f.write("Aerial Gym Simulator - Isaac Sim 6.0 / Isaac Lab 3.0 Evidence\n")
    f.write("=" * 60 + "\n\n")
    for tag, status, detail in test_results:
        f.write(f"[{status}] {tag}: {detail}\n")
    f.write(f"\nTotal: {total}, Passed: {passed}, Failed: {failed}\n")
log(f"\nResults saved to {output_dir}/test_results.txt")

# Print machine-readable summary
summary = {
    "total": total,
    "passed": passed,
    "failed": failed,
    "results": [{"tag": t, "status": s, "detail": d} for t, s, d in test_results],
}
with open(os.path.join(output_dir, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
log(f"JSON results saved to {output_dir}/results.json")
log("\n=== ALL EVIDENCE TESTS COMPLETE ===")
