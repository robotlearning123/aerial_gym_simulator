"""Isaac Lab compatible rl_games runner for aerial_gym.

This module provides rl_games integration using Isaac Lab's RlGamesVecEnvWrapper
instead of the custom AERIALRLGPUEnv.
"""

import os
import yaml
import argparse

import torch

from rl_games.common import env_configurations, vecenv

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def get_args():
    """Parse command line arguments for Isaac Lab compatible training."""
    parser = argparse.ArgumentParser(description="RL Policy")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train network"
    )
    parser.add_argument(
        "--play", action="store_true", help="Play/test network"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument(
        "--file", type=str, default="ppo_aerial_quad.yaml", help="Path to config"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1024, help="Number of environments"
    )
    parser.add_argument(
        "--task", type=str, default="AerialGym-PositionSetpoint-Direct-v0", help="Task ID"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name"
    )
    parser.add_argument(
        "--headless", action="store_true", default=False, help="Run headless"
    )
    parser.add_argument(
        "--rl_device", type=str, default="cuda:0", help="RL device"
    )
    parser.add_argument(
        "--sigma", type=float, default=None, help="Sigma value for fixed sigma"
    )
    parser.add_argument(
        "--track", action="store_true", help="Track with W&B"
    )
    parser.add_argument(
        "--wandb-project-name", type=str, default="rl_games", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity"
    )

    args = parser.parse_args()
    return args


def create_env(task_id, num_envs, headless, rl_device, **kwargs):
    """Create an Isaac Lab environment wrapped for rl_games."""
    import gymnasium as gym

    env = gym.make(
        task_id,
        cfg={
            "scene": {"num_envs": num_envs},
        },
        render_mode="human" if not headless else None,
        **kwargs,
    )

    # Wrap for rl_games
    clip_obs = 10.0
    clip_actions = 1.0
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    return env


def update_config(config, args):
    """Update rl_games config with command line arguments."""
    if args.task is not None:
        config["params"]["config"]["env_name"] = args.task
    if args.experiment_name is not None:
        config["params"]["config"]["name"] = args.experiment_name
    config["params"]["config"]["env_config"] = config["params"]["config"].get("env_config", {})
    config["params"]["config"]["env_config"]["headless"] = args.headless
    config["params"]["config"]["env_config"]["num_envs"] = args.num_envs
    if args.num_envs > 0:
        config["params"]["config"]["num_actors"] = args.num_envs
    if args.seed > 0:
        config["params"]["seed"] = args.seed
        config["params"]["config"]["env_config"]["seed"] = args.seed
    config["params"]["config"]["player"] = {"use_vecenv": True}
    return config


def main():
    """Main entry point for Isaac Lab compatible rl_games training."""
    args = get_args()

    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Register environment
    env_configurations.register(
        args.task,
        {
            "env_creator": lambda **kwargs: create_env(
                args.task,
                args.num_envs,
                args.headless,
                args.rl_device,
                **kwargs,
            ),
            "vecenv_type": "RLGPU",
        },
    )

    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
            config_name, num_actors, **kwargs
        ),
    )

    # Load config
    config_name = args.file
    print(f"Loading config: {config_name}")
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)

    config = update_config(config, vars(args))

    # Run training
    from rl_games.torch_runner import Runner

    runner = Runner()
    try:
        runner.load(config)
    except yaml.YAMLError as exc:
        print(exc)

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if args.track and rank == 0:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )

    runner.run(vars(args))

    if args.track and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
