# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import argparse
import distutils.util


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def parse_sim_params(args, cfg):
    """Parse simulation parameters from config dict.

    Returns a dict of simulation parameters compatible with Isaac Lab's SimulationCfg.
    """
    sim_params = {}

    # Extract sim parameters from config
    if "sim" in cfg:
        sim_cfg = cfg["sim"]
        if isinstance(sim_cfg, dict):
            sim_params.update(sim_cfg)
        else:
            sim_params.update(class_to_dict(sim_cfg))

    # Override with args
    if hasattr(args, "headless") and args.headless is not None:
        if "viewer" not in sim_params:
            sim_params["viewer"] = {}
        sim_params["viewer"]["headless"] = args.headless

    return sim_params


def update_cfg_from_args(cfg, args):
    if cfg is None:
        raise ValueError("cfg is None")
    if hasattr(args, "headless") and args.headless is not None:
        if "viewer" in cfg:
            cfg["viewer"]["headless"] = args.headless
    if hasattr(args, "num_envs") and args.num_envs is not None:
        if "env" in cfg:
            cfg["env"]["num_envs"] = args.num_envs
    return cfg


def parse_device_str(sim_device):
    """Parse device string into device type and device ID."""
    if sim_device.startswith("cuda"):
        parts = sim_device.split(":")
        device_type = "cuda"
        device_id = int(parts[1]) if len(parts) > 1 else 0
    elif sim_device == "cpu":
        device_type = "cpu"
        device_id = -1
    else:
        device_type = "cuda"
        device_id = 0
    return device_type, device_id


def parse_arguments(description="Aerial Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args, unknown_args = parser.parse_known_args()
    print("[aerial_gym] Unknown args: ", unknown_args)

    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    args.use_gpu_pipeline = (args.sim_device_type == "cuda")
    args.use_gpu = (args.sim_device_type == "cuda")
    args.physics_engine = "physx"

    if no_graphics and hasattr(args, "nographics") and args.nographics:
        args.headless = True

    return args


def get_args(additional_parameters=[]):
    custom_parameters = [
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": "64",
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": True,
            "help": "Use warp for rendering",
        },
    ]
    # parse arguments
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters + additional_parameters,
    )

    # name alignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def asset_class_to_AssetOptions(asset_class):
    """Convert asset config class to a dict of options compatible with Isaac Lab.

    Returns a dict that can be used with Isaac Lab's UsdFileCfg or UrdfFileCfg.
    """
    return {
        "collapse_fixed_joints": asset_class.collapse_fixed_joints,
        "replace_cylinder_with_capsule": asset_class.replace_cylinder_with_capsule,
        "flip_visual_attachments": asset_class.flip_visual_attachments,
        "fix_base_link": asset_class.fix_base_link,
        "density": asset_class.density,
        "angular_damping": asset_class.angular_damping,
        "linear_damping": asset_class.linear_damping,
        "max_angular_velocity": asset_class.max_angular_velocity,
        "max_linear_velocity": asset_class.max_linear_velocity,
        "disable_gravity": asset_class.disable_gravity,
    }
