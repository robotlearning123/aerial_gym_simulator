from setuptools import find_packages
from distutils.core import setup

setup(
    name="aerial_gym",
    version="3.0.0",
    author="Mihir Kulkarni",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="mihir.kulkarni@ntnu.no",
    description="Isaac Lab environments for Aerial Robots",
    install_requires=[
        "isaaclab",
        "isaacsim",
        "matplotlib",
        "numpy",
        "torch",
        "warp-lang",
        "trimesh",
        "gymnasium",
        "rl-games",
    ],
)
