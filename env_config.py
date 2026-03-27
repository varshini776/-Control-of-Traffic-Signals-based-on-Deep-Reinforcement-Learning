"""
env_config.py
=============

Configuration for traffic simulation environment.
"""

from dataclasses import dataclass


@dataclass
class EnvConfig:

    # Traffic Flow
    num_phases: int = 4
    arrival_rate: float = 0.3
    """Poisson arrival rate per arm per step (0.1=light, 0.3=medium, 0.5=heavy)"""

    turn_ratio: float = 0.3
    """Fraction of vehicles that turn instead of going straight"""

    saturation_flow: int = 3
    """Max vehicles that can leave per green step per arm"""

    max_queue_capacity: int = 30
    """Max vehicles per arm (prevents overflow)"""

    # Signal Control
    yellow_duration: int = 2
    """Duration of yellow phase (steps)"""

    min_green_steps: int = 5
    """Minimum green duration before switching allowed"""

    # Episode
    max_steps: int = 1000000
    """Maximum steps (set very high for continuous operation)"""    

    # Randomness
    seed: int = 42
    """Random seed for reproducibility"""


# Presets for experiments
def light_traffic(seed=42):
    return EnvConfig(arrival_rate=0.1, seed=seed)


def medium_traffic(seed=42):
    return EnvConfig(arrival_rate=0.3, seed=seed)


def heavy_traffic(seed=42):
    return EnvConfig(arrival_rate=0.5, seed=seed)