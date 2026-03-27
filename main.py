"""
main.py
=======

Main entry point for the enhanced traffic intersection simulator.

Features:
- 4 individual traffic lights (N, S, E, W)
- Individual vehicles that obey traffic lights and turn
- 4-phase traffic control (NS straight, NS turn, EW straight, EW turn)
- 3 control policies: greedy, random, fixed-time
- Real-time visualization with pygame
- Pause/Resume with SPACE
- Reset with R

Usage:
    python main.py --policy greedy   # Run with greedy policy
    python main.py --policy random   # Run with random policy
    python main.py --policy fixed    # Run with fixed-time policy
"""

import argparse
from env_config import EnvConfig, medium_traffic
from env import TrafficEnv
from ui import TrafficUI


def run_ui(policy_name="greedy", traffic="medium"):
    """Run the UI simulation with specified policy and traffic level."""
    
    if traffic == "light":
        cfg = EnvConfig(arrival_rate=0.1)
    elif traffic == "heavy":
        cfg = EnvConfig(arrival_rate=0.5)
    else:  # medium
        cfg = EnvConfig(arrival_rate=0.3)

    env = TrafficEnv(cfg, width=600, height=600)
    ui = TrafficUI(env, policy_name=policy_name)
    ui.run(policy=policy_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚦 Advanced Traffic Intersection Simulator"
    )
    parser.add_argument(
        "--policy",
        choices=["greedy", "random", "fixed", "ddqn"],
        default="greedy",
        help="Control policy for traffic lights (default: greedy)",
    )
    parser.add_argument(
        "--traffic",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="Traffic intensity (default: medium)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🚦 ADVANCED TRAFFIC INTERSECTION SIMULATOR")
    print("=" * 70)
    print(f"Policy: {args.policy.upper()}")
    print(f"Traffic: {args.traffic.upper()}")
    print()
    print("CONTROLS:")
    print("  SPACE - Pause/Resume simulation")
    print("  R     - Reset episode")
    print("  Q     - Quit")
    print()
    print("COLORS:")
    print("  🟦 Blue     - Vehicles (straight)")
    print("  🟩 Cyan     - Vehicles (turning)")
    print("  🟩 Green    - Go")
    print("  🟨 Yellow   - Prepare to stop")
    print("  🟥 Red      - Stop")
    print("=" * 70)
    print()

    run_ui(policy_name=args.policy, traffic=args.traffic)