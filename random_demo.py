"""
random_demo.py — Visualise the environment with a purely random agent

No training is involved. This satisfies the assignment requirement:
"Create a static file that shows the agent taking random actions in the
custom environment."

Usage:
    python random_demo.py              # runs until window closed
    python random_demo.py --steps 200  # run for N steps then exit
    python random_demo.py --no-gui     # terminal-only output
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from environment.custom_env import LesothoHealthEnv, ACTION_TELECONSULT

ACTION_NAMES = [
    "Teleconsult",
    "Mobile clinic",
    "Schedule later",
    "Ignore",
    "Emergency airlift",
]


def run_random_demo(max_steps: int = 200, gui: bool = True, speed: float = 0.1):
    render_mode = "human" if gui else None
    env = LesothoHealthEnv(render_mode=render_mode)
    obs, info = env.reset(seed=0)

    print("\n" + "=" * 65)
    print("  RANDOM AGENT DEMO — Lesotho Telemedicine Environment")
    print("=" * 65)
    print("  No model loaded. Agent selects actions uniformly at random.")
    print("  Watch for poor performance: critical patients ignored,")
    print("  resources wasted, urban bias, and high untreated counts.\n")

    total_reward = 0.0
    step         = 0

    try:
        done = False
        while not done and step < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step         += 1
            done          = terminated or truncated

            # Terminal printout every 10 steps
            if step % 10 == 0 or done:
                print(
                    f"  Step {step:>3d} | "
                    f"Action: {ACTION_NAMES[action]:<18s} | "
                    f"Reward: {reward:+6.1f} | "
                    f"Cumulative: {total_reward:+8.1f} | "
                    f"Queue: {info['queue_size']:>2d} | "
                    f"Critical (untreated): {info['untreated_critical']}"
                )

            if gui:
                time.sleep(speed)

    except KeyboardInterrupt:
        print("\n  Demo interrupted by user.")

    env.close()

    print("\n" + "-" * 65)
    print(f"  Episode ended after {step} steps")
    print(f"  Total reward     : {total_reward:.1f}  (random agents score poorly)")
    print(f"  Treated total    : {info['treated_total']}")
    print(f"    Urban          : {info['treated_urban']}")
    print(f"    Rural          : {info['treated_total'] - info['treated_urban']}")
    print(f"  Fairness score   : {info['fairness_score']:.3f}")
    print(f"  Untreated crit.  : {info['untreated_critical']}")
    print("-" * 65)
    print("  Compare this to a trained agent — the difference shows")
    print("  exactly what reinforcement learning achieves here.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int,   default=200,
                        help="Max steps before auto-exit (default: 200)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable Pygame GUI, print only")
    parser.add_argument("--speed",  type=float, default=0.1,
                        help="Seconds between steps in GUI mode (default: 0.1)")
    args = parser.parse_args()

    run_random_demo(max_steps=args.steps, gui=not args.no_gui, speed=args.speed)
