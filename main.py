"""
main.py — Run the best-performing trained agent with full Pygame GUI

Usage:
    python main.py                        # auto-detects best model
    python main.py --algo ppo --run 3     # run a specific model
    python main.py --random               # random agent demo
    python main.py --episodes 5           # number of episodes
"""

import os
import sys
import argparse
import numpy as np
import time

sys.path.insert(0, os.path.dirname(__file__))


def load_best_model():
    """Read results CSVs and return the best model path and algo."""
    import csv
    best_reward = -1e9
    best_algo   = None
    best_run    = None

    for algo in ["dqn", "ppo", "a2c"]:
        csv_path = f"results/{algo}_results.csv"
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                r = float(row["mean_reward"])
                if r > best_reward:
                    best_reward = r
                    best_algo   = algo
                    best_run    = int(row["run"])

    return best_algo, best_run, best_reward


def load_model(algo: str, run: int):
    if algo == "dqn":
        from stable_baselines3 import DQN
        path = f"models/dqn/run_{run}/best_model"
        if not os.path.exists(path + ".zip"):
            path = f"models/dqn/run_{run}/final"
        return DQN.load(path)

    elif algo == "ppo":
        from stable_baselines3 import PPO
        path = f"models/pg/ppo/run_{run}/best_model"
        if not os.path.exists(path + ".zip"):
            path = f"models/pg/ppo/run_{run}/final"
        return PPO.load(path)

    elif algo == "a2c":
        from stable_baselines3 import A2C
        path = f"models/pg/a2c/run_{run}/best_model"
        if not os.path.exists(path + ".zip"):
            path = f"models/pg/a2c/run_{run}/final"
        return A2C.load(path)

    elif algo == "reinforce":
        import torch
        from training.pg_training import PolicyNet
        from environment.custom_env import LesothoHealthEnv
        env    = LesothoHealthEnv()
        net    = PolicyNet(env.observation_space.shape[0], env.action_space.n, 64)
        path   = f"models/pg/reinforce/run_{run}/policy.pt"
        net.load_state_dict(torch.load(path))
        net.eval()
        env.close()
        return net

    raise ValueError(f"Unknown algo: {algo}")


def predict_reinforce(model, obs):
    import torch
    with torch.no_grad():
        probs = model(torch.FloatTensor(obs).unsqueeze(0))
    return probs.argmax(dim=-1).item()


ACTION_NAMES = [
    "Teleconsult 📞",
    "Mobile clinic 🚑",
    "Schedule later ⏳",
    "Ignore ❌",
    "Emergency airlift 🚁",
]


def run_episode(env, model, algo: str, episode_num: int, verbose: bool = True):
    obs, info = env.reset()
    ep_reward = 0.0
    step      = 0
    done      = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Episode {episode_num}  |  Algorithm: {algo.upper()}")
        print(f"{'='*60}")

    while not done:
        if algo == "reinforce":
            action = predict_reinforce(model, obs)
        elif model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            action    = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        step      += 1
        done       = terminated or truncated

        if verbose and step % 20 == 0:
            print(
                f"  Step {step:>3d} | Action: {ACTION_NAMES[action]:<25s} | "
                f"Reward: {reward:+6.1f} | "
                f"Queue: {info['queue_size']:>2d} | "
                f"Critical: {info['untreated_critical']} | "
                f"Fairness: {info['fairness_score']:.2f}"
            )

    if verbose:
        print(f"\n  Episode complete.")
        print(f"  Total reward    : {ep_reward:.1f}")
        print(f"  Steps taken     : {step}")
        print(f"  Patients treated: {info['treated_total']}")
        print(f"    Urban         : {info['treated_urban']}")
        print(f"    Rural         : {info['treated_total'] - info['treated_urban']}")
        print(f"  Fairness score  : {info['fairness_score']:.3f}")
        reason = "max critical" if terminated else "max steps"
        print(f"  Terminated by   : {reason}")

    return ep_reward, info


def main():
    parser = argparse.ArgumentParser(
        description="Run best RL agent on Lesotho Telemedicine environment"
    )
    parser.add_argument("--algo",     type=str, default=None,
                        choices=["dqn", "ppo", "a2c", "reinforce"],
                        help="Algorithm to load (default: auto best)")
    parser.add_argument("--run",      type=int, default=None,
                        help="Run number to load (default: best run for algo)")
    parser.add_argument("--random",   action="store_true",
                        help="Use random agent (no model)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--no-gui",   action="store_true",
                        help="Disable Pygame GUI")
    args = parser.parse_args()

    from environment.custom_env import LesothoHealthEnv

    render_mode = None if args.no_gui else "human"
    env = LesothoHealthEnv(render_mode=render_mode)

    model = None
    algo  = "random"

    if args.random:
        print("\n[Random Agent Demo — no model loaded]")
        print("This shows the environment without any training.")
        print("Expect poor performance and erratic behaviour.\n")
    else:
        if args.algo is None:
            algo, run, reward = load_best_model()
            if algo is None:
                print("No trained models found. Run training first.")
                print("  python training/dqn_training.py")
                print("  python training/pg_training.py")
                sys.exit(1)
            print(f"\nAuto-selected best model: {algo.upper()} run {run}  "
                  f"(eval reward: {reward:.1f})")
        else:
            algo = args.algo
            run  = args.run or 1

        print(f"Loading {algo.upper()} model (run {run})...")
        model = load_model(algo, run)
        print("Model loaded.\n")

    print("Problem: Healthcare resource allocation across Lesotho's districts.")
    print("Agent:   Decides how to treat each incoming patient given limited resources.")
    print("Rewards: Prioritise critical + rural patients, penalise urban bias & neglect.")
    print("Goal:    Maximise equitable health outcomes within resource constraints.\n")

    all_rewards  = []
    all_fairness = []

    for ep in range(1, args.episodes + 1):
        reward, info = run_episode(env, model, algo, ep, verbose=True)
        all_rewards.append(reward)
        all_fairness.append(info["fairness_score"])
        if render_mode == "human":
            time.sleep(0.5)

    env.close()

    print(f"\n{'='*60}")
    print(f"Summary over {args.episodes} episode(s)")
    print(f"{'='*60}")
    print(f"  Mean episode reward  : {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"  Mean fairness score  : {np.mean(all_fairness):.3f}")
    print(f"  Best episode reward  : {max(all_rewards):.1f}")


if __name__ == "__main__":
    main()
