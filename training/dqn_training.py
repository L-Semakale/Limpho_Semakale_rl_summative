"""
dqn_training.py — Train DQN on LesothoHealthEnv using Stable Baselines3

Runs 10 hyperparameter configurations and saves:
  • models/dqn/run_<n>.zip
  • logs/dqn/run_<n>/  (TensorBoard)
  • results/dqn_results.csv

Usage:
    python training/dqn_training.py
"""

import os
import csv
import time
import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from environment.custom_env import LesothoHealthEnv

os.makedirs("models/dqn",    exist_ok=True)
os.makedirs("logs/dqn",      exist_ok=True)
os.makedirs("results",       exist_ok=True)

TOTAL_TIMESTEPS = 150_000
EVAL_EPISODES   = 20
SEED            = 42

# ── 10 Hyperparameter configurations
CONFIGS = [
    # Run 1: Baseline
    dict(learning_rate=1e-3,  gamma=0.99, batch_size=64,  buffer_size=50_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.2,
         learning_starts=1000, target_update_interval=500, train_freq=4, tau=1.0),
    # Run 2: Lower LR
    dict(learning_rate=5e-4,  gamma=0.99, batch_size=64,  buffer_size=50_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.2,
         learning_starts=1000, target_update_interval=500, train_freq=4, tau=1.0),
    # Run 3: Very low LR
    dict(learning_rate=1e-4,  gamma=0.99, batch_size=64,  buffer_size=50_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.3,
         learning_starts=1000, target_update_interval=500, train_freq=4, tau=1.0),
    # Run 4: High gamma
    dict(learning_rate=1e-3,  gamma=0.999, batch_size=64, buffer_size=50_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.2,
         learning_starts=1000, target_update_interval=500, train_freq=4, tau=1.0),
    # Run 5: Low gamma
    dict(learning_rate=1e-3,  gamma=0.90, batch_size=64,  buffer_size=50_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.1,  exploration_fraction=0.2,
         learning_starts=1000, target_update_interval=500, train_freq=4, tau=1.0),
    # Run 6: Large buffer
    dict(learning_rate=1e-3,  gamma=0.99, batch_size=128, buffer_size=100_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.15,
         learning_starts=2000, target_update_interval=1000, train_freq=4, tau=1.0),
    # Run 7: Small batch, slow target update
    dict(learning_rate=5e-4,  gamma=0.99, batch_size=32,  buffer_size=50_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.2,
         learning_starts=500,  target_update_interval=1000, train_freq=4, tau=1.0),
    # Run 8: Soft target updates (Polyak)
    dict(learning_rate=1e-3,  gamma=0.99, batch_size=64,  buffer_size=50_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.2,
         learning_starts=1000, target_update_interval=1,   train_freq=4, tau=0.005),
    # Run 9: High exploration
    dict(learning_rate=1e-3,  gamma=0.99, batch_size=64,  buffer_size=50_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.2,  exploration_fraction=0.4,
         learning_starts=1000, target_update_interval=500, train_freq=4, tau=1.0),
    # Run 10: More frequent training
    dict(learning_rate=5e-4,  gamma=0.99, batch_size=64,  buffer_size=80_000,
         exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.2,
         learning_starts=1000, target_update_interval=250, train_freq=1, tau=1.0),
]


def evaluate_model(model, n_episodes: int = EVAL_EPISODES):
    """Run n_episodes with the trained model and return mean reward + fairness."""
    env = LesothoHealthEnv()
    rewards, fairness_scores = [], []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        fairness_scores.append(info.get("fairness_score", 0.5))
    env.close()
    return float(np.mean(rewards)), float(np.mean(fairness_scores))


def train_all():
    results = []
    print("=" * 60)
    print("DQN Hyperparameter Search — LesothoHealthEnv")
    print("=" * 60)

    for run_idx, cfg in enumerate(CONFIGS, start=1):
        print(f"\n[Run {run_idx}/10]  LR={cfg['learning_rate']}  gamma={cfg['gamma']}  "
              f"buffer={cfg['buffer_size']}  eps_final={cfg['exploration_final_eps']}")

        log_dir   = f"logs/dqn/run_{run_idx}"
        model_path = f"models/dqn/run_{run_idx}"
        os.makedirs(log_dir, exist_ok=True)

        env      = Monitor(LesothoHealthEnv(), log_dir)
        eval_env = Monitor(LesothoHealthEnv())

        model = DQN(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            seed=SEED,
            tensorboard_log=log_dir,
            **cfg,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_path,
            log_path=log_dir,
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True,
            verbose=0,
        )

        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=False)
        elapsed = time.time() - t0

        mean_reward, fairness = evaluate_model(model)
        print(f"  → Mean reward: {mean_reward:.1f}  |  Fairness: {fairness:.3f}  |  Time: {elapsed:.0f}s")

        results.append({
            "run":                    run_idx,
            "learning_rate":          cfg["learning_rate"],
            "gamma":                  cfg["gamma"],
            "batch_size":             cfg["batch_size"],
            "buffer_size":            cfg["buffer_size"],
            "exploration_final_eps":  cfg["exploration_final_eps"],
            "exploration_fraction":   cfg["exploration_fraction"],
            "target_update_interval": cfg["target_update_interval"],
            "tau":                    cfg["tau"],
            "train_freq":             cfg["train_freq"],
            "mean_reward":            round(mean_reward, 2),
            "fairness_score":         round(fairness, 3),
            "train_time_s":           round(elapsed, 1),
        })

        model.save(f"{model_path}/final")
        env.close()
        eval_env.close()

    # Save CSV
    csv_path = "results/dqn_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Results saved to {csv_path}")

    best = max(results, key=lambda r: r["mean_reward"])
    print(f"\nBest DQN run: #{best['run']}  "
          f"reward={best['mean_reward']}  fairness={best['fairness_score']}")
    return results


if __name__ == "__main__":
    train_all()
