"""
pg_training.py — Train PPO, A2C, and REINFORCE on LesothoHealthEnv

Each algorithm runs 10 hyperparameter configurations.
Saves:
  • models/pg/<algo>/run_<n>.zip
  • logs/pg/<algo>/run_<n>/
  • results/<algo>_results.csv

Usage:
    python training/pg_training.py
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from environment.custom_env import LesothoHealthEnv

TOTAL_TIMESTEPS = 150_000
EVAL_EPISODES   = 20
SEED            = 42

for d in ["models/pg/ppo", "models/pg/a2c", "models/pg/reinforce",
          "logs/pg/ppo",   "logs/pg/a2c",   "logs/pg/reinforce",
          "results"]:
    os.makedirs(d, exist_ok=True)



# PPO — 10 configurations

PPO_CONFIGS = [
    # Run 1: SB3 defaults
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
         gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0),
    # Run 2: Higher entropy (more exploration)
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
         gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01),
    # Run 3: Very high entropy
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
         gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.05),
    # Run 4: Wider clip range
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
         gamma=0.99, gae_lambda=0.95, clip_range=0.3, ent_coef=0.0),
    # Run 5: Narrow clip range (more conservative)
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
         gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.0),
    # Run 6: Shorter rollout
    dict(learning_rate=3e-4, n_steps=512,  batch_size=64, n_epochs=10,
         gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01),
    # Run 7: Longer rollout
    dict(learning_rate=3e-4, n_steps=4096, batch_size=128, n_epochs=10,
         gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0),
    # Run 8: High LR
    dict(learning_rate=1e-3, n_steps=2048, batch_size=64, n_epochs=10,
         gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0),
    # Run 9: Low LR
    dict(learning_rate=1e-4, n_steps=2048, batch_size=64, n_epochs=10,
         gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005),
    # Run 10: Low gamma
    dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
         gamma=0.90, gae_lambda=0.90, clip_range=0.2, ent_coef=0.01),
]


# A2C — 10 configurations

A2C_CONFIGS = [
    # Run 1: SB3 defaults
    dict(learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
         ent_coef=0.0,  vf_coef=0.5, max_grad_norm=0.5),
    # Run 2: More entropy
    dict(learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
         ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5),
    # Run 3: High entropy
    dict(learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
         ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5),
    # Run 4: Higher LR
    dict(learning_rate=2e-3, n_steps=5, gamma=0.99, gae_lambda=1.0,
         ent_coef=0.0,  vf_coef=0.5, max_grad_norm=0.5),
    # Run 5: Lower LR
    dict(learning_rate=2e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
         ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5),
    # Run 6: Longer n_steps
    dict(learning_rate=7e-4, n_steps=20, gamma=0.99, gae_lambda=1.0,
         ent_coef=0.0,  vf_coef=0.5, max_grad_norm=0.5),
    # Run 7: Even longer n_steps
    dict(learning_rate=7e-4, n_steps=50, gamma=0.99, gae_lambda=1.0,
         ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5),
    # Run 8: Higher value function coefficient
    dict(learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
         ent_coef=0.0,  vf_coef=1.0, max_grad_norm=0.5),
    # Run 9: Gradient clipping
    dict(learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
         ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.2),
    # Run 10: Low gamma
    dict(learning_rate=7e-4, n_steps=5, gamma=0.90, gae_lambda=0.90,
         ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5),
]


# REINFORCE — 10 configurations

REINFORCE_CONFIGS = [
    dict(lr=1e-3, gamma=0.99, hidden=64,  n_episodes=800,  entropy_coef=0.0),
    dict(lr=5e-4, gamma=0.99, hidden=64,  n_episodes=800,  entropy_coef=0.0),
    dict(lr=1e-4, gamma=0.99, hidden=64,  n_episodes=800,  entropy_coef=0.0),
    dict(lr=1e-3, gamma=0.95, hidden=64,  n_episodes=800,  entropy_coef=0.0),
    dict(lr=1e-3, gamma=0.90, hidden=64,  n_episodes=800,  entropy_coef=0.0),
    dict(lr=1e-3, gamma=0.99, hidden=128, n_episodes=800,  entropy_coef=0.0),
    dict(lr=1e-3, gamma=0.99, hidden=256, n_episodes=800,  entropy_coef=0.0),
    dict(lr=1e-3, gamma=0.99, hidden=64,  n_episodes=800,  entropy_coef=0.01),
    dict(lr=1e-3, gamma=0.99, hidden=64,  n_episodes=800,  entropy_coef=0.05),
    dict(lr=2e-3, gamma=0.99, hidden=128, n_episodes=1000, entropy_coef=0.01),
]



# Custom REINFORCE implementation (not in SB3)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def run_reinforce(cfg: dict) -> dict:
    env    = LesothoHealthEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy    = PolicyNet(obs_dim, act_dim, cfg["hidden"])
    optimizer = optim.Adam(policy.parameters(), lr=cfg["lr"])
    gamma     = cfg["gamma"]
    ent_coef  = cfg["entropy_coef"]

    episode_rewards = []
    t0 = time.time()

    for episode in range(cfg["n_episodes"]):
        obs, _   = env.reset()
        log_probs, rewards_ep, entropies = [], [], []
        done = False

        while not done:
            obs_t  = torch.FloatTensor(obs).unsqueeze(0)
            probs  = policy(obs_t)
            dist   = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards_ep.append(reward)
            done = terminated or truncated

        # Compute discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns)
        # Normalise returns
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy loss (REINFORCE)
        log_probs_t = torch.stack(log_probs)
        entropy_t   = torch.stack(entropies).mean()
        loss = -(log_probs_t * returns_t).mean() - ent_coef * entropy_t

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        episode_rewards.append(sum(rewards_ep))

        if (episode + 1) % 100 == 0:
            recent = np.mean(episode_rewards[-100:])
            print(f"    ep {episode+1}/{cfg['n_episodes']}  "
                  f"mean reward (last 100): {recent:.1f}")

    env.close()

    # Evaluate
    eval_env   = LesothoHealthEnv()
    eval_r, eval_fair = [], []
    for _ in range(EVAL_EPISODES):
        obs, _ = eval_env.reset()
        ep_r   = 0.0
        done   = False
        while not done:
            with torch.no_grad():
                probs  = policy(torch.FloatTensor(obs).unsqueeze(0))
                action = probs.argmax(dim=-1).item()
            obs, r, te, tr, info = eval_env.step(action)
            ep_r += r
            done  = te or tr
        eval_r.append(ep_r)
        eval_fair.append(info.get("fairness_score", 0.5))
    eval_env.close()

    return {
        "policy":        policy,
        "mean_reward":   float(np.mean(eval_r)),
        "fairness_score": float(np.mean(eval_fair)),
        "train_time_s":  round(time.time() - t0, 1),
        "episode_rewards": episode_rewards,
    }



# Generic SB3 evaluation helper

def evaluate_sb3(model, n=EVAL_EPISODES):
    env = LesothoHealthEnv()
    rewards, fairness = [], []
    for _ in range(n):
        obs, _ = env.reset()
        ep_r   = 0.0
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(int(action))
            ep_r += r
            done  = te or tr
        rewards.append(ep_r)
        fairness.append(info.get("fairness_score", 0.5))
    env.close()
    return float(np.mean(rewards)), float(np.mean(fairness))



# PPO training loop

def train_ppo():
    results = []
    print("\n" + "=" * 60)
    print("PPO Hyperparameter Search — LesothoHealthEnv")
    print("=" * 60)

    for run_idx, cfg in enumerate(PPO_CONFIGS, start=1):
        print(f"\n[PPO Run {run_idx}/10]  LR={cfg['learning_rate']}  "
              f"n_steps={cfg['n_steps']}  ent_coef={cfg['ent_coef']}  "
              f"clip={cfg['clip_range']}")

        log_dir    = f"logs/pg/ppo/run_{run_idx}"
        model_path = f"models/pg/ppo/run_{run_idx}"
        os.makedirs(log_dir, exist_ok=True)

        env      = Monitor(LesothoHealthEnv(), log_dir)
        eval_env = Monitor(LesothoHealthEnv())

        model = PPO(
            "MlpPolicy", env, verbose=0, seed=SEED,
            tensorboard_log=log_dir, **cfg,
        )
        cb = EvalCallback(eval_env, best_model_save_path=model_path,
                          log_path=log_dir, eval_freq=10_000,
                          n_eval_episodes=10, verbose=0)

        t0 = time.time()
        model.learn(TOTAL_TIMESTEPS, callback=cb, progress_bar=False)
        elapsed = time.time() - t0

        mean_r, fair = evaluate_sb3(model)
        print(f"  → Mean reward: {mean_r:.1f}  |  Fairness: {fair:.3f}  |  {elapsed:.0f}s")

        results.append({
            "run": run_idx, **cfg,
            "mean_reward": round(mean_r, 2),
            "fairness_score": round(fair, 3),
            "train_time_s": round(elapsed, 1),
        })
        model.save(f"{model_path}/final")
        env.close(); eval_env.close()

    _save_csv(results, "results/ppo_results.csv")
    _print_best(results, "PPO")
    return results



# A2C training loop

def train_a2c():
    results = []
    print("\n" + "=" * 60)
    print("A2C Hyperparameter Search — LesothoHealthEnv")
    print("=" * 60)

    for run_idx, cfg in enumerate(A2C_CONFIGS, start=1):
        print(f"\n[A2C Run {run_idx}/10]  LR={cfg['learning_rate']}  "
              f"n_steps={cfg['n_steps']}  ent_coef={cfg['ent_coef']}")

        log_dir    = f"logs/pg/a2c/run_{run_idx}"
        model_path = f"models/pg/a2c/run_{run_idx}"
        os.makedirs(log_dir, exist_ok=True)

        env      = Monitor(LesothoHealthEnv(), log_dir)
        eval_env = Monitor(LesothoHealthEnv())

        model = A2C(
            "MlpPolicy", env, verbose=0, seed=SEED,
            tensorboard_log=log_dir, **cfg,
        )
        cb = EvalCallback(eval_env, best_model_save_path=model_path,
                          log_path=log_dir, eval_freq=10_000,
                          n_eval_episodes=10, verbose=0)

        t0 = time.time()
        model.learn(TOTAL_TIMESTEPS, callback=cb, progress_bar=False)
        elapsed = time.time() - t0

        mean_r, fair = evaluate_sb3(model)
        print(f"  → Mean reward: {mean_r:.1f}  |  Fairness: {fair:.3f}  |  {elapsed:.0f}s")

        results.append({
            "run": run_idx, **cfg,
            "mean_reward": round(mean_r, 2),
            "fairness_score": round(fair, 3),
            "train_time_s": round(elapsed, 1),
        })
        model.save(f"{model_path}/final")
        env.close(); eval_env.close()

    _save_csv(results, "results/a2c_results.csv")
    _print_best(results, "A2C")
    return results



# REINFORCE training loop

def train_reinforce():
    results = []
    print("\n" + "=" * 60)
    print("REINFORCE Hyperparameter Search — LesothoHealthEnv")
    print("=" * 60)

    for run_idx, cfg in enumerate(REINFORCE_CONFIGS, start=1):
        print(f"\n[REINFORCE Run {run_idx}/10]  lr={cfg['lr']}  "
              f"gamma={cfg['gamma']}  hidden={cfg['hidden']}  "
              f"ent_coef={cfg['entropy_coef']}")

        out = run_reinforce(cfg)
        print(f"  → Mean reward: {out['mean_reward']:.1f}  |  "
              f"Fairness: {out['fairness_score']:.3f}  |  {out['train_time_s']}s")

        # Save torch model
        model_dir = f"models/pg/reinforce/run_{run_idx}"
        os.makedirs(model_dir, exist_ok=True)
        torch.save(out["policy"].state_dict(), f"{model_dir}/policy.pt")

        # Save reward curve
        np.save(f"{model_dir}/episode_rewards.npy",
                np.array(out["episode_rewards"]))

        results.append({
            "run":            run_idx,
            "lr":             cfg["lr"],
            "gamma":          cfg["gamma"],
            "hidden":         cfg["hidden"],
            "n_episodes":     cfg["n_episodes"],
            "entropy_coef":   cfg["entropy_coef"],
            "mean_reward":    round(out["mean_reward"],    2),
            "fairness_score": round(out["fairness_score"], 3),
            "train_time_s":   out["train_time_s"],
        })

    _save_csv(results, "results/reinforce_results.csv")
    _print_best(results, "REINFORCE")
    return results


def _save_csv(rows: list, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Results saved to {path}")


def _print_best(results: list, algo: str):
    best = max(results, key=lambda r: r["mean_reward"])
    print(f"\nBest {algo} run: #{best['run']}  "
          f"reward={best['mean_reward']}  fairness={best['fairness_score']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c", "reinforce", "all"],
                        default="all")
    args = parser.parse_args()

    if args.algo in ("ppo",      "all"): train_ppo()
    if args.algo in ("a2c",      "all"): train_a2c()
    if args.algo in ("reinforce","all"): train_reinforce()
