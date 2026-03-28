# AI-Driven Telemedicine Resource Allocation — Lesotho

Reinforcement learning agents learn to allocate scarce healthcare resources
(telemedicine slots, mobile clinics, emergency airlifts) equitably across
Lesotho's 10 districts, prioritising rural and critical patients.

## Quick start

```bash
pip install -r requirements.txt

# 1. Watch random agent (no training required)
python random_demo.py

# 2. Train all algorithms
python training/dqn_training.py
python training/pg_training.py --algo all

# 3. Run best trained agent
python main.py
```

## Project structure

```
environment/
  custom_env.py   — Gymnasium environment (obs, actions, rewards, dynamics)
  rendering.py    — Pygame visualisation

training/
  dqn_training.py — DQN: 10 hyperparameter runs
  pg_training.py  — PPO, A2C, REINFORCE: 10 runs each

models/
  dqn/            — saved DQN checkpoints
  pg/             — saved PPO / A2C / REINFORCE checkpoints

results/          — CSV tables of hyperparameter experiments
logs/             — TensorBoard logs

main.py           — run best agent with GUI
random_demo.py    — random agent demo (no model)
```

## Environment details

| Feature | Value |
|---|---|
| Observation space | Box(12,) float32 |
| Action space | Discrete(5) |
| Max steps / episode | 200 |
| Terminal condition | ≥ 8 untreated critical patients |

### Actions
0. Teleconsult — requires connectivity > 0.4  
1. Mobile clinic dispatch — blocked by weather events  
2. Schedule (defer) — only suitable for low-severity  
3. Ignore — penalised, especially for critical patients  
4. Emergency airlift — limited budget, critical only  

### Key reward signals
- Treat critical rural patient: **+35** (severity +20, rural +15)  
- Ignore critical patient: **−20**  
- Urban-only bias detected: **−15**  
- Successful teleconsult: **+10**  

## Algorithms

| Algorithm | Library | Notes |
|---|---|---|
| DQN | Stable Baselines3 | Value-based, experience replay |
| PPO | Stable Baselines3 | Policy gradient, clipped objective |
| A2C | Stable Baselines3 | Actor-critic, synchronous |
| REINFORCE | Custom (PyTorch) | Vanilla policy gradient |
