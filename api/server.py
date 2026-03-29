"""
api/server.py — Flask REST API for the Lesotho Telemedicine RL Agent

Serializes environment state and agent decisions to JSON so any
frontend, mobile app, or external service can consume them.

Endpoints:
  GET  /api/state          — current environment state as JSON
  POST /api/step           — advance one step, returns new state + action
  POST /api/reset          — reset episode
  GET  /api/status         — agent/model info
  GET  /api/history        — full episode action/reward history

Usage:
    python api/server.py --algo ppo --run 1
    python api/server.py --random          # random agent, no model needed
"""

import os
import sys
import json
import time
import argparse
import threading
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from environment.custom_env import LesothoHealthEnv, DISTRICTS, N_ACTIONS

app = Flask(__name__)
CORS(app)  # allow frontend on any port to call this API

# ── Global agent state (thread-safe via lock) ─────────────────────────────
_lock    = threading.Lock()
_env     = None
_model   = None
_algo    = "random"
_obs     = None
_done    = False
_history = []          # list of step dicts
_episode = 0

ACTION_NAMES = [
    "teleconsult",
    "mobile_clinic",
    "schedule",
    "ignore",
    "emergency_airlift",
]

ACTION_LABELS = [
    "Teleconsult 📞",
    "Mobile Clinic 🚑",
    "Schedule ⏳",
    "Ignore ❌",
    "Emergency Airlift 🚁",
]


# ── Serialization helpers ──────────────────────────────────────────────────

def _district_info(dist_id: int) -> dict:
    d = DISTRICTS[dist_id]
    return {
        "id":           dist_id,
        "name":         d[0],
        "is_rural":     d[1],
        "connectivity": round(d[2], 2),
        "position":     {"x": round(d[3], 3), "y": round(d[4], 3)},
    }


def _patient_to_dict(p: dict) -> dict:
    severity_map = {0: "low", 1: "medium", 2: "critical"}
    age_map      = {0: "child", 1: "adult", 2: "elderly"}
    return {
        "severity":       p["severity"],
        "severity_label": severity_map[p["severity"]],
        "district":       _district_info(p["district"]),
        "is_rural":       p["is_rural"],
        "wait_steps":     p["wait"],
        "connectivity":   round(p["connectivity"], 3),
        "age_group":      age_map.get(p["age_group"], "adult"),
    }


def _env_state_to_json(info: dict) -> dict:
    """Convert full environment state to a clean JSON-serialisable dict."""
    env = _env
    patients   = [_patient_to_dict(p) for p in env._patient_queue]
    by_district = {i: [] for i in range(len(DISTRICTS))}
    for p in patients:
        by_district[p["district"]["id"]].append(p)

    districts_out = []
    for i, d in enumerate(DISTRICTS):
        districts_out.append({
            **_district_info(i),
            "patients":   by_district[i],
            "is_blocked": i in env._blocked_districts,
            "mobile_en_route": any(dist == i for _, dist in env._mobile_busy),
        })

    fairness = info.get("fairness_score", 1.0)
    treated  = info.get("treated_total", 0)
    urban    = info.get("treated_urban", 0)

    return {
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "episode":      _episode,
        "step":         info.get("step", 0),
        "max_steps":    200,
        "done":         _done,
        "resources": {
            "mobile_clinics": {
                "available": info.get("mobile_left", 2),
                "total":     2,
            },
            "tele_slots": {
                "available": info.get("tele_slots", 4),
                "total":     4,
            },
            "airlift_budget": {
                "available": info.get("airlift_budget", 3),
                "total":     3,
            },
        },
        "queue": {
            "size":             info.get("queue_size", 0),
            "critical_count":   info.get("untreated_critical", 0),
            "patients":         patients,
        },
        "districts":   districts_out,
        "weather": {
            "severity":          round(info.get("weather", 0), 3),
            "blocked_districts": list(info.get("blocked_districts", [])),
        },
        "metrics": {
            "episode_reward":  round(info.get("episode_reward", 0), 2),
            "treated_total":   treated,
            "treated_urban":   urban,
            "treated_rural":   treated - urban,
            "fairness_score":  round(fairness, 3),
            "fairness_status": (
                "good"    if fairness > 0.7 else
                "warning" if fairness > 0.4 else
                "poor"
            ),
        },
        "agent": {
            "algorithm": _algo,
            "model_loaded": _model is not None,
        },
    }


def _step_result_to_json(action: int, reward: float, info: dict) -> dict:
    return {
        "action": {
            "id":     action,
            "key":    ACTION_NAMES[action],
            "label":  ACTION_LABELS[action],
        },
        "reward":       round(reward, 3),
        "state":        _env_state_to_json(info),
    }


# ── Flask routes ──────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status":       "running",
        "algorithm":    _algo,
        "model_loaded": _model is not None,
        "episode":      _episode,
        "version":      "1.0.0",
        "endpoints": [
            "GET  /api/status",
            "GET  /api/state",
            "POST /api/reset",
            "POST /api/step",
            "GET  /api/history",
        ],
    })


@app.route("/api/state", methods=["GET"])
def get_state():
    with _lock:
        if _env is None or _obs is None:
            return jsonify({"error": "Environment not initialised. Call POST /api/reset first."}), 400
        _, info = _env.observation_space, _env._get_info()
        return jsonify(_env_state_to_json(info))


@app.route("/api/reset", methods=["POST"])
def reset_env():
    global _obs, _done, _history, _episode
    with _lock:
        obs, info = _env.reset()
        _obs      = obs
        _done     = False
        _history  = []
        _episode += 1
        return jsonify({
            "message": "Episode reset",
            "episode": _episode,
            "state":   _env_state_to_json(info),
        })


@app.route("/api/step", methods=["POST"])
def step_env():
    global _obs, _done
    with _lock:
        if _done:
            return jsonify({"error": "Episode done. Call POST /api/reset to start a new one."}), 400

        # Choose action
        if _model is None:
            action = _env.action_space.sample()
        elif _algo == "reinforce":
            import torch
            with torch.no_grad():
                probs  = _model(torch.FloatTensor(_obs).unsqueeze(0))
                action = int(probs.argmax(dim=-1).item())
        else:
            action, _ = _model.predict(_obs, deterministic=True)
            action    = int(action)

        # Override via request body if provided (for manual control)
        body = request.get_json(silent=True) or {}
        if "action" in body:
            action = int(body["action"])

        obs, reward, terminated, truncated, info = _env.step(action)
        _obs  = obs
        _done = terminated or truncated

        result = _step_result_to_json(action, reward, info)
        _history.append({
            "step":    info["step"],
            "action":  ACTION_NAMES[action],
            "reward":  round(reward, 3),
            "fairness": round(info.get("fairness_score", 1.0), 3),
            "queue":   info["queue_size"],
        })

        return jsonify(result)


@app.route("/api/history", methods=["GET"])
def get_history():
    with _lock:
        return jsonify({
            "episode": _episode,
            "steps":   len(_history),
            "history": _history,
            "summary": {
                "total_reward": round(sum(h["reward"] for h in _history), 2),
                "mean_reward":  round(np.mean([h["reward"] for h in _history]) if _history else 0, 3),
                "final_fairness": _history[-1]["fairness"] if _history else 1.0,
            }
        })


@app.route("/api/autorun", methods=["POST"])
def autorun():
    """Run a full episode automatically and return the trajectory."""
    global _obs, _done, _history, _episode
    with _lock:
        obs, info = _env.reset()
        _obs      = obs
        _done     = False
        _history  = []
        _episode += 1

        trajectory = []
        done = False
        while not done:
            if _model is None:
                action = _env.action_space.sample()
            elif _algo == "reinforce":
                import torch
                with torch.no_grad():
                    probs  = _model(torch.FloatTensor(_obs).unsqueeze(0))
                    action = int(probs.argmax(dim=-1).item())
            else:
                action, _ = _model.predict(_obs, deterministic=True)
                action    = int(action)

            obs, reward, terminated, truncated, info = _env.step(action)
            _obs  = obs
            done  = terminated or truncated
            _done = done

            trajectory.append({
                "step":    info["step"],
                "action":  ACTION_NAMES[action],
                "reward":  round(reward, 3),
                "fairness": round(info.get("fairness_score", 1.0), 3),
                "queue":   info["queue_size"],
                "critical": info["untreated_critical"],
            })

        total_r = sum(s["reward"] for s in trajectory)
        return jsonify({
            "episode":       _episode,
            "steps":         len(trajectory),
            "total_reward":  round(total_r, 2),
            "fairness_score": trajectory[-1]["fairness"] if trajectory else 1.0,
            "trajectory":    trajectory,
            "final_info":    {
                "treated_total": info["treated_total"],
                "treated_urban": info["treated_urban"],
                "treated_rural": info["treated_total"] - info["treated_urban"],
            }
        })


# ── Entry point ───────────────────────────────────────────────────────────

def _load_model(algo: str, run: int):
    if algo == "random":
        return None
    if algo == "dqn":
        from stable_baselines3 import DQN
        path = f"models/dqn/run_{run}/best_model"
        if not os.path.exists(path + ".zip"):
            path = f"models/dqn/run_{run}/final"
        return DQN.load(path)
    if algo == "ppo":
        from stable_baselines3 import PPO
        path = f"models/pg/ppo/run_{run}/best_model"
        if not os.path.exists(path + ".zip"):
            path = f"models/pg/ppo/run_{run}/final"
        return PPO.load(path)
    if algo == "a2c":
        from stable_baselines3 import A2C
        path = f"models/pg/a2c/run_{run}/best_model"
        if not os.path.exists(path + ".zip"):
            path = f"models/pg/a2c/run_{run}/final"
        return A2C.load(path)
    if algo == "reinforce":
        import torch
        from training.pg_training import PolicyNet
        env = LesothoHealthEnv()
        net = PolicyNet(env.observation_space.shape[0], env.action_space.n, 64)
        net.load_state_dict(torch.load(f"models/pg/reinforce/run_{run}/policy.pt"))
        net.eval()
        env.close()
        return net
    raise ValueError(f"Unknown algo: {algo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="random",
                        choices=["random", "dqn", "ppo", "a2c", "reinforce"])
    parser.add_argument("--run",  type=int, default=1)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    _algo  = args.algo
    _env   = LesothoHealthEnv()
    _model = _load_model(args.algo, args.run)

    # Initialise first episode
    _obs, info = _env.reset()

    print(f"\n{'='*55}")
    print(f"  Lesotho Telemedicine RL — API Server")
    print(f"{'='*55}")
    print(f"  Algorithm  : {_algo.upper()}")
    print(f"  Model      : {'loaded' if _model else 'random agent'}")
    print(f"  Base URL   : http://localhost:{args.port}/api")
    print(f"\n  Endpoints:")
    print(f"    GET  /api/status")
    print(f"    GET  /api/state")
    print(f"    POST /api/reset")
    print(f"    POST /api/step")
    print(f"    POST /api/autorun")
    print(f"    GET  /api/history")
    print(f"\n  Open frontend/index.html in your browser to see")
    print(f"  the live dashboard consuming this API.\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)
