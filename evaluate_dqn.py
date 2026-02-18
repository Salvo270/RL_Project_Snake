"""Comprehensive evaluation suite for the Snake DQN agent.

Key goals:
- Robustness: evaluate on multiple random seeds.
- Statistical reliability: run multiple episodes per seed.
- Engineering metrics: reward, fruit efficiency, safety events, win rate.
- Artifacts: JSON/CSV summary + GIF of best episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import environments_fully_observable
from dqn_agent import DQNAgent, DQNConfig


@dataclass
class EvaluationConfig:
    board_size: int = 7
    n_actions: int = 4
    max_steps: int = 500
    n_episodes_per_seed: int = 20
    seeds: Sequence[int] = (42, 100, 2024, 777, 99)
    greedy: bool = True
    save_gif: bool = True
    gif_filename: str = "best_agent_gameplay.gif"
    summary_json: str = "evaluation_summary.json"
    episodes_csv: str = "evaluation_episodes.csv"


def get_env(n: int = 1, size: int = 7):
    return environments_fully_observable.OriginalSnakeEnvironment(n, size)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def scalar_reward(value) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr[0])


def save_gif(frames: Sequence[np.ndarray], filename: str) -> None:
    if not frames:
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    image = ax.imshow(frames[0], origin="lower", cmap="viridis")

    def update(frame):
        image.set_data(frame)
        return [image]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=140,
        blit=True,
    )
    ani.save(filename, writer="pillow", fps=7)
    plt.close(fig)


def evaluate_one_episode(agent: DQNAgent, env, max_steps: int, greedy: bool) -> Dict[str, float | int | list]:
    state = np.asarray(env.to_state(), dtype=np.float32)

    total_reward = 0.0
    fruit_count = 0
    wall_hits = 0
    self_hits = 0
    wins = 0
    frames: List[np.ndarray] = []

    for _ in range(max_steps):
        frames.append(env.boards[0].copy())

        action = agent.select_actions(state, greedy=greedy)
        reward_tensor = env.move(action.reshape(-1, 1))
        reward = scalar_reward(reward_tensor)

        total_reward += reward
        if np.isclose(reward, env.FRUIT_REWARD):
            fruit_count += 1
        elif np.isclose(reward, env.HIT_WALL_REWARD):
            wall_hits += 1
        elif np.isclose(reward, env.ATE_HIMSELF_REWARD):
            self_hits += 1
        elif np.isclose(reward, env.WIN_REWARD):
            wins += 1

        state = np.asarray(env.to_state(), dtype=np.float32)

    return {
        "total_reward": float(total_reward),
        "avg_reward_per_step": float(total_reward / max_steps),
        "fruit_count": int(fruit_count),
        "fruit_rate": float(fruit_count / max_steps),
        "wall_hits": int(wall_hits),
        "wall_hit_rate": float(wall_hits / max_steps),
        "self_hits": int(self_hits),
        "self_hit_rate": float(self_hits / max_steps),
        "wins": int(wins),
        "win_rate": float(wins / max_steps),
        "frames": frames,
    }


def build_agent_for_eval(weights_path: str, board_size: int, n_actions: int) -> DQNAgent:
    state_shape = (board_size, board_size, 4)
    config = DQNConfig(epsilon_start=0.0, epsilon_end=0.0)
    agent = DQNAgent(state_shape=state_shape, n_actions=n_actions, config=config)

    dummy = np.zeros((1, *state_shape), dtype=np.float32)
    _ = agent.online_network(dummy, training=False)

    agent.online_network.load_weights(weights_path)
    agent.target_network.set_weights(agent.online_network.get_weights())
    return agent


def summarize(episode_metrics: List[Dict[str, float | int]]) -> Dict[str, float]:
    rewards = np.array([row["total_reward"] for row in episode_metrics], dtype=np.float32)
    fruit_rates = np.array([row["fruit_rate"] for row in episode_metrics], dtype=np.float32)
    safety_rates = np.array(
        [row["wall_hit_rate"] + row["self_hit_rate"] for row in episode_metrics],
        dtype=np.float32,
    )

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    ci95 = float(1.96 * std_reward / np.sqrt(max(1, len(rewards))))

    return {
        "episodes": float(len(episode_metrics)),
        "reward_mean": mean_reward,
        "reward_std": std_reward,
        "reward_ci95": ci95,
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "fruit_rate_mean": float(np.mean(fruit_rates)),
        "fruit_rate_std": float(np.std(fruit_rates)),
        "safety_event_rate_mean": float(np.mean(safety_rates)),
        "safety_event_rate_std": float(np.std(safety_rates)),
    }


def run_evaluation(weights_path: str, cfg: EvaluationConfig) -> Dict[str, object]:
    agent = build_agent_for_eval(weights_path, cfg.board_size, cfg.n_actions)

    all_rows: List[Dict[str, float | int]] = []
    best_reward = -np.inf
    best_frames: List[np.ndarray] = []

    print("\nStarting evaluation...")
    print(
        f"{'seed':<6} {'episode':<8} {'reward':<10} {'fruit_rate':<10} "
        f"{'wall_rate':<10} {'self_rate':<10}"
    )
    print("-" * 68)

    for seed in cfg.seeds:
        set_global_seed(seed)

        for episode_idx in range(cfg.n_episodes_per_seed):
            env = get_env(n=1, size=cfg.board_size)
            metrics = evaluate_one_episode(
                agent=agent,
                env=env,
                max_steps=cfg.max_steps,
                greedy=cfg.greedy,
            )

            row = {
                "seed": int(seed),
                "episode": int(episode_idx),
                "total_reward": float(metrics["total_reward"]),
                "avg_reward_per_step": float(metrics["avg_reward_per_step"]),
                "fruit_count": int(metrics["fruit_count"]),
                "fruit_rate": float(metrics["fruit_rate"]),
                "wall_hits": int(metrics["wall_hits"]),
                "wall_hit_rate": float(metrics["wall_hit_rate"]),
                "self_hits": int(metrics["self_hits"]),
                "self_hit_rate": float(metrics["self_hit_rate"]),
                "wins": int(metrics["wins"]),
                "win_rate": float(metrics["win_rate"]),
            }
            all_rows.append(row)

            print(
                f"{seed:<6} {episode_idx:<8} {row['total_reward']:<10.3f} "
                f"{row['fruit_rate']:<10.3f} {row['wall_hit_rate']:<10.3f} {row['self_hit_rate']:<10.3f}"
            )

            if row["total_reward"] > best_reward:
                best_reward = row["total_reward"]
                best_frames = metrics["frames"]

    overall_summary = summarize(all_rows)

    by_seed = {}
    for seed in cfg.seeds:
        seed_rows = [row for row in all_rows if row["seed"] == seed]
        by_seed[int(seed)] = summarize(seed_rows)

    with open(cfg.episodes_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "config": {
            "board_size": cfg.board_size,
            "n_actions": cfg.n_actions,
            "max_steps": cfg.max_steps,
            "n_episodes_per_seed": cfg.n_episodes_per_seed,
            "seeds": list(cfg.seeds),
            "greedy": cfg.greedy,
            "weights_path": weights_path,
        },
        "overall": overall_summary,
        "by_seed": by_seed,
        "artifacts": {
            "episodes_csv": str(Path(cfg.episodes_csv).resolve()),
            "summary_json": str(Path(cfg.summary_json).resolve()),
            "gif": str(Path(cfg.gif_filename).resolve()) if cfg.save_gif else None,
        },
    }

    with open(cfg.summary_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if cfg.save_gif and best_frames:
        save_gif(best_frames, cfg.gif_filename)

    print("\n=== Overall Summary ===")
    print(json.dumps(overall_summary, indent=2))
    print(f"\nSaved summary: {cfg.summary_json}")
    print(f"Saved episodes: {cfg.episodes_csv}")
    if cfg.save_gif:
        print(f"Saved GIF: {cfg.gif_filename}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN Snake agent")
    parser.add_argument("--weights", required=True, help="Path to model weights (.h5)")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024, 777, 99])
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = EvaluationConfig(
        board_size=args.board_size,
        max_steps=args.max_steps,
        n_episodes_per_seed=args.episodes_per_seed,
        seeds=tuple(args.seeds),
        save_gif=not args.no_gif,
    )
    run_evaluation(weights_path=args.weights, cfg=cfg)