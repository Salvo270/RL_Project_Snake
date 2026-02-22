import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, List, Dict
import csv
import json
from matplotlib import animation

# Custom modules
import environments_fully_observable
import environments_partially_observable
from baseline import BaselineAgent
from dqn_agent import DQNAgent, DQNConfig
from po_dqn_agent import PartiallyObservableDQNAgent, PODQNConfig
from ppo_agent import PPOAgent, PPOConfig

# Global seeds for reproducibility
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

# Global configuration
SEED = 0
BOARD_SIZE = 7
MASK_SIZE = 2
N_PARALLEL_BOARDS = 1000
GAMMA = 0.9

# Local results directory
BASE_OUTPUT_DIR = Path("./results")

OUTPUTS = {
    'fo_dqn': BASE_OUTPUT_DIR / 'fo_dqn',
    'po_dqn': BASE_OUTPUT_DIR / 'po_dqn',
    'po_safety': BASE_OUTPUT_DIR / 'po_safety',
    'ppo': BASE_OUTPUT_DIR / 'ppo',
    'comparison': BASE_OUTPUT_DIR / 'comparison'
}

for output_dir in OUTPUTS.values():
    output_dir.mkdir(parents=True, exist_ok=True)

print("✓ Setup complete. Output directories created.")

# Configurations
STATE_SHAPE_FO = (7, 7, 4)
N_ACTIONS = 4

@dataclass
class EvaluationConfig:
    board_size: int
    n_actions: int
    max_steps: int
    n_episodes_per_seed: int
    seeds: Sequence[int]
    greedy: bool
    save_gif: bool
    gif_filename: str

@dataclass
class EvalConfigPO:
    weights_path: str
    output_dir: Path = OUTPUTS['po_dqn']
    board_size: int = 7
    mask_size: int = 2
    max_steps: int = 500
    n_episodes_per_seed: int = 20
    seeds: Sequence[int] = (42, 100, 2024, 777, 99)
    save_gif: bool = True

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def run_evaluation(weights_path: str, cfg: EvaluationConfig):
    """Evaluate FO DQN agent."""
    env = environments_fully_observable.OriginalSnakeEnvironment(1, cfg.board_size)

    config = DQNConfig(gamma=GAMMA, learning_rate=1e-4, epsilon_decay_steps=10000 // 2)
    agent = DQNAgent(state_shape=STATE_SHAPE_FO, n_actions=cfg.n_actions, config=config)

    dummy_input = np.zeros((1, *STATE_SHAPE_FO), dtype=np.float32)
    agent.online_network(dummy_input)
    agent.online_network.load_weights(weights_path)

    all_results = []
    all_rewards_per_step = []
    best_gif_reward = -np.inf
    best_gif_frames = []

    print(f"Evaluating FO DQN Agent ({len(cfg.seeds)} seeds, {cfg.n_episodes_per_seed} episodes/seed)...")

    for seed in cfg.seeds:
        set_seed(seed)
        for ep_idx in range(cfg.n_episodes_per_seed):
            total_reward = 0.0
            frames = []
            rewards_per_step = []
            state = np.asarray(env.to_state(), dtype=np.float32)

            for t in range(cfg.max_steps):
                frames.append(env.boards[0].copy())
                action = agent.select_actions(state, greedy=cfg.greedy)
                reward_tensor = env.move(action.reshape(-1, 1))
                reward = float(np.asarray(reward_tensor).flatten()[0])
                next_state = np.asarray(env.to_state(), dtype=np.float32)

                total_reward += reward
                rewards_per_step.append(reward)
                state = next_state

                if (np.isclose(reward, env.HIT_WALL_REWARD) or
                    np.isclose(reward, env.ATE_HIMSELF_REWARD) or
                    np.isclose(reward, env.WIN_REWARD)):
                    break

            all_rewards_per_step.append(rewards_per_step)
            all_results.append({
                "seed": seed,
                "episode": ep_idx,
                "total_reward": total_reward
            })

            if total_reward > best_gif_reward:
                best_gif_reward = total_reward
                best_gif_frames = frames

    # Save results to CSV
    csv_path = OUTPUTS['fo_dqn'] / "evaluation_episodes.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "episode", "total_reward"])
        writer.writeheader()
        writer.writerows(all_results)

    # Generate GIF if requested
    if cfg.save_gif and best_gif_frames:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")
        img = ax.imshow(best_gif_frames[0], origin="lower")

        def update(frame):
            img.set_data(frame)
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=best_gif_frames, interval=150, blit=True)
        ani.save(cfg.gif_filename, writer="pillow", fps=6)
        plt.close(fig)

    # Compute summary
    rewards = [r["total_reward"] for r in all_results]
    best_reward = max(rewards) if rewards else -np.inf
    summary = {
        "overall": {
            "episodes": len(rewards),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_ci95": 1.96 * np.std(rewards) / np.sqrt(len(rewards)),
            "fruit_rate_mean": np.mean([len([r for r in rs if r > 0]) / len(rs) for rs in all_rewards_per_step if rs]),
            "safety_event_rate_mean": np.mean([len([r for r in rs if r < 0]) / len(rs) for rs in all_rewards_per_step if rs]),
            "best_reward": best_reward
        }
    }

    json_path = OUTPUTS['fo_dqn'] / "evaluation_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)

    return summary

def get_env_single(n: int, po: bool = False):
    if po:
        return environments_partially_observable.OriginalSnakeEnvironment(n, BOARD_SIZE, MASK_SIZE)
    else:
        return environments_fully_observable.OriginalSnakeEnvironment(n, BOARD_SIZE)

def evaluate_po_episode(agent, env, max_steps):
    raw_state = np.asarray(env.to_state(), dtype=np.float32)
    agent.stacker.reset(raw_state)

    total_reward = 0.0
    frames = []
    rewards_per_step = []

    for t in range(max_steps):
        frames.append(env.boards[0].copy())
        current_stacked_state = agent.stacker.buffer
        action = agent.select_actions(current_stacked_state, greedy=True)
        reward_tensor = env.move(action.reshape(-1, 1))
        reward = float(np.asarray(reward_tensor).flatten()[0])
        next_raw_state = np.asarray(env.to_state(), dtype=np.float32)
        agent.stacker.update(next_raw_state)

        total_reward += reward
        rewards_per_step.append(reward)

        if (np.isclose(reward, env.HIT_WALL_REWARD) or
            np.isclose(reward, env.ATE_HIMSELF_REWARD) or
            np.isclose(reward, env.WIN_REWARD)):
            break

    return {
        "total_reward": total_reward,
        "steps": t + 1,
        "frames": frames,
        "rewards_per_step": rewards_per_step
    }

def run_po_evaluation(cfg: EvalConfigPO):
    env_dummy = get_env_single(1, po=True)
    raw_shape = tuple(np.asarray(env_dummy.to_state()).shape[1:])

    agent = PartiallyObservableDQNAgent(raw_shape, N_ACTIONS, 1, PODQNConfig(epsilon_start=0.0, epsilon_end=0.0))

    if not os.path.exists(cfg.weights_path):
        print(f"Error: Weights not found at {cfg.weights_path}")
        return

    dummy_input = np.zeros((1, *agent.stacked_shape), dtype=np.float32)
    agent.online_network(dummy_input)
    agent.online_network.load_weights(cfg.weights_path)

    all_results = []
    all_rewards_per_step = []
    best_gif_reward = -np.inf
    best_gif_frames = []

    print(f"Evaluating PO DQN Agent ({len(cfg.seeds)} seeds, {cfg.n_episodes_per_seed} episodes/seed)...")

    for seed in cfg.seeds:
        set_seed(seed)
        for ep_idx in range(cfg.n_episodes_per_seed):
            env = get_env_single(1, po=True)
            metrics = evaluate_po_episode(agent, env, cfg.max_steps)

            all_rewards_per_step.append(metrics["rewards_per_step"])
            all_results.append({
                "seed": seed,
                "episode": ep_idx,
                "total_reward": metrics["total_reward"]
            })

            if metrics["total_reward"] > best_gif_reward:
                best_gif_reward = metrics["total_reward"]
                best_gif_frames = metrics["frames"]

    # Compute statistics
    rewards = [r["total_reward"] for r in all_results]
    best_reward = max(rewards) if rewards else -np.inf
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    ci95 = 1.96 * std_reward / np.sqrt(len(rewards)) if len(rewards) > 0 else 0.0

    print(f"Episodes: {len(rewards)}")
    print(f"Avg Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"95% CI: [{mean_reward - ci95:.2f}, {mean_reward + ci95:.2f}]")
    print(f"Best Episode Reward: {best_reward:.2f}")

    # Save to CSV
    csv_path = cfg.output_dir / "evaluation_po_episodes.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "episode", "total_reward"])
        writer.writeheader()
        writer.writerows(all_results)

    # Generate plots
    generate_po_plots(all_results, all_rewards_per_step, cfg.output_dir)

    if cfg.save_gif and best_gif_frames:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")
        img = ax.imshow(best_gif_frames[0], origin="lower")

        def update(frame):
            img.set_data(frame)
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=best_gif_frames, interval=150, blit=True)
        ani.save(str(cfg.output_dir / "best_po_gameplay.gif"), writer="pillow", fps=6)
        plt.close(fig)

def generate_po_plots(results: List[Dict], rewards_per_step: List, output_dir: Path):
    sns.set_theme(style="whitegrid")

    seeds = [r['seed'] for r in results]
    rewards = [r['total_reward'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=seeds, y=rewards, palette="Blues", ax=ax)
    ax.set_title("PO Agent Robustness Across Seeds (POMDP)", fontsize=13)
    ax.set_xlabel("Random Seed", fontsize=12)
    ax.set_ylabel("Total Reward per Episode", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(output_dir / "po_robustness_boxplot.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(str(output_dir / "po_robustness_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    max_len = max(len(r) for r in rewards_per_step)
    padded = np.full((len(rewards_per_step), max_len), np.nan)
    for i, r_list in enumerate(rewards_per_step):
        padded[i, :len(r_list)] = r_list

    mean_curve = np.nanmean(padded, axis=0)
    window = 5
    if len(mean_curve) > window:
        smooth_mean = np.convolve(mean_curve, np.ones(window) / window, mode='valid')
        x_axis = np.arange(len(smooth_mean))
    else:
        smooth_mean = mean_curve
        x_axis = np.arange(len(mean_curve))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_axis, smooth_mean, color='#2980b9', linewidth=2.5, label='Mean Reward (Smoothed)')
    ax.fill_between(x_axis, smooth_mean * 0.9, smooth_mean * 1.1, color='#2980b9', alpha=0.2)
    ax.set_title("PO Agent: Temporal Reward Evolution", fontsize=13)
    ax.set_xlabel("Step within Episode", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "po_temporal_dynamics.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(str(output_dir / "po_temporal_dynamics.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fo_po_comparison(output_dir: Path):
    import pandas as pd

    fo_csv = OUTPUTS['fo_dqn'] / "evaluation_episodes.csv"
    po_csv = OUTPUTS['po_dqn'] / "evaluation_po_episodes.csv"

    if not os.path.exists(fo_csv):
        print(f"Warning: {fo_csv} not found. Skipping FO vs PO comparison.")
        return

    df_fo = pd.read_csv(fo_csv)
    df_fo['Type'] = 'Fully Observable'

    if os.path.exists(po_csv):
        df_po = pd.read_csv(po_csv)
        df_po['Type'] = 'Partially Observable'
        df_combined = pd.concat([df_fo[['total_reward', 'Type']], df_po[['total_reward', 'Type']]], ignore_index=True)
    else:
        df_combined = df_fo[['total_reward', 'Type']]

    mean_fo = df_fo['total_reward'].mean()
    if os.path.exists(po_csv):
        mean_po = df_po['total_reward'].mean()
        gap_pct = ((mean_fo - mean_po) / mean_fo) * 100
    else:
        gap_pct = 0.0

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(x="Type", y="total_reward", data=df_combined, palette=["#4CAF50", "#F44336"], ax=ax, width=0.5)
    sns.stripplot(x="Type", y="total_reward", data=df_combined, color=".2", alpha=0.4, size=3, jitter=True, ax=ax)

    ax.set_title(f"Information Horizon Impact on Performance\nPerformance Gap: -{gap_pct:.1f}%", fontsize=14)
    ax.set_ylabel("Total Reward per Episode", fontsize=12)
    ax.set_xlabel("")

    if os.path.exists(po_csv):
        for i, mean_val in enumerate([mean_fo, mean_po]):
            ax.text(i, mean_val, f'μ={mean_val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(str(output_dir / "fo_vs_po_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(str(output_dir / "fo_vs_po_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison plot saved to {output_dir}")

def evaluate_ppo():
    WEIGHTS_PATH = str(OUTPUTS['ppo'] / "ppo_iter_2000.weights.h5")
    PDF_SAVE_PATH = str(OUTPUTS['ppo'] / "ppo_eval_2000_analysis.pdf")
    GIF_SAVE_PATH = str(OUTPUTS['ppo'] / "ppo_best_gameplay.gif")

    N_EPISODES = 100
    MAX_STEPS = 500

    def create_and_save_gif(frames, filename):
        if not frames:
            return
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis("off")
        img = ax.imshow(frames[0], origin="lower", cmap="viridis")

        def update(frame):
            img.set_data(frame)
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=150, blit=True)
        ani.save(filename, writer="pillow", fps=7)
        plt.close(fig)

    print("\nInitializing Environment and Agent for PPO...")
    env = environments_partially_observable.OriginalSnakeEnvironment(n_boards=1, board_size=7, mask_size=2)
    raw_shape = tuple(np.asarray(env.to_state()).shape[1:])

    agent = PPOAgent(raw_state_shape=raw_shape, n_actions=4, n_boards=1, n_stack=4, config=PPOConfig())

    agent.model(np.zeros((1, *agent.stacked_shape)))
    agent.load_weights(WEIGHTS_PATH)
    print(f"✓ Weights loaded successfully from: {WEIGHTS_PATH}")

    rewards, lengths, fruits, wall_hits = [], [], [], []
    best_reward = -np.inf
    best_frames = []

    print(f"\n▶️ Starting PPO evaluation over {N_EPISODES} episodes...")
    for ep in range(N_EPISODES):
        env.boards[0].fill(env.EMPTY)
        env.boards[0, [0, -1], :] = env.WALL
        env.boards[0, :, [0, -1]] = env.WALL
        env.bodies[0] = []

        avail = np.argwhere(env.boards[0] == env.EMPTY)
        if len(avail) > 0:
            env.boards[0][tuple(avail[np.random.choice(len(avail))])] = env.HEAD

        avail = np.argwhere(env.boards[0] == env.EMPTY)
        if len(avail) > 0:
            env.boards[0][tuple(avail[np.random.choice(len(avail))])] = env.FRUIT

        agent.stacker.reset(np.asarray(env.to_state(), dtype=np.float32))

        ep_rew, ep_len, ep_fruits, ep_walls = 0, 0, 0, 0
        current_episode_frames = []

        for step in range(MAX_STEPS):
            current_episode_frames.append(env.boards[0].copy())

            action, _, _ = agent.select_action(agent.stacker.buffer, training=False)
            r_tensor = env.move(np.array([[int(action)]]))
            r = float(np.array(r_tensor).flatten()[0])

            ep_rew += r
            ep_len += 1
            if np.isclose(r, env.FRUIT_REWARD): ep_fruits += 1
            if np.isclose(r, env.HIT_WALL_REWARD): ep_walls += 1

            agent.stacker.update(np.asarray(env.to_state(), dtype=np.float32))

            if np.isclose(r, env.HIT_WALL_REWARD) or np.isclose(r, env.ATE_HIMSELF_REWARD) or np.isclose(r, env.WIN_REWARD):
                current_episode_frames.append(env.boards[0].copy())
                break

        rewards.append(ep_rew)
        lengths.append(ep_len)
        fruits.append(ep_fruits)
        wall_hits.append(ep_walls)

        if ep_rew > best_reward:
            best_reward = ep_rew
            best_frames = current_episode_frames

    print("\nGenerating PPO evaluation plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('PPO Agent Evaluation Metrics (100 Episodes)', fontsize=18, fontweight='bold', y=0.98)

    def moving_average(data, window_size=10):
        if len(data) < window_size: return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window = 10

    ax = axes[0, 0]
    ax.plot(rewards, color='#1f77b4', alpha=0.25, linestyle='-', marker='o', markersize=3, label='Raw Data')
    if len(rewards) >= window:
        ax.plot(range(window-1, len(rewards)), moving_average(rewards, window), color='#1f77b4', linewidth=2.5, label=f'Trend (MA {window})')
    ax.set_title("Total Reward per Episode", fontsize=14, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12)

    ax = axes[0, 1]
    ax.plot(lengths, color='#2ca02c', alpha=0.25, linestyle='-', marker='o', markersize=3, label='Raw Data')
    if len(lengths) >= window:
        ax.plot(range(window-1, len(lengths)), moving_average(lengths, window), color='#2ca02c', linewidth=2.5, label=f'Trend (MA {window})')
    ax.set_title("Episode Length (Survival)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Steps", fontsize=12)

    ax = axes[1, 0]
    ax.plot(fruits, color='#9467bd', alpha=0.25, linestyle='-', marker='o', markersize=3, label='Raw Data')
    if len(fruits) >= window:
        ax.plot(range(window-1, len(fruits)), moving_average(fruits, window), color='#9467bd', linewidth=2.5, label=f'Trend (MA {window})')
    ax.set_title("Fruits Eaten per Episode", fontsize=14, fontweight='bold')
    ax.set_ylabel("Fruits Count", fontsize=12)

    ax = axes[1, 1]
    ax.plot(wall_hits, color='#d62728', alpha=0.6, linestyle='-', marker='x', markersize=4, label='Collisions')
    ax.set_title("Wall Hits per Episode (Safety)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Hits", fontsize=12)
    ax.set_yticks([0, 1])

    for ax in axes.flat:
        ax.set_xlabel("Episode", fontsize=12)
        ax.legend(loc='upper right', fontsize=10, frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')
        ax.tick_params(colors='#333333')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PDF_SAVE_PATH, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Generating GIF for the best PPO episode...")
    create_and_save_gif(best_frames, GIF_SAVE_PATH)

    print("\n" + "=" * 60)
    print("PPO EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f" PDF Report saved to: {PDF_SAVE_PATH}")
    print(f" Best Episode GIF saved to: {GIF_SAVE_PATH}")
    print("\nAVERAGE STATISTICS:")
    print(f"   Mean Reward:       {np.mean(rewards):.2f}")
    print(f"   Mean Steps Survived: {np.mean(lengths):.1f}")
    print(f"   Mean Fruits Eaten: {np.mean(fruits):.1f}")
    print(f"   Best Episode Reward: {best_reward:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    # FO DQN Evaluation
    print("\n" + "="*50)
    print("FO DQN EVALUATION")
    print("="*50)
    weights_file = str(OUTPUTS['fo_dqn'] / "dqn_fo_final.weights.h5")
    if os.path.exists(weights_file):
        eval_config = EvaluationConfig(
            board_size=7,
            n_actions=4,
            max_steps=1000,
            n_episodes_per_seed=20,
            seeds=(42, 100, 2024, 777, 99),
            greedy=True,
            save_gif=True,
            gif_filename=str(OUTPUTS['fo_dqn'] / "fo_agent_gameplay.gif")
        )
        fo_report = run_evaluation(weights_path=weights_file, cfg=eval_config)
        metrics = fo_report["overall"]
        print(f"Episodes: {metrics['episodes']}")
        print(f"Avg Reward: {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f}")
        print(f"95% CI: [{metrics['reward_mean'] - metrics['reward_ci95']:.2f}, {metrics['reward_mean'] + metrics['reward_ci95']:.2f}]")
        print(f"Fruit Rate: {metrics['fruit_rate_mean']:.4f}")
        print(f"Crash Rate: {metrics['safety_event_rate_mean']:.4f}")
        print(f"Best Episode Reward: {metrics['best_reward']:.2f}")
    else:
        print(f"Weights file not found: {weights_file}")

    # PO DQN Evaluation
    print("\n" + "="*50)
    print("PO DQN EVALUATION")
    print("="*50)
    weights_file_po = str(OUTPUTS['po_dqn'] / "po_dqn_final.weights.h5")
    if os.path.exists(weights_file_po):
        eval_cfg = EvalConfigPO(weights_path=weights_file_po)
        run_po_evaluation(eval_cfg)
        print("\n✓ PO Evaluation complete.")
    else:
        print(f"Weights file not found: {weights_file_po}")

    # FO vs PO Comparison
    print("\n" + "="*50)
    print("FO vs PO COMPARISON")
    print("="*50)
    plot_fo_po_comparison(OUTPUTS['comparison'])

    # PPO Evaluation
    print("\n" + "="*50)
    print("PPO EVALUATION")
    print("="*50)
    evaluate_ppo()