"""
Partially Observable Snake Agent - Evaluation & Analysis Suite.
"""

import argparse
import csv
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence

# Project Imports (Ensure these are in your path)
import environments_partially_observable
from po_dqn_agent import PODQNConfig, PartiallyObservableDQNAgent

# --- Configuration ---
@dataclass
class EvalConfig:
    weights_path: str
    board_size: int = 7
    mask_size: int = 2          
    max_steps: int = 500        
    n_episodes_per_seed: int = 20
    seeds: Sequence[int] = (42, 100, 2024, 777, 99)
    save_gif: bool = True
    output_dir: str = "eval_results_po"

# --- Helper Functions ---
def get_env(n: int, board_size: int, mask_size: int):
    return environments_partially_observable.OriginalSnakeEnvironment(n, board_size, mask_size)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

def save_gif(frames: List[np.ndarray], filename: str):
    if not frames: return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    img = ax.imshow(frames[0], origin="lower", cmap="viridis", interpolation='nearest')
    
    def update(frame):
        img.set_data(frame)
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=150, blit=True)
    ani.save(filename, writer="pillow", fps=6)
    plt.close(fig)

# --- Core Evaluation Logic ---
def evaluate_episode(agent: PartiallyObservableDQNAgent, env, max_steps: int) -> Dict:
    # 1. Initialize State & Memory
    raw_state = np.asarray(env.to_state(), dtype=np.float32)
    agent.stacker.reset(raw_state)
    
    total_reward = 0.0
    steps = 0
    frames = []
    rewards_per_step = [] # Track reward at each step for the new plot
    termination_reason = "max_steps" 

    for t in range(max_steps):
        frames.append(env.boards[0].copy())
        
        # 2. Get Current Stacked State
        current_stacked_state = agent.stacker.buffer 
        
        # 3. Select Action (Greedy)
        action = agent.select_actions(current_stacked_state, greedy=True)
        
        # 4. Environment Step
        reward_tensor = env.move(action.reshape(-1, 1))
        reward = float(np.asarray(reward_tensor).flatten()[0])
        
        # 5. Update Stacker
        next_raw_state = np.asarray(env.to_state(), dtype=np.float32)
        agent.stacker.update(next_raw_state)
        
        total_reward += reward
        rewards_per_step.append(reward)
        steps += 1
        
        # 6. Check Termination
        if np.isclose(reward, env.HIT_WALL_REWARD):
            termination_reason = "wall_collision"
            break
        elif np.isclose(reward, env.ATE_HIMSELF_REWARD):
            termination_reason = "self_collision"
            break
        elif np.isclose(reward, env.WIN_REWARD):
            termination_reason = "victory"
            break
            
    return {
        "total_reward": total_reward,
        "steps": steps,
        "termination_reason": termination_reason,
        "frames": frames,
        "rewards_per_step": rewards_per_step
    }

def run_analysis(cfg: EvalConfig):
    Path(cfg.output_dir).mkdir(exist_ok=True)
    
    print(f"Loading agent weights from: {cfg.weights_path}")
    env_dummy = get_env(1, cfg.board_size, cfg.mask_size)
    raw_shape = tuple(np.asarray(env_dummy.to_state()).shape[1:])
    
    agent = PartiallyObservableDQNAgent(
        raw_state_shape=raw_shape,
        n_actions=4,
        n_boards=1,
        config=PODQNConfig(epsilon_start=0.0, epsilon_end=0.0) 
    )
    
    # Initialize model
    dummy_input = np.zeros((1, *agent.stacked_shape), dtype=np.float32)
    agent.online_network(dummy_input)
    
    try:
        agent.online_network.load_weights(cfg.weights_path)
    except Exception as e:
        print(f"CRITICAL ERROR loading weights: {e}")
        return

    all_results = []
    all_rewards_per_step = [] # Store lists of rewards for plotting
    
    print(f"\nStarting Evaluation over {len(cfg.seeds)} seeds...")
    print("-" * 65)
    print(f"{'Seed':<6} | {'Avg Reward':<12} | {'Avg Steps':<10} | {'Safety (Wall/Self)%':<20}")
    print("-" * 65)

    best_gif_frames = []
    best_gif_reward = -np.inf

    for seed in cfg.seeds:
        set_global_seed(seed)
        seed_rewards = []
        seed_steps = []
        term_counts = {"wall_collision": 0, "self_collision": 0, "max_steps": 0, "victory": 0}
        
        for ep_idx in range(cfg.n_episodes_per_seed):
            env = get_env(1, cfg.board_size, cfg.mask_size)
            metrics = evaluate_episode(agent, env, cfg.max_steps)
            
            seed_rewards.append(metrics["total_reward"])
            seed_steps.append(metrics["steps"])
            term_counts[metrics["termination_reason"]] += 1
            all_rewards_per_step.append(metrics["rewards_per_step"])
            
            all_results.append({
                "seed": seed,
                "episode": ep_idx,
                "total_reward": metrics["total_reward"],
                "steps": metrics["steps"],
                "termination_reason": metrics["termination_reason"]
            })
            
            if metrics["total_reward"] > best_gif_reward:
                best_gif_reward = metrics["total_reward"]
                best_gif_frames = metrics["frames"]

        # Seed Summary
        avg_r = np.mean(seed_rewards)
        avg_s = np.mean(seed_steps)
        n = cfg.n_episodes_per_seed
        wall_pct = (term_counts['wall_collision'] / n) * 100
        self_pct = (term_counts['self_collision'] / n) * 100
        
        print(f"{seed:<6} | {avg_r:<12.2f} | {avg_s:<10.1f} | Wall: {wall_pct:.0f}% / Self: {self_pct:.0f}%")

    # --- SAVE CSV ---
    csv_path = f"{cfg.output_dir}/evaluation_PO_episodes.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "episode", "total_reward", "steps", "termination_reason"])
        for r in all_results:
            writer.writerow([r["seed"], r["episode"], r["total_reward"], r["steps"], r["termination_reason"]])
    
    print(f"\n[Artifact] Raw CSV data saved to: {csv_path}")

    # Generate Plots (Boxplot + Mean Moving Average Reward)
    generate_plots(all_results, all_rewards_per_step, cfg.output_dir)
    
    if cfg.save_gif and best_gif_frames:
        gif_path = f"{cfg.output_dir}/best_po_agent.gif"
        save_gif(best_gif_frames, gif_path)
        print(f"[Artifact] Best gameplay GIF saved to: {gif_path}")

def generate_plots(results: List[Dict], rewards_history: List[List[float]], output_dir: str):
    sns.set_theme(style="whitegrid")
    
    # 1. Boxplot (Agent Robustness)
    seeds = [r['seed'] for r in results]
    total_rewards = [r['total_reward'] for r in results]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=seeds, y=total_rewards, palette="Blues")
    plt.title("Agent Robustness across Seeds (POMDP)", fontsize=14)
    plt.xlabel("Random Seed", fontsize=12)
    plt.ylabel("Total Reward Distribution", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_boxplot.png", dpi=300)
    plt.close()
    
    # 2. Mean Moving Average Reward vs Steps
    # We need to align all episodes to the max length or cut to min length.
    # Here we pad with NaN to calculate mean ignoring short episodes ending.
    max_len = max(len(r) for r in rewards_history)
    padded_rewards = np.full((len(rewards_history), max_len), np.nan)
    
    for i, r_list in enumerate(rewards_history):
        padded_rewards[i, :len(r_list)] = r_list
        
    # Calculate Mean and Std ignoring NaNs
    mean_curve = np.nanmean(padded_rewards, axis=0)
    std_curve = np.nanstd(padded_rewards, axis=0)
    
    # Smoothing for cleaner visualization
    window = 5
    if len(mean_curve) > window:
        smooth_mean = np.convolve(mean_curve, np.ones(window)/window, mode='valid')
        # Adjust x-axis for valid convolution
        x_axis = np.arange(len(smooth_mean))
    else:
        smooth_mean = mean_curve
        x_axis = np.arange(len(mean_curve))

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, smooth_mean, color='#2980b9', linewidth=2, label='Mean Reward (Smoothed)')
    plt.fill_between(x_axis, 
                     smooth_mean - std_curve[:len(smooth_mean)]*0.2, # Scaled std for visibility
                     smooth_mean + std_curve[:len(smooth_mean)]*0.2, 
                     color='#2980b9', alpha=0.2, label='Variance')
    
    plt.title("Mean Reward Evolution over Episode Steps", fontsize=14)
    plt.xlabel("Step Number within Episode", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_reward_evolution.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="dqn_snake_po_final.weights.h5")
    args = parser.parse_args()
    
    config = EvalConfig(weights_path=args.weights)
    run_analysis(config)