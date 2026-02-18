import numpy as np
import tensorflow as tf
import environments_fully_observable 
from baseline import BaselineAgent

def run_evaluation(n_boards=100, n_steps=1000):
    """
    Evaluates the BaselineAgent and prints performance metrics.
    """
    # Initialize the environment with a fixed seed for reproducibility 
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Board size 7 as specified in the course materials
    board_size = 7 
    env = environments_fully_observable.OriginalSnakeEnvironment(n_boards, board_size)
    
    # Initialize the heuristic agent
    agent = BaselineAgent()
    cumulative_rewards = np.zeros(n_boards)
    
    print(f"Starting Baseline Evaluation: {n_boards} boards, {n_steps} steps.")

    for _ in range(n_steps):
        # Select actions for all boards using the heuristic logic 
        actions = agent.select_actions(env)
        
        # Apply actions to the environment 
        rewards = env.move(np.array(actions).reshape(-1, 1))
        
        # Accumulate rewards for final reporting 
        cumulative_rewards += np.array(rewards).flatten()

    # Calculate final metrics
    avg_reward = np.mean(cumulative_rewards)
    std_reward = np.std(cumulative_rewards)

    print("\n--- Evaluation Results ---")
    print(f"Average Total Reward: {avg_reward:.4f}")
    print(f"Standard Deviation:  {std_reward:.4f}")

if __name__ == "__main__":
    run_evaluation()