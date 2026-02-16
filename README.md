# Snake Reinforcement Learning Project


> **Student**: Salvatore Ferracane (m. 2154255)
> 
> **Course**: Reinforcement Learning
> 
> **Master Degree**: Control Systems Engineering
>
>  University of Padua, Italy
> 
> **a.y.** 2025/2026


---

## Overview

This project implements and evaluates multiple Reinforcement Learning (RL) agents to solve the classic **Snake** game under different observability conditions. The objective is to maximize cumulative reward by collecting fruits and surviving as long as possible while avoiding wall and self-collisions.

The project was developed as part of a Reinforcement Learning course and focuses on the **practical implementation, evaluation, and comparison of value-based and policy-based RL methods**, highlighting the impact of observability and architecture on performance.

🛠️ **Installation & Dependencies**

To run this project, ensure you have **Python 3.8+** installed. The project relies on deep learning frameworks and visualization tools.

You can install all the required packages at once by running the following command in your terminal:

```bash
pip install numpy tensorflow matplotlib seaborn tqdm ipykernel
```

📂 **Project Structure**
```
.
├── main.ipynb                        # Main entry point
├── dqn_agent.py                      # Standard DQN Agent
├── po_dqn_agent.py                   # Partially Observable DQN Agent
├── ppo_agent.py                      # PPO Agent with LSTM
├── environments_fully_observable.py  # FO Snake Environment
├── environments_partially_observable.py # PO Snake Environment
├── evaluate_dqn.py                   # Evaluation Logic (FO)
├── evaluate_po.py                    # Evaluation Logic (PO)
└── evaluate_ppo.py                   # Evaluation Logic (PPO)
```

### 🚀 Usage & Execution

The entire project workflow—including training loops, evaluation metrics, and graph generation—is centralized within the **`main.ipynb`** notebook.

To reproduce the experiments:
1. Open `main.ipynb` in Jupyter Notebook, Jupyter Lab, or VS Code.
2. Install the dependencies listed above.
3. **Run the cells sequentially** to load the environments and agents.
4. Navigate to the specific section (Fully Observable DQN, Partially Observable DDDQN, or PPO) to trigger training or evaluation.

 ⚠️ **Performance Note: PPO Agent**
 
Please be aware that training the **PPO Agent with LSTM memory** is computationally intensive due to the complexity of Backpropagation Through Time (BPTT) and the on-policy nature of the algorithm.

> **Hardware Benchmark:** On a standard laptop equipped with a CPU supporting **AVX2 FMA** instructions, the complete PPO training process required approximately **3 days** to reach optimal convergence. Using a discrete GPU (CUDA-enabled) is highly recommended to significantly reduce this time.

---

## Environment and Reward Design

The environment is a discrete **7×7 grid**. The agent (snake) moves deterministically in four directions.

A **shaped reward function** guides learning:

| Event | Reward |
|------|--------|
| Win episode | +1.0 |
| Collect fruit | +0.5 |
| Step | 0.0 |
| Wall collision | −0.1 |
| Self collision | −0.2 |

This asymmetric design penalizes unsafe behaviors while avoiding incentives for passive survival.

---

## Project Structure and Methodology

The analysis follows a progressive approach:

1. Heuristic baseline (deterministic reference)
2. Fully Observable RL agent (Double DQN)
3. Partially Observable RL agent (Dueling Double DQN)
4. Bonus: PPO agent (policy gradient)

Each stage increases complexity and highlights specific RL challenges.

---

# Heuristic Baseline

## Objective

Provide a **performance lower bound and benchmark** for RL agents.

### Optimal Pathfinding and Safety Fallback

The heuristic agent relies on a hierarchical decision strategy designed to combine optimal fruit acquisition with long-term survivability.

First, the environment grid is modeled as an unweighted graph, where each cell represents a node and valid movements define the edges. Within this representation, the agent applies the Breadth-First Search (BFS) algorithm to compute the shortest collision-free path from the snake’s head to the fruit. This guarantees optimal pathfinding whenever a safe path exists.

However, in many configurations the fruit may be temporarily unreachable due to obstruction by the snake’s own body. In these cases, a safety-oriented fallback mechanism is activated. The agent evaluates all admissible actions and uses a Flood Fill algorithm to estimate the size of the reachable free space associated with each action. The action leading to the largest free region is selected, as it maximizes the available maneuvering space and reduces the probability of future entrapment.

This hierarchical combination allows the agent to pursue optimal paths when feasible, while reverting to conservative, space-preserving behavior in constrained scenarios, effectively balancing immediate reward maximization with long-term survival.


---

## Baseline Performance

| Metric | Value |
|------|--------|
| Avg Reward | 47.125 |
| Std Dev | 11.566 |

Equivalent to approximately:

**94 fruits per episode**

This defines a strong non-learning benchmark.

---

# Fully Observable Reinforcement Learning Agent

## State Representation

In the Fully Observable setting, the agent has access to the entire grid at every timestep. This means the observation contains complete information about the environment, including the snake’s body, the fruit position, and the boundaries. As a result, the observation fully represents the true environment state, satisfying the Markov property. This property is critical for value-based reinforcement learning methods, as it ensures that optimal decisions can be made based solely on the current observation without requiring additional memory of past states.

---

## Algorithm: Double Deep Q-Network (DDQN)

The learning agent is based on the Double Deep Q-Network (DDQN) algorithm, which was selected to address a known limitation of the standard Deep Q-Network (DQN): overestimation bias. In standard DQN, the same network is used both to select and evaluate actions, which can lead to systematically inflated Q-value estimates and unstable learning.

DDQN resolves this issue by decoupling these two roles. The online network is used to select the best action, while a separate target network is used to evaluate that action. This separation reduces overoptimistic value estimates and leads to more stable and reliable training. As a consequence, the agent is able to learn more accurate value functions and develop more effective policies for long-term survival and reward maximization.

---

## Architecture

Convolutional Neural Network:

- Feature extraction from spatial grid
- Fully connected layers
- Q-value output per action

---

## Training Configuration

Key parameters:

| Parameter | Value |
|---|---|
| Discount factor | 0.99 |
| Learning rate | 1e-4 |
| Batch size | 256 |
| Replay buffer | 200,000 |
| Training horizon | 10,000 |

---

## Results

| Agent | Avg Reward per Step |
|---|---|
| Heuristic | 0.047 |
| Standard DQN | 0.1208 |
| Double DQN | 0.1226 |

Improvement over heuristic: ~160%

**Learning Progress of Doubl-DQN Agent**

![Miglior Agente PO](./results_fo/final_agent_gameplay_FO.gif)


---

## Interpretation

The agent learns:

- Long-term survival strategies
- Efficient spatial planning
- Near-optimal policies

The environment is fully solvable.

---

# Partially Observable Reinforcement Learning Agent

## Motivation

In the Partially Observable setting, the agent no longer has access to the full grid. Instead, it receives only a local observation window centered on the snake’s head. This limited field of view removes access to global information, such as the full body topology and distant obstacles.

Because the observation no longer fully represents the true environment state, the Markov property is violated. The same observation may correspond to different underlying states, depending on information outside the visible region. This transforms the problem into a **Partially Observable Markov Decision Process (POMDP)**, where optimal decision-making requires inferring hidden information from observation history.

---

## Algorithm: Dueling Double DQN

To address the increased uncertainty introduced by partial observability, the DDQN architecture was extended with temporal processing and value decomposition mechanisms.

### Frame Stacking

To compensate for the lack of full state information, the agent uses **frame stacking**, where the last 4 observations are concatenated into a single input tensor.

This provides a finite temporal memory, allowing the network to infer short-term dynamics such as:

- Snake velocity  
- Movement direction  
- Relative motion of nearby body segments  

This temporal context improves state representation compared to a single static frame.

---

### Dueling Architecture

In addition to frame stacking, a **Dueling Network Architecture** is used to improve value estimation.

The network is split into two separate streams:

- The **state value function** `V(s)`, which estimates how favorable a state is independently of the action  
- The **advantage function** `A(s,a)`, which estimates the relative benefit of each possible action  

These two components are combined to produce the final Q-values.

This decomposition improves learning efficiency and stability, particularly in partially observable environments where many actions may produce similar immediate outcomes. It allows the agent to identify dangerous or favorable states even when the best action is uncertain.


**Architecture Summary**

CNN → Dense → Dueling streams → Q-values

---

## Training Configuration

Similar to fully observable case but adapted:

| Parameter | Value |
|---|---|
| Discount factor | 0.99 |
| Learning rate | 1e-4 |
| Batch size | 256 |
| Replay buffer | 200,000 |
| Training horizon | 12,000 |
| Frame stack | 4 |

---

## Partially Observable Results

Evaluation over 5 seeds:

| Metric | Value |
|---|---|
| Avg Reward | 2.98 |
| Avg Steps | 24.6 |
| Wall collision | 0% |
| Self collision | 100% |

**Best Agent**

![Miglior Agente PO](./eval_results_po/best_po_agent.gif)


---


### Dominant Failure Mechanism: "Ghost Tail" Effect

The primary failure mechanism observed in the partially observable agent arises from hidden state information outside the observation window.

As the snake grows in length, parts of its body inevitably move beyond the visible region. Because the agent only receives a local observation centered on the head, it loses access to the full body configuration and can no longer track the position of its tail.

This creates a phenomenon known as **perceptual aliasing**, where identical observations correspond to different underlying environment states. From the agent’s perspective, a region may appear safe because the tail is not visible, while in reality that same region is occupied.

As a consequence, the agent selects actions that are locally valid but globally unsafe. This leads to unavoidable self-collisions, a failure mode referred to in this project as the **"Ghost Tail" effect**.


---

# Fully Observable vs Partially Observable Comparison

| Metric | Fully Observable | Partially Observable |
|---|---|---|
| Reward | 132.13 | 2.98 |
| Steps | ~280 | 24.6 |
| Wall collisions | 0% | 0% |
| Self collisions | <5% | 100% |

Performance drop: 97.7%

---


# Architectural Comparison and Limitation Analysis

**Fully Observable Agent**

The fully observable agent demonstrates true global planning capability. With access to the complete grid, it learns spatially optimal policies that preserve free space, avoid self-entrapment, and maximize long-term survival. This confirms that the DDQN architecture is sufficient when the Markov property is satisfied.

**Partially Observable Agent**

In contrast, the partially observable agent exhibits purely reactive behavior. It receives only a local observation, it cannot reconstruct the global topology of the snake’s body. As a result, it optimizes immediate safety and reward but fails to maintain long-term survivability.

This reveals a fundamental architectural limitation that I have tried to solve with ```Frame stacking```. However, this **finite temporal memory** is sufficient to recover short-term motion information such as velocity and direction, but insufficient to reconstruct the full spatial configuration of the environment. In fact, due to the poor performance and low efficiency of the Dueling DDQN in the partially observable environment, a different approach was also explored. Specifically, a **policy-gradient agent based on Proximal Policy Optimization (PPO)** was developed to investigate whether an alternative learning paradigm could achieve more robust behavior under **partial observability**.


---

# PPO Agent (Bonus)


**Motivation**

The PPO agent was introduced to address the key limitations observed in value-based methods, particularly under partial observability. The Dueling DDQN showed low performance and instability, mainly due to its limited memory capacity and difficulty in learning robust policies from incomplete state information. To overcome these limitations, a different reinforcement learning paradigm was adopted. Specifically, a **policy-gradient method**, Proximal Policy Optimization (PPO), was implemented.

Unlike value-based methods, PPO directly learns the policy function, optimizing the probability of selecting actions that maximize the expected cumulative reward. Moreover, Its clipped objective function improves training stability and prevents destructive policy updates.

**Goal**

The objective of this agent was to achieve more stable and effective behavior in the partially observable environment, where value-based architectures struggled.

**Evaluation**

The PPO agent was systematically evaluated and its performance was quantitatively compared against the DQN-based agents (DDQN and Dueling DDQN) to assess the impact of the different learning paradigm.


---

### Key Conclusions

Main findings:

1. Double DQN successfully solves the fully observable Snake environment.

2. Partial observability introduces structural performance limits.

3. The primary failure mechanism is hidden state information.

4. Feed-forward architectures cannot solve topology-dependent POMDPs.

5. Observability is a critical factor in reinforcement learning performance.


---







