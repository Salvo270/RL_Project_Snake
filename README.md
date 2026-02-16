# Snake Reinforcement Learning Project

---

**Student**: Salvatore Ferracane

**Course**: Reinforcement Learning

**a.y**: 2025/2026

---

## Overview

This project implements and evaluates multiple Reinforcement Learning (RL) agents to solve the classic **Snake** game under different observability conditions. The objective is to maximize cumulative reward by collecting fruits and surviving as long as possible while avoiding wall and self-collisions.

The project was developed as part of a Reinforcement Learning course and focuses on the **practical implementation, evaluation, and comparison of value-based and policy-based RL methods**, highlighting the impact of observability and architecture on performance.

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

## Architecture

Hierarchical decision logic:

### 1. Optimal Pathfinding

The grid is modeled as a graph.

Breadth-First Search (BFS) computes the shortest safe path to the fruit.

### 2. Safety Fallback

If the fruit is unreachable:

Flood Fill estimates free space for each possible action.

The agent selects the action maximizing future survivability.

This balances:

- Optimality (fruit collection)
- Safety (space preservation)

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

Full grid visibility:


This satisfies the Markov property.

---

## Algorithm: Double Deep Q-Network (DDQN)

Motivation:

Standard DQN suffers from **overestimation bias**.

DDQN separates:

Action selection and action evaluation.

This stabilizes learning.

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

Improvement over heuristic:

~160%

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

Real-world systems often lack full observability.

The agent receives only:


This converts the environment into a:

**Partially Observable Markov Decision Process (POMDP)**

---

## Algorithm: Dueling Double DQN

Enhancements:

### Frame Stacking

Stack last 4 frames

Provides short-term temporal memory.

### Dueling Architecture

Separates:

State value:

V(s)

and

Action advantage:

A(s,a)

Improves value estimation.

---

## Architecture Summary

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

# Partially Observable Results

Evaluation over 5 seeds:

| Metric | Value |
|---|---|
| Avg Reward | 2.98 |
| Avg Steps | 24.6 |
| Wall collision | 0% |
| Self collision | 100% |

---

# Failure Analysis: Ghost Tail Effect

Dominant failure mechanism.

Cause:

Hidden state information.

As the snake grows:

Parts of its body exit the observation window.

The agent cannot detect its tail.

This produces:

Perceptual aliasing

Identical observations correspond to different true states.

Result:

Self-collision becomes inevitable.

---

# Fully Observable vs Partially Observable Comparison

| Metric | Fully Observable | Partially Observable |
|---|---|---|
| Reward | 132.13 | 2.98 |
| Steps | ~280 | 24.6 |
| Wall collisions | 0% | 0% |
| Self collisions | <5% | 100% |

Performance drop:

97.7%

---

# Engineering Interpretation

Fully Observable Agent:

Global planning capability

Learns optimal spatial policies

Partially Observable Agent:

Reactive behavior only

Cannot reconstruct global topology

Fundamental architectural limitation.

Frame stacking provides:

Finite memory

Insufficient for topology reconstruction.

Recurrent networks would be required.

Example:

LSTM / GRU

---

# Bonus: PPO Agent

Motivation:

Address memory and stability limitations of value-based methods.

Approach:

Policy Gradient method:

Proximal Policy Optimization

Goal:

Improve performance under partial observability.

Comparison performed with DQN-based agents.

---

# Key Conclusions

Main findings:

1. Double DQN successfully solves the fully observable Snake environment.

2. Partial observability introduces structural performance limits.

3. The primary failure mechanism is hidden state information.

4. Feed-forward architectures cannot solve topology-dependent POMDPs.

5. Observability is a critical factor in reinforcement learning performance.

---

# Project Contributions

Implemented:

Heuristic path-planning agent

Double DQN agent

Dueling Double DQN agent

PPO agent

Evaluated:

Fully observable vs partially observable environments

Analyzed:

Architectural limitations

Failure modes

Impact of observability

---


---

# Summary

This project demonstrates, in a controlled engineering setting, how:

Reinforcement Learning performance depends critically on:

Architecture

Observability

Memory

and state representation.

It highlights the transition from:

Reactive policies

to

Strategic planning systems.






