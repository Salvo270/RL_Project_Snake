"""
Partially Observable Snake Agent: Optimized Frame Stacking + Dueling Double DQN.
Corrected for terminal state logic and memory management.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf
from dqn_agent import ReplayBuffer

@dataclass
class PODQNConfig:
    """Hyper-parameters for POMDP stability."""
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 256
    buffer_capacity: int = 200_000
    warmup_steps: int = 5_000
    train_frequency: int = 4
    target_update_frequency: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 12_000 # Extended for uncertainty
    gradient_clip_norm: float = 10.0
    use_double_dqn: bool = True
    frame_stack: int = 4

class FrameStackState:
    """
    High-performance Frame Stacker using pre-allocated NumPy buffers.
    Avoids expensive concatenations at each step.
    """
    def __init__(self, n_boards: int, h: int, w: int, c: int, k: int) -> None:
        self.n_boards = n_boards
        self.k = k
        self.c = c
        # Pre-allocate buffer: (N, H, W, C*K)
        self.buffer = np.zeros((n_boards, h, w, c * k), dtype=np.float32)

    def reset(self, raw_state: np.ndarray) -> np.ndarray:
        """Initialize the buffer by tiling the first observation."""
        # raw_state shape: (N, H, W, C)
        for i in range(self.k):
            self.buffer[..., i*self.c : (i+1)*self.c] = raw_state
        return self.buffer.copy()

    def update(self, raw_state: np.ndarray) -> np.ndarray:
        """Shift the buffer and insert the newest frame."""
        # Shift old frames to the left: [f1, f2, f3, f4] -> [f2, f3, f4, f4]
        self.buffer[..., :-self.c] = self.buffer[..., self.c:]
        # Insert new frame at the end
        self.buffer[..., -self.c:] = raw_state
        return self.buffer.copy()



def build_dueling_q_network(state_shape: Tuple[int, ...], n_actions: int) -> tf.keras.Model:
    """
    Refined Dueling Architecture for small receptive fields (5x5).
    """
    inputs = tf.keras.Input(shape=state_shape)
    
    # Feature Extractor: Reduced depth to avoid signal loss on 5x5 grids
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    # --- Dueling Streams ---
    # Value Stream: V(s)
    v_hidden = tf.keras.layers.Dense(128, activation="relu")(x)
    value = tf.keras.layers.Dense(1, activation=None)(v_hidden)

    # Advantage Stream: A(s, a)
    a_hidden = tf.keras.layers.Dense(128, activation="relu")(x)
    advantage = tf.keras.layers.Dense(n_actions, activation=None)(a_hidden)

    # Centering Advantage to ensure identifiability
    # Q(s,a) = V(s) + (A(s,a) - Mean(A))
    mean_advantage = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
    advantage_centered = tf.keras.layers.Subtract()([advantage, mean_advantage])
    
    q_values = tf.keras.layers.Add()([value, advantage_centered])

    return tf.keras.Model(inputs=inputs, outputs=q_values, name="Dueling_DQN_Snake")



class PartiallyObservableDQNAgent:
    def __init__(self, raw_state_shape: Tuple[int, ...], n_actions: int, n_boards: int, config: PODQNConfig | None = None) -> None:
        self.config = config if config is not None else PODQNConfig()
        self.n_actions = n_actions
        self.n_boards = n_boards
        
        h, w, c = raw_state_shape
        self.stacked_shape = (h, w, c * self.config.frame_stack)

        self.online_network = build_dueling_q_network(self.stacked_shape, n_actions)
        self.target_network = build_dueling_q_network(self.stacked_shape, n_actions)
        self.target_network.set_weights(self.online_network.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.loss_fn = tf.keras.losses.Huber()
        
        self.replay_buffer = ReplayBuffer(capacity=self.config.buffer_capacity, state_shape=self.stacked_shape)
        self.stacker = FrameStackState(n_boards, h, w, c, self.config.frame_stack)
        
        self.step_count = 0

    def epsilon(self) -> float:
        ratio = min(1.0, self.step_count / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + ratio * (self.config.epsilon_end - self.config.epsilon_start)

    def select_actions(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        q_values = self.online_network(states, training=False).numpy()
        if greedy or np.random.rand() > self.epsilon():
            return np.argmax(q_values, axis=1).astype(np.int32)
        return np.random.randint(0, self.n_actions, size=self.n_boards).astype(np.int32)

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        gamma = tf.constant(self.config.gamma, dtype=tf.float32)
        with tf.GradientTape() as tape:
            # Online Q-values
            q_values = self.online_network(states, training=True)
            q_selected = tf.reduce_sum(q_values * tf.one_hot(actions, self.n_actions), axis=1)

            # Double DQN Logic
            target_q_next = self.target_network(next_states, training=False)
            online_q_next = self.online_network(next_states, training=False)
            best_actions = tf.argmax(online_q_next, axis=1, output_type=tf.int32)
            q_next = tf.reduce_sum(target_q_next * tf.one_hot(best_actions, self.n_actions), axis=1)

            # Bellman Equation with proper terminal handling
            y = rewards + (1.0 - dones) * gamma * q_next
            loss = self.loss_fn(y, q_selected)

        grads = tape.gradient(loss, self.online_network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.online_network.trainable_variables))
        return loss

    def train_on_env_step(self, env) -> Dict:
        # 1. State Acquisition
        raw_state = np.asarray(env.to_state(), dtype=np.float32)
        state = self.stacker.reset(raw_state) if self.step_count == 0 else self.stacker.buffer.copy()

        # 2. Interaction
        actions = self.select_actions(state)
        # Reshape actions to (N, 1) for the environment
        rewards_tensor = env.move(actions.reshape(-1, 1))
        rewards = np.asarray(rewards_tensor).flatten()

        # 3. Next State Acquisition
        next_raw_state = np.asarray(env.to_state(), dtype=np.float32)
        next_state = self.stacker.update(next_raw_state)

        # 4. CRITICAL: Correct Terminal Logic (Any death = Done)
        terminals = [env.HIT_WALL_REWARD, env.ATE_HIMSELF_REWARD, env.WIN_REWARD]
        dones = np.isin(rewards, [float(t) for t in terminals]).astype(np.float32)

        # 5. Experience Replay
        self.replay_buffer.add_batch(state, actions, rewards, next_state, dones)

        # 6. Optimization
        loss = None
        if self.step_count > self.config.warmup_steps and self.step_count % self.config.train_frequency == 0:
            batch = self.replay_buffer.sample(self.config.batch_size)
            loss = self._train_step(**batch)

        if self.step_count % self.config.target_update_frequency == 0:
            self.target_network.set_weights(self.online_network.get_weights())

        self.step_count += 1
        return {"loss": float(loss.numpy()) if loss else None, "mean_reward": np.mean(rewards)}