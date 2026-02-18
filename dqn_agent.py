"""Robust DQN for the Snake environment with optimized vectorization and terminal logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class DQNConfig:
    """Hyper-parameters for a stable DQN training loop."""
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 256
    buffer_capacity: int = 200_000
    warmup_steps: int = 5_000
    train_frequency: int = 4
    target_update_frequency: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    gradient_clip_norm: float = 10.0
    use_double_dqn: bool = True


class ReplayBuffer:
    """Vectorized replay buffer for efficient batched environment transitions."""

    def __init__(self, capacity: int, state_shape: Tuple[int, ...]) -> None:
        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape)

        self._states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Optimized batch insertion using NumPy slicing."""
        n_items = states.shape[0]
        
        # Calculate indices for circular buffer insertion
        indices = (np.arange(self._ptr, self._ptr + n_items)) % self.capacity
        
        self._states[indices] = states
        self._actions[indices] = actions
        self._rewards[indices] = rewards
        self._next_states[indices] = next_states
        self._dones[indices] = dones

        self._ptr = (self._ptr + n_items) % self.capacity
        self._size = min(self._size + n_items, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, tf.Tensor]:
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "states": tf.convert_to_tensor(self._states[indices], dtype=tf.float32),
            "actions": tf.convert_to_tensor(self._actions[indices], dtype=tf.int32),
            "rewards": tf.convert_to_tensor(self._rewards[indices], dtype=tf.float32),
            "next_states": tf.convert_to_tensor(self._next_states[indices], dtype=tf.float32),
            "dones": tf.convert_to_tensor(self._dones[indices], dtype=tf.float32),
        }


def build_q_network(state_shape: Tuple[int, ...], n_actions: int) -> tf.keras.Model:
    """Refined CNN Q-network for $7 \times 7$ grid observations."""
    
    inputs = tf.keras.Input(shape=state_shape)
    # Reduced layers to prevent overfitting on small board [cite: 5]
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_actions, activation=None)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="snake_dqn")


class DQNAgent:
    """Production-style DQN agent with target net and improved terminal logic."""

    def __init__(self, state_shape: Tuple[int, ...], n_actions: int, config: DQNConfig | None = None) -> None:
        self.config = config if config is not None else DQNConfig()
        self.n_actions = int(n_actions)

        self.online_network = build_q_network(state_shape=state_shape, n_actions=self.n_actions)
        self.target_network = build_q_network(state_shape=state_shape, n_actions=self.n_actions)
        self.target_network.set_weights(self.online_network.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            state_shape=state_shape,
        )

        self.step_count = 0

    def epsilon(self) -> float:
        ratio = min(1.0, self.step_count / max(1, self.config.epsilon_decay_steps))
        return float(self.config.epsilon_start + ratio * (self.config.epsilon_end - self.config.epsilon_start))

    def select_actions(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        q_values = self.online_network(states, training=False).numpy()
        greedy_actions = np.argmax(q_values, axis=1)

        if greedy:
            return greedy_actions.astype(np.int32)

        eps = self.epsilon()
        random_actions = np.random.randint(0, self.n_actions, size=states.shape[0])
        explore_mask = np.random.random(size=states.shape[0]) < eps
        actions = np.where(explore_mask, random_actions, greedy_actions)
        return actions.astype(np.int32)

    def observe_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.replay_buffer.add_batch(
            states=states,
            actions=actions.reshape(-1),
            rewards=rewards.reshape(-1),
            next_states=next_states,
            dones=dones.reshape(-1),
        )

    @tf.function
    def _train_step_graph(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
    ) -> tf.Tensor:
        gamma = tf.constant(self.config.gamma, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_values = self.online_network(states, training=True)
            action_mask = tf.one_hot(actions, depth=self.n_actions, dtype=tf.float32)
            q_selected = tf.reduce_sum(q_values * action_mask, axis=1)

            target_q_next = self.target_network(next_states, training=False)

            if self.config.use_double_dqn:
                online_next = self.online_network(next_states, training=False)
                next_actions = tf.argmax(online_next, axis=1, output_type=tf.int32)
                next_mask = tf.one_hot(next_actions, depth=self.n_actions, dtype=tf.float32)
                q_next = tf.reduce_sum(target_q_next * next_mask, axis=1)
            else:
                q_next = tf.reduce_max(target_q_next, axis=1)

            td_target = rewards + (1.0 - dones) * gamma * q_next
            per_item_loss = self.loss_fn(td_target, q_selected)
            loss = tf.reduce_mean(per_item_loss)

        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.gradient_clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.online_network.trainable_variables))
        return loss

    def maybe_train(self) -> float | None:
        if len(self.replay_buffer) < self.config.warmup_steps:
            return None
        if self.step_count % self.config.train_frequency != 0:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        loss = self._train_step_graph(**batch)

        if self.step_count % self.config.target_update_frequency == 0:
            self.target_network.set_weights(self.online_network.get_weights())

        return float(loss.numpy())

    def train_on_env_step(self, env) -> Dict[str, float | None]:
        """Collect transitions and define 'done' based on terminal rewards."""
        states = np.asarray(env.to_state(), dtype=np.float32)
        actions = self.select_actions(states, greedy=False)

        rewards_tensor = env.move(actions.reshape(-1,1))
        rewards = np.asarray(rewards_tensor, dtype=np.float32).reshape(-1)
        next_states = np.asarray(env.to_state(), dtype=np.float32)

        # Logica 'done' migliorata: collisioni o vittoria resettano lo stato logico 
        terminals = (env.HIT_WALL_REWARD, env.ATE_HIMSELF_REWARD, env.WIN_REWARD) 
        dones = np.isin(rewards, [float(t) for t in terminals]).astype(np.float32)

        self.observe_batch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

        self.step_count += 1
        loss = self.maybe_train()

        return {
            "loss": loss,
            "mean_reward": float(np.mean(rewards)),
            "epsilon": self.epsilon(),
            "buffer_size": float(len(self.replay_buffer)),
        }