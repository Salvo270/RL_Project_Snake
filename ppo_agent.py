"""
PPO Agent with LSTM for Partially Observable Snake Environment.
Optimized for POMDPs with temporal memory and efficient exploration.
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    lstm_units: int = 128
    dense_units: int = 256


class FrameStacker:
    """Stacks last N observations for temporal context."""
    
    def __init__(self, raw_state_shape: Tuple[int, ...], n_stack: int = 4):
        self.raw_shape = raw_state_shape
        self.n_stack = n_stack
        self.buffer = None
        
    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        """Initialize with repeated first observation."""
        single = initial_state[0] if len(initial_state.shape) == 4 else initial_state
        self.buffer = np.repeat(single[np.newaxis, ...], self.n_stack, axis=0)
        return self.buffer
    
    def update(self, new_state: np.ndarray) -> np.ndarray:
        """Roll buffer and add new observation."""
        single = new_state[0] if len(new_state.shape) == 4 else new_state
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1] = single
        return self.buffer
    
    @property
    def stacked_shape(self) -> Tuple[int, ...]:
        return (self.n_stack, *self.raw_shape)


def build_actor_critic_lstm(
    state_shape: Tuple[int, ...],
    n_actions: int,
    config: PPOConfig
) -> keras.Model:
    """
    Shared architecture with LSTM for temporal dependencies.
    Returns both policy logits and value estimate.
    
    state_shape: (n_stack, height, width, channels)
    """
    
    # Input: (batch, n_stack, height, width, channels)
    state_input = layers.Input(shape=state_shape, name='state')
    
    # TimeDistributed CNN to process each frame independently
    conv1 = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        name='conv1'
    )(state_input)
    
    conv2 = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        name='conv2'
    )(conv1)
    
    conv3 = layers.TimeDistributed(
        layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
        name='conv3'
    )(conv2)
    
    # Flatten spatial dimensions for each timestep
    flattened = layers.TimeDistributed(layers.Flatten(), name='flatten')(conv3)
    
    # LSTM for temporal integration across stacked frames
    lstm_out = layers.LSTM(
        config.lstm_units,
        return_sequences=False,
        name='lstm_memory'
    )(flattened)
    
    # Shared dense layer
    shared = layers.Dense(config.dense_units, activation='relu', name='shared_dense')(lstm_out)
    shared = layers.LayerNormalization()(shared)
    
    # Policy head (actor)
    policy = layers.Dense(128, activation='relu', name='policy_dense')(shared)
    policy_logits = layers.Dense(n_actions, activation=None, name='policy_logits')(policy)
    
    # Value head (critic)
    value = layers.Dense(128, activation='relu', name='value_dense')(shared)
    value_output = layers.Dense(1, activation=None, name='value')(value)
    
    model = keras.Model(
        inputs=state_input,
        outputs=[policy_logits, value_output],
        name='ActorCritic_LSTM'
    )
    
    return model


class PPOAgent:
    """PPO Agent with LSTM for Partially Observable Environments."""
    
    def __init__(
        self,
        raw_state_shape: Tuple[int, ...],
        n_actions: int,
        n_boards: int = 1,
        n_stack: int = 4,
        config: PPOConfig = None
    ):
        self.raw_state_shape = raw_state_shape
        self.n_actions = n_actions
        self.n_boards = n_boards
        self.config = config or PPOConfig()
        
        # Frame stacking for temporal context
        self.stacker = FrameStacker(raw_state_shape, n_stack)
        self.stacked_shape = self.stacker.stacked_shape
        
        # Build network
        self.model = build_actor_critic_lstm(
            self.stacked_shape,
            n_actions,
            self.config
        )
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.max_grad_norm
        )
        
        # Training metrics
        self.metrics = {
            'policy_loss': keras.metrics.Mean(),
            'value_loss': keras.metrics.Mean(),
            'entropy': keras.metrics.Mean(),
            'total_loss': keras.metrics.Mean(),
            'kl_divergence': keras.metrics.Mean(),
        }
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        Returns: (action, log_prob, value)
        """
        state_tensor = tf.convert_to_tensor(state[np.newaxis, ...], dtype=tf.float32)
        
        logits, value = self.model(state_tensor, training=False)
        
        if training:
            # Sample from categorical distribution
            action_dist = tf.random.categorical(logits, num_samples=1)
            action = int(action_dist[0, 0])
        else:
            # Greedy action
            action = int(tf.argmax(logits[0]))
        
        # Compute log probability
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        action_log_prob = log_probs[0, action]
        
        return action, float(action_log_prob), float(value[0, 0])
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation.
        Returns: (advantages, returns)
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        
        # Bootstrap from next value
        values_extended = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value_t = 0.0
                last_gae = 0.0
            else:
                next_value_t = values_extended[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value_t - values[t]
            last_gae = delta + self.config.gamma * self.config.lambda_gae * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    @tf.function
    def train_step(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        old_log_probs: tf.Tensor,
        advantages: tf.Tensor,
        returns: tf.Tensor
    ) -> dict:
        """Single PPO training step."""
        
        with tf.GradientTape() as tape:
            # Forward pass
            logits, values = self.model(states, training=True)
            values = tf.squeeze(values, axis=-1)
            
            # Policy loss (clipped)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            action_log_probs = tf.reduce_sum(
                log_probs * tf.one_hot(actions, self.n_actions),
                axis=-1
            )
            
            ratio = tf.exp(action_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(
                ratio,
                1.0 - self.config.clip_ratio,
                1.0 + self.config.clip_ratio
            )
            
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )
            
            # Value loss (clipped)
            value_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Entropy bonus for exploration
            probs = tf.nn.softmax(logits, axis=-1)
            entropy = -tf.reduce_mean(
                tf.reduce_sum(probs * log_probs, axis=-1)
            )
            
            # Total loss
            total_loss = (
                policy_loss +
                self.config.value_coef * value_loss -
                self.config.entropy_coef * entropy
            )
            
            # KL divergence for monitoring
            kl = tf.reduce_mean(old_log_probs - action_log_probs)
        
        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': total_loss,
            'kl_divergence': kl
        }
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> dict:
        """
        Update policy using collected trajectories.
        Performs multiple epochs with minibatch updates.
        """
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset_state()
        
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        # Multiple epochs
        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start in range(0, n_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]
                
                batch_states = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
                batch_actions = tf.convert_to_tensor(actions[batch_idx], dtype=tf.int32)
                batch_old_log_probs = tf.convert_to_tensor(old_log_probs[batch_idx], dtype=tf.float32)
                batch_advantages = tf.convert_to_tensor(advantages[batch_idx], dtype=tf.float32)
                batch_returns = tf.convert_to_tensor(returns[batch_idx], dtype=tf.float32)
                
                metrics = self.train_step(
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns
                )
                
                # Update running metrics
                for key, value in metrics.items():
                    self.metrics[key].update_state(value)
        
        # Return average metrics
        return {key: float(metric.result()) for key, metric in self.metrics.items()}
    
    def save_weights(self, filepath: str):
        """Save model weights."""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath: str):
        """Load model weights."""
        self.model.load_weights(filepath)