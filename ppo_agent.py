"""
PPO Agent with LSTM for Partially Observable Snake Environment.
Optimized for A100 GPU (Batched Inference & XLA Compilation).
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# OPTIMIZATION 1: Force GPU and enable XLA compilation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError:
        pass

# Enable XLA for major speedup on A100
tf.config.optimizer.set_jit(True)


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
        single = initial_state[0] if len(initial_state.shape) == 4 else initial_state
        self.buffer = np.repeat(single[np.newaxis, ...], self.n_stack, axis=0)
        return self.buffer
    
    def update(self, new_state: np.ndarray) -> np.ndarray:
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
    """Shared architecture with LSTM for temporal dependencies."""
    
    state_input = layers.Input(shape=state_shape, name='state')
    
    conv1 = layers.TimeDistributed(
        layers.Conv2D(24, (3, 3), activation='relu', padding='same'), name='conv1'
    )(state_input)
    
    conv2 = layers.TimeDistributed(
        layers.Conv2D(48, (3, 3), activation='relu', padding='same'), name='conv2'
    )(conv1)
    
    conv3 = layers.TimeDistributed(
        layers.Conv2D(48, (2, 2), activation='relu', padding='same'), name='conv3'
    )(conv2)
    
    flattened = layers.TimeDistributed(layers.Flatten(), name='flatten')(conv3)
    
    # CRITICAL FIX for XLA + LSTM: unroll=True bypasses the incompatible CuDNN kernel
    lstm_out = layers.LSTM(
        96, 
        return_sequences=False, 
        unroll=True, 
        name='lstm_memory'
    )(flattened)
    
    shared = layers.Dense(config.dense_units, activation='relu', name='shared_dense')(lstm_out)
    shared = layers.LayerNormalization()(shared)
    
    policy = layers.Dense(128, activation='relu', name='policy_dense')(shared)
    policy_logits = layers.Dense(n_actions, activation=None, name='policy_logits')(policy)
    
    value = layers.Dense(128, activation='relu', name='value_dense')(shared)
    value_output = layers.Dense(1, activation=None, name='value')(value)
    
    model = keras.Model(
        inputs=state_input,
        outputs=[policy_logits, value_output],
        name='ActorCritic_LSTM'
    )
    
    return model


class PPOAgent:
    """PPO Agent with LSTM - OPTIMIZED for A100."""
    
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
        
        self.stacker = FrameStacker(raw_state_shape, n_stack)
        self.stacked_shape = self.stacker.stacked_shape
        
        with tf.device('/GPU:0'):
            self.model = build_actor_critic_lstm(
                self.stacked_shape,
                n_actions,
                self.config
            )
        
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.max_grad_norm
        )
        
        self.metrics = {
            'policy_loss': keras.metrics.Mean(),
            'value_loss': keras.metrics.Mean(),
            'entropy': keras.metrics.Mean(),
            'total_loss': keras.metrics.Mean(),
            'kl_divergence': keras.metrics.Mean(),
        }

    @tf.function(jit_compile=True)
    def _select_action_batch_tf(self, states_tensor, training):
        """Compiled TensorFlow graph for fast batch inference on GPU."""
        logits, values = self.model(states_tensor, training=False)
        
        if training:
            action_dist = tf.random.categorical(logits, num_samples=1)
            actions = tf.squeeze(action_dist, axis=-1)
        else:
            actions = tf.argmax(logits, axis=-1)
            
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        
        # Gather log_probs for the chosen actions
        indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=actions.dtype), actions], axis=1)
        action_log_probs = tf.gather_nd(log_probs, indices)
        
        return actions, action_log_probs, tf.squeeze(values, axis=-1)

    def select_action_batch(self, states: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Takes a batch of states (N_BOARDS, ...) and returns actions, log_probs, and values."""
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions, log_probs, values = self._select_action_batch_tf(states_tensor, tf.constant(training))
        return actions.numpy(), log_probs.numpy(), values.numpy()
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Legacy single-state inference for compatibility."""
        actions, log_probs, values = self.select_action_batch(state[np.newaxis, ...], training)
        return int(actions[0]), float(log_probs[0]), float(values[0])
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        values_extended = np.append(values, next_value)
        
        deltas = rewards + self.config.gamma * values_extended[1:] * (1 - dones) - values
        
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.config.gamma * self.config.lambda_gae * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    @tf.function(jit_compile=True, reduce_retracing=True)
    def train_step(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        old_log_probs: tf.Tensor,
        advantages: tf.Tensor,
        returns: tf.Tensor
    ) -> dict:
        """Single PPO training step - XLA JIT compiled."""
        
        with tf.GradientTape() as tape:
            logits, values = self.model(states, training=True)
            values = tf.squeeze(values, axis=-1)
            
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            action_log_probs = tf.reduce_sum(
                log_probs * tf.one_hot(actions, self.n_actions), axis=-1
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
            
            value_loss = tf.reduce_mean(tf.square(returns - values))
            
            probs = tf.nn.softmax(logits, axis=-1)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1))
            
            total_loss = (
                policy_loss +
                self.config.value_coef * value_loss -
                self.config.entropy_coef * entropy
            )
            
            kl = tf.reduce_mean(old_log_probs - action_log_probs)
        
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
        """Update policy using collected trajectories."""
        
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        for metric in self.metrics.values():
            metric.reset_state()
        
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)
            
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
                
                for key, value in metrics.items():
                    self.metrics[key].update_state(value)
        
        return {key: float(metric.result()) for key, metric in self.metrics.items()}
    
    def save_weights(self, filepath: str):
        """Save model weights."""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath: str):
        """Load model weights."""
        self.model.load_weights(filepath)