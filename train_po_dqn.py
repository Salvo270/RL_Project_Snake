"""Training loop for partially observable Snake with Dueling Double DQN."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from tqdm import trange

from po_dqn_agent import PODQNConfig, PartiallyObservableDQNAgent


def run_training_po(
    env_factory: Callable[[int], object],
    iterations: int = 12_000,
    n_boards: int = 1_000,
    config: PODQNConfig | None = None,
    save_path: str = "dqn_snake_po_final.weights.h5"
) -> List[Dict[str, float | None]]:
    
    env = env_factory(n_boards)
    # Otteniamo lo shape (5, 5, 4)
    raw_state_shape = tuple(np.asarray(env.to_state()).shape[1:])

    agent = PartiallyObservableDQNAgent(
        raw_state_shape=raw_state_shape,
        n_actions=4,
        n_boards=n_boards,
        config=config,
    )

    logs: List[Dict[str, float | None]] = []
    
    # Progress bar parlante
    pbar = trange(iterations, desc="PO-Dueling-DQN Training")
    
    for i in pbar:
        metrics = agent.train_on_env_step(env)
        logs.append(metrics)
        
        # Aggiornamento feedback ogni 10 step
        if i % 10 == 0:
            loss_val = metrics["loss"] if metrics["loss"] is not None else 0.0
            pbar.set_postfix({
                "reward": f"{metrics['mean_reward']:.4f}",
                "loss": f"{loss_val:.6f}",
                "eps": f"{metrics['epsilon']:.2f}"
            })
            
        # Checkpoint periodico (Strategia di disaster recovery)
        if i > 0 and i % 3000 == 0:
            agent.online_network.save_weights(f"dqn_po_checkpoint_{i}.weights.h5")

    # Salvataggio finale
    agent.online_network.save_weights(save_path)
    print(f"\nTraining completato. Pesi salvati in: {save_path}")
    
    return logs