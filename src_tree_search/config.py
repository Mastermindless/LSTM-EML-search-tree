"""Central hyperparameters for the LSTM-EML search-tree system."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Vocabulary / tree
    max_tokens: int = 31          # hard budget on sequence length (depth ~5)
    max_depth: int = 6

    # mpmath precision
    target_digits: int = 64       # digits of ground truth to match against
    mp_dps: int = 96              # mpmath working precision (digits + guard)
    max_mp_dps: int = 1024        # ceiling for curriculum

    # Model
    embed_dim: int = 32
    hidden: int = 128
    num_layers: int = 1

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    total_steps: int = 20_000
    entropy_beta: float = 2e-2
    entropy_beta_min: float = 2e-3
    entropy_decay_steps: int = 10_000
    grad_clip: float = 1.0

    # Reward
    invalid_reward: float = 0.0

    # Curriculum
    curriculum_threshold: float = 0.80   # fraction-matched to advance
    curriculum_factor: float = 2.0


CONFIG = Config()
