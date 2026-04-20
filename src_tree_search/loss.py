"""Digit-prefix reward + scale-homogeneous continuous loss.

The tree evaluation happens on CPU (mpmath is the bottleneck); the REINFORCE
loss itself is a tiny tensor op on the model's device.
"""
from __future__ import annotations

import mpmath
import torch

from eml_tree import evaluate_tree
from targets import target_digits, value_digits
from tokenizer import parse_prefix


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def evaluate_rollout(
    tokens: torch.Tensor,        # (B, T) long (CPU is fine here)
    lengths: torch.Tensor,       # (B,) long
    target_name: str,
    target_digits_n: int,
    mp_dps: int,
    invalid_reward: float = 0.0,
) -> torch.Tensor:
    """Return (B,) float reward in [0, 1] = matched_digits / target_digits_n."""
    tokens_cpu = tokens.detach().cpu().tolist()
    lens_cpu = lengths.detach().cpu().tolist()
    gt = target_digits(target_name, target_digits_n, mp_dps)

    rewards: list[float] = []
    for seq, L in zip(tokens_cpu, lens_cpu):
        trimmed = seq[:L]
        tree = parse_prefix(trimmed)
        if tree is None:
            rewards.append(invalid_reward)
            continue
        val = evaluate_tree(tree, mp_dps)
        if val is None:
            rewards.append(invalid_reward)
            continue
        if isinstance(val, mpmath.mpf):
            val = mpmath.mpc(val, 0)
        got = value_digits(val, target_digits_n)
        rewards.append(_common_prefix_len(got, gt) / target_digits_n)

    return torch.tensor(rewards, dtype=torch.float32)


def reinforce_loss(
    log_probs: torch.Tensor,     # (B, T)
    rewards: torch.Tensor,       # (B,)
    entropies: torch.Tensor,     # (B, T)
    entropy_beta: float,
) -> tuple[torch.Tensor, dict]:
    """Policy-gradient loss with entropy bonus and leave-one-out baseline."""
    rewards = rewards.to(log_probs.device)
    # Leave-one-out baseline is more robust than batch-mean for small B.
    if rewards.shape[0] > 1:
        baseline = (rewards.sum() - rewards) / (rewards.shape[0] - 1)
    else:
        baseline = rewards.mean()
    advantage = (rewards - baseline).detach()

    pg = -(advantage.unsqueeze(1) * log_probs).sum(-1).mean()
    ent = entropies.sum(-1).mean()
    loss = pg - entropy_beta * ent

    stats = {
        "reward_mean": rewards.mean().item(),
        "reward_max": rewards.max().item(),
        "entropy": ent.item(),
        "pg_loss": pg.item(),
    }
    return loss, stats


def scale_homogeneous_loss(val: "mpmath.mpc", target: "mpmath.mpc",
                           eps: float = 1e-30) -> float:
    """Diagnostic only — log10(|v - t| + eps).  Not used for backprop."""
    diff = mpmath.mpc(val) - mpmath.mpc(target)
    return float(mpmath.log10(abs(diff) + eps))
