"""Digit-prefix reward + scale-homogeneous continuous loss.

The tree evaluation happens on CPU (mpmath is the bottleneck); the REINFORCE
loss itself is a tiny tensor op on the model's device.
"""
from __future__ import annotations

import mpmath
import torch

from eml_tree import evaluate_tree
from targets import channel_digits, target_digits
from tokenizer import parse_prefix


# Indices into the (re, im, abs) tuple returned by `channel_digits`.
CHANNEL_NAMES = ("Re", "Im", "abs")


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _best_channel_match(
    val: "mpmath.mpf | mpmath.mpc", gt: str, n_sig: int
) -> tuple[int, int]:
    """Return `(best_lcp, best_channel_idx)` for a rollout value against the
    ground-truth digit string under the channel-agnostic encoder.
    """
    channels = channel_digits(val, n_sig)
    best_lcp = -1
    best_idx = 0
    for idx, ch in enumerate(channels):
        lcp = _common_prefix_len(ch, gt)
        if lcp > best_lcp:
            best_lcp = lcp
            best_idx = idx
    return best_lcp, best_idx


def evaluate_rollout(
    tokens: torch.Tensor,        # (B, T) long (CPU is fine here)
    lengths: torch.Tensor,       # (B,) long
    target_name: str,
    target_digits_n: int,
    mp_dps: int,
    invalid_reward: float = 0.0,
) -> torch.Tensor:
    """Return (B,) float reward in [0, 1].

    Reward model (channel-agnostic; see manuscript §5.4):
        R(T) = max{ LCP(gt, ch) : ch ∈ {Re(v), Im(v), |v|} } / target_digits_n

    The max over projections removes the need to pre-register which
    component of a complex tree output carries the target, preserving the
    unsupervised-discovery property of Part 2.
    """
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
        best_lcp, _ = _best_channel_match(val, gt, target_digits_n)
        rewards.append(best_lcp / target_digits_n)

    return torch.tensor(rewards, dtype=torch.float32)


def evaluate_rollout_verbose(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    target_name: str,
    target_digits_n: int,
    mp_dps: int,
    invalid_reward: float = 0.0,
) -> tuple[torch.Tensor, list[str | None]]:
    """Same as `evaluate_rollout` but also returns the winning channel name
    (``"Re" | "Im" | "abs" | None``) per sample, for diagnostics.
    """
    tokens_cpu = tokens.detach().cpu().tolist()
    lens_cpu = lengths.detach().cpu().tolist()
    gt = target_digits(target_name, target_digits_n, mp_dps)

    rewards: list[float] = []
    channels: list[str | None] = []
    for seq, L in zip(tokens_cpu, lens_cpu):
        trimmed = seq[:L]
        tree = parse_prefix(trimmed)
        if tree is None:
            rewards.append(invalid_reward)
            channels.append(None)
            continue
        val = evaluate_tree(tree, mp_dps)
        if val is None:
            rewards.append(invalid_reward)
            channels.append(None)
            continue
        best_lcp, idx = _best_channel_match(val, gt, target_digits_n)
        rewards.append(best_lcp / target_digits_n)
        channels.append(CHANNEL_NAMES[idx])

    return torch.tensor(rewards, dtype=torch.float32), channels


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
