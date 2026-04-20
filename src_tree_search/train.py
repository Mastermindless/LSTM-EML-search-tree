"""REINFORCE training loop for the LSTM-EML search tree.

Usage
-----
    python train.py --target pi --device mps --steps 20000
    python train.py --target e  --device cpu --steps 5000

Device strategy
---------------
- Neural forward/backward : user-selected device (mps / cuda / cpu).
- mpmath tree evaluation  : always CPU (only pure-Python code runs there).
- Only the short integer token tensor crosses the device boundary per step.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from config import CONFIG
from loss import evaluate_rollout, reinforce_loss
from lstm_generator import LSTM_EML_Generator
from targets import list_targets
from tokenizer import parse_prefix, disabled_leaves_for_target


def pick_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def entropy_beta_schedule(step: int) -> float:
    t = min(step / CONFIG.entropy_decay_steps, 1.0)
    return CONFIG.entropy_beta * (1 - t) + CONFIG.entropy_beta_min * t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="pi", choices=list_targets())
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--steps", type=int, default=CONFIG.total_steps)
    p.add_argument("--batch", type=int, default=CONFIG.batch_size)
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[device] {device}   [target] {args.target}")

    targets = list_targets()
    target_id = targets.index(args.target)

    model = LSTM_EML_Generator(
        n_targets=len(targets),
        embed_dim=CONFIG.embed_dim,
        hidden=CONFIG.hidden,
        num_layers=CONFIG.num_layers,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=CONFIG.lr)
    disabled = disabled_leaves_for_target(args.target)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    best_reward = 0.0
    best_tree = None
    digits_n = CONFIG.target_digits

    t0 = time.time()
    for step in range(args.steps):
        target_ids = torch.full((args.batch,), target_id, dtype=torch.long)
        rollout = model.sample(
            target_ids, max_tokens=CONFIG.max_tokens,
            disabled_leaves=disabled,
        )
        rewards = evaluate_rollout(
            rollout.tokens, rollout.lengths,
            target_name=args.target,
            target_digits_n=digits_n,
            mp_dps=CONFIG.mp_dps,
            invalid_reward=CONFIG.invalid_reward,
        )

        beta = entropy_beta_schedule(step)
        loss, stats = reinforce_loss(
            rollout.log_probs, rewards, rollout.entropies, beta
        )

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
        optim.step()

        batch_max = stats["reward_max"]
        if batch_max > best_reward:
            best_reward = batch_max
            idx = int(rewards.argmax().item())
            seq = rollout.tokens[idx, : rollout.lengths[idx]].tolist()
            best_tree = parse_prefix(seq)

        if step % 50 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            print(
                f"step={step:5d}  r_mean={stats['reward_mean']:.3f}  "
                f"r_max={stats['reward_max']:.3f}  best={best_reward:.3f}  "
                f"H={stats['entropy']:.2f}  beta={beta:.3f}  "
                f"digits={digits_n}  t={elapsed:.1f}s"
            )
            if best_tree is not None:
                print(f"   best_tree = {best_tree!r}")

        if best_reward >= CONFIG.curriculum_threshold and digits_n < CONFIG.max_mp_dps // 2:
            new_n = int(digits_n * CONFIG.curriculum_factor)
            print(f"[curriculum] advancing target digits {digits_n} -> {new_n}")
            digits_n = new_n
            best_reward = 0.0

    ckpt = out_dir / f"lstm_eml_{args.target}.pt"
    torch.save({
        "model": model.state_dict(),
        "target": args.target,
        "target_id": target_id,
        "best_reward": best_reward,
        "best_tree_repr": repr(best_tree) if best_tree else None,
    }, ckpt)
    print(f"[saved] {ckpt}   best_tree={best_tree!r}")


if __name__ == "__main__":
    main()
