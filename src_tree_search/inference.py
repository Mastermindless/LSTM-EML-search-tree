"""Inference: load a trained checkpoint and print the top-k discovered trees.

Usage
-----
    python inference.py --ckpt checkpoints/lstm_eml_pi.pt --k 5
"""
from __future__ import annotations

import argparse

import mpmath
import torch

from config import CONFIG
from eml_tree import evaluate_tree, tree_depth, tree_size
from loss import evaluate_rollout_verbose
from lstm_generator import LSTM_EML_Generator
from targets import list_targets, target_digits
from tokenizer import disabled_leaves_for_target, parse_prefix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--digits", type=int, default=CONFIG.target_digits)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    targets = list_targets()
    target_name = ckpt["target"]
    target_id = ckpt["target_id"]

    model = LSTM_EML_Generator(
        n_targets=len(targets),
        embed_dim=CONFIG.embed_dim,
        hidden=CONFIG.hidden,
        num_layers=CONFIG.num_layers,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    target_ids = torch.full((args.samples,), target_id, dtype=torch.long)
    disabled = disabled_leaves_for_target(target_name)
    rollout = model.sample(target_ids, max_tokens=CONFIG.max_tokens,
                           disabled_leaves=disabled, temperature=1.0)

    rewards, channels = evaluate_rollout_verbose(
        rollout.tokens, rollout.lengths,
        target_name=target_name,
        target_digits_n=args.digits,
        mp_dps=CONFIG.mp_dps,
    )

    top = torch.topk(rewards, min(args.k, rewards.shape[0]))
    gt = target_digits(target_name, args.digits, CONFIG.mp_dps)
    print(f"target={target_name}   digits={args.digits}")
    print(f"ground truth (|τ|): {gt[:60]}…\n")

    for rank, (r, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        seq = rollout.tokens[idx, : rollout.lengths[idx]].tolist()
        tree = parse_prefix(seq)
        if tree is None:
            continue
        val = evaluate_tree(tree, CONFIG.mp_dps)
        val_str = mpmath.nstr(val, 20) if val is not None else "—"
        ch = channels[idx] or "—"
        print(f"#{rank+1}  reward={r:.4f}  channel={ch:>3s}  "
              f"depth={tree_depth(tree)}  size={tree_size(tree)}")
        print(f"       tree : {tree!r}")
        print(f"       value: {val_str}\n")


if __name__ == "__main__":
    main()
