"""LSTM controller that generates EML prefix-notation token sequences.

Core idea
---------
At each step the LSTM emits logits over the full vocabulary.  An arity tracker
masks illegal moves so the output is *always* a well-formed tree:

    remaining_slots = 1                    # root
    remaining_slots -= 1                   # per emitted token
    remaining_slots += arity(token)        # per operator emitted
    stop when remaining_slots == 0

This eliminates the need for post-hoc repair and keeps the sampler vectorised
over the batch dimension.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import ARITIES, EML_ID, EOS_ID, VOCAB_SIZE


@dataclass
class Rollout:
    tokens: torch.Tensor        # (B, T) long  — padded with EOS
    log_probs: torch.Tensor     # (B, T) float — 0.0 on padded positions
    entropies: torch.Tensor     # (B, T) float — 0.0 on padded positions
    lengths: torch.Tensor       # (B,) long   — effective length (pre-padding)


class LSTM_EML_Generator(nn.Module):
    def __init__(
        self,
        n_targets: int,
        embed_dim: int = 32,
        hidden: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.target_embed = nn.Embedding(n_targets, embed_dim)
        self.tok_embed = nn.Embedding(VOCAB_SIZE + 1, embed_dim)  # +1 for BOS
        self.bos_id = VOCAB_SIZE
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, VOCAB_SIZE)

        arities = torch.tensor(ARITIES, dtype=torch.long)
        self.register_buffer("arities", arities)

    # --------------------------------------------------------------------- #
    def _step(self, emb: torch.Tensor, state):
        out, state = self.lstm(emb.unsqueeze(1), state)
        logits = self.head(out.squeeze(1))                      # (B, V)
        return logits, state

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _init_state(self, B: int, device: torch.device):
        z = torch.zeros(self.num_layers, B, self.hidden, device=device)
        return (z.clone(), z.clone())

    # --------------------------------------------------------------------- #
    def sample(
        self,
        target_ids: torch.Tensor,
        max_tokens: int,
        disabled_leaves: set[int] | None = None,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Rollout:
        """Sample a batch of rollouts on the model's device.

        target_ids : (B,) long — index into the target embedding table
        max_tokens : hard cap on sequence length
        disabled_leaves : token ids whose logit is set to -inf (e.g. C_pi
                          when the target is π — prevents trivial cheats).
        """
        device = next(self.parameters()).device
        B = target_ids.shape[0]

        # Init state with target embedding as first input (replaces h_0).
        tgt_emb = self.target_embed(target_ids.to(device))      # (B, E)
        state = self._init_state(B, device)
        # Prime the LSTM with the target embedding.
        _, state = self.lstm(tgt_emb.unsqueeze(1), state)

        cur = torch.full((B,), self.bos_id, dtype=torch.long, device=device)
        remaining = torch.ones(B, dtype=torch.long, device=device)   # 1 slot = root
        done = torch.zeros(B, dtype=torch.bool, device=device)

        tok_buf: list[torch.Tensor] = []
        logp_buf: list[torch.Tensor] = []
        ent_buf: list[torch.Tensor] = []
        lengths = torch.zeros(B, dtype=torch.long, device=device)

        neg_inf = torch.tensor(float("-inf"), device=device)

        for t in range(max_tokens):
            emb = self.tok_embed(cur)                            # (B, E)
            logits, state = self._step(emb, state)               # (B, V)
            logits = logits / max(temperature, 1e-6)

            # Illegal moves: EOS before tree closes; operators when only one slot
            # remains would require more depth than we can afford.
            budget_left = max_tokens - t - 1
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, EOS_ID] = True                               # never pick EOS mid-tree
            # Forbid EML if picking it would push us past the token budget:
            #   after choosing EML, remaining becomes `remaining + 1`, and we
            #   must still fill those slots with leaves within `budget_left`.
            no_eml = (remaining + 1) > budget_left
            mask[no_eml, EML_ID] = True
            if disabled_leaves:
                for lid in disabled_leaves:
                    mask[:, lid] = True

            logits = logits.masked_fill(mask, float("-inf"))

            # Handle already-done rows: force them to EOS so their log_prob = 0.
            if done.any():
                logits[done] = neg_inf
                logits[done, EOS_ID] = 0.0

            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)
            if greedy:
                next_tok = logits.argmax(dim=-1)
            else:
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

            lp = log_probs_all.gather(1, next_tok.unsqueeze(1)).squeeze(1)
            # Entropy (masked positions contribute 0 due to -inf → prob=0).
            safe = probs.clamp_min(1e-12)
            ent = -(probs * safe.log()).sum(-1)

            # Zero-out done rows.
            lp = lp.masked_fill(done, 0.0)
            ent = ent.masked_fill(done, 0.0)

            tok_buf.append(next_tok)
            logp_buf.append(lp)
            ent_buf.append(ent)

            # Update arity budget.
            arity = self.arities[next_tok]
            # subtract 1 slot consumed, add arity new slots.
            remaining = torch.where(done, remaining, remaining - 1 + arity)
            lengths = torch.where(done, lengths, lengths + 1)
            newly_done = remaining == 0
            done = done | newly_done

            cur = next_tok
            if bool(done.all()):
                break

        tokens = torch.stack(tok_buf, dim=1)
        log_probs = torch.stack(logp_buf, dim=1)
        entropies = torch.stack(ent_buf, dim=1)

        return Rollout(tokens=tokens, log_probs=log_probs,
                       entropies=entropies, lengths=lengths)
