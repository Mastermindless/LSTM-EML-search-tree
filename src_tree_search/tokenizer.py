"""Vocabulary + prefix-notation parser for EML search trees.

A token sequence is a depth-first preorder traversal of the tree:

    [EML, left-subtree, right-subtree]

Parsing is a single O(T) stack walk; sampling is an arity-guided generator
(stop when outstanding-operand count hits zero).
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath

from eml_tree import EML, Constant, EMLNode


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TokenDef:
    name: str
    arity: int                      # 2 = EML, 0 = leaf
    value: object | None = None     # leaf value; None for operators


_TOKENS: list[TokenDef] = [
    TokenDef("EML",    arity=2),
    TokenDef("C_0",    arity=0, value=0),
    TokenDef("C_1",    arity=0, value=1),
    TokenDef("C_-1",   arity=0, value=-1),
    TokenDef("C_2",    arity=0, value=2),
    TokenDef("C_i",    arity=0, value=complex(0, 1)),
    TokenDef("C_-i",   arity=0, value=complex(0, -1)),
    TokenDef("C_e",    arity=0, value="_E_"),        # resolved lazily to mpmath.e
    TokenDef("C_pi",   arity=0, value="_PI_"),       # resolved lazily to mpmath.pi
    TokenDef("EOS",    arity=0, value=None),         # terminator; parser stops
]

# Fast lookup tables
TOKEN_NAMES: list[str] = [t.name for t in _TOKENS]
NAME_TO_ID: dict[str, int] = {t.name: i for i, t in enumerate(_TOKENS)}
ARITIES: list[int] = [t.arity for t in _TOKENS]

VOCAB_SIZE: int = len(_TOKENS)
EML_ID: int = NAME_TO_ID["EML"]
EOS_ID: int = NAME_TO_ID["EOS"]
LEAF_IDS: list[int] = [i for i, t in enumerate(_TOKENS) if t.arity == 0 and t.name != "EOS"]


# --------------------------------------------------------------------------- #
# Leaf materialisation
# --------------------------------------------------------------------------- #
def _leaf_node(token_id: int) -> EMLNode:
    tok = _TOKENS[token_id]
    if tok.arity != 0:
        raise ValueError(f"Token {tok.name} is not a leaf")
    v = tok.value
    if v == "_E_":
        return Constant(mpmath.e, label="e")
    if v == "_PI_":
        return Constant(mpmath.pi, label="pi")
    if v is None:
        raise ValueError("EOS is not a materialisable leaf")
    return Constant(v, label=tok.name[2:])


def disabled_leaves_for_target(target_name: str) -> set[int]:
    """Prevent trivial-cheat leaves (e.g. token `C_pi` when target is `pi`)."""
    mapping = {"pi": "C_pi", "e": "C_e"}
    banned = mapping.get(target_name)
    return {NAME_TO_ID[banned]} if banned else set()


# --------------------------------------------------------------------------- #
# Prefix parser  — tokens -> EMLNode
# --------------------------------------------------------------------------- #
def parse_prefix(tokens: list[int]) -> EMLNode | None:
    """Parse a prefix-order token list into an EMLNode tree.

    Returns None if the sequence is malformed (dangling operator, extra tokens,
    or empty).  Complexity: O(T), one pass with an explicit stack.
    """
    if not tokens:
        return None

    # Walk tokens; build partially-complete subtrees on a stack.
    stack: list[list] = []          # each item: [op_id, [children…]]
    root: EMLNode | None = None

    for tid in tokens:
        if tid == EOS_ID:
            break
        arity = ARITIES[tid]
        if arity == 0:
            node: EMLNode = _leaf_node(tid)
        else:
            stack.append([tid, []])
            continue

        # Bubble completed nodes up the stack.
        while True:
            if not stack:
                if root is not None:
                    return None  # extra token after a completed root
                root = node
                break
            frame = stack[-1]
            frame[1].append(node)
            if len(frame[1]) == ARITIES[frame[0]]:
                stack.pop()
                if frame[0] == EML_ID:
                    node = EML(frame[1][0], frame[1][1])
                else:
                    return None  # unknown operator
                continue
            break

    if stack or root is None:
        return None  # incomplete tree
    return root


def tokens_required(tree: EMLNode) -> list[int]:
    """Inverse parser (tree -> prefix token sequence).  Used for logging."""
    out: list[int] = []

    def rec(n: EMLNode) -> None:
        if isinstance(n, EML):
            out.append(EML_ID)
            rec(n.left)
            rec(n.right)
        else:
            # Match Constant back to its token name by label.
            label = getattr(n, "_label", None)
            name_map = {"0": "C_0", "1": "C_1", "-1": "C_-1", "2": "C_2",
                        "i": "C_i", "-i": "C_-i", "e": "C_e", "pi": "C_pi"}
            out.append(NAME_TO_ID[name_map.get(label, "C_1")])
    rec(tree)
    out.append(EOS_ID)
    return out


if __name__ == "__main__":
    seq = [EML_ID, NAME_TO_ID["C_0"], NAME_TO_ID["C_-1"], EOS_ID]
    t = parse_prefix(seq)
    print(t, "=", t.evaluate() if t else None)   # EML(0,-1) = exp(0)-log(-1) = 1 - i*pi
