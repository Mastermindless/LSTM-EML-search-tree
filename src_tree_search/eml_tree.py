"""EML node framework — the tree substrate evaluated by mpmath.

    EML(x, y) := exp(x) - ln(y)

Everything downstream (tokenizer, parser, loss) treats trees as opaque
`EMLNode` objects with a single `evaluate() -> mpmath.mpc` contract.
"""
from __future__ import annotations

from typing import Union

import mpmath


Number = Union[int, float, complex, "mpmath.mpf", "mpmath.mpc"]


class EMLNode:
    def evaluate(self) -> "mpmath.mpc":
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class Constant(EMLNode):
    __slots__ = ("value", "_label")

    def __init__(self, value: Number, label: str | None = None):
        if isinstance(value, complex) or (hasattr(value, "imag") and value.imag != 0):
            self.value = mpmath.mpc(value)
        else:
            self.value = mpmath.mpf(value)
        self._label = label if label is not None else str(value)

    def evaluate(self) -> "mpmath.mpc":
        return self.value

    def __repr__(self) -> str:
        return self._label


class EML(EMLNode):
    __slots__ = ("left", "right")

    def __init__(self, left: EMLNode, right: EMLNode):
        self.left = left
        self.right = right

    def evaluate(self) -> "mpmath.mpc":
        x = self.left.evaluate()
        y = self.right.evaluate()
        return mpmath.exp(x) - mpmath.log(y)

    def __repr__(self) -> str:
        return f"EML({self.left!r}, {self.right!r})"


def evaluate_tree(node: EMLNode, dps: int) -> "mpmath.mpc | None":
    """Evaluate a tree at working precision `dps`. Returns None on math failure.

    A single `try/except` wraps the full recursion so domain errors
    (log(0), overflow) are isolated from the training loop.
    """
    prev = mpmath.mp.dps
    mpmath.mp.dps = dps
    try:
        return node.evaluate()
    except (ValueError, ZeroDivisionError, OverflowError, mpmath.libmp.NoConvergence):
        return None
    finally:
        mpmath.mp.dps = prev


def tree_depth(node: EMLNode) -> int:
    if isinstance(node, EML):
        return 1 + max(tree_depth(node.left), tree_depth(node.right))
    return 0


def tree_size(node: EMLNode) -> int:
    if isinstance(node, EML):
        return 1 + tree_size(node.left) + tree_size(node.right)
    return 1
