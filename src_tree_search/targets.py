"""Ground-truth registry for transcendental constants.

All targets are pre-computed once at the maximum precision the training loop
will ever need, then sliced per sample via `target_digits(name, n)`.

Reference for the EML-family constants: A. Odrzywolek, symbolic-regression work
on nested exp/log identities. See repo:
  https://github.com/VA00/SymbolicRegressionPackage
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

import mpmath


# --------------------------------------------------------------------------- #
# Digit-string helper
# --------------------------------------------------------------------------- #
def _digits(val: "mpmath.mpf | mpmath.mpc", n_sig: int) -> str:
    """Return the first `n_sig` significant digits of |val| as a plain string.

    For complex values we concatenate real|imag digits so the prefix-match
    reward sees *both* components.
    """
    if isinstance(val, mpmath.mpc):
        return _digits(mpmath.re(val), n_sig) + "|" + _digits(mpmath.im(val), n_sig)
    if val == 0:
        return "0" * n_sig
    s = mpmath.nstr(val, n_sig, strip_zeros=False)
    s = s.replace(".", "").replace("-", "")
    if "e" in s or "E" in s:
        s = s.split("e")[0].split("E")[0]
    # Pad or truncate
    return (s + "0" * n_sig)[:n_sig]


# --------------------------------------------------------------------------- #
# Target registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Target:
    name: str
    producer: Callable[[], "mpmath.mpf | mpmath.mpc"]
    is_complex: bool = False


_REGISTRY: dict[str, Target] = {
    "pi":      Target("pi",      lambda: mpmath.pi),
    "e":       Target("e",       lambda: mpmath.e),
    "phi":     Target("phi",     lambda: (1 + mpmath.sqrt(5)) / 2),
    "gamma":   Target("gamma",   lambda: mpmath.euler),
    "ln2":     Target("ln2",     lambda: mpmath.log(2)),
    "sqrt2":   Target("sqrt2",   lambda: mpmath.sqrt(2)),
    "i":       Target("i",       lambda: mpmath.mpc(0, 1), True),
    "e_pi":    Target("e_pi",    lambda: mpmath.exp(mpmath.pi)),
    "pi_sq":   Target("pi_sq",   lambda: mpmath.pi ** 2),
    # Odrzywolek-style EML identity targets
    "ln_neg1": Target("ln_neg1", lambda: mpmath.log(-1), True),           # = i*pi
    "exp_i":   Target("exp_i",   lambda: mpmath.exp(mpmath.mpc(0, 1)), True),
}


def list_targets() -> list[str]:
    return sorted(_REGISTRY.keys())


@functools.lru_cache(maxsize=None)
def target_digits(name: str, n_sig: int, mp_dps: int) -> str:
    """Return pre-computed digit string of target `name` at `n_sig` significant
    digits, computed with mpmath working precision `mp_dps`.

    LRU cache means repeated calls during training are O(1).
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown target '{name}'. Known: {list_targets()}")
    prev = mpmath.mp.dps
    mpmath.mp.dps = max(mp_dps, n_sig + 10)
    try:
        val = _REGISTRY[name].producer()
        return _digits(val, n_sig)
    finally:
        mpmath.mp.dps = prev


def target_is_complex(name: str) -> bool:
    return _REGISTRY[name].is_complex


def value_digits(val: "mpmath.mpf | mpmath.mpc", n_sig: int) -> str:
    """Public wrapper for `_digits` — used by the loss module on tree outputs."""
    return _digits(val, n_sig)


if __name__ == "__main__":
    for t in list_targets():
        print(f"{t:>8s} -> {target_digits(t, 40, 64)}")
