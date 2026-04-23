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
def _digits(val: "mpmath.mpf | mpmath.mpc | int | float", n_sig: int) -> str:
    """Return the first `n_sig` significant digits of |val| as a plain string.

    Scalar-only encoder.  Complex inputs are first projected onto their
    magnitude `|val|`; callers that need per-component encodings should use
    `channel_digits()` instead.
    """
    if isinstance(val, mpmath.mpc):
        val = abs(val)   # project to magnitude for the scalar path
    if val == 0:
        return "0" * n_sig
    s = mpmath.nstr(val, n_sig, strip_zeros=False)
    s = s.replace(".", "").replace("-", "")
    if "e" in s or "E" in s:
        s = s.split("e")[0].split("E")[0]
    # Pad or truncate
    return (s + "0" * n_sig)[:n_sig]


def channel_digits(
    val: "mpmath.mpf | mpmath.mpc | int | float", n_sig: int
) -> tuple[str, str, str]:
    """Return `(re_digits, im_digits, abs_digits)` — three parallel
    digit-string projections of a scalar or complex value, each of length
    `n_sig`.

    Rationale: the reward encoder is *channel-agnostic*.  Given only a
    target digit string, the training signal should not bake in an
    assumption about which component of a complex tree output carries the
    target (e.g. π emerging as the imaginary part of EML(0, −1) = 1 − iπ).
    By exposing all three mathematically meaningful projections we let the
    policy gradient pull on whichever channel matches, without supervision
    over tree structure.
    """
    re_val = mpmath.re(val) if isinstance(val, mpmath.mpc) else val
    im_val = mpmath.im(val) if isinstance(val, mpmath.mpc) else 0
    abs_val = abs(val) if isinstance(val, mpmath.mpc) else mpmath.fabs(val)
    return (
        _digits(re_val, n_sig),
        _digits(im_val, n_sig),
        _digits(abs_val, n_sig),
    )


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
    """Return the canonical real-valued digit string of target `name`.

    The canonical form is `|τ|` (magnitude for complex targets, |τ| for
    real ones).  This choice makes the training signal channel-agnostic:
    the reward function projects each rollout value through (Re, Im, |·|)
    and takes the best prefix match, so complex targets collapse onto a
    single real string without losing information about what we are
    searching for.

    Caveat: constants whose *magnitudes* coincide (e.g. |i| = |1| = 1)
    become indistinguishable under this encoding.  This is a deliberate
    trade-off in favour of unsupervised discovery; see the manuscript
    §5.4 for details.
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
    """Scalar wrapper for `_digits` — retained for API compatibility.

    Prefer `channel_digits()` for reward computation: the channel-agnostic
    encoder needs all three projections simultaneously.
    """
    return _digits(val, n_sig)


if __name__ == "__main__":
    for t in list_targets():
        print(f"{t:>8s} -> {target_digits(t, 40, 64)}")

    # Sanity check: the canonical π derivation EML(0, -1) = 1 - iπ should
    # achieve a perfect prefix match under the channel-agnostic encoder via
    # its imaginary channel.
    def _lcp(a: str, b: str) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    print("\nSanity check (channel-agnostic encoder):")
    mpmath.mp.dps = 80
    v = mpmath.exp(0) - mpmath.log(-1)             # EML(0, -1) = 1 - i*pi
    re_s, im_s, abs_s = channel_digits(v, 40)
    gt = target_digits("pi", 40, 64)
    print(f"  v = EML(0,-1)    = {mpmath.nstr(v, 20)}")
    print(f"  target (pi)      : {gt[:40]}")
    print(f"  channel Re       : {re_s}  lcp={_lcp(re_s, gt)}")
    print(f"  channel Im       : {im_s}  lcp={_lcp(im_s, gt)}")
    print(f"  channel |·|      : {abs_s}  lcp={_lcp(abs_s, gt)}")
    print(f"  reward (max/40)  : {max(_lcp(re_s, gt), _lcp(im_s, gt), _lcp(abs_s, gt)) / 40:.3f}")
