"""Approximate numerical validator for LSTM-EML search output.

Why this exists
---------------
`validate_eml.py` only flags an expression as a "hit" if its full value matches
one of the base TOKENS (pi, e, 1, -1, 0, i, -i) within a tight symbolic-grade
tolerance.  That check fails on every real `train.py` output because:

  * EML(x, y) = exp(x) - ln(y) is evaluated in floating mpmath — rounding
    errors accumulate with tree depth, so "symbolic equality" is unreachable.
  * Most discovered trees do NOT converge to a pure target.  They converge to
    something with the TARGET appearing in *one component* (typically the
    imaginary part, because ln(-1) = i*pi is the only way the vocabulary can
    surface pi without the C_pi leaf).  That matches how the training reward
    works: `targets.py::_digits` concatenates "real|imag" for complex values,
    and the digit-prefix reward in `loss.py` rewards any substring match.

What this script does
---------------------
For each `best_tree = EML(...)` line it:
  1. Evaluates the expression in mpmath at high precision.
  2. Compares val, Re(val), Im(val), |val|, and |Im(val)| against every target.
  3. Reports the closest target with an *absolute tolerance* (approximate, not
     symbolic) AND the number of matching leading digits (same metric the
     training reward uses, for apples-to-apples comparison).
  4. Runs a "positive-control" section first: hand-built EML trees whose
     values are provably the targets, to confirm the evaluator itself is sound.

This is the correct shape of validation for a numeric search: agree on a
tolerance, report proximity, and distinguish "model found pi as a component"
from "model found pi as the whole value".
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mpmath

mpmath.mp.dps = 50


# --------------------------------------------------------------------------- #
# Paste model output here (same style as `validate_eml.py`).
# Lines may be `best_tree = EML(...)` (train.py stdout) or bare `EML(...)`.
# --------------------------------------------------------------------------- #
MODEL_OUTPUT = """
best_tree = pi
best_tree = EML(0, -1)
best_tree = EML(0, EML(EML(0, EML(pi, 1)), 1))
best_tree = EML(1, EML(-1, e))
best_tree = EML(EML(e, 1), 1)
best_tree = EML(1, EML(-1, e))
best_tree = EML(EML(0, 1), EML(EML(EML(i, i), e), -i))
best_tree = EML(EML(e, 1), 2)
best_tree = EML(i, EML(0, e))
best_tree = EML(1, EML(-1, e))
best_tree = EML(1, EML(-1, e))
best_tree = EML(EML(e, 1), -1)
best_tree = EML(-1, EML(0, e))
best_tree = EML(1, EML(-1, e))
best_tree = EML(EML(0, -1), e)
best_tree = EML(1, EML(-1, e))
best_tree = EML(1, EML(-1, e))
best_tree = EML(EML(e, -1), -1)
best_tree = EML(EML(e, -1), -1)
best_tree = EML(EML(e, EML(0, 1)), i)
best_tree = EML(EML(0, -1), EML(-1, i))
best_tree = EML(EML(1, EML(-i, 2)), -i)
best_tree = EML(EML(1, EML(-i, 2)), -i)
best_tree = EML(EML(i, EML(0, 2)), EML(0, i))
best_tree = EML(i, EML(EML(0, EML(1, EML(e, e))), -i))
best_tree = EML(EML(i, EML(0, 2)), EML(-1, EML(-i, -1)))
"""


# --------------------------------------------------------------------------- #
# EML operator + leaf bindings used by eval()
# --------------------------------------------------------------------------- #
def EML(x, y):
    """EML(x, y) = exp(x) - ln(y)  (principal branch for complex y)."""
    return mpmath.exp(x) - mpmath.log(y)


e  = mpmath.e
pi = mpmath.pi
i  = mpmath.mpc(0, 1)


TARGETS: dict[str, "mpmath.mpf | mpmath.mpc"] = {
    "0":  mpmath.mpf(0),
    "1":  mpmath.mpf(1),
    "-1": mpmath.mpf(-1),
    "2":  mpmath.mpf(2),
    "pi": mpmath.pi,
    "e":  mpmath.e,
    "i":  mpmath.mpc(0, 1),
    "-i": mpmath.mpc(0, -1),
}


# --------------------------------------------------------------------------- #
# Approximate matching
# --------------------------------------------------------------------------- #
@dataclass
class Match:
    target: str                  # name of target that matched best
    component: str               # which component matched: val / re / im / abs / abs_im
    abs_err: float               # |component - target_val|
    digits: int                  # leading matching significant digits


def _n_digits_match(a, b, max_digits: int = 50) -> int:
    """Leading matching significant digits of the *real* numbers a, b.

    Mirrors the prefix-match metric used by `loss.py` but without the
    real|imag concatenation trick — we call this per-component instead.
    """
    if a == 0 and b == 0:
        return max_digits
    if a == 0 or b == 0:
        return 0
    # log10 of relative error → count of agreeing digits.
    rel = abs(mpmath.mpf(a) - mpmath.mpf(b)) / max(abs(mpmath.mpf(a)), abs(mpmath.mpf(b)))
    if rel == 0:
        return max_digits
    return max(0, int(-mpmath.log10(rel)))


def _components(val) -> dict[str, "mpmath.mpf"]:
    """Break a scalar into scalar components we can compare target-by-target."""
    if isinstance(val, mpmath.mpc):
        re_, im_ = mpmath.re(val), mpmath.im(val)
    else:
        re_, im_ = mpmath.mpf(val), mpmath.mpf(0)
    return {
        "re":      re_,
        "im":      im_,
        "abs":     mpmath.sqrt(re_ * re_ + im_ * im_),
        "abs_im":  abs(im_),
    }


def classify(val, tol: float) -> list[Match]:
    """Return all (target, component) pairs within `tol` absolute error.

    Sorted by abs_err ascending, so the first entry is the closest hit.
    We check per-component because LSTM-EML rollouts routinely produce values
    like `e - i*pi` where the TARGET (pi) lives in just the imaginary part.
    """
    hits: list[Match] = []
    comps = _components(val)

    for name, tgt in TARGETS.items():
        # Compare whole value (real vs complex) only when shapes match.
        tgt_is_complex = isinstance(tgt, mpmath.mpc)
        whole_err = float(abs(mpmath.mpc(val) - mpmath.mpc(tgt)))
        if whole_err < tol:
            hits.append(Match(name, "val", whole_err,
                              _n_digits_match(mpmath.mpc(val).real, mpmath.mpc(tgt).real)))

        # Compare components against the target's real value (the "imag" targets
        # i and -i we cover via the whole-value check above).
        if not tgt_is_complex:
            for comp_name, comp_val in comps.items():
                err = float(abs(comp_val - tgt))
                if err < tol:
                    hits.append(Match(name, comp_name, err,
                                      _n_digits_match(comp_val, tgt)))

    # Rank by "informativeness", not just raw error:
    #   1. whole-value matches beat component matches (tree as a whole hits target)
    #   2. non-"0" targets beat "0" (matching zero is usually a trivial artefact:
    #      a pure-real value's imag part is 0, so "im matches 0" tells us nothing)
    #   3. smaller abs_err as the tie-breaker
    def _rank(m: Match) -> tuple:
        return (m.component != "val", m.target == "0", m.abs_err)
    hits.sort(key=_rank)
    return hits


# --------------------------------------------------------------------------- #
# Line parsing
# --------------------------------------------------------------------------- #
_ASSIGN_RE = re.compile(r"best_tree\s*=\s*(.+?)\s*$")
_BARE_RE   = re.compile(r"^\s*(EML\([^|]+?\))\s*(?:\||$)")


def extract_expressions(text: str) -> list[str]:
    """Pull EML expressions from a train.py log OR from a `validate_eml.py`
    table (which lists the bare expression in the first column, sometimes
    followed by `| ... | ...` diagnostic columns — we strip those)."""
    out: list[str] = []
    for line in text.splitlines():
        m = _ASSIGN_RE.search(line)
        if m:
            out.append(m.group(1).strip())
            continue
        m = _BARE_RE.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


# --------------------------------------------------------------------------- #
# Positive controls — hand-built EML trees that *provably* hit each target.
# --------------------------------------------------------------------------- #
# Derivations (using only the EML operator and the listed leaves):
#
#   1   = EML(0, 1)                       = exp(0) - ln(1)       = 1 - 0
#   e   = EML(1, 1)                       = exp(1) - ln(1)       = e - 0
#   0   = EML(0, e)                       = exp(0) - ln(e)       = 1 - 1
#   -1  = EML(0, EML(2, 1))               = 1 - ln(e^2) = 1 - 2
#   -i  and  i  require the C_i / C_-i leaves (no construction without them).
#   pi  requires the C_pi leaf and an "identity-through-pi" scramble:
#         EML(pi, 1)         = e^pi
#         EML(0, e^pi)       = 1 - pi
#         EML(1-pi, 1)       = e^(1-pi)
#         EML(0, e^(1-pi))   = 1 - (1 - pi) = pi
#       →  EML(0, EML(EML(0, EML(pi, 1)), 1))
#
# The pi-without-pi-leaf column is deliberately blank: it is not possible to
# synthesize a purely-real pi from {0, 1, -1, 2, i, -i, e} with only the EML
# operator, because EML never multiplies, so the i*pi produced by ln(-1) can
# never be "flattened" back to the real line.
POSITIVE_CONTROLS: list[tuple[str, str]] = [
    ("1",  "EML(0, 1)"),
    ("e",  "EML(1, 1)"),
    ("0",  "EML(0, e)"),
    ("-1", "EML(0, EML(2, 1))"),
    ("pi", "EML(0, EML(EML(0, EML(pi, 1)), 1))"),   # uses C_pi leaf
]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _format_val(v, sig: int = 8) -> str:
    if isinstance(v, mpmath.mpc):
        return f"({mpmath.nstr(mpmath.re(v), sig)} + {mpmath.nstr(mpmath.im(v), sig)}j)"
    return mpmath.nstr(v, sig)


def report(title: str, rows: list[tuple[str, str]], tol: float) -> None:
    print(f"\n=== {title}   (tolerance={tol:g}) ===")
    hdr = f"{'Label':<6} | {'Expression':<52} | {'Evaluation':<32} | {'Closest match':<28} | Digits"
    print(hdr)
    print("-" * len(hdr))
    for label, expr in rows:
        try:
            val = eval(expr, {"__builtins__": {}}, {"EML": EML, "e": e, "pi": pi, "i": i})
        except Exception as ex:
            print(f"{label:<6} | {expr:<52} | ERROR: {ex}")
            continue

        hits = classify(val, tol)
        if hits:
            best = hits[0]
            match_str = f"{best.target}  via {best.component}  (err={best.abs_err:.2g})"
            digits = best.digits
        else:
            match_str = "— (no target within tolerance)"
            digits = 0

        print(f"{label:<6} | {expr:<52} | {_format_val(val):<32} | {match_str:<28} | {digits}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", nargs="?", type=Path,
                    help="Optional log file with `best_tree = ...` lines. "
                         "If omitted, uses the MODEL_OUTPUT string at the top "
                         "of this file (or stdin if data is piped in).")
    ap.add_argument("--tol", type=float, default=1e-10,
                    help="Absolute error threshold for 'match' (default 1e-10). "
                         "Raise it if you want to include looser, noisier hits.")
    ap.add_argument("--skip-controls", action="store_true",
                    help="Skip the positive-control sanity block.")
    args = ap.parse_args()

    if not args.skip_controls:
        report("POSITIVE CONTROLS (hand-built, should all hit cleanly)",
               POSITIVE_CONTROLS, args.tol)

    # Input precedence: explicit file > non-empty piped stdin > embedded MODEL_OUTPUT.
    if args.input is not None:
        text = args.input.read_text()
        source = str(args.input)
    elif not sys.stdin.isatty() and (piped := sys.stdin.read()).strip():
        text = piped
        source = "<stdin>"
    else:
        text = MODEL_OUTPUT
        source = "<embedded MODEL_OUTPUT>"

    exprs = extract_expressions(text)
    if not exprs:
        print(f"\n[warn] no `best_tree = …` lines found in {source}.")
        return

    print(f"\n[source] {source}   ({len(exprs)} expression{'s' if len(exprs)!=1 else ''})")
    rows = [(f"#{k+1}", expr) for k, expr in enumerate(exprs)]
    report("MODEL OUTPUTS", rows, args.tol)


if __name__ == "__main__":
    main()
