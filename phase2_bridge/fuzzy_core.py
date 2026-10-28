"""Minimal dependency-free fuzzy inference (zero-order Takagi-Sugeno) for junction control."""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence, Tuple

MF = Callable[[float], float]


def ramp_up(a: float, b: float) -> MF:
    """Membership 0 below a, 1 above b, linear in between."""
    def f(x: float) -> float:
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        return (x - a) / (b - a)
    return f


def ramp_down(a: float, b: float) -> MF:
    """Membership 1 below a, 0 above b, linear in between."""
    up = ramp_up(a, b)
    return lambda x: 1.0 - up(x)


def tri(a: float, b: float, c: float) -> MF:
    """Triangle with feet at a and c and peak at b (a < b < c)."""
    def f(x: float) -> float:
        if x <= a or x >= c:
            return 0.0
        return (x - a) / (b - a) if x < b else (c - x) / (c - b)
    return f


class TSK:
    """Zero-order TSK system: rule firing = min over antecedents, output = weighted average."""

    def __init__(self, sets: Mapping[str, Mapping[str, MF]],
                 rules: Sequence[Tuple[Mapping[str, str], float]], default: float = 0.0) -> None:
        for ante, _ in rules:
            for var, label in ante.items():
                if var not in sets or label not in sets[var]:
                    raise ValueError(f"unknown fuzzy set {var}.{label}")
        self.sets = sets
        self.rules = list(rules)
        self.default = default

    def trace(self, x: Mapping[str, float]) -> Tuple[float, List[Tuple[int, float]]]:
        """Return (output, [(rule index, firing strength), ...])."""
        num = den = 0.0
        fired: List[Tuple[int, float]] = []
        for i, (ante, z) in enumerate(self.rules):
            w = 1.0
            for var, label in ante.items():
                w = min(w, self.sets[var][label](x[var]))
            if w > 0.0:
                fired.append((i, w))
                num += w * z
                den += w
        return (num / den if den > 1e-12 else self.default), fired

    def infer(self, x: Mapping[str, float]) -> float:
        return self.trace(x)[0]
