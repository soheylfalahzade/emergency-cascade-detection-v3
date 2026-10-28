"""Perception-health estimate from the emergency-alarm stream (no ground truth needed).

Genuine emergency vehicles give long-lived tracks; spurious alarms give short blips. A high blip rate
means the emergency channel of the detector is unreliable, whatever the weather says.
Heuristic mechanism; parameters are NOT calibrated on real detector output."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List


class AlarmChurnHealth:
    def __init__(self, window_s: float = 60.0, merge_gap_s: float = 5.0, short_s: float = 6.0,
                 rate_max_per_min: float = 4.0) -> None:
        if min(window_s, merge_gap_s, short_s, rate_max_per_min) <= 0.0:
            raise ValueError("all parameters must be positive")
        self.window_s, self.merge_gap_s = window_s, merge_gap_s
        self.short_s, self.rate_max = short_s, rate_max_per_min
        self._span: Dict[str, List[float]] = {}
        self._blips: Deque[float] = deque()

    def _close(self, span: List[float], now: float) -> None:
        if span[1] - span[0] < self.short_s:
            self._blips.append(now)

    def update(self, t: float, tids: Iterable[str]) -> float:
        for tid in set(tids):
            sp = self._span.get(tid)
            if sp is not None and t - sp[1] > self.merge_gap_s:
                self._close(sp, t)
                sp = None
            if sp is None:
                self._span[tid] = [t, t]
            else:
                sp[1] = t
        for tid in [k for k, sp in self._span.items() if t - sp[1] > self.merge_gap_s]:
            self._close(self._span.pop(tid), t)
        while self._blips and t - self._blips[0] > self.window_s:
            self._blips.popleft()
        return self.health

    @property
    def rate_per_min(self) -> float:
        return len(self._blips) * 60.0 / self.window_s

    @property
    def health(self) -> float:
        return min(max(1.0 - self.rate_per_min / self.rate_max, 0.0), 1.0)
