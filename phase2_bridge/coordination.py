"""In-memory message bus (with optional delay and loss) and a multi-junction runner."""
from __future__ import annotations

import dataclasses
import random
from typing import Dict, Iterable, List, Mapping, Tuple

from junction_controller import FuzzyJunctionController
from junction_model import Decision, JunctionObs, Message


class MessageBus:
    def __init__(self, delay_s: float = 0.0, drop_p: float = 0.0, seed: int = 0) -> None:
        self.delay_s, self.drop_p = delay_s, drop_p
        self._rng = random.Random(seed)
        self._pending: List[Tuple[float, Message]] = []
        self.sent = 0
        self.dropped = 0

    def publish(self, msgs: Iterable[Message], now: float) -> None:
        for m in msgs:
            self.sent += 1
            if self._rng.random() < self.drop_p:
                self.dropped += 1
                continue
            self._pending.append((now + self.delay_s, m))

    def collect(self, dst: str, now: float) -> Tuple[Message, ...]:
        ready = [m for t, m in self._pending if m.dst == dst and t <= now]
        self._pending = [(t, m) for t, m in self._pending if not (m.dst == dst and t <= now)]
        return tuple(ready)


class NetworkController:
    """Steps every junction controller; messages published now are visible at the next step."""

    def __init__(self, controllers: Mapping[str, FuzzyJunctionController], bus: MessageBus) -> None:
        self.controllers: Dict[str, FuzzyJunctionController] = dict(controllers)
        self.bus = bus

    def step(self, t: float, dt: float, raw_obs: Mapping[str, JunctionObs]) -> Dict[str, Decision]:
        decisions: Dict[str, Decision] = {}
        for name, ctl in self.controllers.items():
            o = dataclasses.replace(raw_obs[name], inbox=self.bus.collect(name, t))
            decisions[name] = ctl.step(o, dt)
        for d in decisions.values():
            self.bus.publish(d.outbox, t)
        return decisions
