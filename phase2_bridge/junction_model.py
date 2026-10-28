"""Data types shared by the junction controller, the message bus and the CARLA adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

EMERGENCY_CLASSES: Tuple[str, ...] = ("ambulance", "police", "firetruck")


@dataclass(frozen=True)
class EmergencyTrack:
    """One tracked emergency vehicle seen by, or reported to, a junction."""
    tid: str
    cls: str
    approach: str
    eta_s: float          # seconds until it reaches the stop line at its current speed
    queue_ahead: int      # stopped vehicles between it and the stop line
    conf: float           # evidence in [0, 1]
    track_age_s: float
    source: str = "sensor"


@dataclass(frozen=True)
class ApproachObs:
    name: str
    queue: int = 0
    wait_s: float = 0.0   # longest waiting time on this approach
    capacity: int = 20    # storage in vehicles, used for normalisation


@dataclass(frozen=True)
class Message:
    """Neighbour message. kind: 'link_state' (occupancy of a link) or 'emergency' (advance notice)."""
    kind: str
    src: str
    dst: str
    t: float
    link: str
    occupancy: float = 0.0
    eta_s: float = 0.0
    cls: str = ""
    conf: float = 0.0
    tid: str = ""


@dataclass(frozen=True)
class JunctionObs:
    t: float
    approaches: Mapping[str, ApproachObs]
    emergencies: Tuple[EmergencyTrack, ...] = ()
    visibility: float = 1.0   # perception health in [0, 1]
    inbox: Tuple[Message, ...] = ()


@dataclass(frozen=True)
class Decision:
    t: float
    phase: int
    yellow: bool              # True: `phase` is the outgoing phase showing yellow
    mode: str
    reason: str
    pressure: Mapping[int, float]
    plan: Tuple[int, ...]
    outbox: Tuple[Message, ...]

    def to_dict(self) -> dict:
        return dict(t=self.t, phase=self.phase, yellow=self.yellow, mode=self.mode, reason=self.reason,
                    pressure={int(k): round(float(v), 4) for k, v in self.pressure.items()},
                    plan=list(self.plan), n_msgs=len(self.outbox))


@dataclass(frozen=True)
class ModeConfig:
    tau_s: float = 2.0             # smoothing time constant of the visibility signal
    normal_min: float = 0.6
    static_max: float = 0.3
    recover_min: float = 0.45
    enter_static_s: float = 3.0
    recover_s: float = 10.0
    enter_degraded_s: float = 2.0
    leave_degraded_s: float = 5.0
    leave_degraded_margin: float = 0.1


@dataclass
class JunctionConfig:
    name: str
    phases: Dict[int, Tuple[str, ...]]
    in_link: Dict[str, str] = field(default_factory=dict)        # approach -> incoming link id
    out_link: Dict[str, str] = field(default_factory=dict)       # approach -> outgoing link id (through movement)
    upstream_of: Dict[str, str] = field(default_factory=dict)    # incoming link -> junction feeding it
    downstream_of: Dict[str, str] = field(default_factory=dict)  # outgoing link -> junction receiving it
    link_travel_s: Dict[str, float] = field(default_factory=dict)
    min_green_s: float = 10.0
    max_green_s: float = 60.0
    yellow_s: float = 3.0
    emerg_min_green_s: float = 4.0
    emerg_max_green_s: float = 45.0
    clear_grace_s: float = 2.0
    headway_s: float = 2.0
    pass_buffer_s: float = 3.0
    switch_margin: float = 0.15
    replan_margin: float = 3.0
    tie_eps: float = 1e-6
    confirm_conf: float = 0.6
    min_track_age_s: float = 1.0
    degraded_confirm_conf: float = 0.8
    degraded_min_track_age_s: float = 2.0
    degraded_switch_margin: float = 0.25
    degraded_min_green_s: float = 15.0
    static_green_s: Dict[int, float] = field(default_factory=dict)
    static_allows_emergency: bool = False
    state_msg_period_s: float = 1.0
    link_state_stale_s: float = 10.0
    notice_expire_s: float = 8.0

    def __post_init__(self) -> None:
        for p in self.phases:
            self.static_green_s.setdefault(p, 30.0)
        self.validate()

    def validate(self) -> None:
        if not self.phases:
            raise ValueError("no phases")
        seen: Dict[str, int] = {}
        for p, apps in self.phases.items():
            for a in apps:
                if a in seen:
                    raise ValueError(f"approach {a} is in phases {seen[a]} and {p}")
                seen[a] = p
        if not (0.0 < self.emerg_min_green_s <= self.min_green_s <= self.max_green_s):
            raise ValueError("need 0 < emerg_min_green <= min_green <= max_green")
        if self.degraded_min_green_s > self.max_green_s:
            raise ValueError("degraded_min_green exceeds max_green")
        for p, g in self.static_green_s.items():
            if not (self.min_green_s <= g <= self.max_green_s):
                raise ValueError(f"static green {g} of phase {p} outside [min_green, max_green]")
