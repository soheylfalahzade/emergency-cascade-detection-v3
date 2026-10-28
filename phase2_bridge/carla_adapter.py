"""CARLA I/O for one junction. `carla` is imported lazily so the controller core stays testable without it.

NOT verified against a live CARLA server. Run inspect_carla_junction.py first and check the spec.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

from junction_model import (EMERGENCY_CLASSES, ApproachObs, Decision, EmergencyTrack,
                            JunctionConfig, JunctionObs)
from perception_noise import Observation, PerceptionNoise


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def cls_of(type_id: str) -> str:
    t = type_id.lower()
    if "ambulance" in t:
        return "ambulance"
    if "firetruck" in t or "fire" in t:
        return "firetruck"
    if "police" in t:
        return "police"
    return "car"


def load_spec(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_config(spec: dict) -> JunctionConfig:
    phases = {int(k): tuple(v) for k, v in spec["phases"].items()}
    return JunctionConfig(name=spec.get("name", "J"), phases=phases, **dict(spec.get("config", {})))


def visibility_from_weather(w) -> float:
    """Heuristic proxy from carla.WeatherParameters. UNCALIBRATED: tune against measured detector degradation."""
    fog = min(max(w.fog_density / 100.0, 0.0), 1.0)
    rain = min(max(w.precipitation / 100.0, 0.0), 1.0)
    night = 1.0 if w.sun_altitude_angle < -10.0 else 0.0
    return min(max(1.0 - 0.9 * fog - 0.5 * rain - 0.25 * night, 0.0), 1.0)


class CarlaJunctionIO:
    def __init__(self, world, spec: dict, noise: Optional[PerceptionNoise] = None,
                 queue_len_m: float = 50.0, approach_len_m: float = 70.0, stop_speed_mps: float = 0.5) -> None:
        import carla  # lazy
        self.carla, self.world, self.noise = carla, world, noise
        self.lights = {a: [world.get_actor(i) for i in info["lights"]] for a, info in spec["approaches"].items()}
        self.stop_xy = {a: [tuple(p) for p in info["stop_xy"]] for a, info in spec["approaches"].items()}
        self.heading = {a: float(info["yaw"]) for a, info in spec["approaches"].items()}
        self.capacity = {a: int(info.get("capacity", 12)) for a, info in spec["approaches"].items()}
        self.spec = spec
        self.queue_len_m, self.approach_len_m, self.stop_speed = queue_len_m, approach_len_m, stop_speed_mps
        self.wait_since: Dict[str, float] = {}
        self.first_seen: Dict[str, float] = {}
        self._last_t: Optional[float] = None

    def take_control(self) -> None:
        for tls in self.lights.values():
            for tl in tls:
                tl.freeze(True)

    def release_control(self) -> None:
        for tls in self.lights.values():
            for tl in tls:
                tl.freeze(False)

    def _assign(self, x: float, y: float, yaw: float) -> Tuple[Optional[str], float]:
        fwd = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))
        best: Tuple[Optional[str], float] = (None, 0.0)
        best_d = 1e18
        for a, pts in self.stop_xy.items():
            if abs(wrap_deg(yaw - self.heading[a])) > 45.0:
                continue
            for sx, sy in pts:
                dx, dy = sx - x, sy - y
                d = math.hypot(dx, dy)
                if d > self.approach_len_m or dx * fwd[0] + dy * fwd[1] < -1.0:
                    continue
                if d < best_d:
                    best_d, best = d, (a, d)
        return best

    def read_obs(self, t: float, visibility: float = 1.0) -> JunctionObs:
        gt: Dict[str, dict] = {}
        for v in self.world.get_actors().filter("vehicle.*"):
            tf, vel = v.get_transform(), v.get_velocity()
            a, d = self._assign(tf.location.x, tf.location.y, tf.rotation.yaw)
            if a is None:
                continue
            gt[str(v.id)] = dict(id=str(v.id), lane=a, cls=cls_of(v.type_id), dist_m=d,
                                 speed=math.hypot(vel.x, vel.y))
        if self.noise is not None:
            dt = 1.0 if self._last_t is None else max(t - self._last_t, 1e-3)
            seen = self.noise.step(t, dt, list(gt.values()), list(self.lights))
        else:
            seen = [Observation(v["id"], v["lane"], v["cls"], 1.0) for v in gt.values()]
        self._last_t = t
        per: Dict[str, List[Tuple[Observation, dict]]] = {a: [] for a in self.lights}
        for o in seen:
            per.setdefault(o.lane, []).append((o, gt.get(o.vid, dict(dist_m=30.0, speed=5.0))))
        approaches: Dict[str, ApproachObs] = {}
        tracks: List[EmergencyTrack] = []
        for a, lst in per.items():
            lst.sort(key=lambda p: p[1]["dist_m"])
            queue, wait = 0, 0.0
            for o, info in lst:
                stopped = info["speed"] < self.stop_speed and info["dist_m"] <= self.queue_len_m
                if stopped:
                    queue += 1
                    if o.vid in gt:
                        wait = max(wait, t - self.wait_since.setdefault(o.vid, t))
                else:
                    self.wait_since.pop(o.vid, None)
            for i, (o, info) in enumerate(lst):
                if o.cls in EMERGENCY_CLASSES:
                    ahead = sum(1 for _, inf in lst[:i] if inf["speed"] < self.stop_speed)
                    tracks.append(EmergencyTrack(
                        tid=o.vid, cls=o.cls, approach=a, eta_s=info["dist_m"] / max(info["speed"], 2.0),
                        queue_ahead=ahead, conf=o.conf, track_age_s=t - self.first_seen.setdefault(o.vid, t)))
            approaches[a] = ApproachObs(a, queue=queue, wait_s=wait, capacity=self.capacity.get(a, 12))
        return JunctionObs(t=t, approaches=approaches, emergencies=tuple(tracks), visibility=visibility)

    def apply(self, decision: Decision) -> None:
        st = self.carla.TrafficLightState
        green = set(self.spec["phases"][str(decision.phase)])
        for a, tls in self.lights.items():
            state = (st.Yellow if decision.yellow else st.Green) if a in green else st.Red
            for tl in tls:
                tl.set_state(state)
