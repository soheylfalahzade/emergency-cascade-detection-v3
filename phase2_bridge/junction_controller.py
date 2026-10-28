"""Fuzzy junction controller.

Layers: (1) safety state machine (min/max green, mandatory yellow); (2) perception-aware modes
(normal / degraded / static fixed-time); (3) emergency arbitration by minimum expected weighted
delay over service orders, with deterministic tie-breaks; (4) fuzzy pressure for ordinary traffic
with anti-spillback from neighbour messages; (5) advance notices to downstream junctions.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fuzzy_core import TSK, ramp_down, ramp_up, tri
from junction_model import (EMERGENCY_CLASSES, Decision, EmergencyTrack, JunctionConfig,
                            JunctionObs, Message, ModeConfig)

NORMAL, DEGRADED, STATIC = "normal", "degraded", "static"


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def build_pressure_tsk() -> TSK:
    """Phase pressure in [0, 1] from queue, waiting time and downstream occupancy (all normalised)."""
    sets = {
        "queue": {"EMPTY": ramp_down(0.02, 0.1), "LOW": tri(0.02, 0.2, 0.45),
                  "MED": tri(0.3, 0.5, 0.75), "HIGH": ramp_up(0.55, 0.85)},
        "wait": {"SHORT": ramp_down(0.2, 0.5), "MED": tri(0.2, 0.5, 0.8), "LONG": ramp_up(0.5, 0.9)},
        "down": {"FREE": ramp_down(0.3, 0.6), "BUSY": tri(0.3, 0.6, 0.85), "FULL": ramp_up(0.6, 0.9)},
    }
    rules: List[Tuple[Dict[str, str], float]] = [({"queue": "EMPTY"}, 0.0)]
    for q, row in (("LOW", (0.25, 0.15, 0.05)), ("MED", (0.6, 0.4, 0.1)), ("HIGH", (1.0, 0.6, 0.15))):
        for d, val in zip(("FREE", "BUSY", "FULL"), row):
            rules.append(({"queue": q, "down": d}, val))
    for d, val in zip(("FREE", "BUSY", "FULL"), (0.9, 0.7, 0.4)):
        rules.append(({"wait": "LONG", "down": d}, val))
    return TSK(sets, rules, default=0.0)


def build_urgency_tsk() -> TSK:
    """Urgency weight in (0, 1] of one emergency vehicle from ETA, evidence and queue ahead."""
    sets = {
        "eta": {"NEAR": ramp_down(8, 20), "MID": tri(8, 25, 50), "FAR": ramp_up(35, 60)},
        "conf": {"LOW": ramp_down(0.4, 0.6), "MED": tri(0.4, 0.65, 0.9), "HIGH": ramp_up(0.7, 0.9)},
        "queue": {"SHORT": ramp_down(2, 6), "LONG": ramp_up(3, 8)},
    }
    rules: List[Tuple[Dict[str, str], float]] = [
        ({"conf": "HIGH", "eta": "NEAR"}, 1.0), ({"conf": "HIGH", "eta": "MID"}, 0.8),
        ({"conf": "HIGH", "eta": "FAR"}, 0.5), ({"conf": "MED", "eta": "NEAR"}, 0.7),
        ({"conf": "MED", "eta": "MID"}, 0.5), ({"conf": "MED", "eta": "FAR"}, 0.3),
        ({"conf": "LOW"}, 0.1), ({"conf": "HIGH", "queue": "LONG"}, 0.95),
    ]
    return TSK(sets, rules, default=0.1)


class _Dwell:
    """Accumulates time while a condition holds; resets when it does not."""

    def __init__(self) -> None:
        self.t = 0.0

    def update(self, cond: bool, dt: float) -> float:
        self.t = self.t + dt if cond else 0.0
        return self.t


class ModeManager:
    """Normal / degraded / static selection from a smoothed visibility signal, with hysteresis."""

    def __init__(self, cfg: ModeConfig) -> None:
        self.cfg = cfg
        self.mode = NORMAL
        self.smoothed: Optional[float] = None
        self._low, self._rec, self._deg, self._nor = _Dwell(), _Dwell(), _Dwell(), _Dwell()

    def update(self, visibility: float, dt: float) -> str:
        c = self.cfg
        v = _clamp(visibility, 0.0, 1.0)
        if self.smoothed is None:
            self.smoothed = v
        else:
            self.smoothed += (1.0 - math.exp(-dt / c.tau_s)) * (v - self.smoothed)
        vs = self.smoothed
        d_low = self._low.update(vs < c.static_max, dt)
        d_rec = self._rec.update(vs >= c.recover_min, dt)
        d_deg = self._deg.update(vs < c.normal_min, dt)
        d_nor = self._nor.update(vs >= c.normal_min + c.leave_degraded_margin, dt)
        if self.mode != STATIC and d_low >= c.enter_static_s:
            self.mode = STATIC
        elif self.mode == STATIC and d_rec >= c.recover_s:
            self.mode = DEGRADED if vs < c.normal_min else NORMAL
        elif self.mode == NORMAL and d_deg >= c.enter_degraded_s:
            self.mode = DEGRADED
        elif self.mode == DEGRADED and d_nor >= c.leave_degraded_s:
            self.mode = NORMAL
        return self.mode


@dataclass(frozen=True)
class _Demand:
    eta_s: float
    service_s: float
    weight: float
    age_s: float
    tracks: Tuple[str, ...]


class FuzzyJunctionController:
    """One controller per junction. Call step() once per control period."""

    def __init__(self, cfg: JunctionConfig, modes: Optional[ModeConfig] = None,
                 phase0: Optional[int] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.modes = ModeManager(modes or ModeConfig())
        self.phase = min(cfg.phases) if phase0 is None else phase0
        if self.phase not in cfg.phases:
            raise ValueError("phase0 is not a known phase")
        self.t_in_phase = 0.0
        self._yellow_left = 0.0
        self._pending: Optional[int] = None
        self._hold: Optional[int] = None
        self._hold_until = -1.0
        self._link_occ: Dict[str, Tuple[float, float]] = {}
        self._notices: Dict[Tuple[str, str], Message] = {}
        self._sent: Dict[Tuple[str, str], float] = {}
        self._last_state_t = -1e9
        self._approach_phase = {a: p for p, apps in cfg.phases.items() for a in apps}
        self._approach_by_in_link = {link: a for a, link in cfg.in_link.items()}
        self._pressure_tsk = build_pressure_tsk()
        self._urgency_tsk = build_urgency_tsk()

    # ------------------------------------------------------------------ main step
    def step(self, obs: JunctionObs, dt: float) -> Decision:
        cfg = self.cfg
        mode = self.modes.update(obs.visibility, dt)
        self._ingest(obs)
        pressure = self._pressures(obs)
        if mode == STATIC and not cfg.static_allows_emergency:
            self._hold = None
        demands: Dict[int, _Demand] = {}
        confirmed: List[EmergencyTrack] = []
        if mode != STATIC or cfg.static_allows_emergency:
            demands, confirmed = self._demands(obs, mode)
            demands = self._apply_cap(demands)
        plan: Tuple[int, ...] = ()
        emergency = False
        if demands:
            target, plan, reason = self._emergency_target(demands, obs.t)
            emergency = True
        elif self._hold is not None and obs.t < self._hold_until:
            target, reason, emergency = self._hold, "emergency hold (clearance grace)", True
        elif mode == STATIC:
            self._hold = None
            target, reason = self._static_target()
        else:
            self._hold = None
            target, reason = self._fuzzy_target(pressure, mode)
        phase, yellow = self._advance(target, emergency, dt, mode)
        outbox = self._outbox(obs, confirmed, target)
        return Decision(t=obs.t, phase=phase, yellow=yellow, mode=mode, reason=reason,
                        pressure=pressure, plan=plan, outbox=outbox)

    # ------------------------------------------------------------ safety state machine
    def _advance(self, target: int, emergency: bool, dt: float, mode: str) -> Tuple[int, bool]:
        cfg = self.cfg
        if self._yellow_left > 0.0:
            self._yellow_left -= dt
            if self._yellow_left <= 1e-9:
                if self._pending is not None:
                    self.phase = self._pending
                self._pending, self._yellow_left, self.t_in_phase = None, 0.0, 0.0
                return self.phase, False
            return self.phase, True
        self.t_in_phase += dt
        if target == self.phase:
            return self.phase, False
        if emergency:
            min_g = cfg.emerg_min_green_s
        elif mode == STATIC:
            min_g = 0.0
        elif mode == DEGRADED:
            min_g = cfg.degraded_min_green_s
        else:
            min_g = cfg.min_green_s
        if self.t_in_phase >= min_g:
            self._pending, self._yellow_left = target, cfg.yellow_s
            return self.phase, True
        return self.phase, False

    def _current(self) -> Tuple[int, float]:
        if self._yellow_left > 0.0 and self._pending is not None:
            return self._pending, 1e9
        return self.phase, self.t_in_phase

    # ------------------------------------------------------------------ ordinary traffic
    def _downstream(self, approach: str, t: float) -> float:
        link = self.cfg.out_link.get(approach)
        if link is None:
            return 0.0
        rec = self._link_occ.get(link)
        if rec is None or t - rec[0] > self.cfg.link_state_stale_s:
            return 0.0
        return rec[1]

    def _pressures(self, obs: JunctionObs) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for p, apps in self.cfg.phases.items():
            present = [obs.approaches[a] for a in apps if a in obs.approaches]
            cap = sum(a.capacity for a in present)
            qn = _clamp(sum(a.queue for a in present) / cap, 0.0, 1.0) if cap > 0 else 0.0
            wn = _clamp(max((a.wait_s for a in present), default=0.0) / 60.0, 0.0, 1.0)
            dn = max((self._downstream(a, obs.t) for a in apps), default=0.0)
            out[p] = self._pressure_tsk.infer({"queue": qn, "wait": wn, "down": dn})
        return out

    def _fuzzy_target(self, pressure: Dict[int, float], mode: str) -> Tuple[int, str]:
        cfg = self.cfg
        cur, t_in = self._current()
        others = [p for p in sorted(cfg.phases) if p != cur]
        if not others:
            return cur, "single phase"
        best = max(others, key=lambda p: (pressure[p], -p))
        if self._yellow_left <= 0.0 and t_in >= cfg.max_green_s:
            return best, f"max green {cfg.max_green_s:.0f}s reached"
        margin = cfg.degraded_switch_margin if mode == DEGRADED else cfg.switch_margin
        if pressure[best] - pressure[cur] > margin:
            return best, f"fuzzy pressure {pressure[best]:.2f} > {pressure[cur]:.2f}+{margin:.2f}"
        return cur, f"fuzzy keep (pressure {pressure[cur]:.2f})"

    def _static_target(self) -> Tuple[int, str]:
        cur, t_in = self._current()
        order = sorted(self.cfg.phases)
        g = self.cfg.static_green_s[cur]
        if self._yellow_left <= 0.0 and t_in >= g:
            return order[(order.index(cur) + 1) % len(order)], f"static cycle: {g:.0f}s elapsed"
        return cur, "static cycle: hold"

    # ------------------------------------------------------------------ emergencies
    def _notice_tracks(self, obs: JunctionObs) -> List[EmergencyTrack]:
        out: List[EmergencyTrack] = []
        for m in self._notices.values():
            a = self._approach_by_in_link.get(m.link)
            if a is None:
                continue
            ap = obs.approaches.get(a)
            out.append(EmergencyTrack(tid=f"notice:{m.src}:{m.tid}", cls=m.cls, approach=a,
                                      eta_s=max(m.eta_s - (obs.t - m.t), 0.0),
                                      queue_ahead=ap.queue if ap else 0, conf=m.conf * 0.9,
                                      track_age_s=1e3, source="notice"))
        return out

    def _demands(self, obs: JunctionObs, mode: str) -> Tuple[Dict[int, _Demand], List[EmergencyTrack]]:
        cfg = self.cfg
        strict = mode != NORMAL
        conf_min = cfg.degraded_confirm_conf if strict else cfg.confirm_conf
        age_min = cfg.degraded_min_track_age_s if strict else cfg.min_track_age_s
        sensor = list(obs.emergencies)
        have = {(t.approach, t.cls) for t in sensor}
        tracks = sensor + [n for n in self._notice_tracks(obs) if (n.approach, n.cls) not in have]
        by_phase: Dict[int, List[Tuple[EmergencyTrack, float]]] = {}
        confirmed: List[EmergencyTrack] = []
        for tr in tracks:
            p = self._approach_phase.get(tr.approach)
            if p is None or tr.cls not in EMERGENCY_CLASSES:
                continue
            if tr.conf < conf_min or tr.track_age_s < age_min:
                continue
            u = self._urgency_tsk.infer({"eta": _clamp(tr.eta_s, 0.0, 120.0), "conf": _clamp(tr.conf, 0.0, 1.0),
                                         "queue": float(min(tr.queue_ahead, 15))})
            by_phase.setdefault(p, []).append((tr, u))
            if tr.source == "sensor":
                confirmed.append(tr)
        demands: Dict[int, _Demand] = {}
        for p, lst in by_phase.items():
            us = [u for _, u in lst]
            top = max(us)
            q_ahead = max(t.queue_ahead for t, _ in lst)
            demands[p] = _Demand(eta_s=min(max(t.eta_s, 0.0) for t, _ in lst),
                                 service_s=q_ahead * cfg.headway_s + cfg.pass_buffer_s,
                                 weight=min(1.5, top + 0.25 * (sum(us) - top)),
                                 age_s=max(t.track_age_s for t, _ in lst),
                                 tracks=tuple(t.tid for t, _ in lst))
        return demands, confirmed

    def _apply_cap(self, demands: Dict[int, _Demand]) -> Dict[int, _Demand]:
        if self._yellow_left <= 0.0 and self.phase in demands and self.t_in_phase >= self.cfg.emerg_max_green_s:
            return {p: d for p, d in demands.items() if p != self.phase}
        return demands

    def _switch_cost(self, cur: int, nxt: int, t_in: float) -> float:
        if nxt == cur:
            return 0.0
        return max(0.0, self.cfg.emerg_min_green_s - t_in) + self.cfg.yellow_s

    def _plan_costs(self, demands: Dict[int, _Demand], cur: int,
                    t_in: float) -> Dict[int, Tuple[float, Tuple[int, ...]]]:
        """For each possible first phase: (minimum expected weighted delay, best service order)."""
        phases = sorted(demands, key=lambda p: (-demands[p].weight, p))[:4]
        best: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
        for perm in itertools.permutations(phases):
            t = self._switch_cost(cur, perm[0], t_in)
            total = 0.0
            for i, p in enumerate(perm):
                d = demands[p]
                total += d.weight * max(0.0, t - d.eta_s)
                t += d.service_s
                if i + 1 < len(perm):
                    t += self.cfg.yellow_s
            if perm[0] not in best or total < best[perm[0]][0] - 1e-9:
                best[perm[0]] = (total, perm)
        return best

    def _emergency_target(self, demands: Dict[int, _Demand], now: float) -> Tuple[int, Tuple[int, ...], str]:
        cfg = self.cfg
        cur, t_in = self._current()
        best = self._plan_costs(demands, cur, t_in)
        bestc = min(c for c, _ in best.values())
        cands = [p for p, (c, _) in best.items() if c <= bestc + cfg.tie_eps]
        if self._hold in best and best[self._hold][0] <= bestc + cfg.replan_margin:
            first, how = self._hold, "hold"
        elif len(cands) == 1:
            first, how = cands[0], "min-delay"
        else:
            first = max(cands, key=lambda p: (p == cur, demands[p].age_s, -p))
            how = "tie-break"
        cost, perm = best[first]
        self._hold, self._hold_until = first, now + cfg.clear_grace_s
        return first, perm, f"emergency[{how}]: phase {first} first, plan {perm}, expected weighted delay {cost:.1f}"

    # ------------------------------------------------------------------ neighbours
    def _ingest(self, obs: JunctionObs) -> None:
        for m in obs.inbox:
            if m.kind == "link_state":
                self._link_occ[m.link] = (m.t, _clamp(m.occupancy, 0.0, 1.0))
            elif m.kind == "emergency":
                self._notices[(m.tid, m.link)] = m
        for k, m in list(self._notices.items()):
            if m.eta_s - (obs.t - m.t) < -self.cfg.notice_expire_s:
                del self._notices[k]

    def _outbox(self, obs: JunctionObs, confirmed: List[EmergencyTrack], target: int) -> Tuple[Message, ...]:
        cfg = self.cfg
        out: List[Message] = []
        if obs.t - self._last_state_t >= cfg.state_msg_period_s:
            self._last_state_t = obs.t
            for a, link in cfg.in_link.items():
                up = cfg.upstream_of.get(link)
                ap = obs.approaches.get(a)
                if up and ap and ap.capacity > 0:
                    out.append(Message("link_state", cfg.name, up, obs.t, link,
                                       occupancy=_clamp(ap.queue / ap.capacity, 0.0, 1.0)))
        for tr in confirmed:
            if self._approach_phase.get(tr.approach) != target:
                continue
            link = cfg.out_link.get(tr.approach)
            dst = cfg.downstream_of.get(link) if link else None
            if not dst:
                continue
            key = (tr.tid, link)
            if obs.t - self._sent.get(key, -1e9) < 5.0:
                continue
            self._sent[key] = obs.t
            eta = tr.eta_s + tr.queue_ahead * cfg.headway_s + cfg.link_travel_s.get(link, 10.0)
            out.append(Message("emergency", cfg.name, dst, obs.t, link, eta_s=eta, cls=tr.cls,
                               conf=tr.conf, tid=tr.tid))
        return tuple(out)
