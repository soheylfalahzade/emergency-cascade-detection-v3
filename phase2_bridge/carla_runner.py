"""Closed-loop CARLA runner for ONE junction (client on Windows, venv carla_venv).

Policies: fixed | preempt (fixed + naive preemption) | maxpressure | fuzzy (our controller).
Every result line is tagged source=CARLA. Metrics come from CARLA ground truth, not from the
(possibly noisy) observations the controller sees. Run in CARLA 0.9.16 on one junction (Town05_Opt).

Known limits: two phases only, no protected left turn, no separate all-red (controller yellow only),
other junctions keep their own CARLA light programs, Traffic Manager is not bit-exact deterministic.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from typing import Dict, Optional

import carla

from carla_adapter import CarlaJunctionIO, build_config, load_spec, visibility_from_weather, wrap_deg
from junction_controller import FuzzyJunctionController
from junction_model import EMERGENCY_CLASSES, Decision
from perception_noise import NoiseConfig, PerceptionNoise

EXCLUDE = ("ambulance", "police", "firetruck", "carlacola", "cybertruck", "european_hgv", "fusorosa")


# ----------------------------------------------------------------------------- policies
class _Machine:
    def __init__(self, yellow: float = 3.0, phase0: int = 0) -> None:
        self.yellow, self.phase, self.t, self.yel, self.pend = yellow, phase0, 0.0, 0.0, None

    def advance(self, dt: float, want: Optional[int]):
        if self.yel > 0.0:
            self.yel -= dt
            if self.yel <= 1e-9:
                self.phase, self.pend, self.yel, self.t = self.pend, None, 0.0, 0.0
                return self.phase, False
            return self.phase, True
        self.t += dt
        if want is not None and want != self.phase:
            self.pend, self.yel = want, self.yellow
            return self.phase, True
        return self.phase, False


class FixedTime(_Machine):
    def __init__(self, green: float = 30.0, **kw) -> None:
        super().__init__(**kw)
        self.green = green

    def _cycle(self, dt: float) -> Optional[int]:
        return 1 - self.phase if (self.yel <= 0.0 and self.t + dt >= self.green) else None

    def step(self, obs, dt: float) -> Decision:
        ph, y = self.advance(dt, self._cycle(dt))
        return Decision(obs.t, ph, y, "fixed", "", {}, (), ())


class FixedPreempt(FixedTime):
    def __init__(self, phase_of: Dict[str, int], **kw) -> None:
        super().__init__(**kw)
        self.phase_of = phase_of

    def step(self, obs, dt: float) -> Decision:
        want_em = None
        for tr in obs.emergencies:
            if tr.conf >= 0.6 and tr.track_age_s >= 1.0 and tr.cls in EMERGENCY_CLASSES:
                want_em = self.phase_of[tr.approach]
                break
        reason, want = "", None
        if want_em is not None:
            reason = f"emergency[preempt]: phase {want_em}"
            if want_em == self.phase and self.yel <= 0.0:
                self.t = min(self.t, self.green - 2.0 * dt)
            elif self.yel <= 0.0 and self.t >= 4.0:
                want = want_em
        else:
            want = self._cycle(dt)
        ph, y = self.advance(dt, want)
        return Decision(obs.t, ph, y, "fixed+preempt", reason, {}, (), ())


class MaxPressure(_Machine):
    def __init__(self, phases: Dict[int, tuple], min_green: float = 10.0, max_green: float = 60.0, **kw) -> None:
        super().__init__(**kw)
        self.phases, self.min_green, self.max_green = phases, min_green, max_green

    def step(self, obs, dt: float) -> Decision:
        pr = {p: float(sum(obs.approaches[a].queue for a in apps if a in obs.approaches))
              for p, apps in self.phases.items()}
        want = None
        if self.yel <= 0.0:
            cur, oth = self.phase, 1 - self.phase
            if self.t + dt >= self.max_green or (self.t + dt >= self.min_green and pr[oth] > pr[cur]):
                want = oth
        ph, y = self.advance(dt, want)
        return Decision(obs.t, ph, y, "maxpressure", "", pr, (), ())


class FuzzyPolicy:
    def __init__(self, cfg) -> None:
        self.ctl = FuzzyJunctionController(cfg, phase0=0)

    def step(self, obs, dt: float) -> Decision:
        return self.ctl.step(obs, dt)


def make_policy(name: str, spec: dict, cfg):
    phases = {int(k): tuple(v) for k, v in spec["phases"].items()}
    phase_of = {a: p for p, apps in phases.items() for a in apps}
    if name == "fixed":
        return FixedTime()
    if name == "preempt":
        return FixedPreempt(phase_of)
    if name == "maxpressure":
        return MaxPressure(phases)
    if name == "fuzzy":
        return FuzzyPolicy(cfg)
    raise SystemExit("unknown policy: " + name)


# ----------------------------------------------------------------------------- helpers
def make_weather(name: str):
    if name == "Fog":
        w = carla.WeatherParameters.ClearNoon
        w.fog_density, w.fog_distance = 70.0, 10.0
        return w
    return getattr(carla.WeatherParameters, name)


def amb_spawn_tf(world, spec: dict, app: str, dist: float):
    sx, sy = spec["approaches"][app]["stop_xy"][0]
    wp = world.get_map().get_waypoint(carla.Location(sx, sy, 0.0), project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    prev = wp.previous(dist)
    if not prev:
        raise SystemExit("no waypoint %.0f m upstream of %s" % (dist, app))
    tf = prev[0].transform
    if abs(wrap_deg(tf.rotation.yaw - spec["approaches"][app]["yaw"])) > 30.0:
        print("WARN: ambulance spawn yaw %.0f differs from approach yaw %.0f"
              % (tf.rotation.yaw, spec["approaches"][app]["yaw"]))
    tf.location.z += 0.5
    return tf


def resolve_lights(world, spec: dict) -> dict:
    """Light actor ids change after reload_world; re-find each approach's light by its stop-line position."""
    stops = {}
    for tl in list(world.get_actors().filter("traffic.traffic_light*")):
        stops[tl.id] = [(w.transform.location.x, w.transform.location.y) for w in tl.get_stop_waypoints()]
    out = copy.deepcopy(spec)
    for a, info in out["approaches"].items():
        best, best_d = None, 1e9
        for tid, pts in stops.items():
            if not pts:
                continue
            d = sum(min(math.hypot(px - sx, py - sy) for px, py in pts) for sx, sy in info["stop_xy"]) \
                / len(info["stop_xy"])
            if d < best_d:
                best, best_d = tid, d
        if best is None or best_d > 3.0:
            raise SystemExit("no traffic light with stop line near %s (best %.1f m); re-run inspect" % (a, best_d))
        print("%s: spec light %s -> light %d (stop-line mismatch %.2f m)" % (a, info["lights"], best, best_d))
        info["lights"] = [best]
    return out


def run_one(client, spec: dict, policy_name: str, seed: int, args) -> dict:
    t_wall = time.time()
    cfg = build_config(spec)
    cur = client.get_world().get_map().name
    world = client.load_world(args.map) if not cur.endswith(args.map) else client.reload_world()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds, s.no_rendering_mode = True, args.dt, args.no_render
    world.apply_settings(s)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    rng = random.Random(seed)
    actors, io, amb, vo = [], None, None, None
    try:
        spec = resolve_lights(world, spec)
        pts = [p for info in spec["approaches"].values() for p in info["stop_xy"]]
        cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

        # background traffic
        bps = sorted([b for b in world.get_blueprint_library().filter("vehicle.*")
                      if int(b.get_attribute("number_of_wheels")) == 4
                      and not any(k in b.id for k in EXCLUDE)], key=lambda b: b.id)
        def toward(p) -> bool:
            dx, dy = cx - p.location.x, cy - p.location.y
            n, yw = math.hypot(dx, dy), math.radians(p.rotation.yaw)
            return n > 1e-6 and (dx * math.cos(yw) + dy * math.sin(yw)) / n > 0.85

        sps = [p for p in world.get_map().get_spawn_points()
               if 40.0 <= math.hypot(p.location.x - cx, p.location.y - cy) <= 180.0
               and (args.any_spawn or toward(p))]
        print("spawn points used: %d (any_spawn=%s)" % (len(sps), args.any_spawn))
        sps.sort(key=lambda p: (round(p.location.x, 1), round(p.location.y, 1)))
        rng.shuffle(sps)
        n_spawned = 0
        for sp in sps:
            if n_spawned >= args.n_traffic:
                break
            bp = rng.choice(bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
            v = world.try_spawn_actor(bp, sp)
            if v is None:
                continue
            v.set_autopilot(True, args.tm_port)
            actors.append(v)
            n_spawned += 1
        world.tick()
        n_inflow, next_spawn = 0, 0.0
        coll = []
        prev_dec, red_until, red_steps = None, -1.0, 0

        noise = None
        if args.p_miss > 0 or args.p_miss_emerg > 0 or args.p_miss_far > 0 or args.p_emerg_as_car > 0 or args.fp_per_min > 0:
            noise = PerceptionNoise(NoiseConfig(p_miss=args.p_miss, p_miss_emerg=args.p_miss_emerg, p_miss_far=args.p_miss_far, far_m=args.far_m, p_emerg_as_car=args.p_emerg_as_car,
                                                fp_emerg_per_min=args.fp_per_min, hold_s=args.hold_s, seed=seed))
        vis = 1.0
        if args.weather:
            world.set_weather(make_weather(args.weather))
            if args.use_visibility:
                vis = visibility_from_weather(world.get_weather())
        if args.hybrid_port:   # PATCH_HYBRID_VISION: detector on a camera frame supplies the emergency class
            import queue as _q
            import numpy as _np
            from carla_capture import cam_matrix
            from vision_bridge import DetClient, VisionObserver
            _tgt = spec["approaches"][args.amb_approach]
            _opp = [a_ for a_, i_ in spec["approaches"].items() if abs(wrap_deg(i_["yaw"] - _tgt["yaw"] - 180.0)) < 30.0]
            _oi = spec["approaches"][_opp[0]]
            _ox = sum(p[0] for p in _oi["stop_xy"]) / len(_oi["stop_xy"])
            _oy = sum(p[1] for p in _oi["stop_xy"]) / len(_oi["stop_xy"])
            _bp = world.get_blueprint_library().find("sensor.camera.rgb")
            _bp.set_attribute("image_size_x", "1280"); _bp.set_attribute("image_size_y", "720"); _bp.set_attribute("fov", str(args.fov))
            _cam = world.spawn_actor(_bp, carla.Transform(carla.Location(_ox, _oy, args.cam_height),
                                                          carla.Rotation(pitch=args.cam_pitch, yaw=_oi["yaw"])))
            actors.append(_cam)
            _qq = _q.Queue()
            _cam.listen(_qq.put)
            vo = VisionObserver(world, carla, _cam, _qq, DetClient(args.det_host, args.hybrid_port), cam_matrix(1280, 720, args.fov),
                                1280, 720, target_lane=args.amb_approach, det_conf=args.det_conf, warmup=args.warmup,
                                log_path=args.det_log)
            noise = vo
        io = CarlaJunctionIO(world, spec, noise)
        io.take_control()
        pol = make_policy(policy_name, spec, cfg)
        world.get_spectator().set_transform(carla.Transform(carla.Location(cx, cy, 70.0),
                                                            carla.Rotation(pitch=-90.0)))

        info = spec["approaches"][args.amb_approach]
        fwd = (math.cos(math.radians(info["yaw"])), math.sin(math.radians(info["yaw"])))
        mid = (sum(p[0] for p in info["stop_xy"]) / len(info["stop_xy"]),
               sum(p[1] for p in info["stop_xy"]) / len(info["stop_xy"]))
        spc = max(1, round(args.control_dt / args.dt))
        ctl_dt = spc * args.dt
        n_ctl = int(args.duration / ctl_dt)
        amb_tries, amb_t0, crossed, t_cross, amb_stopped = 0, None, False, None, 0.0
        amb_t_eff = args.amb_t + random.Random(seed + 12345).uniform(0.0, args.amb_jitter)
        stop_t, cont, seen = defaultdict(float), defaultdict(float), set()
        near, counted, crossed_n = set(), set(), 0
        box_jam_s, box_stop_max = 0.0, 0
        max_wait = switches = false_pre = 0.0
        stopped_sum, win_steps, modes, prev_y = 0.0, 0, defaultdict(int), False

        for k in range(n_ctl):
            t = k * ctl_dt
            if amb is None and t >= amb_t_eff and amb_tries < 10:
                amb = world.try_spawn_actor(world.get_blueprint_library().find("vehicle.ford.ambulance"),
                                            amb_spawn_tf(world, spec, args.amb_approach, args.amb_dist))
                if amb is None:
                    amb_tries += 1
                else:
                    amb.set_autopilot(True, args.tm_port)
                    tm.auto_lane_change(amb, False)
                    tm.ignore_lights_percentage(amb, 0.0)
                    try:
                        tm.set_route(amb, ["Straight"] * 4)
                    except Exception as e:
                        print("WARN set_route:", e)
                    try:
                        amb.set_light_state(carla.VehicleLightState(
                            carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2))
                    except Exception as e:
                        print("WARN siren lights:", e)
                    actors.append(amb)
                    if args.trace:
                        cs = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),
                                               carla.Transform(), attach_to=amb)
                        cs.listen(lambda e: coll.append((e.other_actor.type_id.split(".")[-1],
                                                         round(e.transform.location.x, 1),
                                                         round(e.transform.location.y, 1))))
                        actors.append(cs)
                    amb_t0 = t
                    world.tick()

            obs = io.read_obs(t, vis)
            dec = pol.step(obs, ctl_dt)
            if args.all_red > 0.0:
                if prev_dec is not None and prev_dec.yellow and not dec.yellow and dec.phase != prev_dec.phase:
                    red_until = t + args.all_red
                if t < red_until:
                    for tls in io.lights.values():
                        for tl in tls:
                            tl.set_state(carla.TrafficLightState.Red)
                    red_steps += 1
                else:
                    io.apply(dec)
            else:
                io.apply(dec)
            prev_dec = dec
            if args.hud:
                world.debug.draw_string(carla.Location(cx, cy, 12.0), "%s t=%3.0fs ph=%d%s %s" % (
                    policy_name, t, dec.phase, "Y" if dec.yellow else "", dec.mode), draw_shadow=False,
                    color=carla.Color(255, 255, 0), life_time=ctl_dt + 0.05)

            # ground-truth metrics
            in_win = t >= args.warmup
            present, now_assigned, n_stopped = False, set(), 0
            box_n = box_stop = 0
            amb_v = amb_d = amb_xy = None
            others = []
            for v in list(world.get_actors().filter("vehicle.*")):
                tf, vel = v.get_transform(), v.get_velocity()
                speed = math.hypot(vel.x, vel.y)
                others.append((v.id, v.type_id, tf.location.x, tf.location.y, speed))
                if abs(tf.location.x - cx) <= 12.0 and abs(tf.location.y - cy) <= 12.0:
                    box_n += 1
                    box_stop += 1 if speed < 0.3 else 0
                a, d = io._assign(tf.location.x, tf.location.y, tf.rotation.yaw)
                if amb is not None and v.id == amb.id:
                    present = a is not None
                    proj = (mid[0] - tf.location.x) * fwd[0] + (mid[1] - tf.location.y) * fwd[1]
                    amb_v, amb_d, amb_xy = speed, proj, (tf.location.x, tf.location.y)
                    if not crossed:
                        if proj <= 0.0:
                            crossed, t_cross = True, t
                        elif speed < io.stop_speed:
                            amb_stopped += ctl_dt
                    continue
                if a is not None and d <= 12.0:
                    near.add(v.id)
                elif a is None and v.id in near and v.id not in counted:
                    counted.add(v.id)
                    if in_win:
                        crossed_n += 1
                if a is None or not in_win:
                    continue
                now_assigned.add(v.id)
                seen.add(v.id)
                if d <= io.queue_len_m and speed < io.stop_speed:
                    n_stopped += 1
                    stop_t[v.id] += ctl_dt
                    cont[v.id] += ctl_dt
                    max_wait = max(max_wait, cont[v.id])
                else:
                    cont[v.id] = 0.0
            for vid in list(cont):
                if vid not in now_assigned:
                    cont[vid] = 0.0
            if args.trace:
                near_txt = ""
                if amb_xy is not None and not crossed and amb_v < 0.3:
                    cand = [(math.hypot(o[2] - amb_xy[0], o[3] - amb_xy[1]), o) for o in others if o[0] != amb.id]
                    if cand:
                        dmin, o = min(cand, key=lambda c: c[0])
                        near_txt = " nearest=%s(%s) d=%.1f v=%.1f" % (o[0], o[1].split(".")[-1], dmin, o[4])
                print("TR t=%.0f ph=%d%s %s amb_v=%s amb_d=%s box=%d/stopped=%d coll=%d%s" % (
                    t, dec.phase, "Y" if dec.yellow else "", dec.mode,
                    "-" if amb_v is None else "%.1f" % amb_v, "-" if amb_d is None else "%.0f" % amb_d,
                    box_n, box_stop, len(coll), near_txt))
            if in_win:
                win_steps += 1
                box_stop_max = max(box_stop_max, box_stop)
                if box_stop >= 8:
                    box_jam_s += ctl_dt
                stopped_sum += n_stopped
                modes[dec.mode] += 1
                if dec.yellow and not prev_y:
                    switches += 1
                if dec.reason.startswith("emergency[") and not present:
                    false_pre += ctl_dt
            prev_y = dec.yellow
            for i in range(spc):
                if i == spc - 1 and args.inflow > 0 and sps:   # spawn just before the last tick so the
                    while t >= next_spawn:                      # next read sees a valid transform
                        next_spawn += 1.0 / args.inflow
                        bp = rng.choice(bps)
                        if bp.has_attribute("color"):
                            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
                        v = world.try_spawn_actor(bp, rng.choice(sps))
                        if v is not None:
                            v.set_autopilot(True, args.tm_port)
                            actors.append(v)
                            n_inflow += 1
                world.tick()

        win_s = max(win_steps * ctl_dt, 1e-9)
        return dict(
            source="CARLA", map=args.map, spec=spec.get("name"), policy=policy_name, seed=seed,
            duration_s=args.duration, warmup_s=args.warmup, control_dt=ctl_dt, n_traffic=n_spawned,
            n_inflow=n_inflow, all_red_s=args.all_red, all_red_steps=red_steps,
            noise=dict(p_miss=args.p_miss, p_miss_emerg=args.p_miss_emerg, p_miss_far=args.p_miss_far, far_m=args.far_m, p_emerg_as_car=args.p_emerg_as_car, fp_per_min=args.fp_per_min),
            weather=args.weather, visibility=vis,
            amb_approach=args.amb_approach, amb_t=amb_t_eff, amb_spawned=amb is not None, amb_crossed=crossed,
            amb_stopped_s=amb_stopped,
            amb_time_to_cross_s=(t_cross - amb_t0) if (crossed and amb_t0 is not None) else None,
            norm_stopped_s_per_veh=(sum(stop_t.values()) / len(seen)) if seen else None,
            norm_stopped_veh_mean=stopped_sum / max(win_steps, 1), norm_n_veh=len(seen),
            norm_crossed=crossed_n, norm_crossed_per_min=crossed_n / (win_s / 60.0),
            box_jam_s=box_jam_s, box_stop_max=box_stop_max,
            max_wait_s=max_wait, switches_per_h=switches / win_s * 3600.0, false_preempt_s=false_pre,
            modes={k: v * ctl_dt for k, v in modes.items()}, wall_s=time.time() - t_wall,
            **(vo.stats() if vo is not None else {}))
    finally:
        try:
            if vo is not None:
                vo.close()
            if io is not None:
                io.release_control()
            client.apply_batch([carla.command.DestroyActor(a) for a in actors])
            world.tick()
            s = world.get_settings()
            s.synchronous_mode, s.fixed_delta_seconds, s.no_rendering_mode = False, None, False
            world.apply_settings(s)
            tm.set_synchronous_mode(False)
        except Exception as e:
            print("WARN cleanup:", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="junction_spec_town05_idx14.json")
    ap.add_argument("--map", default="Town05_Opt")
    ap.add_argument("--policies", default="fixed,preempt,maxpressure,fuzzy")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--duration", type=float, default=160.0)
    ap.add_argument("--warmup", type=float, default=20.0)
    ap.add_argument("--amb-t", type=float, default=40.0)
    ap.add_argument("--amb-jitter", type=float, default=66.0)
    ap.add_argument("--any-spawn", action="store_true")
    ap.add_argument("--amb-approach", default="A0")
    ap.add_argument("--amb-dist", type=float, default=30.0)
    ap.add_argument("--n-traffic", type=int, default=40)
    ap.add_argument("--inflow", type=float, default=0.5, help="new background vehicles per second")
    ap.add_argument("--control-dt", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--p-miss", type=float, default=0.0)
    ap.add_argument("--p-emerg-as-car", type=float, default=0.0)
    ap.add_argument("--fp-per-min", type=float, default=0.0)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--p-miss-far", type=float, default=0.0)
    ap.add_argument("--p-miss-emerg", type=float, default=-1.0, help="miss chance for emergency classes (<0: same as --p-miss)")
    ap.add_argument("--far-m", type=float, default=60.0)
    ap.add_argument("--weather", default="", help="CARLA preset name (ClearNoon, HardRainNoon, ClearNight, Fog, ...)")
    ap.add_argument("--use-visibility", action="store_true", help="feed the (uncalibrated) weather visibility proxy to the controller")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--out", default="results_carla.jsonl")
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--all-red", type=float, default=0.0, help="seconds of all-red after each yellow")
    ap.add_argument("--hybrid-port", type=int, default=0, help="detector server port (WSL); 0 = off")
    ap.add_argument("--det-host", default="127.0.0.1")
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--cam-height", type=float, default=16.0)
    ap.add_argument("--cam-pitch", type=float, default=-28.0)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--det-log", default="")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    for seed in [int(x) for x in args.seeds.split(",")]:
        for pol in args.policies.split(","):
            res = run_one(client, spec, pol, seed, args)
            line = json.dumps(res)
            print(line)
            with open(args.out, "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
