"""Step 4 of the Phase-2 plan: record camera frames + ground truth from CARLA (Town05_Opt, one junction)
so the Phase-1 detectors can be run on them (in WSL) and real miss / confusion / false-alarm rates measured.

One weather per process (a fresh process per run; multi-run processes were unreliable).
Writes <out>/<weather>_s<seed>/img_XXXXXX.png and gt.jsonl (one JSON line per frame).
GT per vehicle: id, cls, bbox [x1,y1,x2,y2] in pixels (clipped), dist_m, speed, inst_px (pixels of that vehicle in the
instance-segmentation image; -1 if disabled). Axis remap: UE4 (x fwd, y right, z up) -> camera (y, -z, x).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import time

import numpy as np

import carla
from carla_adapter import cls_of, load_spec, visibility_from_weather, wrap_deg

EXCLUDE = ("ambulance", "police", "firetruck", "carlacola", "cybertruck", "european_hgv", "fusorosa")


def cam_matrix(w: int, h: int, fov: float) -> np.ndarray:
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def project(pts_world: np.ndarray, world_to_cam: np.ndarray, K: np.ndarray):
    """pts_world: (N,3). Returns (u, v, depth) arrays; depth<=0 means behind the camera."""
    hom = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1).T      # 4xN
    pc = world_to_cam @ hom                                                         # UE4 camera frame
    std = np.stack([pc[1], -pc[2], pc[0]])                                          # (right, down, forward)
    z = std[2]
    zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K[0, 0] * std[0] / zs + K[0, 2]
    v = K[1, 1] * std[1] / zs + K[1, 2]
    return u, v, z


def signal_plan(t: float, phases: dict, g: float, y: float, r: float):
    """Two-phase fixed program. Returns (phase_key, {approach: 'Green'|'Yellow'|'Red'})."""
    keys = sorted(phases)
    cyc = g + y + r
    cur = keys[int(t // cyc) % len(keys)]
    loc = t % cyc
    out = {}
    for p in keys:
        for a in phases[p]:
            if p != cur:
                out[a] = "Red"
            elif loc < g:
                out[a] = "Green"
            elif loc < g + y:
                out[a] = "Yellow"
            else:
                out[a] = "Red"
    return cur, out


def resolve_lights(world, spec: dict) -> dict:
    """Light actor ids change after reload_world; re-find each approach's light by its stop-line position."""
    stops = {}
    for tl in list(world.get_actors().filter("traffic.traffic_light*")):
        stops[tl.id] = [(w.transform.location.x, w.transform.location.y) for w in tl.get_stop_waypoints()]
    out = json.loads(json.dumps(spec))
    for a, info in out["approaches"].items():
        best, best_d = None, 1e9
        for tid, pts in stops.items():
            if not pts:
                continue
            d = sum(min(math.hypot(px - sx, py - sy) for px, py in pts) for sx, sy in info["stop_xy"]) / len(info["stop_xy"])
            if d < best_d:
                best, best_d = tid, d
        if best is None or best_d > 3.0:
            raise SystemExit("no traffic light with stop line near %s (best %.1f m)" % (a, best_d))
        info["lights"] = [best]
    return out


def make_weather(name: str):
    if name == "Fog":
        w = carla.WeatherParameters.ClearNoon
        w.fog_density, w.fog_distance = 70.0, 10.0
        return w
    return getattr(carla.WeatherParameters, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="junction_spec_town05_idx14.json")
    ap.add_argument("--map", default="Town05_Opt")
    ap.add_argument("--weather", default="ClearNoon")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-approach", default="A0", help="approach whose oncoming traffic the camera looks at")
    ap.add_argument("--cam-height", type=float, default=7.0)
    ap.add_argument("--cam-pitch", type=float, default=-12.0)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--duration", type=float, default=140.0)
    ap.add_argument("--warmup", type=float, default=20.0)
    ap.add_argument("--every", type=int, default=20, help="capture every N ticks (dt 0.05 -> 20 = 1 s)")
    ap.add_argument("--max-dist", type=float, default=120.0)
    ap.add_argument("--min-area", type=float, default=150.0)
    ap.add_argument("--n-traffic", type=int, default=20)
    ap.add_argument("--inflow", type=float, default=0.25)
    ap.add_argument("--emerg-every", type=float, default=12.0)
    ap.add_argument("--emerg-dists", default="30,35,25", help="metres upstream of the target stop line to try when spawning")
    ap.add_argument("--no-inst", action="store_true")
    ap.add_argument("--signal", choices=["fixed", "default"], default="fixed",
                    help="fixed: we freeze the 4 lights and run a 2-phase fixed program; default: leave CARLA's own program")
    ap.add_argument("--green", type=float, default=20.0)
    ap.add_argument("--yellow", type=float, default=3.0)
    ap.add_argument("--allred", type=float, default=2.0)
    ap.add_argument("--out", default="captures")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    args = ap.parse_args()
    dt = 0.05

    spec = load_spec(args.spec)
    tgt = spec["approaches"][args.target_approach]
    opp = [a for a, i in spec["approaches"].items()
           if abs(wrap_deg(i["yaw"] - tgt["yaw"] - 180.0)) < 30.0]
    if not opp:
        raise SystemExit("no opposite approach found for " + args.target_approach)
    oinfo = spec["approaches"][opp[0]]
    ox = sum(p[0] for p in oinfo["stop_xy"]) / len(oinfo["stop_xy"])
    oy = sum(p[1] for p in oinfo["stop_xy"]) / len(oinfo["stop_xy"])
    pts = [p for i in spec["approaches"].values() for p in i["stop_xy"]]
    cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    cur = client.get_world().get_map().name
    world = client.load_world(args.map) if not cur.endswith(args.map) else client.reload_world()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, dt
    world.apply_settings(s)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    rng = random.Random(args.seed)
    wname = args.weather
    world.set_weather(make_weather(wname))
    vis = visibility_from_weather(world.get_weather())

    outdir = os.path.join(args.out, "%s_s%d" % (wname, args.seed))
    os.makedirs(outdir, exist_ok=True)
    gt_f = open(os.path.join(outdir, "gt.jsonl"), "w", encoding="utf-8")
    actors, emerg = [], []
    try:
        lib = world.get_blueprint_library()
        bps = sorted([b for b in lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4
                      and not any(k in b.id for k in EXCLUDE)], key=lambda b: b.id)

        def toward(p) -> bool:
            dx, dy = cx - p.location.x, cy - p.location.y
            n, yw = math.hypot(dx, dy), math.radians(p.rotation.yaw)
            return n > 1e-6 and (dx * math.cos(yw) + dy * math.sin(yw)) / n > 0.85

        sps = [p for p in world.get_map().get_spawn_points()
               if 40.0 <= math.hypot(p.location.x - cx, p.location.y - cy) <= 180.0 and toward(p)]
        sps.sort(key=lambda p: (round(p.location.x, 1), round(p.location.y, 1)))
        rng.shuffle(sps)

        def spawn_car(sp):
            bp = rng.choice(bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
            v = world.try_spawn_actor(bp, sp)
            if v is not None:
                v.set_autopilot(True, args.tm_port)
                actors.append(v)
            return v

        n0 = 0
        for sp in sps:
            if n0 >= args.n_traffic:
                break
            n0 += 1 if spawn_car(sp) is not None else 0

        # camera: at the opposite approach's stop line, looking along that approach's heading (toward the target's oncoming traffic)
        cam_tf = carla.Transform(carla.Location(ox, oy, args.cam_height),
                                 carla.Rotation(pitch=args.cam_pitch, yaw=oinfo["yaw"]))
        bp = lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(args.width))
        bp.set_attribute("image_size_y", str(args.height))
        bp.set_attribute("fov", str(args.fov))
        cam = world.spawn_actor(bp, cam_tf)
        actors.append(cam)
        q_rgb: "queue.Queue" = queue.Queue()
        cam.listen(q_rgb.put)
        q_inst = None
        if not args.no_inst:
            bpi = lib.find("sensor.camera.instance_segmentation")
            bpi.set_attribute("image_size_x", str(args.width))
            bpi.set_attribute("image_size_y", str(args.height))
            bpi.set_attribute("fov", str(args.fov))
            cami = world.spawn_actor(bpi, cam_tf)
            actors.append(cami)
            q_inst = queue.Queue()
            cami.listen(q_inst.put)
        world.get_spectator().set_transform(cam_tf)
        K = cam_matrix(args.width, args.height, args.fov)

        lights, cur_state = {}, {}
        phases_d = {int(k): tuple(v) for k, v in spec["phases"].items()}
        if args.signal == "fixed":
            spec_l = resolve_lights(world, spec)
            for a_, info_ in spec_l["approaches"].items():
                lights[a_] = [world.get_actor(i_) for i_ in info_["lights"]]
                for tl in lights[a_]:
                    tl.freeze(True)

        # emergency vehicle schedule: classes x approaches, spawn upstream of each approach's stop line
        eclasses = [("*ambulance*", "ambulance"), ("*firetruck*", "firetruck"), ("*police*", "police")]
        apps = sorted(spec["approaches"])
        e_idx, e_fail, next_e, next_in, frames, n_in, inst_ok, inst_tot = 0, 0, args.warmup, 0.0, 0, 0, 0, 0
        e_idx = args.seed % len(eclasses)
        e_start = e_idx
        n_ticks = int(args.duration / dt)
        t0 = time.time()
        for k in range(n_ticks):
            t = k * dt
            if t >= next_in and args.inflow > 0 and sps:
                next_in = t + 1.0 / args.inflow
                if spawn_car(rng.choice(sps)) is not None:
                    n_in += 1
            if args.signal == "fixed" and k % 20 == 0:
                _, plan = signal_plan(t, phases_d, args.green, args.yellow, args.allred)
                for a_, name_ in plan.items():
                    if cur_state.get(a_) != name_:
                        cur_state[a_] = name_
                        for tl in lights.get(a_, []):
                            tl.set_state(getattr(carla.TrafficLightState, name_))
            if args.emerg_every > 0 and t >= next_e:
                pat, _ = eclasses[e_idx % len(eclasses)]
                cand = sorted(lib.filter(pat), key=lambda b: b.id)
                ev = None
                if cand:
                    sx, sy = tgt["stop_xy"][e_idx % len(tgt["stop_xy"])]
                    wp = world.get_map().get_waypoint(carla.Location(sx, sy, 0.0), project_to_road=True,
                                                      lane_type=carla.LaneType.Driving)
                    for dist_up in [float(x_) for x_ in args.emerg_dists.split(',')]:
                        prev = wp.previous(dist_up)
                        if not prev:
                            continue
                        tf = prev[0].transform
                        tf.location.z += 0.5
                        ev = world.try_spawn_actor(cand[0], tf)
                        if ev is not None:
                            break
                if ev is None:
                    e_fail += 1
                    next_e = t + 3.0
                else:
                    ev.set_autopilot(True, args.tm_port)
                    tm.auto_lane_change(ev, False)
                    tm.vehicle_percentage_speed_difference(ev, rng.uniform(0.0, 40.0))
                    try:
                        tm.set_route(ev, ["Straight"] * 4)
                    except Exception as e_:
                        print("WARN set_route:", e_)
                    try:
                        ev.set_light_state(carla.VehicleLightState(
                            carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2))
                    except Exception:
                        pass
                    actors.append(ev)
                    emerg.append((ev, t))
                    e_idx += 1
                    next_e = t + args.emerg_every
            for ev, te in list(emerg):
                if t - te > 45.0:
                    ev.destroy()
                    emerg.remove((ev, te))
                    if ev in actors:
                        actors.remove(ev)

            fr = world.tick()
            img = q_rgb.get(timeout=10.0)
            ins = q_inst.get(timeout=10.0) if q_inst is not None else None
            while img.frame < fr:
                img = q_rgb.get(timeout=10.0)
            if ins is not None:
                while ins.frame < fr:
                    ins = q_inst.get(timeout=10.0)
            if t < args.warmup or k % args.every != 0:
                continue

            w2c = np.array(cam.get_transform().get_inverse_matrix())
            idarr = None
            if ins is not None:
                a4 = np.frombuffer(ins.raw_data, dtype=np.uint8).reshape(args.height, args.width, 4)
                idarr = a4[:, :, 1].astype(np.int32) + 256 * a4[:, :, 0].astype(np.int32)   # BGRA: G low byte, B high byte
            camloc = cam.get_transform().location
            veh = []
            for v in list(world.get_actors().filter("vehicle.*")):
                tf, vel = v.get_transform(), v.get_velocity()
                d = tf.location.distance(camloc)
                if d > args.max_dist:
                    continue
                verts = np.array([[p.x, p.y, p.z] for p in v.bounding_box.get_world_vertices(tf)])
                u, vv, z = project(verts, w2c, K)
                if np.any(z <= 0.1):
                    continue
                x1, y1, x2, y2 = float(u.min()), float(vv.min()), float(u.max()), float(vv.max())
                full = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
                cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
                cx2, cy2 = min(x2, args.width - 1.0), min(y2, args.height - 1.0)
                clip = max(cx2 - cx1, 0.0) * max(cy2 - cy1, 0.0)
                if clip < args.min_area or full <= 0 or clip / full < 0.5:
                    continue
                inst_px = -1
                if idarr is not None:
                    inst_px = int(np.count_nonzero(idarr[int(cy1):int(cy2) + 1, int(cx1):int(cx2) + 1] == (v.id & 0xFFFF)))
                    inst_tot += 1
                    inst_ok += 1 if inst_px > 0 else 0
                veh.append(dict(id=v.id, cls=cls_of(v.type_id), type=v.type_id,
                                bbox=[round(cx1, 1), round(cy1, 1), round(cx2, 1), round(cy2, 1)],
                                dist_m=round(d, 1), speed=round(math.hypot(vel.x, vel.y), 2), inst_px=inst_px))
            name = "img_%06d.png" % fr
            img.save_to_disk(os.path.join(outdir, name))
            gt_f.write(json.dumps(dict(file=name, frame=fr, t=round(t, 2), weather=wname, visibility=round(vis, 3),
                                       seed=args.seed, signal=dict(mode=args.signal, state=dict(cur_state)), cam=dict(x=ox, y=oy, z=args.cam_height, yaw=oinfo["yaw"],
                                                                pitch=args.cam_pitch, fov=args.fov,
                                                                w=args.width, h=args.height),
                                       vehicles=veh)) + "\n")
            gt_f.flush()
            frames += 1
        print(json.dumps(dict(source="CARLA", map=args.map, weather=wname, seed=args.seed, frames=frames,
                              inflow_spawned=n_in, initial=n0, emergency_spawned=e_idx - e_start, emergency_spawn_failed=e_fail,
                              inst_checked=inst_tot, inst_visible_frac=(inst_ok / inst_tot) if inst_tot else None,
                              wall_s=round(time.time() - t0, 1), out=outdir)))
    finally:
        gt_f.close()
        try:
            for _g in lights.values():
                for _tl in _g:
                    _tl.freeze(False)
            for a in actors:
                if hasattr(a, "stop"):
                    a.stop()
            client.apply_batch([carla.command.DestroyActor(a) for a in actors])
            world.tick()
            s = world.get_settings()
            s.synchronous_mode, s.fixed_delta_seconds = False, None
            world.apply_settings(s)
            tm.set_synchronous_mode(False)
        except Exception as e:
            print("WARN cleanup:", e)


if __name__ == "__main__":
    main()
