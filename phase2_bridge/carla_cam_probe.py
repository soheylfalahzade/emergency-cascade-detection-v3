"""Camera probe: renders RGB + depth from several poses around the junction and prints depth along the centre column,
to tell 'geometry missing / wall in front' from 'geometry present but rendered black'."""
from __future__ import annotations

import json
import math
import os
import queue
import sys

import numpy as np

import carla
from carla_adapter import load_spec

SPEC = sys.argv[1] if len(sys.argv) > 1 else "junction_spec_town05_idx14.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "probe"
W, H, FOV = 1280, 720, 60.0


def decode_depth(raw: bytes, h: int, w: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4).astype(np.float64)   # BGRA
    return (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0


def main() -> None:
    spec = load_spec(SPEC)
    pts = [p for i in spec["approaches"].values() for p in i["stop_xy"]]
    cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    a2 = spec["approaches"]["A2"]
    ox = sum(p[0] for p in a2["stop_xy"]) / len(a2["stop_xy"])
    oy = sum(p[1] for p in a2["stop_xy"]) / len(a2["stop_xy"])
    poses = {
        "spec_h7_p-12": (ox, oy, 7.0, -12.0, a2["yaw"]),
        "spec_h15_p-25": (ox, oy, 15.0, -25.0, a2["yaw"]),
        "oblique_h25_p-30": (cx + 45.0, cy, 25.0, -30.0, 180.0),
        "topdown_h70": (cx, cy, 70.0, -90.0, 0.0),
    }
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    if not client.get_world().get_map().name.endswith("Town05_Opt"):
        client.load_world("Town05_Opt")
    world = client.get_world()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    w = world.get_weather()
    print("weather: sun_altitude=%.1f cloudiness=%.1f fog=%.1f" % (w.sun_altitude_angle, w.cloudiness, w.fog_density))
    os.makedirs(OUT, exist_ok=True)
    lib = world.get_blueprint_library()
    actors = []
    try:
        for name, (x, y, z, pitch, yaw) in poses.items():
            tf = carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch=pitch, yaw=yaw))
            qs = {}
            for kind in ("rgb", "depth"):
                bp = lib.find("sensor.camera." + kind)
                bp.set_attribute("image_size_x", str(W))
                bp.set_attribute("image_size_y", str(H))
                bp.set_attribute("fov", str(FOV))
                c = world.spawn_actor(bp, tf)
                actors.append(c)
                q = queue.Queue()
                c.listen(q.put)
                qs[kind] = (c, q)
            last = {}
            for _ in range(20):
                fr = world.tick()
                for kind, (_, q) in qs.items():
                    im = q.get(timeout=10.0)
                    while im.frame < fr:
                        im = q.get(timeout=10.0)
                    last[kind] = im
            last["rgb"].save_to_disk(os.path.join(OUT, name + ".png"))
            d = decode_depth(last["depth"].raw_data, H, W)
            rows = [60, 150, 250, 350, 450, 520, 600, 700]
            print("pose %-18s depth(m) at x=640, rows %s:" % (name, rows))
            print("   ", [round(float(d[r, 640]), 1) for r in rows])
            for kind, (c, _) in qs.items():
                c.stop()
                c.destroy()
                actors.remove(c)
    finally:
        try:
            for c in actors:
                c.destroy()
            s = world.get_settings()
            s.synchronous_mode, s.fixed_delta_seconds = False, None
            world.apply_settings(s)
        except Exception as e:
            print("WARN cleanup:", e)


if __name__ == "__main__":
    main()
