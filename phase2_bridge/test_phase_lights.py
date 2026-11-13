"""Controlled traffic-light test: labels stop points and lamps in the CARLA
window, greens each approach alone, then cycles the two phases.
Usage: python test_phase_lights.py [spec.json] [hold_seconds]
Needs a running CARLA server on 127.0.0.1:2000 with the map already loaded."""
import json
import sys
import time

import carla

SPEC = sys.argv[1] if len(sys.argv) > 1 else "junction_spec_draft.json"
HOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

spec = json.load(open(SPEC))
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30)
world = client.get_world()
print("map:", world.get_map().name)

xs, ys = [], []
for d in spec["approaches"].values():
    for x, y in d["stop_xy"]:
        xs.append(x)
        ys.append(y)
cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
print("centre from stop points: (%.1f, %.1f)" % (cx, cy))

lights = {}
for name, d in spec["approaches"].items():
    lights[name] = []
    for lid in d["lights"]:
        actor = world.get_actor(lid)
        if actor is None:
            raise SystemExit("light id %d not found (wrong map?)" % lid)
        lights[name].append(actor)

world.get_spectator().set_transform(
    carla.Transform(carla.Location(cx, cy, 55.0),
                    carla.Rotation(pitch=-90.0, yaw=0.0)))

steps = [("only " + n, {n}) for n in sorted(spec["approaches"])]
steps += [("phase " + p, set(g)) for p, g in [("0", spec["phases"]["0"]),
                                               ("1", spec["phases"]["1"])]]
life = HOLD * len(steps) + 30
for name, d in spec["approaches"].items():
    for x, y in d["stop_xy"]:
        world.debug.draw_point(carla.Location(x, y, 1.0), size=0.3,
                               color=carla.Color(255, 0, 0), life_time=life)
        world.debug.draw_string(carla.Location(x, y, 2.5), "stop " + name,
                                draw_shadow=False,
                                color=carla.Color(255, 255, 0), life_time=life)
    for a in lights[name]:
        loc = a.get_transform().location
        world.debug.draw_string(carla.Location(loc.x, loc.y, loc.z + 1.5),
                                "L%d=%s" % (a.id, name), draw_shadow=False,
                                color=carla.Color(0, 255, 255), life_time=life)

try:
    for group in lights.values():
        for a in group:
            a.freeze(True)
    for label, green in steps:
        for name, group in lights.items():
            st = carla.TrafficLightState.Green if name in green \
                else carla.TrafficLightState.Red
            for a in group:
                a.set_state(st)
        time.sleep(1.0)
        world.wait_for_tick()
        print(label, "-> green:", sorted(green))
        for name, group in lights.items():
            print("   ", name, [str(a.get_state()) for a in group])
        time.sleep(HOLD)
finally:
    for group in lights.values():
        for a in group:
            a.freeze(False)
    print("lights released")
