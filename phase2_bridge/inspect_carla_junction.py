"""Read-only: list CARLA traffic-light groups and write a draft junction spec. Verify the draft by hand."""
import argparse
import json

import carla


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--index", type=int, default=0, help="which light group to export")
    ap.add_argument("--out", default="junction_spec_draft.json")
    args = ap.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    print("map:", world.get_map().name)
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    print("traffic lights:", len(lights))
    groups = {}
    for tl in lights:
        try:
            g = list(tl.get_group_traffic_lights())
        except Exception:
            g = [tl]
        ids = tuple(sorted({x.id for x in g} | {tl.id}))
        groups[ids] = [world.get_actor(i) for i in ids]
    glist = sorted(groups.items(), key=lambda kv: kv[0][0])
    for i, (ids, tls) in enumerate(glist):
        cx = sum(t.get_location().x for t in tls) / len(tls)
        cy = sum(t.get_location().y for t in tls) / len(tls)
        print(f"[{i}] lights={len(tls)} ids={list(ids)} centre=({cx:.0f},{cy:.0f})")
    ids, tls = glist[args.index]
    approaches = {}
    for tl in tls:
        try:
            stops = [(w.transform.location.x, w.transform.location.y, w.transform.rotation.yaw)
                     for w in tl.get_stop_waypoints()]
        except Exception:
            stops = []
        if not stops:
            tf = tl.get_transform()
            stops = [(tf.location.x, tf.location.y, tf.rotation.yaw)]
        binn = int(round(stops[0][2] / 90.0)) % 4
        a = approaches.setdefault(f"A{binn}", dict(lights=[], stop_xy=[], yaw=binn * 90.0 if binn < 2 else binn * 90.0 - 360.0,
                                                   capacity=12))
        a["lights"].append(tl.id)
        a["stop_xy"] += [[round(x, 2), round(y, 2)] for x, y, _ in stops]
        print(f"light {tl.id}: stops={len(stops)} yaw={stops[0][2]:.0f} -> A{binn}")
    phases = {}
    for p, pair in ((0, ("A0", "A2")), (1, ("A1", "A3"))):
        present = [a for a in pair if a in approaches]
        if present:
            phases[str(p)] = present
        else:
            print(f"WARNING: no approaches for phase {p}")
    spec = dict(_note="DRAFT: verify approach bins, light ids and stop points before use",
                name=f"group{args.index}", approaches=approaches, phases=phases, config={})
    with open(args.out, "w") as f:
        json.dump(spec, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
