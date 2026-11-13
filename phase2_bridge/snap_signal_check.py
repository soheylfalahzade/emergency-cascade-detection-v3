"""Signal-head colour check (CARLA, Windows venv). For each approach a camera 22 m before the stop line looks along the approach (narrow FOV, 1920x1080).
The same camera is rendered in phase 0 and phase 1 with the world otherwise frozen. Pixels that change between the two renders are the signal heads
visible from that approach. If the approach-to-light mapping is right, those heads are green in the phases in which the approach is green and red otherwise."""
import json, math, os, queue, sys
import numpy as np

OUT = "figs_sim"


def classify(rgb):
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    green = (g > r + 25) & (g > b + 10)
    red = (r > g + 40) & (r > b + 40)
    return green, red


def analyse(f0, f1, thr=60):
    """f0, f1: HxWx3 uint8 RGB frames of the same camera in two phases. Returns dict with changed-pixel colour counts and a crop box."""
    changed = np.abs(f0.astype(int) - f1.astype(int)).sum(2) > thr
    n = int(changed.sum())
    if n < 6:
        return dict(n_changed=n, g0=0, r0=0, g1=0, r1=0, box=None)
    g0, r0 = classify(f0)
    g1, r1 = classify(f1)
    ys, xs = np.nonzero(changed)
    box = (int(np.percentile(xs, 1)), int(np.percentile(ys, 1)), int(np.percentile(xs, 99)), int(np.percentile(ys, 99)))
    return dict(n_changed=n, g0=int((g0 & changed).sum()), r0=int((r0 & changed).sum()), g1=int((g1 & changed).sum()), r1=int((r1 & changed).sum()), box=box)


def verdict(res, green_in_phase0):
    """Expected: green in phase 0 and red in phase 1 if the approach is green in phase 0 (and the reverse otherwise)."""
    if res["box"] is None:
        return "NO VISIBLE CHANGE (cannot confirm)"
    a_g, a_r, b_g, b_r = (res["g0"], res["r0"], res["g1"], res["r1"]) if green_in_phase0 else (res["g1"], res["r1"], res["g0"], res["r0"])
    return "CONSISTENT" if (a_g > a_r and b_r > b_g) else "INCONSISTENT"


def run():
    import cv2
    import carla
    sys.path.insert(0, ".")
    from carla_runner import resolve_lights
    os.makedirs(OUT, exist_ok=True)
    spec0 = json.load(open("junction_spec_town05_idx14.json"))
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120)
    world = client.load_world("Town05_Opt")
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 0.05
    world.apply_settings(st)
    world.tick()
    spec = resolve_lights(world, spec0)
    lights = {a: [world.get_actor(i) for i in info["lights"]] for a, info in spec["approaches"].items()}
    for tls in lights.values():
        for tl in tls:
            tl.freeze(True)
    W, H, FOV = 1920, 1080, 40
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(W)); bp.set_attribute("image_size_y", str(H)); bp.set_attribute("fov", str(FOV))
    phases = sorted(int(k) for k in spec["phases"])

    def set_phase(ph):
        green = set(spec["phases"][str(ph)])
        for a, tls in lights.items():
            for tl in tls:
                tl.set_state(carla.TrafficLightState.Green if a in green else carla.TrafficLightState.Red)
        for _ in range(6):
            world.tick()

    def grab(q):
        target = world.get_snapshot().frame
        while True:
            im = q.get(timeout=30)
            if im.frame >= target:
                break
        return np.frombuffer(im.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3][:, :, ::-1].copy()      # BGR -> RGB

    cols, report = [], []
    try:
        for a in sorted(spec["approaches"]):
            info = spec["approaches"][a]
            sx, sy = info["stop_xy"][0]
            yaw = float(info["yaw"])
            fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            cam = world.spawn_actor(bp, carla.Transform(carla.Location(sx - 22 * fx, sy - 22 * fy, 4.5), carla.Rotation(pitch=-3.0, yaw=yaw)))
            q = queue.Queue()
            cam.listen(q.put)
            frames = {}
            for ph in phases[:2]:
                set_phase(ph)
                world.tick()
                frames[ph] = grab(q)
            api = {}
            for ph in phases[:2]:
                set_phase(ph)
                api[ph] = str(lights[a][0].get_state()).split(".")[-1]
            cam.stop(); cam.destroy()
            res = analyse(frames[phases[0]], frames[phases[1]])
            green0 = a in spec["phases"][str(phases[0])]
            v = verdict(res, green0)
            report.append("%s: %s | API state phase %d=%s, phase %d=%s | changed pixels=%d | colour of changed pixels: phase %d green=%d red=%d; phase %d green=%d red=%d" % (
                a, v, phases[0], api[phases[0]], phases[1], api[phases[1]], res["n_changed"], phases[0], res["g0"], res["r0"], phases[1], res["g1"], res["r1"]))
            tiles = []
            for ph in phases[:2]:
                img = frames[ph]
                if res["box"] is not None:
                    x0, y0, x1, y1 = res["box"]
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    w = max(x1 - x0 + 120, 320); h = int(w * 9 / 16)
                    xa, ya = max(0, min(W - w, cx - w // 2)), max(0, min(H - h, cy - h // 2))
                    crop = img[ya:ya + h, xa:xa + w]
                else:
                    crop = img[: H // 3, W // 3: 2 * W // 3]
                crop = cv2.resize(crop, (480, 270), interpolation=cv2.INTER_CUBIC)
                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                    cv2.putText(crop, "%s phase %d (API: %s) %s" % (a, ph, api[ph], "" if ph != phases[0] else v), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)
                tiles.append(crop)
            cols.append(cv2.vconcat(tiles))
        cv2.imwrite(os.path.join(OUT, "fig_sim_signal_check.png"), cv2.hconcat(cols))
        open(os.path.join(OUT, "signal_check.txt"), "w", encoding="utf-8").write("\n".join(report) + "\n")
        print("\n".join(report))
    finally:
        for tls in lights.values():
            for tl in tls:
                tl.freeze(False)
        st.synchronous_mode, st.fixed_delta_seconds = False, None
        world.apply_settings(st)


if __name__ == "__main__":
    run()
