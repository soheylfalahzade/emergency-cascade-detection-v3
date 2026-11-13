"""Top-down snapshots of the junction for each signal phase (CARLA, Windows venv). The light heads in the render are independent
evidence of which lights are green; the text labels (approach name + API state) are overlays at the stop lines from the spec."""
import json, os, queue, sys
import cv2
import numpy as np
import carla
sys.path.insert(0, ".")
from carla_capture import cam_matrix, project
from carla_runner import resolve_lights

OUT = "figs_sim"
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
pts = [p for info in spec["approaches"].values() for p in info["stop_xy"]]
cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
W, H, FOV, HEIGHT = 1280, 720, 90, 32
bp = world.get_blueprint_library().find("sensor.camera.rgb")
bp.set_attribute("image_size_x", str(W)); bp.set_attribute("image_size_y", str(H)); bp.set_attribute("fov", str(FOV))
cam = world.spawn_actor(bp, carla.Transform(carla.Location(cx, cy, HEIGHT), carla.Rotation(pitch=-90.0, yaw=0.0)))
q = queue.Queue()
cam.listen(q.put)
K = cam_matrix(W, H, FOV)
imgs = []
try:
    for ph in sorted(int(k) for k in spec["phases"]):
        green = set(spec["phases"][str(ph)])
        for a, tls in lights.items():
            for tl in tls:
                tl.set_state(carla.TrafficLightState.Green if a in green else carla.TrafficLightState.Red)
        for _ in range(4):
            world.tick()
        target = world.get_snapshot().frame
        while True:
            im = q.get(timeout=30)
            if im.frame >= target:
                break
        arr = np.frombuffer(im.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3].copy()
        w2c = np.array(cam.get_transform().get_inverse_matrix())
        for a, info in spec["approaches"].items():
            x, y = info["stop_xy"][0]
            u, v, z = project(np.array([[x, y, 0.3]]), w2c, K)
            api = str(lights[a][0].get_state()).split(".")[-1]
            p = (int(u[0]), int(v[0]))
            cv2.circle(arr, p, 7, (0, 0, 0), -1); cv2.circle(arr, p, 5, (255, 255, 255), -1)
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                cv2.putText(arr, "%s (API: %s)" % (a, api), (p[0] + 12, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
        cap = "phase %d: green for %s | CARLA simulation, Town05_Opt, top-down view" % (ph, ",".join(sorted(green)))
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
            cv2.putText(arr, cap, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, "fig_sim_junction_phase%d.png" % ph), arr)
        imgs.append(arr)
    both = cv2.resize(cv2.hconcat(imgs), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, "fig_sim_junction_phases.png"), both)
    print("wrote", len(imgs), "phase snapshots to", OUT)
finally:
    try:
        cam.stop(); cam.destroy()
    except Exception:
        pass
    for tls in lights.values():
        for tl in tls:
            tl.freeze(False)
    st.synchronous_mode, st.fixed_delta_seconds = False, None
    world.apply_settings(st)
