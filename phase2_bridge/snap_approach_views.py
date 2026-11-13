import json, math, os, queue, sys
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
W, H, FOV = 960, 540, 60
bp = world.get_blueprint_library().find("sensor.camera.rgb")
bp.set_attribute("image_size_x", str(W)); bp.set_attribute("image_size_y", str(H)); bp.set_attribute("fov", str(FOV))
K = cam_matrix(W, H, FOV)


def text(img, s, p, scale=0.6):
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(img, s, p, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


rows = []
try:
    for ph in sorted(int(k) for k in spec["phases"]):
        green = set(spec["phases"][str(ph)])
        for a, tls in lights.items():
            for tl in tls:
                tl.set_state(carla.TrafficLightState.Green if a in green else carla.TrafficLightState.Red)
        for _ in range(4):
            world.tick()
        cells = []
        for a in sorted(spec["approaches"]):
            info = spec["approaches"][a]
            sx, sy = info["stop_xy"][0]
            yaw = float(info["yaw"])
            fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            cam = world.spawn_actor(bp, carla.Transform(carla.Location(sx - 22 * fx, sy - 22 * fy, 4.5), carla.Rotation(pitch=-6.0, yaw=yaw)))
            q = queue.Queue()
            cam.listen(q.put)
            world.tick(); world.tick()
            target = world.get_snapshot().frame
            while True:
                im = q.get(timeout=30)
                if im.frame >= target:
                    break
            arr = np.frombuffer(im.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3].copy()
            w2c = np.array(cam.get_transform().get_inverse_matrix())
            tl = lights[a][0]
            loc = tl.get_transform().location
            u, v, z = project(np.array([[loc.x, loc.y, loc.z]]), w2c, K)
            api = str(tl.get_state()).split(".")[-1]
            if z[0] > 0 and 0 <= u[0] < W and 0 <= v[0] < H:
                p = (int(u[0]), int(v[0]))
                cv2.circle(arr, p, 22, (0, 0, 0), 4); cv2.circle(arr, p, 22, (255, 255, 255), 2)
            else:
                text(arr, "assigned light actor outside the view", (20, H - 20), 0.55)
            text(arr, "phase %d | %s (API: %s)" % (ph, a, api), (20, 34), 0.8)
            cam.stop(); cam.destroy()
            cells.append(cv2.resize(arr, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
        rows.append(cv2.hconcat(cells))
    cv2.imwrite(os.path.join(OUT, "fig_sim_signal_heads.png"), cv2.vconcat(rows))
    print("wrote fig_sim_signal_heads.png")
finally:
    for tls in lights.values():
        for tl in tls:
            tl.freeze(False)
    st.synchronous_mode, st.fixed_delta_seconds = False, None
    world.apply_settings(st)
