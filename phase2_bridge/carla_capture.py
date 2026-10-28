"""Record a top-down CARLA camera view of a junction to mp4 while the controller runs elsewhere.
UNTESTED against a live server."""
import argparse
import time

import carla
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=2000)
ap.add_argument("--x", type=float, required=True)
ap.add_argument("--y", type=float, required=True)
ap.add_argument("--z", type=float, default=30.0)
ap.add_argument("--pitch", type=float, default=-70.0)
ap.add_argument("--yaw", type=float, default=0.0)
ap.add_argument("--w", type=int, default=1280)
ap.add_argument("--h", type=int, default=720)
ap.add_argument("--fov", type=float, default=90.0)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--out", default="carla_capture.mp4")
a = ap.parse_args()
client = carla.Client(a.host, a.port)
client.set_timeout(15.0)
world = client.get_world()
bp = world.get_blueprint_library().find("sensor.camera.rgb")
bp.set_attribute("image_size_x", str(a.w))
bp.set_attribute("image_size_y", str(a.h))
bp.set_attribute("fov", str(a.fov))
bp.set_attribute("sensor_tick", str(1.0 / a.fps))
cam = world.spawn_actor(bp, carla.Transform(carla.Location(x=a.x, y=a.y, z=a.z), carla.Rotation(pitch=a.pitch, yaw=a.yaw)))
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.w, a.h))
count = [0]


def on_img(img):
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))[:, :, :3]
    vw.write(np.ascontiguousarray(arr))
    count[0] += 1


cam.listen(on_img)
try:
    time.sleep(a.seconds)
finally:
    cam.stop()
    cam.destroy()
    vw.release()
    print("frames", count[0], "->", a.out)
