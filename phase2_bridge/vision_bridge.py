"""Hybrid perception for the CARLA runner: queue/wait/distance/speed come from the CARLA API (as before), the
EMERGENCY class evidence comes from a real detector run on the camera frame.
Plugs into CarlaJunctionIO as its `noise` object (same step() interface as PerceptionNoise).
ORACLE parts (stated in every result): track association (detections are matched to CARLA actors by projected-box IoU),
distance and speed of a detected vehicle (taken from the actor), the lane of a true vehicle. Camera sees only the
junction's target approach (default A0): emergency vehicles on other approaches are never detected.
False alarms (emergency-class detections with IoU<0.3 to every projected actor, e.g. parked props) are attributed to the
target approach and tracked by quantised box position, so a persistent static false alarm accumulates track age."""
from __future__ import annotations

import json
import queue
import socket
import struct
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from perception_noise import EMERGENCY, Observation

CLS = ["ambulance", "police", "firetruck", "car"]


class DetClient:
    def __init__(self, host: str, port: int, timeout: float = 120.0) -> None:
        self.s = socket.create_connection((host, port), timeout=timeout)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _recv(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            c = self.s.recv(n - len(buf))
            if not c:
                raise ConnectionError("detector server closed")
            buf += c
        return bytes(buf)

    def ping(self) -> bool:
        self.s.sendall(b"PING")
        return self._recv(4) == b"PONG"

    def detect(self, bgr: np.ndarray) -> List[List[float]]:
        h, w, _ = bgr.shape
        data = np.ascontiguousarray(bgr).tobytes()
        self.s.sendall(b"FRM1" + struct.pack("!III", w, h, len(data)) + data)
        n = struct.unpack("!I", self._recv(4))[0]
        return json.loads(self._recv(n).decode())

    def close(self) -> None:
        try:
            self.s.close()
        except OSError:
            pass


def iou_one(a, bs: np.ndarray) -> np.ndarray:
    if len(bs) == 0:
        return np.zeros(0)
    x1, y1 = np.maximum(a[0], bs[:, 0]), np.maximum(a[1], bs[:, 1])
    x2, y2 = np.minimum(a[2], bs[:, 2]), np.minimum(a[3], bs[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


class VisionObserver:
    """Same interface as PerceptionNoise: step(t, dt, vehicles, lanes) -> List[Observation]."""

    def __init__(self, world, carla_mod, cam, q: "queue.Queue", client: DetClient, K: np.ndarray, width: int, height: int,
                 target_lane: str = "A0", det_conf: float = 0.5, min_area: float = 400.0, max_dist: float = 100.0,
                 warmup: float = 20.0, log_path: str = "") -> None:
        self.world, self.cam, self.q, self.client, self.K = world, cam, q, client, K
        self.w, self.h = width, height
        self.target_lane, self.det_conf, self.min_area, self.max_dist, self.warmup = target_lane, det_conf, min_area, max_dist, warmup
        self.log = open(log_path, "w", encoding="utf-8") if log_path else None
        self.n_frames = 0
        self.ms, self.lag_max = 0.0, 0
        self.vis_f = self.found_f = self.emerg_f = self.cls_f = 0
        self.first_vis_t = self.first_emerg_t = None
        self.fp_emerg_n = self.fp_emerg_frames = self.car_as_emerg_n = self.ambig_emerg_n = 0
        self.n_car_obs = 0
        self._first = True   # PATCH_FIRST_FRAME: camera is spawned after the last tick, so step 0 has no image yet

    def _projected(self) -> List[dict]:
        from carla_capture import project
        w2c = np.array(self.cam.get_transform().get_inverse_matrix())
        camloc = self.cam.get_transform().location
        out = []
        for v in self.world.get_actors().filter("vehicle.*"):
            tf = v.get_transform()
            d = tf.location.distance(camloc)
            if d > self.max_dist:
                continue
            verts = np.array([[p.x, p.y, p.z] for p in v.bounding_box.get_world_vertices(tf)])
            u, vv, z = project(verts, w2c, self.K)
            if np.any(z <= 0.1):
                continue
            x1, y1, x2, y2 = float(u.min()), float(vv.min()), float(u.max()), float(vv.max())
            full = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, self.w - 1.0), min(y2, self.h - 1.0)
            clip = max(cx2 - cx1, 0.0) * max(cy2 - cy1, 0.0)
            if clip < self.min_area or full <= 0 or clip / full < 0.5:
                continue
            out.append(dict(id=str(v.id), type=v.type_id, bbox=(cx1, cy1, cx2, cy2), dist=d))
        out.sort(key=lambda p: (round(p["dist"], 2), p["type"]))
        return out

    def _frame(self):
        target = self.world.get_snapshot().frame
        if self._first:
            self._first = False
            if self.q.empty():
                return None
        while True:
            im = self.q.get(timeout=20.0)
            if im.frame >= target:
                break
        self.lag_max = max(self.lag_max, im.frame - target)
        a = np.frombuffer(im.raw_data, dtype=np.uint8).reshape(self.h, self.w, 4)
        return np.ascontiguousarray(a[:, :, :3])           # BGRA -> BGR (what cv2.imread gave the offline evaluation)

    def step(self, t: float, dt: float, vehicles: Iterable[dict], lanes: Iterable[str]) -> List[Observation]:
        from carla_adapter import cls_of
        gt = {str(v["id"]): v for v in vehicles}
        bgr = self._frame()
        if bgr is None:
            raw = []
        else:
            t0 = time.time()
            raw = self.client.detect(bgr)
            self.ms += (time.time() - t0) * 1000.0
            self.n_frames += 1
        proj = self._projected()
        pbox = np.array([p["bbox"] for p in proj], dtype=float).reshape(-1, 4)
        dets = [d for d in raw if d[4] >= self.det_conf]
        dets.sort(key=lambda d: (-d[4], d[0], d[1]))
        used, match = set(), {}
        fps = []
        in_win = t >= self.warmup
        for d in dets:
            cls = CLS[int(d[5])]
            ious = iou_one(d[:4], pbox)
            order = [j for j in np.argsort(-ious, kind="stable") if ious[j] >= 0.5 and j not in used]
            if order:
                j = int(order[0])
                used.add(j)
                match[proj[j]["id"]] = (cls, float(d[4]))
                if cls in EMERGENCY and cls_of(proj[j]["type"]) not in EMERGENCY and in_win:
                    self.car_as_emerg_n += 1
            elif cls in EMERGENCY:
                if len(ious) and ious.max() >= 0.3:
                    if in_win:
                        self.ambig_emerg_n += 1
                elif in_win:
                    fps.append((cls, float(d[4]), d[:4]))
        em_dbg = []
        for p in proj:
            gcls = cls_of(p["type"])
            if gcls in EMERGENCY and in_win:
                self.vis_f += 1
                if self.first_vis_t is None:
                    self.first_vis_t = t
                m = match.get(p["id"])
                if m is not None:
                    self.found_f += 1
                    if m[0] in EMERGENCY:
                        self.emerg_f += 1
                        if self.first_emerg_t is None:
                            self.first_emerg_t = t
                    if m[0] == gcls:
                        self.cls_f += 1
                em_dbg.append(dict(id=p["id"], cls=gcls, dist=round(p["dist"], 1), bbox=[round(x, 1) for x in p["bbox"]],
                                   det=None if m is None else [m[0], round(m[1], 3)]))
        if fps:
            self.fp_emerg_frames += 1
            self.fp_emerg_n += len(fps)
        if self.log is not None and (em_dbg or fps):
            self.log.write(json.dumps(dict(t=t, em=em_dbg, fp=[[c, round(cf, 3), [round(x, 1) for x in b]] for c, cf, b in fps])) + "\n")
        out: List[Observation] = []
        for vid in sorted(gt, key=lambda k: (str(gt[k]["lane"]), round(float(gt[k]["dist_m"]), 2), str(gt[k]["cls"]))):
            g = gt[vid]
            m = match.get(vid)
            if m is not None and m[0] in EMERGENCY:
                out.append(Observation(vid, g["lane"], m[0], m[1]))
            else:
                out.append(Observation(vid, g["lane"], "car", 0.9))
        for c, cf, b in sorted(fps, key=lambda f: (f[0], f[2][0], f[2][1])):
            key = "fp_%s_%d_%d" % (c, int((b[0] + b[2]) / 2) // 40, int((b[1] + b[3]) / 2) // 40)
            out.append(Observation(key, self.target_lane, c, cf, True))
        return out

    def stats(self) -> dict:
        return dict(perception="hybrid(api queue/dist/speed + detector emergency class; oracle association)",
                    det_frames=self.n_frames, det_ms_mean=self.ms / max(self.n_frames, 1), det_lag_max=self.lag_max,
                    det_conf_thr=self.det_conf, det_amb_vis_frames=self.vis_f, det_amb_found_frames=self.found_f,
                    det_amb_emerg_frames=self.emerg_f, det_amb_cls_frames=self.cls_f,
                    det_first_vis_t=self.first_vis_t, det_first_emerg_t=self.first_emerg_t,
                    det_fp_emerg_frames=self.fp_emerg_frames, det_fp_emerg_n=self.fp_emerg_n,
                    det_car_as_emerg_n=self.car_as_emerg_n, det_ambig_emerg_n=self.ambig_emerg_n)

    def close(self) -> None:
        try:
            self.cam.stop()
        except Exception:
            pass
        self.client.close()
        if self.log is not None:
            self.log.close()
