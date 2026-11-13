"""Detector server (run in WSL, env traffic_env). Windows CARLA client sends raw BGR frames, gets boxes back.
Protocol: request = 'FRM1' + '!III' w, h, nbytes + raw BGR bytes. Reply = '!I' nbytes + JSON list of
[x1,y1,x2,y2,conf,cls_idx] (cls order of Phase 1 / eval_on_carla: ambulance, police, firetruck, car). b'PING' -> b'PONG'."""
import argparse
import json
import socket
import struct
import time

import numpy as np


def recv_all(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return bytes(buf)


def serve(model_fn, host, port, once=False):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print("det_server listening", host, port, flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                head = recv_all(conn, 4)
                if head == b"PING":
                    conn.sendall(b"PONG")
                    continue
                if head != b"FRM1":
                    raise ValueError("bad magic %r" % head)
                w, h, nb = struct.unpack("!III", recv_all(conn, 12))
                img = np.frombuffer(recv_all(conn, nb), dtype=np.uint8).reshape(h, w, 3)
                dets = model_fn(img)
                body = json.dumps([[round(float(x), 2) for x in d] for d in dets]).encode()
                conn.sendall(struct.pack("!I", len(body)) + body)
        except (ConnectionError, ValueError) as e:
            print("client done:", e, flush=True)
        finally:
            conn.close()
        if once:
            break


def make_model(weights, imgsz, device, conf):
    from ultralytics import YOLO
    m = YOLO(weights)
    m.predict(np.zeros((720, 1280, 3), np.uint8), imgsz=imgsz, conf=conf, verbose=False, device=device)  # warm-up
    def fn(img):
        r = m.predict(img, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
        b, c, k = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
        order = np.lexsort((b[:, 1], b[:, 0], -c)) if len(c) else []
        return [list(b[i]) + [c[i], k[i]] for i in order]
    return fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    serve(make_model(a.weights, a.imgsz, a.device, a.conf), a.host, a.port)
