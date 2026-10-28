"""Shared plotting helpers (headless, CPU only)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "Portfolio/Smart_Traffic_System"
RES = ROOT / "results_v3"
P1_OUT = RES / "figures"
P2_OUT = ROOT / "phase2_bridge" / "outputs"
for _d in (P1_OUT, P2_OUT):
    _d.mkdir(parents=True, exist_ok=True)
for _p in (ROOT / "stacking", ROOT / "phase2_bridge", ROOT / "figures_pipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PAL = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7",
           sky="#56B4E9", yellow="#F0E442", grey="#7F7F7F", black="#222222")
CLASSES = ["ambulance", "police", "firetruck", "car"]
CLASS_COL = dict(ambulance=PAL["blue"], police=PAL["orange"], firetruck=PAL["red"], car=PAL["green"])
MODELS = ["yolov8n", "rtdetr_l", "fasterrcnn"]
MODEL_LABEL = dict(yolov8n="YOLOv8n", rtdetr_l="RT-DETR-l", fasterrcnn="Faster R-CNN")
MODEL_COL = dict(yolov8n=PAL["blue"], rtdetr_l=PAL["red"], fasterrcnn=PAL["green"])


def setup_style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 110, "pdf.fonttype": 42, "ps.fonttype": 42})


def skip(msg):
    print("SKIP:", msg)
    sys.exit(0)


def save(fig, name, out=P1_OUT):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", out / f"{name}.png")


def watermark(fig, text, alpha=0.10):
    fig.text(0.5, 0.5, text, ha="center", va="center", rotation=25, fontsize=24, color="#8a0000",
             alpha=alpha, zorder=100)


def hex_to_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def fig_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


class VideoOut:
    """Streaming mp4 writer (frames are RGB arrays); never keeps frames in memory."""

    def __init__(self, path, fps, size):
        import cv2
        self.cv2, self.size, self.path = cv2, tuple(size), path
        self.w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), self.size)
        if not self.w.isOpened():
            raise RuntimeError(f"cannot open video writer for {path}")

    def write_rgb(self, rgb):
        if (rgb.shape[1], rgb.shape[0]) != self.size:
            rgb = self.cv2.resize(rgb, self.size)
        self.w.write(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))

    def close(self):
        self.w.release()
        print("saved", self.path)


def save_gif(frames, path, fps=6):
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(str(path), save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
    print("saved", path)
