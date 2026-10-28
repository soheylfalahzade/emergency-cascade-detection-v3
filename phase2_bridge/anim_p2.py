import os
import sys

sys.path.insert(0, os.path.expanduser("~/Portfolio/Smart_Traffic_System/figures_pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.patches as mp

from common import *
from toy_sim import APPS, CAP, JUNCTIONS, PHASES, POLICIES, Scenario, run_sim

setup_style()
SEED = 11
sc = Scenario(duration_s=600.0, lam=0.12, n_emergency=3, corridor_share=0.67)
_, log_f = run_sim(POLICIES["Fuzzy (full)"](SEED), sc, SEED, log=True)
_, log_x = run_sim(POLICIES["Fixed-time"](SEED), sc, SEED, log=True)
OUT = {"A1": (-1, 0), "A3": (1, 0), "A0": (0, -1), "A2": (0, 1)}
OFF = {"A1": (0, -0.07), "A3": (0, 0.07), "A0": (0.07, 0), "A2": (-0.07, 0)}
LCOL = {"G": "#2ca02c", "Y": "#e6b800", "R": "#d62728"}


def draw_j(ax, snap, jn):
    ax.cla()
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.25, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(mp.Rectangle((-1, -0.15), 2, 0.3, fc="#d9d9d9", ec="none"))
    ax.add_patch(mp.Rectangle((-0.15, -1), 0.3, 2, fc="#d9d9d9", ec="none"))
    j = snap["j"][JUNCTIONS[jn]]
    for a in APPS:
        ox, oy = OUT[a]
        fx, fy = OFF[a]
        sx, sy = fx + 0.17 * ox, fy + 0.17 * oy
        L = min(j["q"][APPS.index(a)] / CAP, 1.2) * 0.75
        ax.plot([sx, sx + ox * L], [sy, sy + oy * L], lw=5, color="#555555", solid_capstyle="butt")
        green = a in PHASES[j["phase"]]
        c = LCOL["Y"] if (green and j["yellow"]) else LCOL["G"] if green else LCOL["R"]
        ax.add_patch(mp.Circle((sx, sy), 0.05, fc=c, ec="black", lw=0.6, zorder=5))
        for e in snap["ems"]:
            if e["j"] == jn and e["a"] == a:
                d = 0.07 + 0.045 * e["ahead"] if e["state"] == "queued" else 0.14 + 0.03 * max(e["eta"], 0.0)
                ax.plot([sx + ox * d], [sy + oy * d], marker="*", ms=13, mfc=CLASS_COL.get(e["cls"], "red"), mec="black", zorder=6)
    ax.set_title(f"{JUNCTIONS[jn]}  [{j['mode']}]", fontsize=9)
    ax.text(0, -1.05, j["reason"][:46], ha="center", va="top", fontsize=6)


fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), dpi=100)
sup = fig.suptitle("", fontsize=10)
watermark(fig, "abstract queue model - not CARLA", alpha=0.07)
vw = VideoOut(P2_OUT / "p2_corridor_fuzzy_vs_fixed.mp4", 10, (1280, 720))
gif = []
for k in range(0, min(len(log_f), len(log_x)), 2):
    for jn in range(3):
        draw_j(axes[0, jn], log_f[k], jn)
        draw_j(axes[1, jn], log_x[k], jn)
    sup.set_text(f"top: fuzzy controller | bottom: fixed-time | t = {log_f[k]['t']:.0f} s | same demand and emergency vehicles "
                 f"(stars: blue ambulance, orange police, red firetruck)")
    rgb = fig_to_rgb(fig)
    vw.write_rgb(rgb)
    if (k // 2) % 3 == 0:
        gif.append(rgb[::2, ::2])
vw.close()
save_gif(gif, P2_OUT / "p2_corridor_fuzzy_vs_fixed.gif", fps=6)
