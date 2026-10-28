import os
import sys

sys.path.insert(0, os.path.expanduser("~/Portfolio/Smart_Traffic_System/figures_pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from junction_controller import build_pressure_tsk, build_urgency_tsk

setup_style()
pt, ut = build_pressure_tsk(), build_urgency_tsk()


def mfs(tsk, dom, title, name):
    fig, axes = plt.subplots(1, len(dom), figsize=(3.0 * len(dom), 2.5))
    for ax, (var, (lo, hi)) in zip(np.atleast_1d(axes), dom.items()):
        xs = np.linspace(lo, hi, 300)
        for lab, f in tsk.sets[var].items():
            ax.plot(xs, [f(x) for x in xs], label=lab)
        ax.set_title(var)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=6)
    fig.suptitle(title, y=1.04)
    save(fig, name, P2_OUT)


mfs(pt, {"queue": (0, 1), "wait": (0, 1), "down": (0, 1)}, "Membership functions: phase pressure (normalised inputs)", "p2_mf_pressure")
mfs(ut, {"eta": (0, 100), "conf": (0, 1), "queue": (0, 12)}, "Membership functions: emergency urgency", "p2_mf_urgency")

specs = [(pt, "queue", (0, 1), "down", (0, 1), {"wait": 0.3}, "pressure: queue vs downstream occupancy (wait 0.3)"),
         (pt, "wait", (0, 1), "down", (0, 1), {"queue": 0.5}, "pressure: wait vs downstream occupancy (queue 0.5)"),
         (ut, "eta", (0, 100), "conf", (0, 1), {"queue": 0.0}, "urgency: ETA vs evidence (no queue ahead)"),
         (ut, "eta", (0, 100), "conf", (0, 1), {"queue": 10.0}, "urgency: ETA vs evidence (10 vehicles ahead)")]
fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
for ax, (tsk, xv, xr, yv, yr, fixed, title) in zip(axes.ravel(), specs):
    xs, ys = np.linspace(*xr, 50), np.linspace(*yr, 50)
    Z = np.array([[tsk.infer({**fixed, xv: x, yv: y}) for x in xs] for y in ys])
    im = ax.contourf(xs, ys, Z, 20, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(xv)
    ax.set_ylabel(yv)
    ax.set_title(title, fontsize=8)
    ax.grid(False)
save(fig, "p2_fuzzy_surfaces", P2_OUT)
