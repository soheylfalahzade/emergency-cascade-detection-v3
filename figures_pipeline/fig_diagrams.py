from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import *

setup_style()


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333333", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)


def arrow(ax, p, q, text=None, rad=0.0, color="#333333"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.0, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, text, ha="center", fontsize=7)


fig, ax = canvas(11.5, 4.0)
box(ax, 0.1, 2.3, 1.4, 0.9, "video\nframes", "#eeeeee")
box(ax, 2.0, 2.3, 1.9, 0.9, "light detector\nYOLOv8n\n(every frame)", "#cfe8ff")
box(ax, 4.4, 2.3, 1.9, 0.9, "tracker\nByteTrack /\nBoT-SORT", "#cfe8ff")
box(ax, 6.8, 2.3, 1.9, 0.9, "uncertain?\nconf band or\nrare class", "#fff2cc")
box(ax, 9.2, 2.3, 2.2, 0.9, "track-level class vote\n+ per-lane features", "#d9f2d9")
box(ax, 6.6, 0.4, 2.3, 0.9, "heavy detector\nRT-DETR-l\n(escalated only)", "#ffd6d6")
arrow(ax, (1.5, 2.75), (2.0, 2.75))
arrow(ax, (3.9, 2.75), (4.4, 2.75))
arrow(ax, (6.3, 2.75), (6.8, 2.75))
arrow(ax, (8.7, 2.75), (9.2, 2.75), "no")
arrow(ax, (7.75, 2.3), (7.75, 1.3), "yes")
arrow(ax, (8.9, 0.85), (10.3, 2.3), "refined class", rad=-0.2)
ax.text(5.7, 3.7, "Phase 1: detection, tracking and escalation (cascade)", ha="center", fontsize=10, weight="bold")
ax.text(5.7, 0.05, "alternative / baseline: stacking of YOLOv8n, RT-DETR-l, Faster R-CNN (meta-learner on candidate scores)",
        ha="center", fontsize=7, color="#555555")
save(fig, "diagram_phase1")

fig, ax = canvas(12.0, 4.6)
box(ax, 0.1, 3.0, 1.9, 0.9, "CARLA API\nground truth", "#eeeeee")
box(ax, 2.4, 3.0, 2.1, 0.9, "perception error model\n(miss, class swap,\nfalse alarms)", "#fff2cc")
box(ax, 4.9, 3.0, 2.6, 0.9, "lane observations:\nqueue, wait, emergency\ntracks (class, ETA, conf)", "#cfe8ff")
box(ax, 7.9, 3.0, 1.9, 0.9, "mode manager\nnormal / degraded /\nstatic", "#d9f2d9")
box(ax, 4.3, 1.5, 2.6, 0.9, "emergency arbitration\n(min expected weighted delay)", "#ffd6d6")
box(ax, 7.3, 1.5, 2.6, 0.9, "fuzzy pressure (TSK)\n+ anti-spillback", "#cfe8ff")
box(ax, 4.3, 0.1, 2.6, 0.8, "safety state machine\nmin/max green, yellow", "#eeeeee")
box(ax, 7.3, 0.1, 2.6, 0.8, "signal actuation", "#eeeeee")
box(ax, 10.2, 1.5, 1.7, 0.9, "neighbour\nmessages", "#e6d9f2")
arrow(ax, (2.0, 3.45), (2.4, 3.45))
arrow(ax, (4.5, 3.45), (4.9, 3.45))
arrow(ax, (7.5, 3.45), (7.9, 3.45))
arrow(ax, (8.85, 3.0), (8.6, 2.4))
arrow(ax, (6.2, 3.0), (5.6, 2.4))
arrow(ax, (5.6, 1.5), (5.6, 0.9))
arrow(ax, (8.6, 1.5), (6.9, 0.9), rad=0.1)
arrow(ax, (6.9, 0.5), (7.3, 0.5))
arrow(ax, (10.2, 1.95), (9.9, 1.95))
arrow(ax, (9.9, 2.2), (10.2, 2.2))
ax.text(6.0, 4.35, "Phase 2: fuzzy junction control with emergency priority", ha="center", fontsize=10, weight="bold")
save(fig, "diagram_phase2", P2_OUT)
