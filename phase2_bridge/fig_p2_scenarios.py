import os
import sys

sys.path.insert(0, os.path.expanduser("~/Portfolio/Smart_Traffic_System/figures_pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from junction_controller import DEGRADED, NORMAL, STATIC, FuzzyJunctionController
from junction_model import ApproachObs, EmergencyTrack, JunctionConfig, JunctionObs

setup_style()
APPS = ("A0", "A1", "A2", "A3")


def ctl():
    return FuzzyJunctionController(JunctionConfig(name="J", phases={0: ("A0", "A2"), 1: ("A1", "A3")}), phase0=0)


def mkobs(t, q, em, vis=1.0):
    return JunctionObs(t=float(t), approaches={a: ApproachObs(a, q.get(a, 0), 0.0) for a in APPS},
                       emergencies=tuple(em), visibility=vis)


def trk(tid, app, eta, qa=0, cls="ambulance", conf=0.95):
    return EmergencyTrack(tid, cls, app, max(eta, 0.0), qa, conf, 5.0)


def run_case(n, fn):
    c, ds, sm = ctl(), [], []
    for i in range(n):
        ds.append(c.step(fn(float(i)), 1.0))
        sm.append(c.modes.smoothed)
    return ds, np.array(sm, float)


def runs(seq):
    out, s = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            out.append((s, i - s, seq[s]))
            s = i
    return out


def plot_case(ds, events, title, name, vis=None, sm=None):
    n = len(ds)
    fig, axes = plt.subplots(3, 1, figsize=(9, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2, 1.2]))
    ax = axes[0]
    for s, ln, (ph, yel) in runs([(d.phase, d.yellow) for d in ds]):
        ax.barh(0, ln, left=s, color=PAL["yellow"] if yel else (PAL["blue"] if ph == 0 else PAL["orange"]),
                edgecolor="white", height=0.8)
        if ln >= 6:
            ax.text(s + ln / 2, 0, "Y" if yel else f"P{ph}", ha="center", va="center", fontsize=7,
                    color="black" if yel else "white")
    ax.set_yticks([])
    ax.set_ylabel("signal")
    ax.set_title(title)
    ax = axes[1]
    for k, (lab, s, e) in enumerate(events):
        ax.plot([s, min(e, n)], [k, k], lw=5, color=PAL["red"])
        ax.text(s, k + 0.25, lab, fontsize=7)
    ax.set_ylim(-0.6, max(len(events), 1) - 0.2)
    ax.set_yticks([])
    ax.set_ylabel("emergency\ntracks")
    ax = axes[2]
    for s, ln, md in runs([d.mode for d in ds]):
        ax.axvspan(s, s + ln, color={NORMAL: PAL["sky"], DEGRADED: PAL["orange"], STATIC: PAL["grey"]}[md], alpha=0.25, lw=0)
    if vis is not None:
        ax.plot(range(n), [vis(float(i)) for i in range(n)], color=PAL["black"], lw=1, label="visibility (raw)")
    if sm is not None:
        ax.plot(range(n), sm, color=PAL["red"], lw=1, label="smoothed")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, n)
    ax.set_ylabel("mode / visibility")
    ax.set_xlabel("time (s)   [blue band: normal, orange: degraded, grey: static]")
    if vis is not None:
        ax.legend(loc="lower right", ncol=2)
    first = next((d for d in ds if d.reason.startswith("emergency")), None)
    if first is not None:
        fig.text(0.01, -0.01, first.reason[:120], fontsize=6.5)
    watermark(fig, "illustrative controller trace (no traffic model)")
    save(fig, name, P2_OUT)


qa = {"A0": 12, "A2": 12, "A1": 4}
ds, _ = run_case(70, lambda t: mkobs(t, qa, [trk("amb", "A1", 30 - t, 3)] if 5 <= t < 36 else []))
plot_case(ds, [("ambulance on A1 (ETA 30 s)", 5, 36)], "Case A: single emergency vehicle preempts a busy cross phase", "p2_case_a_single")

ds, _ = run_case(60, lambda t: mkobs(t, {"A0": 10, "A2": 10, "A1": 3},
                                     [trk("amb", "A0", 25 - t, 2), trk("fire", "A2", 30 - t, 2, "firetruck")] if t < 28 else []))
plot_case(ds, [("ambulance A0", 0, 28), ("firetruck A2 (same phase, no conflict)", 0, 28)],
          "Case B: opposing emergency vehicles share a phase", "p2_case_b_same_phase")


def case_c(t):
    em = []
    if 3 <= t < 26:
        em.append(trk("amb", "A0", 14 - (t - 3), 6))
    if 3 <= t < 46:
        em.append(trk("pol", "A1", 16 - (t - 3), 0, "police"))
    return mkobs(t, {"A0": 8, "A2": 4, "A1": 8, "A3": 4}, em)


ds, _ = run_case(70, case_c)
plot_case(ds, [("ambulance A0 (6 vehicles ahead)", 3, 26), ("police A1 (no queue ahead)", 3, 46)],
          "Case C: conflicting emergency vehicles, order by minimum expected weighted delay", "p2_case_c_conflict")


def vis_fn(t):
    return 1.0 if t < 60 else 0.1 if t < 200 else 0.9


ds, sm = run_case(320, lambda t: mkobs(t, {"A0": 10, "A2": 10, "A1": 10, "A3": 10},
                                       [trk("amb1", "A1", 30 - (t - 100), 3)] if 100 <= t < 130 else
                                       [trk("amb2", "A1", 30 - (t - 250), 3)] if 250 <= t < 280 else [], vis_fn(t)))
plot_case(ds, [("ambulance during low visibility (ignored: static mode)", 100, 130), ("ambulance after recovery", 250, 280)],
          "Case D: visibility drop, static fixed-time mode and recovery", "p2_case_d_visibility", vis=vis_fn, sm=sm)
