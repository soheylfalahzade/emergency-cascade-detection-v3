"""Logic tests for the phase-2 controller. CPU only; no CARLA, no GPU, no extra packages."""
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordination import MessageBus, NetworkController
from junction_controller import DEGRADED, NORMAL, STATIC, FuzzyJunctionController
from junction_model import ApproachObs, EmergencyTrack, JunctionConfig, JunctionObs, Message
from perception_noise import NoiseConfig, PerceptionNoise, summarize

APPS = ("A0", "A1", "A2", "A3")


def mk(phases=None, **kw):
    cfg = JunctionConfig(name=kw.pop("name", "J"), phases=phases or {0: ("A0", "A2"), 1: ("A1", "A3")}, **kw)
    return FuzzyJunctionController(cfg, phase0=min(cfg.phases))


def obs(t, q=None, w=None, em=(), vis=1.0, inbox=()):
    q, w = q or {}, w or {}
    names = set(APPS) | set(q) | set(w)
    return JunctionObs(t=float(t), approaches={a: ApproachObs(a, q.get(a, 0), w.get(a, 0.0)) for a in names},
                       emergencies=tuple(em), visibility=vis, inbox=tuple(inbox))


def trk(tid, app, eta=10.0, qa=0, conf=0.95, age=5.0, cls="ambulance"):
    return EmergencyTrack(tid, cls, app, eta, qa, conf, age)


def run(ctl, n, f, dt=1.0):
    return [ctl.step(f(i * dt), dt) for i in range(n)]


def first_yellow(ds):
    return next((i for i, d in enumerate(ds) if d.yellow), None)


def t_fuzzy_monotone():
    ctl = mk()
    p = [ctl._pressure_tsk.infer({"queue": 0.8, "wait": 0.3, "down": d}) for d in (0.0, 0.5, 1.0)]
    assert p[0] > p[1] > p[2], p


def t_min_green_and_yellow():
    ds = run(mk(), 20, lambda t: obs(t, q={"A1": 18, "A3": 18}))
    assert first_yellow(ds) == 9, first_yellow(ds)
    assert [d.yellow for d in ds[9:12]] == [True] * 3
    assert ds[12].phase == 1 and not ds[12].yellow
    assert all(d.phase == 0 for d in ds[:12])


def t_max_green():
    ds = run(mk(), 70, lambda t: obs(t, q={"A0": 18, "A2": 18, "A1": 4, "A3": 4}))
    assert first_yellow(ds) == 60, first_yellow(ds)


def t_single_emergency():
    ds = run(mk(), 10, lambda t: obs(t, em=[trk("a", "A1", eta=30.0)]))
    assert first_yellow(ds) == 3, first_yellow(ds)
    assert ds[6].phase == 1 and not ds[6].yellow
    assert ds[0].plan == (1,) and "emergency" in ds[0].reason


def t_opposing_same_phase():
    ds = run(mk(), 20, lambda t: obs(t, em=[trk("a", "A0", eta=8.0), trk("b", "A2", eta=12.0)]))
    assert all(d.phase == 0 and not d.yellow for d in ds)
    assert ds[0].plan == (0,)


def t_conflict_current_first_then_other():
    ds = run(mk(), 20, lambda t: obs(t, em=([trk("a", "A0", eta=5.0)] if t < 8 else []) + [trk("b", "A1", eta=5.0)]))
    assert all(d.phase == 0 and not d.yellow for d in ds[:8])
    assert ds[8].yellow and ds[11].phase == 1 and not ds[11].yellow


def t_scheduling_prefers_short_job():
    ds = run(mk(), 3, lambda t: obs(t, em=[trk("a", "A0", eta=5.0, qa=6), trk("b", "A1", eta=4.0, qa=0)]))
    assert ds[0].plan[0] == 1, ds[0].plan


def t_tie_break_three_phase():
    ph = {0: ("A0",), 1: ("A1",), 2: ("A2",)}
    ctl = mk(phases=ph)
    ctl.phase = 2
    d = ctl.step(obs(0, em=[trk("a", "A0", eta=5.0, age=5.0), trk("b", "A1", eta=5.0, age=5.0)]), 1.0)
    assert d.plan[0] == 0, d.plan
    ctl = mk(phases=ph)
    ctl.phase = 2
    d = ctl.step(obs(0, em=[trk("a", "A0", eta=5.0, age=5.0), trk("b", "A1", eta=5.0, age=9.0)]), 1.0)
    assert d.plan[0] == 1, d.plan


def t_static_entry_and_cycle():
    ds = run(mk(), 40, lambda t: obs(t, vis=0.1))
    assert ds[1].mode == DEGRADED and ds[2].mode == STATIC, [d.mode for d in ds[:4]]
    assert not any(d.yellow for d in ds[:30]) and ds[29].phase == 0
    assert ds[30].yellow and ds[33].phase == 1 and not ds[33].yellow


def t_static_ignores_emergency():
    ds = run(mk(), 30, lambda t: obs(t, em=[trk("a", "A1")], vis=0.1))
    assert not any(d.yellow for d in ds)


def t_mode_hysteresis_no_flapping():
    ds = run(mk(), 60, lambda t: obs(t, vis=0.2 if int(t) % 2 == 0 else 0.5))
    modes = [d.mode for d in ds]
    assert STATIC not in modes
    assert sum(1 for a, b in zip(modes, modes[1:]) if a != b) <= 1


def t_static_recovery():
    ds = run(mk(), 30, lambda t: obs(t, vis=0.1 if t < 10 else 0.9))
    assert ds[19].mode == STATIC and ds[20].mode == NORMAL, [d.mode for d in ds[17:23]]


def t_downstream_full():
    msgs = [Message("link_state", "N", "J", 0.0, "L0", occupancy=0.95), Message("link_state", "N", "J", 0.0, "L2", occupancy=0.95)]
    q = {"A0": 18, "A2": 18, "A1": 10, "A3": 10}
    ds = run(mk(out_link={"A0": "L0", "A2": "L2"}), 20, lambda t: obs(t, q=q, inbox=msgs if t == 0 else ()))
    assert first_yellow(ds) == 9, first_yellow(ds)
    ds = run(mk(out_link={"A0": "L0", "A2": "L2"}), 20, lambda t: obs(t, q=q))
    assert first_yellow(ds) is None


def t_notice_preclears():
    m = Message("emergency", "N", "J", 0.0, "LIN1", eta_s=20.0, cls="ambulance", conf=0.95, tid="x")
    ds = run(mk(in_link={"A1": "LIN1"}), 10, lambda t: obs(t, inbox=(m,) if t == 0 else ()))
    assert ds[6].phase == 1 and not ds[6].yellow


def t_network_messages():
    c1 = mk(name="J1", out_link={"A1": "L12"}, downstream_of={"L12": "J2"}, link_travel_s={"L12": 10.0})
    c2 = mk(name="J2", in_link={"A1": "L12"}, upstream_of={"L12": "J1"})
    net = NetworkController({"J1": c1, "J2": c2}, MessageBus())
    for i in range(10):
        t = float(i)
        net.step(t, 1.0, {"J1": obs(t, em=[trk("amb", "A1", eta=10.0)]), "J2": obs(t, q={"A1": 9})})
    assert net.controllers["J2"].phase == 1
    assert abs(net.controllers["J1"]._link_occ["L12"][1] - 0.45) < 1e-9


def t_invariants_random():
    rnd = random.Random(1)
    ctl = mk()
    ds = []
    for i in range(800):
        em = []
        if rnd.random() < 0.05:
            em.append(trk(f"t{i}", rnd.choice(APPS), eta=rnd.uniform(3, 40), qa=rnd.randint(0, 8),
                          conf=rnd.uniform(0.3, 1.0), age=rnd.uniform(0, 6)))
        vis = [1.0, 0.5, 0.1][(i // 100) % 3]
        ds.append(ctl.step(obs(float(i), q={a: rnd.randint(0, 20) for a in APPS}, w={a: rnd.uniform(0, 60) for a in APPS},
                              em=em, vis=vis), 1.0))
    run_len, prev, green_start = 0, ds[0].phase, -1
    for k, d in enumerate(ds):
        assert d.phase in (0, 1)
        if d.yellow:
            if run_len == 0:
                assert k - green_start >= 4, (k, green_start)
            run_len += 1
            assert d.phase == prev
        else:
            if d.phase != prev:
                assert run_len == 3, (k, run_len)
                green_start = k
            run_len = 0
        prev = d.phase


def t_noise_model_basic():
    n = PerceptionNoise(NoiseConfig(p_miss=1.0, seed=1))
    assert n.step(0.0, 1.0, [dict(id="1", lane="A0", cls="car", dist_m=10.0)], ["A0"]) == []
    n2 = PerceptionNoise(NoiseConfig(p_emerg_as_car=1.0, seed=1))
    o = n2.step(0.0, 1.0, [dict(id="9", lane="A1", cls="ambulance", dist_m=20.0)], ["A1"])
    assert len(o) == 1 and o[0].cls == "car"
    s = summarize(o, ["A0", "A1"])
    assert s["A1"]["count"] == 1 and not s["A1"]["emergency"]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("t_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
