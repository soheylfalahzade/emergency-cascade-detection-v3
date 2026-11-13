import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from coordination import MessageBus, NetworkController
from junction_controller import FuzzyJunctionController
from junction_model import ApproachObs, EmergencyTrack, JunctionObs
from toy_sim import APPS, CAP, POLICIES, Scenario, make_cfg, names, run_sim


def net(n, hops):
    ctl = {nm: FuzzyJunctionController(make_cfg(j, n, hops), phase0=0) for j, nm in enumerate(names(n))}
    return NetworkController(ctl, MessageBus())


def drive(nw, n, steps, q_last=0, em_at=None):
    for t in range(steps):
        raw = {}
        for j, nm in enumerate(names(n)):
            q = {a: 2 for a in APPS}
            if j == n - 1:
                q["A1"] = q_last
            tr = ()
            if em_at is not None and j == em_at:
                tr = (EmergencyTrack("e1", "ambulance", "A1", 20.0, 0, 0.95, float(t)),)
            raw[nm] = JunctionObs(t=float(t), approaches={a: ApproachObs(a, queue=q[a], capacity=CAP) for a in APPS},
                                  emergencies=tr)
        nw.step(float(t), 1.0, raw)


def t_hop2_state_reaches_two_upstream():
    nw = net(3, 2)
    drive(nw, 3, 8, q_last=18)
    occ = nw.controllers["J1"]._link_occ
    assert "L12" in occ and "L23" in occ, sorted(occ)
    assert occ["L23"][1] > 0.8, occ


def t_hop1_only_sees_direct_neighbour():
    nw = net(3, 1)
    drive(nw, 3, 8, q_last=18)
    assert set(nw.controllers["J1"]._link_occ) <= {"L12"}, sorted(nw.controllers["J1"]._link_occ)


def t_never_more_than_two_hops():
    nw = net(5, 2)
    drive(nw, 5, 12, q_last=18)
    for j, nm in enumerate(names(5)):
        allowed = {f"L{j + 1}{j + 2}", f"L{j + 2}{j + 3}"}
        got = set(nw.controllers[nm]._link_occ)
        assert got <= allowed, (nm, sorted(got))


def t_notice_reaches_two_ahead_only():
    nw = net(5, 2)
    drive(nw, 5, 12, em_at=0)
    got = {nm: {k[1] for k in c._notices} for nm, c in nw.controllers.items()}
    assert "L12" in got["J2"] and "L23" in got["J3"], got
    assert not got["J4"] and not got["J5"], got


def t_hop2_pressure_changes_decision():
    def first_switch(hops):
        nw = net(3, hops)
        drive(nw, 3, 1)
        for t in range(1, 40):
            raw = {}
            for j, nm in enumerate(names(3)):
                q = {"A0": 6, "A2": 6, "A1": 8, "A3": 8}
                if j == 2:
                    q["A1"] = 20
                raw[nm] = JunctionObs(t=float(t), approaches={a: ApproachObs(a, queue=q[a], capacity=CAP) for a in APPS})
            d = nw.step(float(t), 1.0, raw)
            if d["J1"].yellow:
                return t
        return None
    a, b = first_switch(1), first_switch(2)
    assert a != b, (a, b)


def t_hop2_sim_runs_n5_deterministic():
    sc = Scenario(n_junc=5, duration_s=300.0, n_emergency=2, corridor_share=1.0)
    a, _ = run_sim(POLICIES["Fuzzy (2-hop)"](2, n=5), sc, 2)
    b, _ = run_sim(POLICIES["Fuzzy (2-hop)"](2, n=5), sc, 2)
    assert np.isfinite(a["ord_delay_s"]) and a["throughput_vph"] > 0
    for k in a:
        if np.isfinite(a[k]):
            assert a[k] == b[k], k


def t_same_demand_hops():
    sc = Scenario(n_junc=5, duration_s=200.0, n_emergency=0)
    arr = {p: run_sim(POLICIES[p](1, n=5), sc, 1)[0]["arrivals"] for p in ("Fuzzy (full)", "Fuzzy (2-hop)", "Fixed-time")}
    assert len(set(arr.values())) == 1, arr


if __name__ == "__main__":
    bad = 0
    for k, f in sorted(globals().items()):
        if k.startswith("t_"):
            try:
                f()
                print("PASS", k)
            except Exception as e:
                bad += 1
                print("FAIL", k, repr(e))
    print("ALL PASSED" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)
