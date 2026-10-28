import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from toy_sim import POLICIES, Scenario, run_sim


def t_all_policies_run():
    for name, mk in POLICIES.items():
        m, _ = run_sim(mk(0), Scenario(duration_s=200.0, n_emergency=1), 0)
        assert np.isfinite(m["ord_delay_s"]) and m["throughput_vph"] > 0, name


def t_deterministic():
    a, _ = run_sim(POLICIES["Fuzzy (full)"](3), Scenario(duration_s=200.0), 3)
    b, _ = run_sim(POLICIES["Fuzzy (full)"](3), Scenario(duration_s=200.0), 3)
    for k in a:
        if np.isfinite(a[k]):
            assert a[k] == b[k], k


def t_same_demand_across_policies():
    sc = Scenario(duration_s=200.0, n_emergency=0)
    arr = {name: run_sim(POLICIES[name](1), sc, 1)[0]["arrivals"] for name in ("Fixed-time", "Max-pressure", "Fuzzy (full)")}
    assert len(set(arr.values())) == 1, arr


def t_emergency_completes():
    m, _ = run_sim(POLICIES["Fuzzy (full)"](0), Scenario(duration_s=400.0, n_emergency=1, corridor_share=1.0), 0)
    assert m["em_incomplete"] == 0 and np.isfinite(m["em_delay_mean"]) and m["em_delay_mean"] < 200.0, m


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
