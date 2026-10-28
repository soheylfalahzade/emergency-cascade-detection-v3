import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import health_policy  # noqa: F401  (registers the policy)
from perception_health import AlarmChurnHealth
from toy_sim import POLICIES, NoiseConfig, Scenario, run_sim


def t_quiet():
    h = AlarmChurnHealth()
    assert all(h.update(float(t), []) == 1.0 for t in range(100))


def t_persistent_track_is_not_a_blip():
    h = AlarmChurnHealth()
    out = [h.update(float(t), ["a"] if t < 30 else []) for t in range(120)]
    assert min(out) == 1.0, min(out)


def t_blips_lower_then_restore_health():
    h = AlarmChurnHealth()
    out = {}
    for t in range(140):
        tids = [f"f{k}" for k in range(6) if t in (10 * k, 10 * k + 1)]
        out[t] = h.update(float(t), tids)
    assert out[60] == 0.0, out[60]
    assert out[130] == 1.0, out[130]


def t_gap_merge():
    h = AlarmChurnHealth()
    for t in range(60):
        h.update(float(t), ["a"] if (t <= 4 or 7 <= t <= 12) else [])
    assert h.rate_per_min == 0.0


def t_policy_smoke_and_behaviour():
    for fp in (0.0, 5.0):
        nz = NoiseConfig(p_miss=0.1, p_emerg_as_car=0.1, fp_emerg_per_min=fp, hold_s=2.0)
        m, _ = run_sim(POLICIES["Fuzzy (health)"](0), Scenario(duration_s=300.0, n_emergency=2, noise=nz), 0)
        assert np.isfinite(m["ord_delay_s"])
        if fp == 0.0:
            assert m.get("mode_static", 0.0) < 0.05, m
        else:
            assert m.get("mode_static", 0.0) > 0.3, m


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
