import numpy as np
from cascade_core import Track, CascadePolicy, sweep
from stats_utils import wilson, bootstrap_ci, reliability

# --- wilson
lo, hi = wilson(11, 11)
assert 0.70 < lo < 0.80 and hi == 1.0, (lo, hi)
lo, hi = wilson(50, 100)
assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61

# --- bootstrap
x = np.random.RandomState(1).binomial(1, 0.9, 500)
a, b = bootstrap_ci(x)
assert a < x.mean() < b and b - a < 0.08

# --- ECE: calibrated vs overconfident
rng = np.random.RandomState(0)
conf = rng.uniform(0, 1, 50000)
_, e_cal = reliability(conf, rng.uniform(0, 1, 50000) < conf)
_, e_bad = reliability(np.full(50000, 0.95), rng.uniform(0, 1, 50000) < 0.5)
assert e_cal < 0.02 and e_bad > 0.4, (e_cal, e_bad)

# --- track vote
t = Track(1)
for c, p in [(3, .9), (3, .8), (0, .4), (3, .9), (3, .85)]:
    t.update(c, p)
v, s = t.vote()
assert v == 3 and s > 0.8

# --- policy: car track stable -> accept; ambulance -> escalate until heavy verifies
pol = CascadePolicy(conf_thr=0.6, stab_thr=0.7, min_obs=3)
assert pol.decide(t, 3, 0.9)[0] == "accept"
amb = Track(2)
for _ in range(5):
    amb.update(0, 0.95)
act, lab, why = pol.decide(amb, 0, 0.95)
assert act == "escalate" and "verify_rare" in why
amb.heavy_cls = 0
assert pol.decide(amb, 0, 0.95) == ("accept", 0, [])
young = Track(3); young.update(3, 0.99)
assert "few_obs" in pol.decide(young, 3, 0.99)[2]
flip = Track(4)
for c in (3, 3, 3, 3):
    flip.update(c, 0.9)
assert "disagree" in pol.decide(flip, 0, 0.9)[2]

# --- sweep on synthetic data
n = 20000
conf = rng.beta(5, 2, n); stab = rng.uniform(0.3, 1, n)
light = rng.uniform(0, 1, n) < conf * stab
heavy = rng.uniform(0, 1, n) < 0.97
rows = sweep(conf, stab, light, heavy, [0, .3, .5, .7, .9, 1.01], [0, .5, .8])
d = {(r["conf_thr"], r["stab_thr"]): r for r in rows}
assert d[(0, 0)]["esc_rate"] == 0 and abs(d[(0, 0)]["acc"] - light.mean()) < 1e-9
assert d[(1.01, 0)]["esc_rate"] == 1 and abs(d[(1.01, 0)]["acc"] - heavy.mean()) < 1e-9
for st in (0, .5, .8):
    e = [d[(ct, st)]["esc_rate"] for ct in (0, .3, .5, .7, .9, 1.01)]
    assert all(e[i] <= e[i + 1] for i in range(len(e) - 1))
assert d[(.7, .5)]["acc"] > light.mean() and d[(.7, .5)]["esc_rate"] < 1
print("ALL TESTS PASSED")
