import numpy as np
from collections import deque


class Track:
    def __init__(self, tid, win=15):
        self.tid = tid
        self.hist = deque(maxlen=win)
        self.heavy_cls = None

    def update(self, cls, conf):
        self.hist.append((int(cls), float(conf)))

    @property
    def n(self):
        return len(self.hist)

    def vote(self):
        if not self.hist:
            return None, 0.0
        w = {}
        for c, p in self.hist:
            w[c] = w.get(c, 0.0) + p
        tot = sum(w.values())
        if tot <= 0:
            cs = [c for c, _ in self.hist]
            b = max(set(cs), key=cs.count)
            return b, cs.count(b) / len(cs)
        b = max(w, key=w.get)
        return b, w[b] / tot


class CascadePolicy:
    def __init__(self, conf_thr=0.6, stab_thr=0.7, min_obs=3, rare=(0, 1, 2), verify_rare=True):
        self.conf_thr, self.stab_thr, self.min_obs = conf_thr, stab_thr, min_obs
        self.rare, self.verify_rare = set(rare), verify_rare

    def decide(self, tr, cls, conf):
        """returns (action, label, reasons); action in {'accept','escalate'}"""
        if tr.heavy_cls is not None:
            return "accept", tr.heavy_cls, []
        v, stab = tr.vote()
        r = []
        if tr.n < self.min_obs:
            r.append("few_obs")
        if conf < self.conf_thr:
            r.append("low_conf")
        if stab < self.stab_thr:
            r.append("unstable")
        if v is not None and int(cls) != v:
            r.append("disagree")
        if self.verify_rare and v in self.rare:
            r.append("verify_rare")
        if r:
            return "escalate", v, r
        return "accept", v, []


def sweep(conf, stab, light_ok, heavy_ok, conf_grid, stab_grid):
    conf, stab = np.asarray(conf), np.asarray(stab)
    light_ok, heavy_ok = np.asarray(light_ok, bool), np.asarray(heavy_ok, bool)
    rows = []
    for ct in conf_grid:
        for st in stab_grid:
            esc = (conf < ct) | (stab < st)
            ok = np.where(esc, heavy_ok, light_ok)
            rows.append(dict(conf_thr=float(ct), stab_thr=float(st), esc_rate=float(esc.mean()), acc=float(ok.mean())))
    return rows
