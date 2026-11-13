"""Summary of a capture's gt.jsonl: frames, classes, visibility (instance pixels), speeds, emergency-vehicle frames.
Usage: python gt_stats.py <capture_dir>   NOT RUN ON REAL DATA YET."""
import json
import os
import statistics as st
import sys
from collections import defaultdict

d = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(d, "gt.jsonl"), encoding="utf-8") if l.strip()]
print("frames:", len(rows), " t range: %.0f-%.0f s" % (rows[0]["t"], rows[-1]["t"]), " weather:", rows[0]["weather"],
      " visibility proxy:", rows[0]["visibility"])
obs = [(r, v) for r in rows for v in r["vehicles"]]
print("vehicle observations:", len(obs), " per frame (mean): %.1f" % (len(obs) / len(rows)))
cls = defaultdict(list)
for r, v in obs:
    cls[v["cls"]].append(v)
print("\nclass: observations | with instance pixels | median dist (m) | median bbox area (px^2)")
for c, vs in sorted(cls.items()):
    vis = [v for v in vs if v["inst_px"] > 0]
    area = [(v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]) for v in vs]
    print("  %-10s %5d | %5d (%.0f%%) | %6.1f | %8.0f" % (c, len(vs), len(vis), 100.0 * len(vis) / len(vs),
                                                          st.median(v["dist_m"] for v in vs), st.median(area)))
sp = [v["speed"] for _, v in obs]
print("\nspeed < 0.3 m/s: %.0f%% of observations;  median speed %.2f m/s" % (100.0 * sum(s < 0.3 for s in sp) / len(sp), st.median(sp)))
per = defaultdict(list)
for r, v in obs:
    per[v["id"]].append((r["t"], v["speed"], v["dist_m"]))
print("\nmost frequently seen ids: id | n | t first-last | speed min/max | dist first->last")
for vid, lst in sorted(per.items(), key=lambda kv: -len(kv[1]))[:8]:
    print("  %5d | %3d | %5.0f-%5.0f | %.1f/%.1f | %.0f->%.0f" % (vid, len(lst), lst[0][0], lst[-1][0],
                                                                  min(s for _, s, _ in lst), max(s for _, s, _ in lst),
                                                                  lst[0][2], lst[-1][2]))
em = [(r, v) for r, v in obs if v["cls"] != "car"]
print("\nemergency-class observations:", len(em))
for r, v in em[:12]:
    print("  t=%5.0f %-9s id=%d dist=%.0f m inst_px=%d bbox=%s" % (r["t"], v["cls"], v["id"], v["dist_m"], v["inst_px"], v["bbox"]))
