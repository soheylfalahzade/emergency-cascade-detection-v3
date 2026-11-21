#!/usr/bin/env python3
"""Builds docs/report/REPORT.md from the result files of this repository.
Numbers come from results_v4_frcnn60/, results_v3/, results_coco/, phase2_bridge/ and docs/leakage_audit/;
nothing in the tables is typed by hand. Prose is written for the labels DEV / FROZEN-ONCE / POST-HOC /
CARLA / abstract queue model (see Section 3.6).
Usage (repo root):  python figures_pipeline/build_report.py
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_refs import REFS

OUT = ROOT / "docs" / "report"
OUT.mkdir(parents=True, exist_ok=True)
V4, V3, CO = ROOT / "results_v4_frcnn60", ROOT / "results_v3", ROOT / "results_coco"


def rd(p):
    return Path(p).read_text(encoding="utf-8-sig")


def jl(p):
    return json.loads(rd(p))


def split_sections(md, level="## "):
    parts, cur, name = {}, [], "_head"
    for line in md.splitlines():
        if line.startswith(level):
            parts[name] = "\n".join(cur).strip()
            name, cur = line[len(level):].strip(), []
        else:
            cur.append(line)
    parts[name] = "\n".join(cur).strip()
    return parts


def find(parts, prefix):
    for k, v in parts.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


def sec(md, start, end):
    """Text between a heading line starting with `start` and the next heading line starting with `end` (heading line itself dropped)."""
    lines, on, out = md.splitlines(), False, []
    for ln in lines:
        if ln.startswith(start) and not on:
            on = True
            continue
        if on and ln.startswith(end):
            break
        if on:
            out.append(ln)
    if not on:
        raise KeyError(start)
    return "\n".join(out).strip()


def fix_paths(t, prefix="../../"):
    t = re.sub(r'(src|href)="(?!https?://|#)', lambda m: f'{m.group(1)}="{prefix}', t)
    t = re.sub(r"\]\((?!https?://|#)", f"]({prefix}", t)
    return t


def f4(x):
    return f"{x:.4f}"


def sgn(x):
    return f"{x:+.4f}"


def ci(c):
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


# ------------------------------------------------------------------ load data
final = rd(V4 / "final_report.md")
fparts = split_sections(final)
cv_tbl = find(fparts, "5-fold CV")
casc_tbl = find(fparts, "Cascade (light")
cost_tbl = find(fparts, "Cost")
hold_tbl = find(fparts, "Holdout")
hold_main = hold_tbl.split("Paired differences")[0].strip()
ext = rd(V4 / "cascade_ext_report.md")
ext_tbl = "\n".join(l for l in ext.splitlines() if l.startswith("|"))
ext_pairs = "\n".join(l for l in ext.splitlines() if l.startswith("- "))
NICE = {"single_yolov8n": "YOLOv8n", "single_rtdetr_l": "RT-DETR-l", "single_fasterrcnn": "Faster R-CNN", "nms_union": "NMS union",
        "wbf": "WBF", "top1_ensemblenet_style": "Top-1", "softavg": "Soft average", "meta_LR": "Stacking: LR", "meta_HGB": "Stacking: HGB",
        "meta_RF": "Stacking: RF", "meta_SVM": "Stacking: SVM", "meta_MLP": "Stacking: MLP", "meta_KNN": "Stacking: KNN",
        "cascade_frozen": "Cascade (frozen policy)"}


def pretty(t):
    out = []
    for ln in t.splitlines():
        m = re.match(r"^\| (\S+) \|(.*)$", ln)
        if m and m.group(1) in NICE:
            ln = f"| {NICE[m.group(1)]} |{m.group(2)}"
        out.append(ln)
    return "\n".join(out)


def esc_ext(t):
    out = []
    for ln in t.splitlines():
        m = re.match(r"^\| (.*?) \| ([0-9.]+) \| ([0-9.]+) \| (.*)$", ln)
        if m:
            ln = "| " + m.group(1).replace("|", "\\|") + f" | {m.group(2)} | {m.group(3)} | {m.group(4)}"
        out.append(ln)
    return "\n".join(out)


def clean_pairs(t):
    def fx(m):
        return f"[{float(m.group(1)):+.4f}, {float(m.group(2)):+.4f}]"
    t = re.sub(r"\[(-?[0-9.]+), (-?[0-9.]+)\]", fx, t)
    return t


ext_tbl = esc_ext(ext_tbl)
ext_pairs = clean_pairs(ext_pairs)
v3final = rd(V3 / "final_report.md")
v3parts = split_sections(v3final)
v3_cv = pretty(find(v3parts, "5-fold CV"))
v3_hold_all = find(v3parts, "Holdout")
v3_hold = pretty(v3_hold_all.split("Paired differences")[0].strip())
v3_pairs = clean_pairs("\n".join(l for l in v3_hold_all.splitlines() if l.startswith("- ")))
proto = jl(V4 / "protocol_frozen.json")
proto3 = jl(V3 / "protocol_frozen.json")
bench = jl(V4 / "bench.json")
hold = jl(V4 / "holdout_final.json")
pd_ = hold["paired_differences"]
coco = jl(CO / "coco_final.json")
cx = jl(CO / "coco_extra.json")
cpt = {k: v["map50"] if isinstance(v, dict) else v for k, v in coco["point"].items()}
cpt.update(cx.get("point", {}) and {k: (v["map50"] if isinstance(v, dict) else v) for k, v in cx["point"].items()})
cci = dict(coco["map50_ci95"])
cci.update(cx.get("map50_ci95", {}))
audit = rd(ROOT / "docs/leakage_audit/audit_holdout_v3.txt")
readme = rd(ROOT / "README.md")
absmd = rd(ROOT / "phase2_bridge/ABSTRACT_MODEL.md")
abs_parts = split_sections(absmd)

heavy, light = cpt["single_rtdetr_l"], cpt["single_yolov8n"]
cv = {}
for ln in cv_tbl.splitlines():
    c = [x.strip() for x in ln.strip("|").split("|")]
    if c and c[0].startswith(("single_", "wbf", "softavg", "meta_", "nms", "top1")):
        cv[c[0]] = float(c[1].split("±")[0])
casc = {}
for ln in casc_tbl.splitlines():
    c = [x.strip() for x in ln.strip("|").split("|")]
    if len(c) >= 4 and c[0] in ("light_only", "heavy_only", "band<0.7"):
        casc[c[0]] = (float(c[1]), float(c[2].split("±")[0]), float(c[3]))
sp = {}
for ln in hold_main.splitlines():
    c = [x.strip() for x in ln.strip("|").split("|")]
    if c and c[0].startswith(("single_", "wbf", "softavg", "meta_", "nms", "top1", "cascade")):
        sp[c[0]] = float(c[1])

# ------------------------------------------------------------------ tables from JSON
cv_tbl = pretty(cv_tbl)
hold_main = pretty(hold_main)

def paired_holdout():
    names = [("cascade_minus_heavy", "Cascade − RT-DETR-l"), ("cascade_minus_light", "Cascade − YOLOv8n"),
             ("meta_minus_heavy", "Stacking (LR) − RT-DETR-l"), ("softavg_minus_heavy", "Soft average − RT-DETR-l"),
             ("wbf_minus_heavy", "WBF − RT-DETR-l"), ("meta_minus_wbf", "Stacking (LR) − WBF")]
    rows = ["| Comparison | Mean difference in mAP50 | 95% group-bootstrap interval | Interval excludes 0 |", "|---|---|---|---|"]
    for k, lab in names:
        d = pd_[k]
        ex = "yes" if not (d["ci95"][0] <= 0 <= d["ci95"][1]) else "no"
        rows.append(f"| {lab} | {sgn(d['mean'])} | {ci(d['ci95'])} | {ex} |")
    return "\n".join(rows)


def coco_main():
    order = [("single_yolov8n", "YOLOv8n"), ("single_rtdetr_l", "RT-DETR-l"), ("single_fasterrcnn", "Faster R-CNN"),
             ("nms_union", "NMS union"), ("wbf", "WBF"), ("softavg", "Soft average"), ("meta_LR", "Stacking (LR)")]
    rows = ["| Method | mAP50 | 95% bootstrap interval |", "|---|---|---|"]
    for k, lab in order:
        i = cci.get(k)
        rows.append(f"| {lab} | {f4(cpt[k])} | {('[%.3f, %.3f]' % tuple(i)) if i else '–'} |")
    return "\n".join(rows)


def coco_paired():
    P = coco["paired"]
    items = [("meta_LR-single_rtdetr_l", "Stacking (LR) − RT-DETR-l", P), ("softavg-single_rtdetr_l", "Soft average − RT-DETR-l", P),
             ("wbf-single_rtdetr_l", "WBF − RT-DETR-l", P), ("meta_LR-wbf", "Stacking (LR) − WBF", P),
             ("cascade_band0.7_transfer-single_rtdetr_l", "Cascade (band<0.7, no recalibration) − RT-DETR-l", P),
             ("single_fasterrcnn-single_rtdetr_l", "Faster R-CNN − RT-DETR-l (added later, POST-HOC)", cx),
             ("softavg-single_fasterrcnn", "Soft average − Faster R-CNN (added later, POST-HOC)", cx)]
    rows = ["| Comparison | Mean difference in mAP50 | 95% bootstrap interval | Interval excludes 0 |", "|---|---|---|---|"]
    for k, lab, src in items:
        d = src[k]
        ex = "yes" if not (d["ci95"][0] <= 0 <= d["ci95"][1]) else "no"
        rows.append(f"| {lab} | {sgn(d['mean'])} | {ci(d['ci95'])} | {ex} |")
    return "\n".join(rows)


def coco_cascade():
    P = coco["paired"]
    rows = ["| Risk level ρ | Images escalated | mAP50 | Cascade − RT-DETR-l | Share of light-to-heavy gap closed (median, 95%) | Share closed by random routing, same escalation (median, 95%) |",
            "|---|---|---|---|---|---|"]
    for r in ("0.333", "0.500", "0.667"):
        d = P[f"cascade_r{r}-single_rtdetr_l"]
        g = P[f"share of light->heavy gap closed by cascade_r{r}"]
        gr = P[f"share of light->heavy gap closed by random_r{r}"]
        rows.append(f"| {r} | {coco['esc_rate'][r]*100:.1f}% | {f4(cpt['cascade_r'+r])} | {sgn(d['mean'])} {ci(d['ci95'])} | "
                    f"{g['median']:.2f} [{g['ci95'][0]:.2f}, {g['ci95'][1]:.2f}] | {gr['median']:+.2f} [{gr['ci95'][0]:+.2f}, {gr['ci95'][1]:+.2f}] |")
    return "\n".join(rows)


def per_class_holdout():
    S = hold["summary"]
    lab = {"single_yolov8n": "YOLOv8n", "single_rtdetr_l": "RT-DETR-l", "single_fasterrcnn": "Faster R-CNN", "nms_union": "NMS union",
           "wbf": "WBF", "top1_ensemblenet_style": "Top-1", "softavg": "Soft average", "meta_LR": "Stacking (LR)", "cascade_frozen": "Cascade (frozen)"}
    rows = ["| Method | mAP50 | AP50 ambulance | AP50 police | AP50 firetruck | AP50 car |", "|---|---|---|---|---|---|"]
    for k, n in lab.items():
        s = S[k]
        rows.append(f"| {n} | {s['map50']['mean']:.4f} | {s['ap50_ambulance']['mean']:.3f} | {s['ap50_police']['mean']:.3f} | "
                    f"{s['ap50_firetruck']['mean']:.3f} | {s['ap50_car']['mean']:.3f} |")
    return "\n".join(rows)


def bench_tbl():
    rows = ["| Model | Parameters (M) | GFLOPs | Median ms per image | Frames per second |", "|---|---|---|---|---|"]
    for k, n in (("yolov8n", "YOLOv8n"), ("rtdetr_l", "RT-DETR-l"), ("fasterrcnn", "Faster R-CNN")):
        b = bench[k]
        rows.append(f"| {n} | {b['params_M']:.1f} | {b['flops_G']:.1f} | {b['ms_median']:.1f} | {b['fps_median']:.1f} |")
    return "\n".join(rows)


# manifest facts (read from the file, not typed)
import csv, collections
man = list(csv.DictReader(open(ROOT / "datasets/folds_v3_manifest.csv", encoding="utf-8-sig", newline="")))
n_man = len(man)
by_set = collections.Counter(r["set"] for r in man)
by_src = collections.Counter(r["src"] for r in man)
by_cls = collections.Counter(r["rarity"] for r in man)
set_rows = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(by_set.items()))
src_rows = "\n".join(f"| `{k}` | {v:,} |" for k, v in sorted(by_src.items(), key=lambda x: -x[1]))
cls_rows = "\n".join(f"| {k} | {by_cls[k]:,} |" for k in ("ambulance", "police", "firetruck", "car"))

# README-derived sections (verified public text; image paths rewritten for docs/report/)
m = re.findall(r"```mermaid\n(.*?)```", readme, re.S)
mer_overview, mer_cascade, mer_control = m[0], m[1], m[2]
carla_clean = fix_paths(sec(readme, "## 4.5", "## 4.6"))
carla_dom = fix_paths(sec(readme, "## 4.6", "## 4.7"))
carla_loop = fix_paths(sec(readme, "## 4.7", "## 4.8"))
carla_renders = fix_paths(sec(readme, "## 4.8", "## 4.9"))
leak = fix_paths(sec(readme, "## 4.10", "## 4.11"))
absreadme = fix_paths(sec(readme, "## 4.11", "## 4.12"))
app_first = fix_paths(sec(readme, "## 4.12", "# Chapter 5"))
disc_p1 = sec(readme, "## 5.1", "## 5.2")
disc_p2 = sec(readme, "## 5.2", "## 5.3")
disc_val = sec(readme, "## 5.3", "## 5.4")
fut = sec(readme, "## 5.5", "---")
sec_control = sec(readme, "## 3.6", "---")
sec_ctrl_body = fix_paths(sec_control.split("```\n", 1)[1] if "```\n" in sec_control else sec_control)
train_tbl = sec(readme, "## 3.3", "## 3.4").split("Training arguments")[0].strip()

abs1 = find(abs_parts, "1. Policies")
abs2 = find(abs_parts, "2. 2-hop minus")
abs3 = find(abs_parts, "3. Where does 2-hop")
abs4 = find(abs_parts, "4. Hop-2 weight")
abs5 = find(abs_parts, "5. Earlier run")

# ------------------------------------------------------------------ document
D = []
def w(s=""):
    D.append(s)

cv_gain = cv["meta_LR"] - cv["single_rtdetr_l"]
saving = casc["band<0.7"][2] / casc["heavy_only"][2]
w(f"""<div align="center">

# Emergency-Vehicle-Aware Traffic Signal Control
### Detection, Cascade Inference and Closed-Loop Evaluation with Emergency-Vehicle Priority

**Soheil Falahzade**

November 2026

![Perception](https://img.shields.io/badge/Part%20I-perception-1f6feb?style=flat-square)
![Control](https://img.shields.io/badge/Part%20II-control%20in%20CARLA-2da44e?style=flat-square)
![Protocol](https://img.shields.io/badge/evaluation-frozen%20protocol%20%2B%20leakage%20audit-d1242f?style=flat-square)
![Simulator](https://img.shields.io/badge/simulator-CARLA%200.9.16-e36209?style=flat-square)

</div>

---

## Abstract

A traffic signal can only give way to an emergency vehicle that it has noticed. This report follows that chain from the camera image to the signal phase. **Part I (perception)** compares three detectors (YOLOv8n, RT-DETR-l, Faster R-CNN) on four vehicle classes (ambulance, police car, fire truck, car) with 5-fold cross-validation and one holdout, tests ensembling and stacking against the best single detector, and studies a light/heavy cascade that sends only uncertain images to the heavy detector. **Part II (control)** compares a fixed-time controller, simple preemption, max-pressure control and a fuzzy controller with emergency arbitration in closed loop in the CARLA simulator, with ground-truth perception, injected perception errors and a real detector in the loop; a separate abstract queue model is used to study message passing between neighbouring junctions.

The main findings are: (i) RT-DETR-l is the most accurate detector (cross-validation mAP50 {f4(cv['single_rtdetr_l'])}; holdout {f4(sp['single_rtdetr_l'])}, POST-HOC). (ii) Ensembling and stacking give no measurable gain on the emergency-vehicle data (stacking minus RT-DETR-l on the holdout: {sgn(pd_['meta_minus_heavy']['mean'])}, interval {ci(pd_['meta_minus_heavy']['ci95'])}); on a COCO benchmark soft averaging and stacking exceed the best single detector by small margins. (iii) The cascade is a speed/accuracy trade-off, not an accuracy gain: in cross-validation it keeps {casc['band<0.7'][1]:.4f} mAP50 at {casc['band<0.7'][2]:.1f} ms per image instead of {casc['heavy_only'][1]:.4f} at {casc['heavy_only'][2]:.1f} ms, and on the holdout it is below the heavy model ({sgn(pd_['cascade_minus_heavy']['mean'])}). (iv) In closed loop in CARLA, simple preemption already reaches the floor of about 2 s of ambulance stopped time; the fuzzy controller matches but does not beat it for the ambulance and reduces ordinary-vehicle stopping time; with the detector in the loop no ambulance advantage of the fuzzy controller was found. (v) A detector trained only on real images almost never finds vehicles in CARLA frames, and fine-tuning closes the gap only within the same junction and vehicle models. (vi) In the abstract queue model, extending messages from one to two junctions each way does not help at the base demand and helps the emergency delay only under saturated demand.

Every number carries an evidence label (DEV, FROZEN-ONCE, POST-HOC, CARLA, abstract queue model; Section 3.6), negative and neutral results are reported as they are, and a perceptual-hash audit of the holdout is included together with its limits.

**Keywords:** emergency vehicle detection, traffic signal control, signal preemption, model ensembling, stacking, cascade inference, CARLA, evaluation protocol, data leakage.

---

## Contents

1. [Introduction](#1-introduction)
2. [Background and related work](#2-background-and-related-work)
3. [Method](#3-method)
4. [Results](#4-results)
5. [Discussion](#5-discussion)
6. [Conclusion and future work](#6-conclusion-and-future-work)
7. [References](#references)
8. [Appendices](#appendices)

> [!IMPORTANT]
> **How to read the numbers.** **DEV**: a choice (checkpoint, threshold, design) was made on the same data, so the number is optimistic. **FROZEN-ONCE**: rules were written to a file before the holdout was touched and the holdout was evaluated once. **POST-HOC**: computed after the holdout had been consumed, or defined after seeing results. **CARLA**: measured in the CARLA simulator on one junction, never a real-world measurement. **Abstract queue model**: output of a simplified queue model, never evidence about CARLA or real traffic.

---

# 1. Introduction

## 1.1 Motivation

Traffic signals control the flow of urban traffic at junctions. Emergency vehicles (ambulances, fire trucks, police cars) behave differently from other traffic: arrival time matters, and waiting at a red signal or in a junction queue costs part of it. Signal preemption temporarily changes the signal plan to clear the path of an emergency vehicle [[nelson,chen24]]. Preemption is not free. It can lengthen queues and delays for ordinary traffic at the same junction and at neighbouring junctions. In a case study of closely spaced junctions on an arterial, a single preemption request had little effect on the corridor, whereas several requests in quick succession had a stronger effect, with arterial travel time rising by about 20 to 30 s in the worst case [[nelson]]. Later work stresses queue spillback and the negative impact on non-priority traffic [[chen24]].

Every preemption starts with awareness that an emergency vehicle is present. Image-based detection of emergency vehicles with deep learning has received attention, with datasets and models for ambulances, fire trucks and police cars [[akdag,maleki,ibrahim]]. Detection and signal control are, however, usually evaluated separately, and it is rarely asked how a perception error (a missed emergency vehicle, or a false alarm) changes the quality of signal control and the delay of ordinary traffic.

## 1.2 Problem statement

The problem has three connected layers.

**Perception.** From a camera image the system must decide whether an emergency vehicle is approaching and of which type. There are two error types: missing the vehicle (preemption is late or never happens) and a false alarm (an unnecessary preemption disturbs ordinary traffic). Accurate models are usually slower and more expensive, so accuracy has to be traded against computation.

**Decision.** At each moment the controller chooses which approach receives green. The decision must account for the emergency vehicle and for the delay of ordinary vehicles, and it must remain acceptable when perception is imperfect.

**Evaluation.** Results of both layers have to be reported with a method that can be trusted. Data leakage between training and test data, repeated use of one test set, and non-reproducible simulation runs can make results too optimistic. Leakage is a documented cause of failed reproducibility in machine-learning-based research [[kapoor]].

## 1.3 Objectives and research questions

The objective is an honest evaluation of the chain from emergency-vehicle detection to signal control with priority. The questions are:

1. **RQ1.** Which detector gives the best trade-off between accuracy and cost on four vehicle classes?
2. **RQ2.** Does combining detectors (ensembling, stacking) improve on the best single detector by a measurable amount?
3. **RQ3.** Can a cascade keep most of the accuracy of the heavy detector at lower expected cost?
4. **RQ4.** Does a fuzzy emergency-aware controller beat fixed-time control, simple preemption and max-pressure, for the emergency vehicle and for ordinary traffic?
5. **RQ5.** How does perception error change the control results, and how large is the gap between a detector trained on real images and CARLA frames?
6. **RQ6.** Do the findings on RQ2 and RQ3 depend on the emergency-vehicle data, or do they appear on a public benchmark (COCO) as well?

## 1.4 Contributions

- A frozen evaluation protocol for the perception part with explicit result labels, and a perceptual-hash leakage audit of the holdout reported together with its limits.
- A cascade evaluated as a speed/accuracy trade-off with an explicit selection rule, including a negative result for cascades that use Faster R-CNN.
- An external check of ensembling, stacking and the cascade on COCO with three pre-trained detectors.
- A closed-loop CARLA study that compares ground-truth perception, injected perception error and a detector in the loop, and documents the domain gap between real and simulated images.
- Identification and repair of a reproducibility defect in the generation of injected perception noise (dependence on the order of the simulator's actor list); results from before the repair are discarded.
- A message-passing study in an abstract queue model, with an explicit statement that it was not run in CARLA.
- Code, result files and figure generators for all of the above in this repository.

## 1.5 Scope and limitations

The control study covers one junction on one simulator map. Results must not be generalised to other junctions or to real traffic. The held-out CARLA capture uses the same junction and the same vehicle models and is therefore not a new environment. Messaging up to two junctions before and after is implemented and evaluated only in the abstract queue model, not in CARLA. No real traffic data were used, and the simulation-to-reality gap is not measured. Reinforcement learning is not used; the comparison policies are rule-based and stabilising controllers.

## 1.6 Organisation

Chapter 2 reviews the background. Chapter 3 describes the data, models, protocols and simulation set-ups. Chapter 4 reports results. Chapter 5 interprets them and lists threats to validity. Chapter 6 concludes. Appendices give the first (15-epoch) Faster R-CNN run, the extended cascade study, a repository map and reproduction commands.

---

# 2. Background and related work

## 2.1 Traffic signal control

Traffic signal control aims to reduce vehicle travel time by coordinating movements at junctions. A survey notes that deployed systems still rely largely on simplified information and rule-based methods, although richer data and more computing power make more intelligent methods possible [[wei]].

**Max-pressure.** Max-pressure control is a family of decentralised controllers for networks of signalised junctions. Varaiya [[varaiya]] introduced it: the phase at each junction is chosen as a function of network queues, and the controller is stabilising if the queues stay bounded. Kouvelas et al. [[kouvelas]] extended it to arterial networks and showed that the decentralised controller does not need the current or future mean demand, although it needs the mean turn ratios. Its simplicity and scalability make it a strong comparison policy for ordinary traffic, which is how it is used here.

**Fuzzy control.** Fuzzy logic has been applied to signal control since the 1970s. Pappis and Mamdani [[pappis]] built a fuzzy controller for a junction of two one-way streets that turns linguistic rules into control decisions and reported lower mean vehicle delay than a vehicle-actuated controller. Chou and Teng [[chou]] proposed another fuzzy controller and studied low, medium and heavy traffic for single and multiple junctions. Fuzzy rules express decision logic in terms close to engineering intuition and combine several criteria such as queue length and waiting time; the price is that rules and membership functions are designed by hand.

**Reinforcement learning.** Reinforcement learning is widely studied for signal control, because existing plans do not provide supervised labels that separate good from bad plans [[wei]]. It is not used in this report.

| Approach | Main idea | Note | Source |
|---|---|---|---|
| Rule-based / fixed plan | predefined plans, simplified information | still common in deployed systems | [[wei]] |
| Max-pressure | phase choice from queues, decentralised | queue stability; needs mean turn ratios | [[varaiya,kouvelas]] |
| Fuzzy | linguistic rules turned into fuzzy sets | better than vehicle-actuated control in an early single-junction test | [[pappis,chou]] |
| Reinforcement learning | learn from the outcome of actions | no supervised labels for good plans | [[wei]] |

## 2.2 Signal preemption for emergency vehicles

Emergency vehicle preemption (EVP) reduces the stopped time of the emergency vehicle at a junction and improves safety [[nelson,chen24]]. Nelson and Bullock [[nelson]] studied preemption on closely spaced junctions along an arterial; one request had a small effect, repeated requests a stronger one. Chen et al. [[chen24]] described a preemption system that takes queue spillback along the route and the negative impact on non-priority traffic into account, with algorithms for restoring the traffic state and a soft return to the normal plan, and reported a clear improvement of emergency travel time with lower impact on non-priority traffic.

Two points matter here. Preemption is a trade-off: the benefit to the emergency vehicle is paid for by ordinary traffic. And such evaluations usually assume that the presence of the emergency vehicle is announced correctly and on time. What happens when the announcement comes from an image detector that makes mistakes is the subject of this report.

## 2.3 Emergency-vehicle detection

Akdag et al. [[akdag]] built a large dataset of critical vehicles (fire truck, police car, ambulance, military police car, hazardous-materials truck, standard vehicle) and compared a vision transformer with traditional convolutional networks for classification; they stress the rarity of emergency vehicles and the different livery across countries, and report an average precision and recall of about 99.39% and 99.34% for the transformer-based model. Maleki et al. [[maleki]] introduced a dataset with classes such as ambulance and fire truck for vehicle detection with YOLO. Ibrahim et al. [[ibrahim]] combined YOLOv8 with an LSTM on audio and video input and report 96.6% accuracy for detection and classification.

These studies evaluate the detector or classifier itself. The effect of its errors on a signal controller in closed loop is not reported in the sources reviewed here.

## 2.4 Object detectors

**Two-stage: Faster R-CNN.** Ren et al. [[ren]] introduced the region proposal network, which shares convolutional features with the detection network and makes region proposals almost free; this architecture underlies several winning entries of ILSVRC and COCO 2015.

**One-stage: YOLO.** Redmon et al. [[redmon]] cast detection as regression to spatially separated boxes and class probabilities, predicted by one network from the full image; the base model ran at 45 frames per second on a GPU. Later members of the family, including YOLOv8 in the Ultralytics library [[ultralytics]], are used here through the light model YOLOv8n.

**Transformer-based: RT-DETR.** Zhao et al. [[zhao]] observed that the speed and accuracy of YOLO detectors suffer from non-maximum suppression (NMS) and presented the real-time end-to-end detector RT-DETR, which does not need NMS; its RT-DETR-R50 variant reaches 53.1% AP on COCO at 108 frames per second on a T4 GPU. Here the RT-DETR-l variant is used.

The general trade-off between these families, accuracy against cost, is measured in Part I as parameters, floating-point operations and latency on a laptop GPU.

## 2.5 Combining models and cascades

**Combining detections.** NMS and its soft variants only select boxes and cannot produce an averaged localisation from several models [[solovyev]]. Solovyev et al. [[solovyev]] proposed weighted boxes fusion (WBF), which uses the confidence of all proposed boxes to build averaged boxes, and applied it to Open Images and COCO.

**Stacking.** Wolpert [[wolpert]] introduced stacked generalisation, in which the outputs of several generalisers are combined by a higher-level model that learns their biases from held-out data (in the cross-validation fashion). He describes it as a more sophisticated version of cross-validation that combines the models instead of letting the winner take all [[wolpert]].

**Cascades.** The cascade idea has a long history in computer vision: Viola and Jones [[viola]] detect objects with a chain of increasingly complex classifiers that discard background regions early and spend more computation on promising regions. The cascade used here differs in that the decision is made per image between two deep detectors.

## 2.6 Simulation and evaluation methodology

**CARLA.** CARLA is an open-source simulator for urban driving research, built for developing, training and validating driving systems, with flexible specification of sensors and environmental conditions [[carla]]. It allows camera images of a junction in several weather conditions and closed-loop signal control. Simulation has its own limits: the appearance of images and the behaviour of vehicles differ from the real world, and this gap is not measured here.

**Cross-validation and bootstrap.** Cross-validation and the bootstrap are two common ways to estimate accuracy and select models; Kohavi [[kohavi]] compares them experimentally. The bootstrap was introduced by Efron [[efron]]: resampling with replacement from the observed data to estimate the sampling distribution of a statistic. Here 5-fold cross-validation is used for selection (hence the label DEV), and bootstrap resampling at the level of image groups gives confidence intervals on the holdout. Group-level resampling is an adaptation of the basic method [[efron]], not a new method.

**External benchmark.** COCO [[lin]] contains images of everyday scenes with 91 object types, 2.5 million labelled instances in 328 thousand images. It is used here only for a supplementary check with three detectors pre-trained on COCO (no training by us) on several vehicle classes, to see whether the comparison of ensembling and cascade depends on emergency-vehicle data. It makes no claim about emergency vehicles and cannot be compared with the COCO leaderboard.

**Leakage and reproducibility.** Data leakage, where information from the test set enters training or model selection, produces over-optimistic results. Kapoor and Narayanan [[kapoor]] surveyed previous research and found leakage affecting at least 294 papers in 17 scientific fields, and proposed a taxonomy of leakage types and model information sheets to prevent it. This motivates the frozen protocol, the separation of the test set, the labelling of results and the leakage audit in this report.

## 2.7 Gap addressed

For each link in the chain "see, decide, act" there are suitable methods: stabilising and fuzzy controllers [[varaiya,kouvelas,pappis,chou]], preemption that accounts for the effect on ordinary traffic [[nelson,chen24]], and detectors and datasets for emergency vehicles [[akdag,maleki,ibrahim]]. This report addresses (a) joint evaluation of detection and control in closed loop, including the effect of perception error; (b) an honest comparison of control policies under error-free, injected and real-detector perception, with negative and neutral results; and (c) strict use of a frozen protocol and leakage audit for the detectors [[kapoor]], with the cost/accuracy trade-off of cascade and model combination [[solovyev]].

---

# 3. Method

## 3.1 System overview

```mermaid
{mer_overview.strip()}
```

## 3.2 Data and splits (Part I)

**Classes:** ambulance, police, fire truck, car. The dataset is merged from several image sources by `datasets/merge_emergency.py` (source folders are listed below) and split by `datasets/make_folds_v3.py`: images are grouped by perceptual-hash similarity (connected components), the groups are assigned to five folds with a stratified group k-fold (seed 42), a holdout is cut out, and three car-only sources (`kitti_yolo`, `miotcd_yolo`, `visdrone_yolo`) are used for training only (script constants `TRAIN_ONLY`, `SEED`, `T_PURGE`). Datasets are not included in this repository (size and licence terms); the preparation scripts are in `datasets/`.

The fold manifest (`datasets/folds_v3_manifest.csv`) lists {n_man:,} grouped images. Counts below are read from the file.

| Set | Images in manifest |
|---|---|
{set_rows}

| Source folder | Images in manifest |
|---|---|
{src_rows}

| Class label of the image group (rarity) | Images in manifest |
|---|---|
{cls_rows}

According to the audit log, each fold directory contains 28,788 images (28,099 training and 689 validation images per fold). The relation between this count and the {n_man:,} manifest rows (the manifest lists grouped images; other images are not listed in it) is not documented in the files of this repository, and it is not interpreted here.

**Holdout:** 381 images (boxes: ambulance {hold['boxes']['ambulance']}, police {hold['boxes']['police']}, firetruck {hold['boxes']['firetruck']}, car {hold['boxes']['car']}). Intervals are 95% group-bootstrap intervals ({hold['bootstrap']} resamples) over image groups from the manifest. Car labels in validation and holdout data are incomplete, so car AP is not a clean metric; police has very few holdout boxes, so any police number needs its interval.

## 3.3 Detectors and training protocol

{train_tbl}

Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`) is trained with SGD (learning rate 0.005, momentum 0.9, weight decay 0.0005, step schedule), mixed precision and batch 4; training arguments of YOLOv8n and RT-DETR-l are stored in `results_v3/train_logs/*/args.yaml`. All methods are scored with the same code: torchmetrics mAP, confidence ≥ 0.02, top 100 detections per image. Software versions are recorded in the frozen protocol (torch {proto['versions']['torch']}, ultralytics {proto['versions']['ultralytics']}).

> [!NOTE]
> **Faster R-CNN was trained twice.** The first run used 15 epochs and the last checkpoint, which did not match the other two models. It was retrained under the shared protocol (60 epochs, patience 15, best checkpoint by mAP50-95). Cross-validation mAP50 changed from 0.8424 to 0.8508 (+0.0084), so the retraining changed no conclusion; first-run numbers are kept in Appendix A. The best checkpoint of every model is selected on the validation fold that the meta-learner and cross-validation also use; cross-validation numbers are therefore DEV.

## 3.4 Ensembles and stacking

- **Ensembles over detections:** NMS union, weighted boxes fusion (WBF) [[solovyev]], soft average, top-1.
- **Stacking** [[wolpert]]: meta-learners (LR, HGB, RF, SVM, MLP, KNN) trained on cross-validated detections (leave-one-fold-out meta). The meta-learner is chosen by a rule fixed beforehand (highest CV mAP50): `{proto['meta']}`.
- Two-model ablations (`stack_cv_2model_*.json`) are DEV.

## 3.5 The light/heavy cascade

The light model (YOLOv8n) answers; images whose light-model confidence falls in an uncertainty band are sent whole to the heavy model (RT-DETR-l). The policy is the smallest escalation rate whose CV mAP50 stays within 0.01 of the best single heavy model. The expected cost per image uses the median latency of Section 3.7.

```mermaid
{mer_cascade.strip()}
```

**Frozen protocol** ({proto['time']}): meta = {proto['meta']}, heavy = {proto['heavy']}, cascade = {proto['policy']} (fallback = {proto['fallback']}). Rules written in the file before the holdout was evaluated:

""" + "\n".join(f"- {r}" for r in proto["rules"]) + f"""

The extended study (Appendix B) also tries Faster R-CNN as the heavy model or as a second tier. It is DEV only, never touches the holdout, and its selection rule is fixed in `protocol_frozen_ext.json`.

## 3.6 Evaluation protocol and result labels

| Label | Meaning |
|---|---|
| **DEV** | a choice (checkpoint, threshold, design) was made while looking at the same data; optimistic |
| **FROZEN-ONCE** | rules written to a protocol file before the holdout was touched; holdout evaluated once |
| **POST-HOC** | computed after the holdout had been consumed, or defined after seeing results |
| **CARLA** | measured in the CARLA simulator on one junction; never a real-world measurement |
| **Abstract queue model** | output of `phase2_bridge/toy_sim.py`; figures carry the watermark "abstract queue model - not CARLA" |

The holdout was consumed by the first (15-epoch Faster R-CNN) run (`results_v3/holdout_final.json`, FROZEN-ONCE) and then evaluated a second time under the protocol of the retrained run; the second evaluation is POST-HOC (`results_v4_frcnn60/holdout_done.flag`: {rd(V4/'holdout_done.flag').strip()}). Any further evaluation on it is POST-HOC, and a new protocol must be frozen before looking at results. Hashes of the scripts used are stored in the protocol files.

## 3.7 Cost measurement

Parameters, floating-point operations (counted with `torch` FlopCounterMode, 1 MAC = 2 FLOPs) and end-to-end latency (batch 1, including pre-processing and NMS, excluding disk read) are measured on {bench['gpu']} for {bench['n_images']} images of fold {bench['fold']}. Laptop GPU timing is unstable; it is used for ordering, not for precise ratios.

## 3.8 External check on COCO

Three COCO-pre-trained detectors (no training by us) are evaluated on the vehicle classes car, motorcycle, bus and truck. A development half and a test half of {coco['n_test_images']:,} images each are drawn with seed 42; everything is frozen on the development half before the test half is read (FROZEN-ONCE). Metric: torchmetrics mAP, not comparable with the COCO leaderboard [[lin]]. Cascade thresholds are calibrated on the development half with an exact (Clopper-Pearson) bound on the risk of losing a heavy-model detection (fixed-sequence procedure; τ = {coco['protocol']['tau']}, δ = {coco['protocol']['delta']}); the risk levels are ρ = 1/3, 1/2, 2/3. Comparisons added after the frozen evaluation (`coco_extra.json`, written later) are POST-HOC.

## 3.9 Leakage audit of the holdout

A perceptual-hash audit (64-bit difference hash, Hamming distance ≤ 4, mirrored copies included) compares the holdout images with the training folds (`stacking/audit_cross.py`; log in `docs/leakage_audit/audit_holdout_v3.txt`). The audit finds near-identical images; it does not recognise different frames of the same video or scene.

## 3.10 Control study in CARLA (Part II)

```mermaid
{mer_control.strip()}
```

{sec_ctrl_body}

## 3.11 Abstract queue model and two-hop messaging

`phase2_bridge/toy_sim.py` is a simplified queue model of a corridor of signalised junctions. It is used to study messages between neighbouring junctions: a controller may receive emergency notices and state from one junction on each side (1-hop) or two junctions on each side (2-hop). The 2-hop variant, with message delay and loss, a notices-only and a state-only variant, and a hop-2 weight, is implemented and evaluated **only in this model**; it was not run in CARLA. Intervals in this model are mean ± 1.96·SE over seeds, paired by seed (normal approximation, not a bootstrap). The number of junctions and the duration are not stored in the CSV files; the script defaults (5 junctions, 900 s, 20 seeds) are the likely settings and are not verified.

## 3.12 Reproducibility

Seeds, scripts and hashes of the frozen protocol are stored with the results. The noise generator in the CARLA study originally drew random numbers in the order of the simulator's actor list, which changes between runs; after sorting inputs with a stable key and an audit, noisy results reproduce exactly, and earlier noisy results are discarded. Appendix D lists commands.

---

# 4. Results

## 4.1 Part I: detectors, ensembles and stacking (DEV)

<p align="center"><img src="../readme_figures/fig_cv_map50.png" alt="Cross-validation mAP50 of detectors, ensembles and stacking" width="860"></p>

RT-DETR-l reaches **{f4(cv['single_rtdetr_l'])}** mAP50 and YOLOv8n **{f4(cv['single_yolov8n'])}**; Faster R-CNN (60 epochs) reaches {f4(cv['single_fasterrcnn'])} (0.8424 in the first 15-epoch run). Stacking with logistic regression reaches {f4(cv['meta_LR'])}, which differs from RT-DETR-l by {sgn(cv_gain)}; soft average and WBF stay below RT-DETR-l.

{cv_tbl}

<p align="center"><img src="../paper_figures/fig_training_curves.png" alt="Training curves per fold" width="860"></p>

*Validation mAP50-95 per epoch for every fold (DEV); the dot marks the selected epoch.*

<p align="center"><img src="../paper_figures/fig_per_class_ap50.png" alt="Per-class AP50" width="860"></p>

*Per-class AP50: cross-validation (DEV, mean ± std over folds) and holdout (POST-HOC, error bars: std over the five training runs, not a confidence interval). Class `car` is omitted because its labels are incomplete.*

<p align="center"><img src="../paper_figures/fig_seed_variability.png" alt="Seed variability of YOLOv8n" width="480"></p>

*Seed-to-seed spread of YOLOv8n best-epoch validation mAP50 per fold (DEV).*

## 4.2 Cascade: accuracy versus cost (DEV)

<p align="center"><img src="../readme_figures/fig_cascade_tradeoff.png" alt="Cascade accuracy versus cost" width="760"></p>

{casc_tbl}

The frozen policy `{proto['policy']}` escalates {casc['band<0.7'][0]*100:.0f}% of the images and reaches **{casc['band<0.7'][1]:.4f}** mAP50 at **{casc['band<0.7'][2]:.1f} ms** per image, against {casc['light_only'][2]:.1f} ms for the light model and {casc['heavy_only'][2]:.1f} ms for the heavy model alone. That is {saving*100:.0f}% of the heavy-model time at a cost of {casc['heavy_only'][1]-casc['band<0.7'][1]:.4f} mAP50 (DEV).

**Negative result (DEV).** The cascade was also tried with Faster R-CNN as the heavy model or as a second tier. Any cascade that sends images to Faster R-CNN costs the same or more and is the same or less accurate than the frozen `r|b1<0.7` policy; sending images to both heavy models does not improve accuracy either. The full table is in Appendix B.

## 4.3 Model cost

<p align="center"><img src="../readme_figures/fig_model_cost.png" alt="Model cost" width="860"></p>

{bench_tbl}

*{bench['gpu']}; {bench['n_images']} images, fold {bench['fold']}; timing is unstable.*

## 4.4 Holdout (POST-HOC)

The holdout had already been consumed by the first run, so these numbers carry the label POST-HOC even though the v4 protocol was frozen before this evaluation ({proto['time']}). The first run's holdout numbers (FROZEN-ONCE) are in Appendix A. Values are means over five fold-models; intervals are 95% group-bootstrap intervals.

<p align="center"><img src="../readme_figures/fig_holdout_map50.png" alt="Holdout mAP50" width="860"></p>

{hold_main}

{per_class_holdout()}

<p align="center"><img src="../readme_figures/fig_holdout_paired.png" alt="Holdout paired differences" width="860"></p>

{paired_holdout()}

**Reading.** Stacking brings no measurable gain over RT-DETR-l ({sgn(pd_['meta_minus_heavy']['mean'])} {ci(pd_['meta_minus_heavy']['ci95'])}). The cascade is worse than the heavy model ({sgn(pd_['cascade_minus_heavy']['mean'])}, interval excludes 0) and better than the light model ({sgn(pd_['cascade_minus_light']['mean'])}, interval excludes 0), so it is a speed/accuracy trade-off, not an accuracy gain. Stacking is better than WBF ({sgn(pd_['meta_minus_wbf']['mean'])}), but WBF is not itself better than RT-DETR-l. The escalation rate on the holdout was {hold['esc_rate_mean']*100:.1f}% (mean over five fold-models) against {casc['band<0.7'][0]*100:.1f}% in cross-validation.

## 4.5 COCO benchmark of the same methods (FROZEN-ONCE on the test half)

<p align="center"><img src="../paper_figures/fig_coco.png" alt="COCO results" width="900"></p>

{coco_main()}

Paired differences on the test half (95% bootstrap intervals):

{coco_paired()}

Risk-controlled cascade on COCO (threshold calibrated on the development half):

{coco_cascade()}

**Reading.** On COCO, soft average and stacking beat RT-DETR-l by a small amount; WBF does not. The cascade closes much more of the light-to-heavy gap than random routing at the same escalation rate but stays below the heavy model. The emergency-vehicle data therefore do not show the small ensemble gain that COCO shows (Chapter 5).

## 4.6 Leakage audit of the holdout (POST-HOC)

{leak}

## 4.7 Part II: policies without perception error (CARLA)

{carla_clean}

## 4.8 Part II: detectors on CARLA frames (domain gap)

{carla_dom}

## 4.9 Part II: injected noise versus detector in the loop (CARLA)

{carla_loop}

## 4.10 Part II: simulator renders (CARLA)

{carla_renders}

## 4.11 Abstract queue model and two-hop messaging (abstract queue model)

> [!WARNING]
> Nothing in this section is evidence about CARLA or real traffic.

{absreadme}

#### Policies on the corridor (demand λ = 0.12)

{abs1}

#### 2-hop minus 1-hop under message delay and loss

{abs2}

#### Where does 2-hop help? Demand sweep

{abs3}

#### Hop-2 weight sweep

{abs4}

#### Earlier 3-junction run

{abs5}

---

# 5. Discussion

## 5.1 Part I

""" + disc_p1.replace("(section 4.9 versus 4.4)", "(Section 4.5 versus Section 4.4)") + """

**Why the COCO result may differ.** The three COCO detectors were trained by others on COCO; the emergency-vehicle detectors were trained and selected on a four-class set where the best single model is already strong (RT-DETR-l) and where the other models are clearly weaker. When one member is much better than the rest, averaging has little to add; on COCO the three members are closer to each other (RT-DETR-l and Faster R-CNN differ by +0.0057, interval including 0). This is an interpretation, not a tested cause.

## 5.2 Part II

""" + disc_p2 + """

## 5.3 Threats to validity

""" + disc_val + f"""

- The abstract queue model is a simplification; its tables are about the model, not about traffic.
- The number of seeds is small (10 in CARLA, 20 in the abstract model); intervals are normal or bootstrap intervals over seeds and not predictions.

## 5.4 Negative and neutral results (summary)

| Question | Result | Label |
|---|---|---|
| Does retraining Faster R-CNN to the shared protocol change conclusions? | No (CV mAP50 0.8424 → 0.8508) | DEV |
| Does stacking beat RT-DETR-l on emergency-vehicle data? | No measurable gain | POST-HOC |
| Does a cascade through Faster R-CNN help? | No: no cheaper, no more accurate | DEV |
| Does the cascade match the heavy model on the holdout? | No: below it | POST-HOC |
| Does the fuzzy controller beat simple preemption for the ambulance? | No (matches it); lower ordinary-vehicle stopping | CARLA |
| Does 2-hop messaging help at base demand? | No (ordinary delay higher in 20 of 20 seeds) | abstract queue model |
| Does the hop-2 weight matter at base demand? | No measured effect | abstract queue model |

---

# 6. Conclusion and future work

## 6.1 Summary

RT-DETR-l is the most accurate of the three detectors; the cascade keeps most of its accuracy for a fraction of its cost in cross-validation ({casc['band<0.7'][2]:.1f} ms against {casc['heavy_only'][2]:.1f} ms) but stays below it on the holdout. Stacking gives no gain on the emergency-vehicle data and a small gain on COCO. In closed loop in CARLA, a fuzzy emergency-aware controller matches simple preemption for the ambulance and lowers ordinary-vehicle stopping time, but does not beat preemption for the ambulance, with or without perception error; perception quality in simulation is dominated by the domain gap, and injected noise gives pessimistic but unvalidated stress tests. In the abstract queue model, messaging two junctions each way is useful only in saturated demand.

## 6.2 Future work

""" + fut + """

---

# References
""")

# ---- fill citations in order of first appearance
body = "\n".join(D)
order = []
def num(m):
    keys = [k.strip() for k in m.group(1).split(",")]
    for k in keys:
        if k not in order:
            order.append(k)
    nums = sorted(order.index(k) + 1 for k in keys)
    return "[" + ", ".join(str(n) for n in nums) + "]"
body_split = body.split("# References")
main_text = re.sub(r"\[\[([a-z0-9_, ]+)\]\]", num, body_split[0])
missing = [k for k in order if k not in REFS]
assert not missing, missing
refs = "\n".join(f"{i+1}. {REFS[k]}" for i, k in enumerate(order))
unused = [k for k in REFS if k not in order]
if unused:
    refs += "\n" + "\n".join(f"{len(order)+j+1}. {REFS[k]}" for j, k in enumerate(unused))
    order += unused

app = f"""

---

# Appendices

## Appendix A. First run with the 15-epoch Faster R-CNN

Frozen protocol of the first run ({proto3['time']}): meta = {proto3['meta']}, heavy = {proto3['heavy']}, cascade = {proto3['policy']} (fallback = {proto3['fallback']}). The Faster R-CNN of this run used 15 epochs and the last checkpoint, which did not match the other models; ensemble, stacking and cascade rows that use it as an input are affected.

**A.1 Cross-validation, first run (DEV; mean ± std over 5 folds)**

{v3_cv}

**A.2 Holdout, first run (FROZEN-ONCE; 381 images; mean over 5 fold-models)**

{v3_hold}

Paired differences (95% group-bootstrap intervals):

{v3_pairs}

## Appendix B. Extended cascade study (DEV)

{ext.split(chr(10)+chr(10))[1]}

{ext_tbl}

Paired per-fold differences versus the best single heavy model:

{ext_pairs}

## Appendix C. Repository map

| Path | Content |
|---|---|
| `datasets/` | dataset preparation, fold generation and training scripts (no data) |
| `results_v4_frcnn60/` | main run: frozen protocols, final report, cross-validation JSON, extended cascade report, holdout results (no detection dumps) |
| `results_v3/` | first run (15-epoch Faster R-CNN): protocol, report, holdout results, training logs |
| `results_coco/` | COCO benchmark results and frozen protocol |
| `stacking/`, `stacking_v4/` | ensembles, stacking, cascade, audits and benchmark scripts |
| `docs/leakage_audit/` | near-duplicate audit logs |
| `figures_pipeline/` | figure scripts and the generators of the README and of this report |
| `docs/readme_figures/`, `docs/paper_figures/`, `docs/media/` | figures and animations |
| `phase2_bridge/` | controllers, CARLA bridge, abstract queue model, tests, Part II results and report |
| `frcnn60/` | Faster R-CNN 60-epoch retraining scripts |
| `requirements_frozen.txt` | frozen package versions |

Datasets and model weights are not committed.

## Appendix D. Reproduction

```bash
pip install -r requirements_frozen.txt
python figures_pipeline/build_root_readme.py   # README and its figures from result files
python figures_pipeline/make_paper_figures.py  # docs/paper_figures from result files
python figures_pipeline/build_report.py        # this report from result files
```

Part II (CARLA 0.9.16, one run per process, seeds and gates) is described in [`phase2_bridge/README.md`](../../phase2_bridge/README.md); the abstract queue model tables are generated by `phase2_bridge/build_abstract_model_doc.py` into [`phase2_bridge/ABSTRACT_MODEL.md`](../../phase2_bridge/ABSTRACT_MODEL.md).

## Appendix E. Statement on data and claims

- Every number in Chapter 4 is read from a result file in this repository; the generator of this report is `figures_pipeline/build_report.py`.
- Claims not made: that the fuzzy controller is safer under false alarms; that it beats simple preemption for the ambulance; any real-world error rate; generalisation to other junctions; absence of leakage (the audit only finds near-identical images); conclusions from ambulance time-to-cross without noting survivor bias; conclusions from the weather-visibility proxy (uncalibrated).
- No real-world emergency-vehicle video ground truth exists; video renders are illustrations, not results.
"""

main_text = (main_text.replace("(section 4.10)", "(Section 4.6)").replace("averages over seeds are in section 4.5", "averages over seeds are in Section 4.7")
             .replace("at roughly half of the heavy model's cost in cross-validation", f"at {saving*100:.0f}% of the heavy model's time in cross-validation"))
(OUT / "REPORT.md").write_text(main_text.rstrip() + "\n\n# References\n\n" + refs + "\n" + app, encoding="utf-8")
print("wrote", OUT / "REPORT.md", "refs", len(order))
