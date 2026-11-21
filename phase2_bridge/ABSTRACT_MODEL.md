# Abstract queue model: corridor results (NOT CARLA, NOT real traffic)

Everything below comes from `toy_sim.py`, a simplified queue model. It says nothing about CARLA or real traffic.

Seeds: 20 (paired by seed). Intervals are mean ± 1.96·SE over seeds (normal approximation, not a bootstrap).
Junction count and duration are NOT stored in the CSV files; the scripts' defaults are 5 junctions and 900 s. Treat those as the likely settings, not as verified.

## 1. Policies on the corridor (demand lam = 0.12, no message delay or loss)

| policy | emergency delay, corridor (s) | ordinary delay (s) | max wait (s) | false-preempt time (s) | throughput (veh/h) |
|---|---|---|---|---|---|
| Fixed-time | 61.54 | 13.76 | 37.65 | 0.00 | 6752.40 |
| Fixed+preempt | 17.96 | 13.93 | 53.60 | 0.00 | 6753.00 |
| Max-pressure | 37.99 | 8.45 | 22.95 | 0.00 | 6798.60 |
| Fuzzy (no comm) | 15.95 | 9.50 | 46.25 | 0.00 | 6783.40 |
| Fuzzy (full) | 5.89 | 10.71 | 53.75 | 109.95 | 6782.60 |
| Fuzzy (2-hop) | 7.34 | 12.04 | 54.35 | 96.10 | 6782.40 |

Means over seeds. `Fuzzy (full)` = 1-hop messages; `Fuzzy (2-hop)` = messages to 2 junctions each way.

Paired differences of `Fuzzy (full)` minus another policy (mean ± 1.96·SE; negative emergency/ordinary delay = Fuzzy better):

| versus | emergency delay (s) | ordinary delay (s) | max wait (s) |
|---|---|---|---|
| Fuzzy (no comm) | -10.06 ± 2.04 | +1.22 ± 0.18 | +7.50 ± 2.39 |
| Fixed+preempt | -12.07 ± 4.05 | -3.22 ± 0.26 | +0.15 ± 3.12 |
| Max-pressure | -32.10 ± 4.37 | +2.26 ± 0.22 | +30.80 ± 1.27 |
| Fixed-time | -55.65 ± 2.27 | -3.04 ± 0.11 | +16.10 ± 0.35 |

## 2. 2-hop minus 1-hop under message delay and loss (lam = 0.12)

Negative is better for the delay columns. 'seeds worse' = seeds where 2-hop was higher than 1-hop.

| message delay (s) | drop prob. | emergency delay (s) | ordinary delay (s) | ordinary: seeds worse | max wait (s) | false-preempt (s) |
|---|---|---|---|---|---|---|
| 0 | 0 | +1.45 ± 1.62 | +1.33 ± 0.11 | 20/20 | +0.60 ± 0.22 | -13.85 ± 5.77 |
| 2 | 0 | +0.49 ± 1.42 | +1.20 ± 0.10 | 20/20 | +0.95 ± 0.46 | -9.65 ± 5.03 |
| 0 | 0.2 | -1.06 ± 2.22 | +1.19 ± 0.11 | 20/20 | +0.95 ± 0.33 | -3.75 ± 8.99 |
| 0 | 0.5 | -1.38 ± 2.22 | +0.93 ± 0.20 | 19/20 | +1.00 ± 0.75 | +7.05 ± 8.74 |

## 3. Where does 2-hop help? Demand sweep (paired by seed against 1-hop)

Variants: `state-only` = hop-2 occupancy enters the pressure term, no 2-ahead notices; `notices-only` = notices, no hop-2 pressure; `both`.

Regime of the 1-hop controller at each demand (means):

| lam | ordinary delay (s) | spillbacks/h | throughput (veh/h) |
|---|---|---|---|
| 0.12 | 10.7 | 0.0 | 6783 |
| 0.2 | 31.1 | 0.0 | 10928 |
| 0.28 | 128.6 | 285.2 | 12211 |
| 0.32 | 203.8 | 783.6 | 11783 |

Difference variant − 1-hop (mean ± 1.96·SE):

| lam | variant | emergency delay (s) | ordinary delay (s) | spillbacks/h | max wait (s) |
|---|---|---|---|---|---|
| 0.12 | 2hop state-only | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 |
| 0.12 | 2hop notices-only | +1.45 ± 1.62 | +1.33 ± 0.11 | +0.00 ± 0.00 | +0.60 ± 0.22 |
| 0.12 | 2hop both | +1.45 ± 1.62 | +1.33 ± 0.11 | +0.00 ± 0.00 | +0.60 ± 0.22 |
| 0.2 | 2hop state-only | +0.56 ± 0.69 | -0.06 ± 0.16 | +0.00 ± 0.00 | -0.15 ± 0.41 |
| 0.2 | 2hop notices-only | -11.04 ± 2.26 | +3.02 ± 0.53 | +0.00 ± 0.00 | +3.45 ± 1.90 |
| 0.2 | 2hop both | -11.20 ± 2.16 | +3.00 ± 0.51 | +0.00 ± 0.00 | +3.60 ± 1.87 |
| 0.28 | 2hop state-only | +0.70 ± 3.27 | -0.51 ± 1.09 | -28.60 ± 64.34 | -1.20 ± 2.35 |
| 0.28 | 2hop notices-only | -44.29 ± 7.23 | +2.39 ± 1.44 | -209.80 ± 130.48 | -1.20 ± 2.35 |
| 0.28 | 2hop both | -43.79 ± 7.06 | +2.38 ± 1.50 | -205.00 ± 122.69 | -1.05 ± 2.06 |
| 0.32 | 2hop state-only | +2.62 ± 3.12 | +1.52 ± 1.26 | +34.80 ± 78.90 | -0.55 ± 2.61 |
| 0.32 | 2hop notices-only | -69.72 ± 11.93 | -1.78 ± 2.71 | -652.40 ± 173.59 | -3.15 ± 2.45 |
| 0.32 | 2hop both | -67.55 ± 12.43 | -0.91 ± 2.69 | -651.40 ± 172.61 | -3.15 ± 2.45 |

## 4. Hop-2 weight sweep (lam = 0.12)

| policy | weight | ordinary delay (s) | emergency delay (s) | false-preempt (s) | max wait (s) |
|---|---|---|---|---|---|
| 1-hop | - | 10.713 | 5.888 | 109.95 | 53.75 |
| 2-hop | 0 | 12.039 | 7.338 | 96.10 | 54.35 |
| 2-hop | 0.25 | 12.039 | 7.338 | 96.10 | 54.35 |
| 2-hop | 0.5 | 12.039 | 7.338 | 96.10 | 54.35 |
| 2-hop | 0.75 | 12.039 | 7.338 | 96.10 | 54.35 |
| 2-hop | 1 | 12.039 | 7.338 | 96.10 | 54.35 |

All 2-hop rows are identical across weights: at this demand the weight had no measured effect.

## 5. Earlier run: 3 junctions (default n_junc=3), 10 seeds, three demands

File: outputs/p2_policy_runs.csv (stored 2026-11-04, before the multi-hop code of 2026-11-13; not re-run with the current code). Different setting from sections 1-4: do not mix them.

| lam | policy | emergency delay (s) | ordinary delay (s) | throughput (veh/h) | max wait (s) |
|---|---|---|---|---|---|
| 0.08 | Fixed+preempt | 2.58 | 12.80 | 2844 | 52.00 |
| 0.08 | Fixed-time | 18.48 | 11.66 | 2839 | 37.20 |
| 0.08 | Fuzzy (full) | 0.55 | 9.46 | 2851 | 53.00 |
| 0.08 | Fuzzy (no comm) | 1.50 | 8.93 | 2851 | 44.00 |
| 0.08 | Fuzzy (no priority) | 16.15 | 8.54 | 2851 | 38.20 |
| 0.08 | Max-pressure | 11.90 | 6.85 | 2855 | 21.30 |
| 0.12 | Fixed+preempt | 5.62 | 15.26 | 4236 | 55.90 |
| 0.12 | Fixed-time | 20.82 | 13.35 | 4249 | 37.50 |
| 0.12 | Fuzzy (full) | 1.55 | 10.12 | 4264 | 51.20 |
| 0.12 | Fuzzy (no comm) | 2.35 | 9.49 | 4266 | 44.00 |
| 0.12 | Fuzzy (no priority) | 17.57 | 9.00 | 4264 | 36.40 |
| 0.12 | Max-pressure | 14.15 | 8.51 | 4272 | 21.00 |
| 0.16 | Fixed+preempt | 7.78 | 18.08 | 5634 | 52.30 |
| 0.16 | Fixed-time | 22.88 | 16.16 | 5654 | 37.80 |
| 0.16 | Fuzzy (full) | 3.58 | 14.63 | 5669 | 51.70 |
| 0.16 | Fuzzy (no comm) | 5.15 | 13.31 | 5670 | 42.50 |
| 0.16 | Fuzzy (no priority) | 23.18 | 12.38 | 5674 | 36.40 |
| 0.16 | Max-pressure | 25.45 | 14.18 | 5670 | 27.20 |

Paired difference `Fuzzy (full)` minus other (mean ± 1.96·SE over 10 seeds):

| lam | versus | emergency delay (s) | ordinary delay (s) |
|---|---|---|---|
| 0.08 | Fuzzy (no comm) | -0.95 ± 0.72 | +0.52 ± 0.19 |
| 0.08 | Fixed+preempt | -2.02 ± 0.99 | -3.34 ± 0.40 |
| 0.08 | Max-pressure | -11.35 ± 2.88 | +2.61 ± 0.21 |
| 0.08 | Fixed-time | -17.93 ± 3.79 | -2.20 ± 0.41 |
| 0.12 | Fuzzy (no comm) | -0.80 ± 1.29 | +0.63 ± 0.28 |
| 0.12 | Fixed+preempt | -4.08 ± 3.08 | -5.14 ± 0.42 |
| 0.12 | Max-pressure | -12.60 ± 2.89 | +1.61 ± 0.38 |
| 0.12 | Fixed-time | -19.27 ± 3.98 | -3.23 ± 0.50 |
| 0.16 | Fuzzy (no comm) | -1.57 ± 2.16 | +1.32 ± 0.76 |
| 0.16 | Fixed+preempt | -4.20 ± 3.22 | -3.45 ± 1.13 |
| 0.16 | Max-pressure | -21.88 ± 6.94 | +0.45 ± 0.92 |
| 0.16 | Fixed-time | -19.30 ± 3.64 | -1.53 ± 0.97 |

## Labels and limits

- All numbers: abstract queue model. They are not evidence about CARLA or real traffic.
- The model was designed together with the controller; agreement between them is not independent validation.
- Messaging up to 2 junctions each way exists only in this model; it was not run in CARLA.
- Intervals use a normal approximation over a small number of seeds.
