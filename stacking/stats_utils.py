import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(x, stat=np.mean, B=2000, seed=0, alpha=0.05):
    x = np.asarray(x)
    rng = np.random.RandomState(seed)
    v = [stat(x[rng.randint(0, len(x), len(x))]) for _ in range(B)]
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def reliability(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        c, a = conf[m].mean(), correct[m].mean()
        ece += m.mean() * abs(c - a)
        rows.append((edges[b], edges[b + 1], int(m.sum()), float(c), float(a)))
    return rows, float(ece)


def plot_reliability(conf, correct, path, bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, ece = reliability(conf, correct, bins)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.plot([r[3] for r in rows], [r[4] for r in rows], "o-")
    plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title(f"ECE={ece:.3f}")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return ece
