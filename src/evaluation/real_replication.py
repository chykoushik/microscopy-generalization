import numpy as np

def weighted_wasserstein(x, y, weights):
    x, y, w = (np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float))
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or (w.shape != (len(y),)) or np.any(w < 0) or (w.sum() <= 0) or (not all((np.isfinite(a).all() for a in [x, y, w]))):
        raise ValueError('Invalid sliced mass')
    scores = []
    w = w / w.sum()
    for j in range(x.shape[1]):
        a = np.sort(x[:, j])
        order = np.argsort(y[:, j], kind='stable')
        b = y[order, j]
        mass = w[order]
        keep = mass > 0
        b = b[keep]
        mass = mass[keep]
        ends = np.cumsum(mass)
        ends[-1] = 1.0
        edges = np.unique(np.clip(np.r_[np.arange(len(a) + 1) / len(a), ends, 0.0, 1.0], 0, 1))
        mid = (edges[:-1] + edges[1:]) / 2
        scores.append(float(np.sum(np.diff(edges) * (a[np.minimum((mid * len(a)).astype(int), len(a) - 1)] - b[np.searchsorted(ends, mid)]) ** 2)))
    return float(np.mean(scores))
