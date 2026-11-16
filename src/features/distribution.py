import numpy as np

def squared_distances(x, y):
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.ndim != 2 or y.ndim != 2 or (not len(x)) or (not len(y)) or (x.shape[1] != y.shape[1]) or (not np.isfinite(x).all()) or (not np.isfinite(y).all()):
        raise ValueError('Invalid distribution matrix')
    result = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2 * x @ y.T
    return np.maximum(result, 0)

def kernel_mean(distances, bandwidths):
    if not bandwidths or any((not np.isfinite(value) or value <= 0 for value in bandwidths)):
        raise ValueError('Invalid kernel bandwidth')
    return float(np.mean([np.exp(-distances / (2 * bandwidth ** 2)).mean() for bandwidth in bandwidths]))

def distribution_scores(x, reference, bandwidths, reference_terms=None):
    xx, xr = (squared_distances(x, x), squared_distances(x, reference))
    if reference_terms is None:
        rr = squared_distances(reference, reference)
        reference_terms = (kernel_mean(rr, bandwidths), float(np.sqrt(rr).mean()))
    mmd = kernel_mean(xx, bandwidths) + reference_terms[0] - 2 * kernel_mean(xr, bandwidths)
    energy = 2 * np.sqrt(xr).mean() - np.sqrt(xx).mean() - reference_terms[1]
    if min(mmd, energy) < -1e-10:
        raise ValueError('Negative distribution discrepancy')
    return dict(mmd_exact=float(max(0, mmd)), energy=float(max(0, energy)))
