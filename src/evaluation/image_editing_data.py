import numpy as np

def crop(a, x, y, size=64):
    return a[..., y - size // 2:y + size // 2, x - size // 2:x + size // 2]

def measured_state(images, nuclei, cytoplasm, assay):
    marker = images[:, 1]
    nuclear = (marker * nuclei).sum((1, 2)) / np.maximum(nuclei.sum((1, 2)), 1)
    if assay == 'NF':
        cyto = (marker * cytoplasm).sum((1, 2)) / np.maximum(cytoplasm.sum((1, 2)), 1)
        return np.log((nuclear + 0.0001) / (cyto + 0.0001))
    return nuclear

def descriptor(images, nuclei, cytoplasm):
    rows = []
    for image, n, c in zip(images, nuclei, cytoplasm):
        values = [np.log1p(n.sum()), np.log1p(c.sum())]
        yy, xx = np.where(n)
        values.extend([np.std(xx), np.std(yy)])
        for channel in image[[0, 2]]:
            for mask in [n, c]:
                a = channel[mask.astype(bool)]
                values.extend([a.mean(), a.std(), *np.quantile(a, [0.1, 0.5, 0.9])])
        rows.append(values)
    return np.asarray(rows, np.float32)
