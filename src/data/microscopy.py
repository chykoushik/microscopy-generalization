import numpy as np

def normalize_pixels(image, percentiles=(0.01, 99.9)):
    image = np.asarray(image)
    if image.ndim != 3 or not np.isfinite(image).all() or (not 0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError('Invalid microscopy pixels')
    bounds = np.percentile(image, percentiles, axis=(1, 2))
    scale = bounds[1] - bounds[0]
    result = np.clip((image.astype(np.float32) - bounds[0, :, None, None]) / np.maximum(scale[:, None, None], 1e-12), 0, 1)
    result[scale == 0] = 0
    return (result.astype(np.float32), bounds.tolist())
