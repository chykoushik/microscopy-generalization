import hashlib
import numpy as np
from PIL import Image, ImageSequence

def rng(key):
    return np.random.default_rng(int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16))

def image(path):
    with Image.open(path) as stream:
        return np.stack([np.array(x) for x in ImageSequence.Iterator(stream)])

def normalize(x):
    low, high = np.quantile(x, [0.01, 0.995])
    return np.clip((x.astype(float) - low) / max(1, high - low), 0, 1)

from scipy import ndimage as ndi
from skimage import filters,feature,morphology,segmentation
def segment(dna, dhe):
    dna = normalize(dna)
    body = normalize(dhe)
    smooth = ndi.gaussian_filter(dna, 1.0)
    mask = smooth > filters.threshold_otsu(smooth)
    mask = morphology.remove_small_objects(mask, min_size=30)
    mask = ndi.binary_fill_holes(mask)
    distance = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(distance, min_distance=5, labels=mask)
    seeds = np.zeros(mask.shape, int)
    for i, (y, x) in enumerate(peaks):
        seeds[y, x] = i + 1
    nuclei = segmentation.watershed(-distance, seeds, mask=mask)
    outer_distance = ndi.distance_transform_edt(nuclei == 0)
    cellmask = ((body > max(0.025, filters.threshold_otsu(body) * 0.5)) | ndi.binary_dilation(nuclei > 0, iterations=4)) & (outer_distance < 25)
    cells = segmentation.watershed(-ndi.gaussian_filter(body, 1), nuclei, mask=cellmask)
    return (nuclei, cells)
