import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, sobel
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed

def segment_nuclei(image, settings):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3 or (not np.isfinite(image).all()):
        raise ValueError('Invalid segmentation image')
    foreground = np.maximum(ndi.gaussian_filter(image[0], settings['sigma']) - ndi.gaussian_filter(image[0], settings['background_sigma']), 0)
    if foreground.max() == 0:
        return (np.zeros(image.shape[1:], np.int32), np.zeros(image.shape[1:], np.int32), [], dict(threshold=0, detected=0, accepted=0, rejections={}))
    threshold = float(threshold_otsu(foreground))
    binary = remove_small_objects(foreground > threshold, min_size=settings['min_area'])
    binary = remove_small_holes(binary, area_threshold=settings['hole_area'])
    distance = ndi.distance_transform_edt(binary)
    peaks = peak_local_max(distance, min_distance=settings['min_distance'], labels=binary, exclude_border=False)
    markers = np.zeros(binary.shape, np.int32)
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-distance, markers, mask=binary).astype(np.int32)
    accepted = []
    rejected = dict(area=0, solidity=0, edge=0)
    margin = settings['min_crop_margin']
    h, w = binary.shape
    for region in regionprops(labels):
        y, x = region.centroid
        if not settings['min_area'] <= region.area <= settings['max_area']:
            rejected['area'] += 1
        elif region.solidity < settings['min_solidity']:
            rejected['solidity'] += 1
        elif min(y, x, h - y, w - x) < margin:
            rejected['edge'] += 1
        else:
            accepted.append(dict(label=int(region.label), y=float(y), x=float(x), nucleus_area=int(region.area), solidity=float(region.solidity), eccentricity=float(region.eccentricity)))
    territories = watershed(sobel(image[1:3].mean(axis=0)), labels, mask=ndi.distance_transform_edt(labels == 0) <= settings['secondary_radius']).astype(np.int32)
    return (labels, territories, accepted, dict(threshold=threshold, detected=int(labels.max()), accepted=len(accepted), rejections=rejected, nuclear_foreground_fraction=float(binary.mean())))
