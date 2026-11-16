import json
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation, feature
from PIL import Image
from sklearn.decomposition import PCA
from evaluation.response_localization_data import tnbc, rng, read

def tnbc_data(root):
    wells = tnbc(root)
    cutoffs = {'MDA231': 19.0, 'MDA468': 21.0}
    for w in wells:
        w['target'] = (w['marker'] > cutoffs[w['context']]).astype(np.int64)
        w['counts'] = w['marker'].astype(np.float32)
        w['cutoff'] = cutoffs[w['context']]
        w['channels'] = ['DAPI', 'Cy5_tubulin']
        w['dataset'] = 'TNBC'
    return wells

def nuclei(dna):
    smooth = ndi.gaussian_filter(dna.astype(float), 1.0)
    mask = smooth > filters.threshold_otsu(smooth)
    mask = morphology.remove_small_objects(mask, min_size=35)
    mask = ndi.binary_fill_holes(mask)
    distance = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(distance, min_distance=6, labels=mask)
    seeds = np.zeros(mask.shape, int)
    for i, (y, x) in enumerate(peaks):
        seeds[y, x] = i + 1
    return segmentation.watershed(-distance, seeds, mask=mask)

def extract_dna(dna, labels, marker):
    distance, indices = ndi.distance_transform_edt(labels == 0, return_indices=True)
    nearest = labels[tuple(indices)]
    ring = (distance >= 2) & (distance <= 7) & (labels == 0)
    vectors = []
    ratios = []
    centers = []
    ids = []
    for region in measure.regionprops(labels, intensity_image=dna):
        if not 35 <= region.area <= 2500 or min(region.bbox[:2]) == 0 or region.bbox[2] >= dna.shape[0] or (region.bbox[3] >= dna.shape[1]):
            continue
        y, x = region.coords.T
        values = dna[y, x].astype(float) / 255
        outer = ring & (nearest == region.label)
        if outer.sum() < 12:
            continue
        vectors.append([np.log1p(region.area), region.eccentricity, region.solidity, region.extent, region.major_axis_length, region.minor_axis_length, region.perimeter, region.euler_number, values.mean(), values.std(), *np.quantile(values, np.linspace(0, 1, 13)), *np.histogram(values, bins=12, range=(0, 1), density=False)[0] / len(values)])
        ratios.append(np.log((float(marker[y, x].mean()) + 1) / (float(marker[outer].mean()) + 1)))
        centers.append(region.centroid)
        ids.append(region.label)
    return (np.array(vectors, dtype=np.float32), np.array(ratios), np.array(centers), np.array(ids))

def second_data(root, output):
    cache = root / '.cache/downloads/missing_marker'
    saved = output / 'BBBC013_FEATURES.npz'
    metadata = output / 'BBBC013_WELLS.json'
    if saved.exists():
        rows = json.loads(metadata.read_text())
        arrays = np.load(saved)
        return [dict(row, cells=arrays['x' + str(i)], marker=arrays['m' + str(i)], target=arrays['y' + str(i)], center=arrays['center' + str(i)], object_ids=arrays['ids' + str(i)]) for i, row in enumerate(rows)]
    doses = [float(v) for v in (cache / 'platemap.txt').read_text().splitlines()[1:] if v.strip()]
    wells = []
    arrays = {}
    rows = []
    for path in sorted((cache / 'images').rglob('Channel2-*.BMP')):
        parts = path.stem.split('-')
        row = parts[-2]
        column = int(parts[-1])
        well = row + str(column).zfill(2)
        marker_path = path.with_name(path.name.replace('Channel2-', 'Channel1-'))
        dna = np.array(Image.open(path))
        marker = np.array(Image.open(marker_path))
        assert dna.shape == marker.shape == (640, 640)
        labels = nuclei(dna)
        cells, values, centers, ids = extract_dna(dna, labels, marker)
        role = 'reference' if row in 'AB' and column in [1, 2] else 'train' if row in 'AB' else 'calibration' if row == 'C' and column in [1, 2] else 'validation' if row == 'C' else 'test'
        index = (ord(row) - 65) * 12 + column - 1
        wells.append(dict(id='BBBC013:' + well, cells=cells, marker=values, center=centers, object_ids=ids, role=role, context='U2OS_single_plate', group=('LY294002' if row in 'EFGH' else 'Wortmannin') + ':' + row, treatment='LY294002' if row in 'EFGH' else 'Wortmannin', dose=doses[index], dataset='BBBC013', channels=['DRAQ_DNA'], provenance=str(path.relative_to(root))))
    cutoff = float(np.quantile(np.concatenate([w['marker'] for w in wells if w['role'] == 'reference']), 0.95))
    for i, w in enumerate(wells):
        w['target'] = (w['marker'] > cutoff).astype(np.int64)
        w['cutoff'] = cutoff
        for key, source in [('x', 'cells'), ('m', 'marker'), ('y', 'target'), ('center', 'center'), ('ids', 'object_ids')]:
            arrays[key + str(i)] = w[source]
        rows.append({k: v for k, v in w.items() if k not in ['cells', 'marker', 'target', 'center', 'object_ids']})
    np.savez_compressed(saved, **arrays)
    metadata.write_text(json.dumps(rows, indent=2))
    return wells

class Channels:

    def fit(self, wells):
        train = [w for w in wells if w['role'] == 'train']
        self.fit_ids = [w['id'] for w in train]
        values = np.concatenate([w['cells'] for w in train])
        self.groups = len(train[0]['channels'])
        raw = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        self.center = np.median(raw, axis=0)
        self.scale = np.maximum(np.quantile(raw, 0.75, axis=0) - np.quantile(raw, 0.25, axis=0), 0.001)
        raw = np.clip((raw - self.center) / self.scale, -12, 12)
        self.pcas = []
        for group in np.array_split(np.arange(raw.shape[1]), self.groups):
            self.pcas.append((group, PCA(n_components=min(24, len(group)), random_state=719).fit(raw[:, group])))
        return self

    def transform(self, x):
        raw = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.clip((raw - self.center) / self.scale, -12, 12)
        values = []
        for group, pca in self.pcas:
            z = pca.transform(raw[:, group])
            z = z / np.sqrt(np.maximum(pca.explained_variance_, 0.1))
            if z.shape[1] < 24:
                z = np.pad(z, ((0, 0), (0, 24 - z.shape[1])))
            values.append(np.clip(z, -8, 8))
        return np.stack(values, axis=1).astype(np.float32)

def arrays(wells, frame, roles=('train',)):
    own = [w for w in wells if w['role'] in roles]
    return dict(x=np.concatenate([frame.transform(w['cells']) for w in own]), y=np.concatenate([w['target'] for w in own]), weights=np.concatenate([np.full(len(w['cells']), 1 / len(w['cells'])) for w in own]), well=np.concatenate([np.full(len(w['cells']), i) for i, w in enumerate(own)]), context=np.concatenate([np.full(len(w['cells']), sorted({v['context'] for v in own}).index(w['context'])) for w in own]), counts=np.concatenate([w.get('counts', np.zeros(len(w['cells']))) for w in own]), wells=own)
