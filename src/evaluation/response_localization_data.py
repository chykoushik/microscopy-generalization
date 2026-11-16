from pathlib import Path
import csv
import hashlib
import h5py
import numpy as np

def rng(key):
    return np.random.default_rng(int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16))

def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def tnbc(root):
    plan = read(root / 'outputs/results/tnbc_source/TNBC_DOWNLOAD_LEDGER.csv')
    roles = {}
    for context in ['MDA231', 'MDA468']:
        controls = sorted([r for r in plan if r['cell_line'] == context and r['role'] == 'control'], key=lambda r: rng('localize|' + context + r['well']).random())
        for i, row in enumerate(controls):
            roles[context + ':' + row['well']] = 'reference' if i < 10 else 'train' if i < 16 else 'validation' if i < 21 else 'calibration' if i < 25 else 'test'
    wells = []
    marker_index = None
    for candidate in (root / '.cache/downloads/tnbc_source/tnbc_cells').rglob('*.ch5'):
        with h5py.File(candidate, 'r') as stream:
            key = 'definition/feature/tertiary__expanded/object_features'
            if key in stream:
                names = [v[0].decode() for v in stream[key][()]]
                marker_index = names.index('spotfeatures_count')
                break
    if marker_index is None:
        raise ValueError('Missing marker definition')
    for row in plan:
        context = row['cell_line']
        path = root / '.cache/downloads/tnbc_source/tnbc_cells' / context / Path(row['destination']).name
        with h5py.File(path, 'r') as stream:
            position = next(iter(stream['sample/0/plate/' + row['plate'] + '/experiment/' + row['well'] + '/position'].values()))
            feature = position['feature']
            if 'primary__primary4/object_features' not in feature:
                continue
            arrays = [feature[key]['object_features'][()] for key in ['primary__primary4', 'secondary__propagate', 'tertiary__expanded']]
            if not len(arrays[0]):
                continue
            if len({len(v) for v in arrays}) != 1:
                raise ValueError('Channel join mismatch')
            centers = [feature[key]['center'][()] for key in ['primary__primary4', 'secondary__propagate', 'tertiary__expanded']]
            if not np.array_equal(centers[0], centers[2]):
                raise ValueError('Marker center mismatch')
            cells = np.concatenate(arrays[:2], axis=1).astype(np.float32)
            marker = arrays[2][:, marker_index].astype(float)
        key = context + ':' + row['well']
        label = int(row['treatment'] != 'DMSO')
        if len(cells) == 0:
            continue
        wells.append(dict(id=key, cells=cells, marker=marker, context=context, label=label, role=row['role'] if label else roles[key], treatment=row['treatment'], group=row['treatment'] if label else key, plate=row['plate'], provenance=str(path.relative_to(root)), center=centers[0]))
    return wells
