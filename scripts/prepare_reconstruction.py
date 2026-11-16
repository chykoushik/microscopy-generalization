import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import json, csv, hashlib, shutil
import numpy as np
import h5py, torch
from scipy import ndimage as ndi
from PIL import Image
from evaluation.spatial_response_data import image, normalize, rng
from evaluation.image_editing_data import crop, measured_state, descriptor
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/results/fluorescence_reconstruction'
SOURCE = ROOT / 'outputs/results/tnbc_spatial_inputs'

def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def prepare(assay):
    meta = json.loads((SOURCE / (assay + '_wells.json')).read_text())
    held = 'SUM159PT' if assay == 'NF' else 'MDA468'
    meta = [w for w in meta if w['role'] in ['train', 'validation', 'test'] and (w['cell_line'] != held or w['role'] == 'test')]
    scales = []
    if assay == 'NF':
        for w in meta:
            if w['role'] == 'train':
                scales.append(image(ROOT / w['path'])[1].ravel()[::100])
    scale = float(np.quantile(np.concatenate(scales), 0.9995)) if scales else 255.0
    images = []
    nuclei = []
    cytoplasm = []
    cells = []
    rows = []
    fail = []
    for w in meta:
        if assay == 'NF':
            raw = image(ROOT / w['path'])
            seg = np.load(SOURCE / 'segmentations' / (w['image_id'] + '.npz'))
            nm = seg['nuclei']
            cm = seg['cells']
            ids = seg['ids']
            coordinates = []
            for label in ids:
                yy, xx = np.where(nm == label)
                coordinates.append((int(np.rint(xx.mean())), int(np.rint(yy.mean()))))
            input0 = normalize(raw[0])
            input2 = normalize(raw[2])
            target = np.clip(raw[1] / scale, 0, 1)
            original_ids = ids
        else:
            with h5py.File(ROOT / w['path']) as f:
                pos = next(iter(f['sample/0/plate/' + w['plate'] + '/experiment/' + w['well'] + '/position'].values()))
                raw = pos['image/channel'][:, 0, 0]
                regions = pos['image/region'][:, 0, 0]
                nm = regions[0]
                cm = regions[1]
                objects = pos['object/primary__primary4'][()]
                ids = objects['obj_label_id']
                secondary = pos['object/secondary__propagate'][()]
                assert np.array_equal(ids, secondary['obj_label_id'])
                centers = pos['feature/primary__primary4/center'][()]
                coordinates = [(int(c['x']), int(c['y'])) for c in centers]
                original_ids = ids
            input0 = raw[0] / 255.0
            input2 = raw[1] / 255.0
            target = raw[2] / 255.0
        field = np.stack([input0, target, input2]).astype(np.float32)
        selected = rng('editing200|' + w['id']).permutation(len(ids))
        kept = 0
        for index in selected:
            label = int(ids[index])
            x, y = coordinates[index]
            if min(x, y) < 32 or x + 32 > field.shape[-1] or y + 32 > field.shape[-2]:
                continue
            n = nm == label
            c = cm == label
            cy = c & ~ndi.binary_dilation(n, iterations=2)
            n0 = crop(n, x, y)
            c0 = crop(c, x, y)
            cy0 = crop(cy, x, y)
            if n0.sum() < 30 or cy0.sum() < 20 or n0.sum() / max(1, n.sum()) < 0.98:
                continue
            role = 'ood_test' if w['cell_line'] == held else w['role']
            images.append(crop(field, x, y))
            nuclei.append(n0)
            cells.append(c0)
            cytoplasm.append(cy0)
            rows.append(dict(assay=assay, id=w['id'] + ':object' + str(label), well_id=w['id'], plate=w['plate'], well=w['well'], context=w['cell_line'], treatment=w['treatment'], group=w['group'], role=role, object_label=label, source_index=int(index), x=x, y=y, source=w['path'], cell_coverage=float(c0.sum() / max(1, c.sum())), marker_clipped_fraction=float((crop(target, x, y) >= 1).mean())))
            kept += 1
            if kept == 24:
                break
        fail.append(dict(assay=assay, well_id=w['id'], available=len(original_ids), selected=kept, status='ok' if kept else 'NO_USABLE_CROPS'))
        print('Field cropped', assay, len(rows), flush=True)
    images = np.asarray(images, np.float32)
    nuclei = np.asarray(nuclei, bool)
    cells = np.asarray(cells, bool)
    cytoplasm = np.asarray(cytoplasm, bool)
    readout = measured_state(images, nuclei, cytoplasm, assay)
    controls = np.array([w['role'] == 'train' and w['treatment'] in ['DMSO', 'untreated'] for w in rows])
    assert controls.any(), set((w['treatment'] for w in rows))
    cutoff = float(np.quantile(readout[controls], 0.95))
    state = readout > cutoff
    for i, w in enumerate(rows):
        w['readout'] = float(readout[i])
        w['state'] = int(state[i])
    np.savez_compressed(OUT / (assay + '_images.npz'), images=images.astype(np.float16), nuclei=nuclei, cytoplasm=cytoplasm, cells=cells, readout=readout, state=state, descriptor=descriptor(images, nuclei, cytoplasm))
    (OUT / (assay + '_cells.json')).write_text(json.dumps(rows, indent=2))
    write(assay + '_EXTRACTION.csv', fail)
    return dict(assay=assay, marker_scale=scale, state_cutoff=cutoff, held_context=held, fields=len({r['well_id'] for r in rows}), cells=len(rows), role_cells={r: sum((w['role'] == r for w in rows)) for r in ['train', 'validation', 'test', 'ood_test']}, role_wells={r: len({w['well_id'] for w in rows if w['role'] == r}) for r in ['train', 'validation', 'test', 'ood_test']})
if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'NF_images.npz').exists() or (OUT/'GAMMA_images.npz').exists():
        raise FileExistsError('Preserve extracted pixels')
    configs=[prepare(assay) for assay in ['NF','GAMMA']]
    (OUT/'CONFIG.json').write_text(json.dumps(dict(seed=200,crop_size=64,max_cells_per_field=24,assays=configs),indent=2))
