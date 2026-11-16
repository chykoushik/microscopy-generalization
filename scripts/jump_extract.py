import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/cache/matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tifffile
from PIL import Image
EVIDENCE = ROOT / 'data/manifests/jump'
OUT = ROOT / 'outputs/results/jump_experiment_v1'
RAW = ROOT / 'data/raw/jump_site2'
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))

def decode(path):
    try:
        return tifffile.imread(path)
    except ValueError as exc:
        if 'imagecodecs' not in str(exc):
            raise
        with Image.open(path) as image:
            return np.array(image)

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def acquire(entry):
    path = RAW / entry['plate'] / f"{entry['well']}_s2_{entry['channel']}.tiff"
    path.parent.mkdir(exist_ok=True)
    partial = path.with_suffix('.part')
    error = None
    for attempt in range(3):
        try:
            if not path.exists():
                if shutil.disk_usage(ROOT).free < 20 * 2 ** 30 + 64 * 2 ** 20:
                    raise RuntimeError('Storage reserve reached')
                offset = partial.stat().st_size if partial.exists() else 0
                request = urllib.request.Request(entry['url'], headers={'Range': f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request, timeout=90) as response:
                    append = offset and response.status == 206
                    if append and (not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise ValueError('Range mismatch')
                    with open(partial, 'ab' if append else 'wb') as f:
                        shutil.copyfileobj(response, f, 1024 * 1024)
                partial.replace(path)
            image = decode(path)
            if image.ndim != 2 or image.dtype != np.uint16 or (not np.isfinite(image).all()):
                raise ValueError('Invalid image schema')
            return entry | dict(status='ok', path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=list(image.shape), dtype=str(image.dtype), error='')
        except Exception as exc:
            error = repr(exc)
            time.sleep(attempt + 1)
    return entry | dict(status='failed', path=path.relative_to(ROOT).as_posix(), bytes=0, sha256='', shape=[], dtype='', error=error)

def download():
    entries = read_csv(EVIDENCE / 'proposed_site2_images.csv')
    assert len({e['url'] for e in entries}) == len(entries) == 1200
    results = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        for future in as_completed([pool.submit(acquire, e) for e in entries]):
            results.append(future.result())
            if len(results) % 50 == 0:
                (OUT / 'acquisition.json').write_text(json.dumps(results, indent=2))
                print('Images verified', len(results), flush=True)
    results.sort(key=lambda e: (e['plate'], e['well'], e['channel']))
    (OUT / 'acquisition.json').write_text(json.dumps(results, indent=2))

def recover():
    records = json.loads((OUT / 'acquisition.json').read_text())
    (OUT / 'acquisition_before_recovery.json').write_text(json.dumps(records, indent=2))
    for i, entry in enumerate(records):
        if entry['status'] != 'ok':
            records[i] = acquire({k: entry[k] for k in ['plate', 'well', 'site', 'channel', 'url']})
    (OUT / 'acquisition.json').write_text(json.dumps(records, indent=2))
    print('Recovery completed', flush=True)

def extract():
    import torch
    from data.microscopy import normalize_pixels
    from features.microscopy import segment_nuclei
    from features.microscopy_encoder import load_encoder, infer_crops
    cfg = json.loads((ROOT / 'configs/real_images_v1.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123001)
    torch.set_num_threads(2)
    model, provenance = load_encoder(ROOT, cfg['encoder'], device)
    order = ['DNA', 'RNA', 'ER', 'AGP', 'Mito']
    provenance.update(channel_order=order, adaptation='All five Cell Painting channels, native 128px crops; no zero channels. Existing DNA watershed, unvalidated JUMP segmentation.', seed=123001)
    (OUT / 'encoder.json').write_text(json.dumps(provenance, indent=2))
    manifest = read_csv(EVIDENCE / 'proposed_site2_images.csv')
    for entry in manifest:
        entry['path'] = (RAW / entry['plate'] / f"{entry['well']}_s2_{entry['channel']}.tiff").relative_to(ROOT).as_posix()
    records = []
    features = OUT / 'cells'
    features.mkdir(exist_ok=True)
    acquisition_order = {(e['plate'], e['well']): i for i, e in reversed(list(enumerate(manifest)))}
    wells = sorted(read_csv(EVIDENCE / 'proposed_split.csv'), key=lambda r: acquisition_order[r['Metadata_Plate'], r['Metadata_Well']])
    for well in wells:
        plate, name = (well['Metadata_Plate'], well['Metadata_Well'])
        target = features / f'{plate}_{name}.npz'
        entries = [e for e in manifest if e['plate'] == plate and e['well'] == name]
        started = time.monotonic()
        while not all(((ROOT / e['path']).exists() for e in entries)):
            try:
                ledger = json.loads((OUT / 'acquisition.json').read_text())
                if len(ledger) == 1200 or any((e['plate'] == plate and e['well'] == name and (e['status'] == 'failed') for e in ledger)):
                    break
            except (ValueError, OSError):
                pass
            if time.monotonic() - started > 3600:
                break
            time.sleep(2)
        record = dict(well, treatment='AMG900' if well['Metadata_JCP2022'] == 'JCP2022_037716' else 'DMSO', site=2)
        try:
            if len(entries) != 5 or {e['channel'] for e in entries} != set(order) or (not all(((ROOT / e['path']).exists() for e in entries))):
                raise ValueError('Missing channels')
            if target.exists():
                data = np.load(target)
                record.update(cells=len(data['embeddings']), status='ok', error='')
            else:
                paths = {e['channel']: ROOT / e['path'] for e in entries}
                images = [decode(paths[ch]) for ch in order]
                if len({im.shape for im in images}) != 1:
                    raise ValueError('Channel dimensions differ')
                image, bounds = normalize_pixels(np.stack(images))
                _, _, eligible, qc = segment_nuclei(image[[0, 3, 4]], cfg['segmentation'])
                rng = np.random.default_rng(int(hashlib.sha256(f'{plate}|{name}|123001'.encode()).hexdigest()[:8], 16))
                selected = [eligible[i] for i in sorted(rng.permutation(len(eligible))[:256])]
                if len(selected) < 32:
                    raise ValueError(f'Insufficient cells: {len(selected)}')
                crops = np.stack([image[:, int(round(c['y'])) - 64:int(round(c['y'])) + 64, int(round(c['x'])) - 64:int(round(c['x'])) + 64] for c in selected])
                values = infer_crops(model, crops, device, 16)
                assert values.shape == (len(selected), 384) and np.isfinite(values).all()
                np.savez_compressed(target, embeddings=values, coordinates=np.array([[c['y'], c['x']] for c in selected]), nucleus_area=np.array([c['nucleus_area'] for c in selected]))
                (features / f'{plate}_{name}.json').write_text(json.dumps(dict(metadata=record, channels=order, paths={ch: str(paths[ch].relative_to(ROOT)) for ch in order}, normalization_bounds=bounds, qc=qc, selected=selected), indent=2))
                record.update(cells=len(values), status='ok', error='')
            print('Well encoded', plate, name, flush=True)
        except Exception as exc:
            record.update(cells=0, status='failed', error=repr(exc))
        records.append(record)
        write_csv(OUT / 'wells.csv', records)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['download', 'extract', 'recover'])
    args = parser.parse_args()
    globals()[args.stage]()
