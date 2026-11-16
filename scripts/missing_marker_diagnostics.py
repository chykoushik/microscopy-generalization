from pathlib import Path
import hashlib
import importlib.util
import json
import pickle
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from sklearn.linear_model import LogisticRegression
from evaluation.missing_marker_data import tnbc_data, rng, read
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'scripts/missing_marker.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)

def area_diagnostic(out, wells):
    train = [w for w in wells if w['role'] == 'train']
    cal = [w for w in wells if w['role'] == 'calibration']
    x = np.concatenate([np.log1p(w['cells'][:, 236:237]) for w in train])
    y = np.concatenate([w['target'] for w in train])
    weights = np.concatenate([np.full(len(w['cells']), 1 / len(w['cells'])) for w in train])
    weights *= len(weights) / weights.sum()
    model = LogisticRegression(C=0.3, class_weight='balanced', random_state=719).fit(x, y, sample_weight=weights)
    cx = np.concatenate([model.decision_function(np.log1p(w['cells'][:, 236:237])) for w in cal])
    cy = np.concatenate([w['target'] for w in cal])
    cw = np.concatenate([np.full(len(w['cells']), 1 / len(w['cells'])) for w in cal])
    cw *= len(cw) / cw.sum()
    calibrator = LogisticRegression(C=1.0, random_state=719).fit(cx[:, None], cy, sample_weight=cw)
    rows = []
    for w in [w for w in wells if w['role'] == 'test']:
        p = calibrator.predict_proba(model.decision_function(np.log1p(w['cells'][:, 236:237]))[:, None])[:, 1]
        entropy = -p * np.log(p + 1e-12) - (1 - p) * np.log(1 - p + 1e-12)
        rows.append(dict(dataset='TNBC', well=w['id'], group=w['group'], context=w['context'], method='DAPI_area_only', interpretation='post-evaluation biological diagnostic; not used to select models', **e.measurements(w['target'], p, entropy)))
    e.write(out / 'AREA_ONLY_DIAGNOSTIC.csv', rows)
    (out / 'area_diagnostic.pkl').write_bytes(pickle.dumps(dict(model=model, calibrator=calibrator, feature='primary roisize index236', fit_ids=[w['id'] for w in train], calibration_ids=[w['id'] for w in cal])))

def shifts(out, wells):
    if (out / 'CONTROL_SHIFT_RESULTS.csv').exists():
        return
    frozen = json.loads((out / 'FROZEN.json').read_text())
    names = list(dict.fromkeys([frozen['TNBC']['strongest_baseline'], frozen['TNBC']['selected'], 'MMSL']))
    rows = []
    metadata = []
    for name in names:
        model = pickle.loads((out / 'models' / ('TNBC_' + name + '.pkl')).read_bytes())
        frame = model.frame
        for context in ['MDA231', 'MDA468']:
            controls = [w for w in wells if w['role'] in ['train', 'reference'] and w['context'] == context and (not w['label'])]
            means = np.array([frame.transform(w['cells']).mean(0).reshape(-1) for w in controls])
            means -= means.mean(0)
            u, s, v = np.linalg.svd(means, full_matrices=False)
            direction = v[0] * s[0] / np.sqrt(max(1, len(means) - 1))
            direction = direction.reshape(frame.groups, 24)
            raw = np.zeros_like(frame.center)
            for i, (group, pca) in enumerate(frame.pcas):
                raw[group] = direction[i, :len(pca.components_)] * np.sqrt(np.maximum(pca.explained_variance_, 0.1)) @ pca.components_ * frame.scale[group]
            metadata.append(dict(method=name, context=context, fit_wells=[w['id'] for w in controls], interpretation='within-line control-well feature variation; not independently verified pure technical nuisance'))
            for magnitude in [1.0, 2.0]:
                for w in [w for w in wells if w['role'] == 'test' and w['context'] == context]:
                    sign = 1 if rng('shift' + w['id']).random() > 0.5 else -1
                    altered = w['cells'] + sign * magnitude * raw
                    p, uncertainty, var = model.predict(altered, np.ones(2, dtype=np.float32), w['cutoff'])
                    rows.append(dict(dataset='TNBC', method=name, well=w['id'], group=w['group'], context=context, magnitude=magnitude, interpretation='simulated source-control feature offset; fixed marker label assumption', **e.measurements(w['target'], p, uncertainty)))
    e.write(out / 'CONTROL_SHIFT_RESULTS.csv', rows)
    (out / 'CONTROL_SHIFT_PROVENANCE.json').write_text(json.dumps(metadata, indent=2))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/results/missing_marker')
    out = ROOT / parser.parse_args().output
    wells = tnbc_data(ROOT)
    area_diagnostic(out, wells)
    shifts(out, wells)
    print('Diagnostics saved')
