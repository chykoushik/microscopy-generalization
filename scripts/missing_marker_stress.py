from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from evaluation.missing_marker_data import Channels, read, rng
from evaluation.missing_marker import Predictor, masks
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'scripts/missing_marker.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)

def saved(out, dataset, method, well, mask):
    token = hashlib.sha256((well + mask).encode()).hexdigest()[:16]
    return np.load(out / 'predictions' / (dataset + '_' + method + '_' + token + '.npz'))

def rare(out):
    if (out / 'RARE_STATE_RESULTS.csv').exists():
        return
    rows = []
    selective = []
    primary = read(out / 'TEST_WELL_RESULTS.csv')
    for row in primary:
        data = saved(out, row['dataset'], row['method'], row['well'], row['mask'])
        truth = data['truth']
        p = data['p']
        u = data['uncertainty']
        if int(row['available_channels']) == len(row['mask']):
            order = np.argsort(u, kind='stable')
            risks = []
            for coverage in np.linspace(0.1, 1, 10):
                chosen = order[:max(1, int(np.ceil(len(order) * coverage)))]
                y = truth[chosen]
                q = p[chosen]
                risks.append(float(np.mean((y - q) ** 2)))
                selective.append(dict(dataset=row['dataset'], method=row['method'], well=row['well'], group=row['group'], context=row['context'], coverage=coverage, Brier_risk=risks[-1], classification_risk=float(np.mean((q >= 0.5) != y)), cells=len(chosen), oracle_prevalence_constant_risk=float(np.mean((y - truth.mean()) ** 2)), interpretation='cell retention inside a well; oracle comparator uses observed full-well prevalence and is not deployable'))
        for fraction, denominator in [(0.01, 100), (0.05, 20), (0.1, 10), (0.25, 4)]:
            pos = np.where(truth == 1)[0]
            neg = np.where(truth == 0)[0]
            n = int(min(200, len(pos) / fraction, len(neg) / (1 - fraction)) // denominator) * denominator
            if n < 20:
                rows.append(dict(dataset=row['dataset'], method=row['method'], well=row['well'], group=row['group'], context=row['context'], mask=row['mask'], fraction=fraction, status='NOT_RUN', reason='Insufficient without-replacement cells for exact fraction and N>=20', cells=n))
                continue
            count = int(round(n * fraction))
            for repeat in range(2):
                generator = rng(row['well'] + str(fraction) + str(repeat))
                indices = np.r_[generator.choice(pos, count, replace=False), generator.choice(neg, n - count, replace=False)]
                result = e.measurements(truth[indices], p[indices], u[indices])
                rows.append(dict(dataset=row['dataset'], method=row['method'], well=row['well'], group=row['group'], context=row['context'], mask=row['mask'], fraction=fraction, repeat=repeat, status='EXECUTED', sample_sha256=hashlib.sha256(indices.tobytes()).hexdigest(), interpretation='same-well stratified marker-state subset; not a new biological replicate', **result))
    e.write(out / 'RARE_STATE_RESULTS.csv', rows)
    e.write(out / 'SELECTIVE_PREDICTION.csv', selective)
    print('Rare shifts complete')

def refit(wells, frame, name):
    original = 'MMSL' if name == 'no_uncertainty' else name
    model = Predictor().fit(wells, frame, original)
    if name == 'no_uncertainty':
        model.name = name
    model.calibrate(wells, passes=1 if name == 'no_uncertainty' else 8)
    return model

def training_shifts(out):
    if (out / 'TRAINING_SHIFT_COMPLETE.json').exists():
        return
    frozen = json.loads((out / 'FROZEN.json').read_text())
    wells = e.datasets(out)['TNBC']
    names = list(dict.fromkeys([frozen['TNBC']['strongest_baseline'], frozen['TNBC']['selected']]))
    frame = pickle.loads((out / 'TNBC_frame.pkl').read_bytes())
    results = []
    fits = []
    design = dict(seed=719, label_well_fractions=[0.25, 0.5], cell_label_fractions=[0.1, 0.25, 0.5], line_transfer=['MDA231_to_MDA468', 'MDA468_to_MDA231'], selected_methods=names, selection='fixed main validation selections; no stress tuning', test_labels_used_for_fitting=False, scarcity='deterministic unstratified groups/cells; unknown labels not used to choose samples', input_preprocessing='full training-only unlabeled features retained for scarcity')
    (out / 'TRAINING_SHIFT_DESIGN.json').write_text(json.dumps(design, indent=2))
    for axis, fractions in [('marker_well_fraction', [0.25, 0.5]), ('cell_label_fraction', [0.1, 0.25, 0.5])]:
        for fraction in fractions:
            altered = copy.deepcopy(wells)
            groups = sorted({w['group'] for w in wells if w['role'] == 'train'}, key=lambda g: rng('scarcity' + g).random())
            selected = set(groups[:max(1, int(np.ceil(fraction * len(groups))))])
            for w in altered:
                if w['role'] != 'train':
                    continue
                if axis == 'marker_well_fraction':
                    if w['group'] not in selected:
                        w['role'] = 'unused'
                else:
                    index = rng('cells' + w['id']).choice(len(w['cells']), max(1, int(np.ceil(len(w['cells']) * fraction))), replace=False)
                    for key in ['cells', 'target', 'counts', 'marker']:
                        w[key] = w[key][index]
            labels = np.concatenate([w['target'] for w in altered if w['role'] == 'train'])
            if len(np.unique(labels)) < 2:
                results.append(dict(dataset='TNBC', axis=axis, fraction=fraction, status='NOT_RUN', reason='Selected training labels contain one class'))
                continue
            for name in names:
                model = refit(altered, frame, name)
                own = e.score(model, altered, 'test', 'TNBC', passes=1 if name == 'no_uncertainty' else 8)
                results += [dict(r, axis=axis, fraction=fraction, status='EXECUTED') for r in own]
                fits.append(dict(axis=axis, fraction=fraction, method=name, train_wells=len(model.fit_ids), train_cells=len(labels), positive_cells=int(labels.sum()), fit_ids=json.dumps(model.fit_ids)))
                (out / 'stress_models' / (axis + str(fraction) + '_' + name + '.pkl')).write_bytes(pickle.dumps(model))
                print(axis, name, 'fitted')
    transfer = []
    for source in ['MDA231', 'MDA468']:
        target = 'MDA468' if source == 'MDA231' else 'MDA231'
        altered = [copy.deepcopy(w) for w in wells if w['context'] == source and w['role'] != 'test' or (w['context'] == target and w['role'] == 'test')]
        source_frame = Channels().fit(altered)
        for name in names:
            model = refit(altered, source_frame, name)
            own = e.score(model, altered, 'test', 'TNBC', passes=1 if name == 'no_uncertainty' else 8)
            transfer += [dict(r, source_line=source, target_line=target, interpretation='joint cell-line/plate shift, not separable technical batch transfer') for r in own]
            fits.append(dict(axis='line_plus_plate', fraction=1, method=name, source=source, train_wells=len(model.fit_ids), fit_ids=json.dumps(model.fit_ids)))
            (out / 'stress_models' / (source + '_to_' + target + '_' + name + '.pkl')).write_bytes(pickle.dumps(model))
            print(source, name, 'transferred')
    results.append(dict(dataset='TNBC', axis='marker_well_fraction', fraction=0, status='NOT_RUN', reason='No source marker or state labels: semantic state supervised fitting unsupported'))
    e.write(out / 'LABEL_SCARCITY_RESULTS.csv', results)
    e.write(out / 'CELL_LINE_TRANSFER.csv', transfer)
    e.write(out / 'STRESS_FIT_PROVENANCE.csv', fits)
    (out / 'TRAINING_SHIFT_COMPLETE.json').write_text(json.dumps(dict(rows=len(results), transfer_rows=len(transfer))))

def quantification(out):
    rows = []
    frozen = json.loads((out / 'FROZEN.json').read_text())
    all_data = e.datasets(out)
    for dataset, wells in all_data.items():
        names = list(dict.fromkeys([frozen[dataset]['strongest_baseline'], frozen[dataset]['selected'], 'MMSL']))
        for name in names:
            model = pickle.loads((out / 'models' / (dataset + '_' + name + '.pkl')).read_bytes())
            mask = np.ones(model.groups, dtype=np.float32)
            calibration = [(w, model.predict(w['cells'], mask, w['cutoff'], passes=1 if name == 'no_uncertainty' else 8)[0]) for w in wells if w['role'] == 'calibration']
            p = np.concatenate([p for w, p in calibration])
            y = np.concatenate([w['target'] for w, p in calibration])
            m0 = p[y == 0].mean()
            m1 = p[y == 1].mean()
            for w in [w for w in wells if w['role'] == 'test']:
                probability = model.predict(w['cells'], mask, w['cutoff'], passes=1 if name == 'no_uncertainty' else 8)[0]
                corrected = float(np.clip((probability.mean() - m0) / (m1 - m0), 0, 1)) if m1 - m0 > 0.001 else None
                rows.append(dict(dataset=dataset, method=name, well=w['id'], group=w['group'], context=w['context'], truth=float(w['target'].mean()), mean_probability=float(probability.mean()), mean_probability_error=float(abs(probability.mean() - w['target'].mean())), PACC=corrected, PACC_error=abs(corrected - w['target'].mean()) if corrected is not None else None, calibration_negative_mean=float(m0), calibration_positive_mean=float(m1), interpretation='standard probabilistic adjusted classify/count; label-shift assumption unverified'))
    e.write(out / 'PREVALENCE_QUANTIFICATION.csv', rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/results/missing_marker')
    args = parser.parse_args()
    out = ROOT / args.output
    (out / 'stress_models').mkdir(exist_ok=True)
    rare(out)
    training_shifts(out)
    quantification(out)
if __name__ == '__main__':
    main()
