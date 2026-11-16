from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from evaluation.missing_marker_data import tnbc_data, second_data, Channels, rng, read
from evaluation.missing_marker import Predictor, masks
BASELINES = ['logistic', 'linear', 'gradient_boosting', 'random_forest', 'MLP', 'class_balanced', 'metric_learning', 'prototype', 'group_DRO', 'masked_MLP', 'marker_regression']
CANDIDATES = ['MMSL', 'no_mask', 'no_prototypes', 'no_consistency', 'no_lattice']

def write(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys((k for row in rows for k in row)))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys or ['status'])
        writer.writeheader()
        writer.writerows(rows)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def datasets(out):
    secondary = second_data(ROOT, out)
    for w in secondary:
        w['counts'] = w['marker'].astype(np.float32)
    return {'TNBC': tnbc_data(ROOT), 'BBBC013': secondary}

def measurements(y, p, u):
    positive = int(y.sum())
    n = len(y)
    valid = 0 < positive < n
    result = dict(cells=n, positives=positive, prevalence=float(y.mean()), AP=float(average_precision_score(y, p)) if valid else None, AUROC=float(roc_auc_score(y, p)) if valid else None, Brier=float(np.mean((y - p) ** 2)), prevalence_estimate=float(p.mean()), prevalence_error=float(abs(p.mean() - y.mean())), mean_uncertainty=float(u.mean()), TP=int(((p >= 0.5) & (y == 1)).sum()), FP=int(((p >= 0.5) & (y == 0)).sum()), FN=int(((p < 0.5) & (y == 1)).sum()))
    result['precision'] = result['TP'] / max(result['TP'] + result['FP'], 1)
    result['recall'] = result['TP'] / positive if positive else None
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if selected.any():
            ece += selected.mean() * abs(p[selected].mean() - y[selected].mean())
    result['ECE'] = float(ece)
    for f in [0.01, 0.05, 0.1]:
        count = max(1, int(np.ceil(n * f)))
        threshold = np.sort(p)[-count]
        above = p > threshold
        tied = p == threshold
        true = y[above].sum() + (count - above.sum()) * y[tied].mean()
        result['precision_top' + str(f)] = float(true / count)
        result['recall_top' + str(f)] = float(true / positive) if positive else None
    return result

def score(model, wells, role, dataset, name=None, export=None, passes=8):
    rows = []
    for w in [w for w in wells if w['role'] == role]:
        for mask in masks(model.groups):
            p, u, var = model.predict(w['cells'], mask, w['cutoff'], passes=passes)
            row = measurements(w['target'], p, u)
            row.update(dataset=dataset, method=name or model.name, well=w['id'], group=w['group'], context=w['context'], treatment=w['treatment'], mask=''.join((str(int(v)) for v in mask)), available_channels=int(mask.sum()), role=role)
            rows.append(row)
            if export:
                token = hashlib.sha256((w['id'] + row['mask']).encode()).hexdigest()[:16]
                np.savez_compressed(export / (dataset + '_' + (name or model.name) + '_' + token + '.npz'), p=p, uncertainty=u, variance=var, truth=w['target'], cell_ids=np.array([w['id'] + ':' + str(i) for i in range(len(p))]))
    return rows

def macro(rows, method):
    own = [r for r in rows if r['method'] == method and r['AP'] is not None]
    return float(np.mean([np.mean([r['AP'] for r in own if r['group'] == g]) for g in sorted({r['group'] for r in own})]))

def protocol(out, all_data):
    manifest = []
    for dataset, wells in all_data.items():
        for w in wells:
            manifest.append(dict(dataset=dataset, well=w['id'], role=w['role'], context=w['context'], treatment=w['treatment'], group=w['group'], cells=len(w['cells']), positives=int(w['target'].sum()), cutoff=w['cutoff'], input_channels=';'.join(w['channels']), provenance=w['provenance']))
    write(out / 'SPLIT_MANIFEST.csv', manifest)
    config = dict(seed=719, epochs=40, batch_size=256, latent=16, per_channel_PCA=24, selection='validation group-macro AP averaged over all nonempty channel subsets', calibration='Platt scaling on independent calibration-control wells for each subset', bootstrap='1000 group resamples, cells and mask views nested', TNBC_marker='Cy3 spotfeatures_count >19 MDA231; >21 MDA468', TNBC_inputs=['DAPI239', 'Cy5_tubulin239'], BBBC013_marker='log GFP nuclear/ring ratio > reference95th percentile', BBBC013_inputs=['DRAQ_DNA morphology only'], test_tuning=False, baselines=BASELINES, candidates=CANDIDATES, alternative='marker-count/ratio distribution tail model if MMSL validation does not beat strongest baseline')
    (out / 'PROTOCOL.json').write_text(json.dumps(config, indent=2))

def baseline_stage(out):
    if (out / 'BASELINES_COMPLETE.json').exists():
        raise RuntimeError('Preserve baseline run')
    all_data = datasets(out)
    protocol(out, all_data)
    rows = []
    fits = []
    reproduction = []
    historical = read(ROOT / 'outputs/results/microscopy_response_localization_v1/BIOLOGICAL_STATE_RESULTS.csv')
    old = [r for r in historical if r['method'] == 'C' and r['label'] == '1' and r['AP']]
    old_model = pickle.loads((ROOT / 'outputs/results/microscopy_response_localization_v1/TNBC_model.pkl').read_bytes())
    for w in all_data['TNBC']:
        if w['role'] != 'test' or not w['label'] or len(np.unique(w['target'])) < 2:
            continue
        p = old_model.scores(w['cells'], w['context'])[0]['C']
        a = average_precision_score(w['target'], p)
        b = roc_auc_score(w['target'], p)
        expected = next((r for r in old if r['well'] == w['id']))
        if abs(a - float(expected['AP'])) > 1e-10 or abs(b - float(expected['AUROC'])) > 1e-10:
            raise ValueError('Historical replay mismatch')
        reproduction.append(dict(well=w['id'], group=w['group'], context=w['context'], cells=len(p), AP=a, AUROC=b, AP_difference=a - float(expected['AP']), AUROC_difference=b - float(expected['AUROC'])))
    write(out / 'PREVIOUS_MARKER_REPRODUCTION.csv', reproduction)
    for dataset, wells in all_data.items():
        frame = Channels().fit(wells)
        (out / (dataset + '_frame.pkl')).write_bytes(pickle.dumps(frame))
        for name in BASELINES:
            model = Predictor().fit(wells, frame, name).calibrate(wells)
            (out / 'models' / (dataset + '_' + name + '.pkl')).write_bytes(pickle.dumps(model))
            own = score(model, wells, 'validation', dataset)
            rows += own
            fits.append(dict(dataset=dataset, method=name, parameters=model.parameters, training_wells=len(model.fit_ids), calibration_wells=len(model.calibration_ids), epochs=model.epochs, validation_AP=macro(own, name)))
            print(dataset, name, 'fitted')
    write(out / 'BASELINE_VALIDATION.csv', rows)
    write(out / 'FIT_RESULTS.csv', fits)
    (out / 'BASELINES_COMPLETE.json').write_text(json.dumps(dict(protocol_sha256=sha(out / 'PROTOCOL.json'), manifest_sha256=sha(out / 'SPLIT_MANIFEST.csv')), indent=2))

def development_stage(out):
    if (out / 'FROZEN.json').exists():
        raise RuntimeError('Preserve frozen run')
    all_data = datasets(out)
    base_rows = read(out / 'BASELINE_VALIDATION.csv')
    for r in base_rows:
        r['AP'] = float(r['AP']) if r['AP'] else None
    rows = []
    fits = read(out / 'FIT_RESULTS.csv')
    frozen = {}
    if not (out / 'PROTOCOL_initial.json').exists():
        (out / 'PROTOCOL_initial.json').write_bytes((out / 'PROTOCOL.json').read_bytes())
    config = json.loads((out / 'PROTOCOL.json').read_text())
    config.update(baselines=BASELINES, alternative_trigger='MMSL fails validation baseline or lattice ablation is no worse', pretest_addendum='Continuous-marker regression added to match supervision of distribution-tail alternative.')
    (out / 'PROTOCOL.json').write_text(json.dumps(config, indent=2))
    for dataset, wells in all_data.items():
        frame = pickle.loads((out / (dataset + '_frame.pkl')).read_bytes())
        if not any((r['dataset'] == dataset and r['method'] == 'marker_regression' for r in base_rows)):
            model = Predictor().fit(wells, frame, 'marker_regression').calibrate(wells)
            (out / 'models' / (dataset + '_marker_regression.pkl')).write_bytes(pickle.dumps(model))
            own = score(model, wells, 'validation', dataset)
            base_rows += own
            fits.append(dict(dataset=dataset, method='marker_regression', parameters=None, training_wells=len(model.fit_ids), calibration_wells=len(model.calibration_ids), epochs=0, validation_AP=macro(own, 'marker_regression')))
        strongest = max(BASELINES, key=lambda m: macro([r for r in base_rows if r['dataset'] == dataset], m))
        baseline_ap = macro([r for r in base_rows if r['dataset'] == dataset], strongest)
        for name in CANDIDATES:
            model = Predictor().fit(wells, frame, name).calibrate(wells)
            (out / 'models' / (dataset + '_' + name + '.pkl')).write_bytes(pickle.dumps(model))
            own = score(model, wells, 'validation', dataset)
            rows += own
            fits.append(dict(dataset=dataset, method=name, parameters=model.parameters, training_wells=len(model.fit_ids), calibration_wells=len(model.calibration_ids), epochs=model.epochs, validation_AP=macro(own, name)))
            if name == 'MMSL':
                no_u = copy.deepcopy(model)
                no_u.name = 'no_uncertainty'
                no_u.calibrate(wells, passes=1)
                (out / 'models' / (dataset + '_no_uncertainty.pkl')).write_bytes(pickle.dumps(no_u))
                own_u = score(no_u, wells, 'validation', dataset, passes=1)
                rows += own_u
            print(dataset, name, 'fitted')
        full_ap = macro([r for r in rows if r['dataset'] == dataset], 'MMSL')
        alternative = 'NOT_RUN'
        lattice_ap = macro([r for r in rows if r['dataset'] == dataset], 'no_lattice')
        if full_ap <= baseline_ap or lattice_ap >= full_ap:
            alternative = 'count_tail' if dataset == 'TNBC' else 'ratio_tail'
            model = Predictor().fit(wells, frame, alternative).calibrate(wells)
            (out / 'models' / (dataset + '_' + alternative + '.pkl')).write_bytes(pickle.dumps(model))
            own = score(model, wells, 'validation', dataset)
            rows += own
            fits.append(dict(dataset=dataset, method=alternative, parameters=model.parameters, training_wells=len(model.fit_ids), calibration_wells=len(model.calibration_ids), epochs=model.epochs, validation_AP=macro(own, alternative)))
            print(dataset, alternative, 'fitted')
        candidates = CANDIDATES + ['no_uncertainty'] + ([] if alternative == 'NOT_RUN' else [alternative])
        selected = max(candidates, key=lambda m: macro([r for r in rows if r['dataset'] == dataset], m))
        frozen[dataset] = dict(strongest_baseline=strongest, baseline_validation_AP=baseline_ap, full_MMSL_AP=full_ap, alternative=alternative, selected=selected, selected_validation_AP=macro([r for r in rows if r['dataset'] == dataset], selected), candidates=candidates, feature_fit_ids=frame.fit_ids, models={p.name: sha(p) for p in (out / 'models').glob(dataset + '_*.pkl')})
    write(out / 'CANDIDATE_VALIDATION.csv', rows)
    write(out / 'BASELINE_VALIDATION.csv', base_rows)
    write(out / 'FIT_RESULTS.csv', fits)
    frozen['manifest_sha256'] = sha(out / 'SPLIT_MANIFEST.csv')
    frozen['protocol_sha256'] = sha(out / 'PROTOCOL.json')
    frozen['code_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'src/evaluation/missing_marker.py', ROOT / 'src/evaluation/missing_marker_data.py']}
    (out / 'FROZEN.json').write_text(json.dumps(frozen, indent=2))
    print('Development frozen')

def evaluation_stage(out):
    if (out / 'TEST_COMPLETE.json').exists():
        raise RuntimeError('Preserve test run')
    frozen = json.loads((out / 'FROZEN.json').read_text())
    rows = []
    if sha(out / 'SPLIT_MANIFEST.csv') != frozen['manifest_sha256']:
        raise ValueError('Manifest changed')
    for name, expected in frozen['code_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen code changed')
    (out / 'TEST_STARTED.json').write_text(json.dumps(dict(frozen_sha256=sha(out / 'FROZEN.json'))))
    all_data = datasets(out)
    for dataset, wells in all_data.items():
        for name, expected in frozen[dataset]['models'].items():
            path = out / 'models' / name
            if sha(path) != expected:
                raise ValueError('Checkpoint changed')
            model = pickle.loads(path.read_bytes())
            rows += score(model, wells, 'test', dataset, export=out / 'predictions', passes=1 if model.name == 'no_uncertainty' else 8)
            print(dataset, model.name, 'evaluated')
    write(out / 'TEST_WELL_RESULTS.csv', rows)
    write(out / 'TNBC_RESULTS.csv', [r for r in rows if r['dataset'] == 'TNBC'])
    write(out / 'SECOND_DATASET_RESULTS.csv', [r for r in rows if r['dataset'] == 'BBBC013'])
    write(out / 'MISSING_CHANNEL_RESULTS.csv', rows)
    write(out / 'BASELINES.csv', [r for r in rows if r['method'] in BASELINES])
    write(out / 'ABLATIONS.csv', [r for r in rows if r['method'] in CANDIDATES + ['no_uncertainty']])
    write(out / 'HELDOUT_DRUG_RESULTS.csv', [r for r in rows if r['dataset'] == 'TNBC' or r['treatment'] == 'LY294002'])
    (out / 'TEST_COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), frozen_sha256=sha(out / 'FROZEN.json')), indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['baseline', 'develop', 'evaluate'], required=True)
    parser.add_argument('--output', default='outputs/results/missing_marker')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(exist_ok=True, parents=True)
    (out / 'models').mkdir(exist_ok=True)
    (out / 'predictions').mkdir(exist_ok=True)
    {'baseline': baseline_stage, 'develop': development_stage, 'evaluate': evaluation_stage}[args.stage](out)
if __name__ == '__main__':
    main()
