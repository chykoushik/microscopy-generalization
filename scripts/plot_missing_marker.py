from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation.missing_marker_data import read, rng, tnbc_data
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'scripts/missing_marker.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
METRICS = ['AP', 'AUROC', 'Brier', 'ECE', 'prevalence_error', 'precision', 'recall', 'precision_top0.05', 'recall_top0.05']
COLORS = {'MMSL': '#b44a32', 'count_tail': '#128c7e', 'ratio_tail': '#128c7e', 'marker_regression': '#5b4b98', 'random_forest': '#de9320', 'linear': '#5b4b98', 'masked_MLP': '#247ba0', 'no_lattice': '#85786a'}

def interval(rows, metric):
    own = [r for r in rows if r.get(metric) not in ['', None] and np.isfinite(float(r[metric]))]
    groups = sorted({r['group'] for r in own})
    values = np.array([np.mean([float(r[metric]) for r in own if r['group'] == g]) for g in groups])
    if not len(values):
        return (None, None, None, 0)
    generator = rng('bootstrap' + metric + '|'.join(groups))
    boot = values[generator.integers(len(values), size=(1000, len(values)))].mean(1)
    return (float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), len(groups))

def summarize(out):
    rows = read(out / 'TEST_WELL_RESULTS.csv')
    summary = []
    for dataset in ['TNBC', 'BBBC013']:
        contexts = ['pooled'] + sorted({r['context'] for r in rows if r['dataset'] == dataset})
        for context in contexts:
            for method in sorted({r['method'] for r in rows if r['dataset'] == dataset}):
                for mask in sorted({r['mask'] for r in rows if r['dataset'] == dataset}):
                    own = [r for r in rows if r['dataset'] == dataset and r['method'] == method and (r['mask'] == mask) and (context == 'pooled' or r['context'] == context)]
                    for metric in METRICS:
                        mean, lo, hi, n = interval(own, metric)
                        summary.append(dict(dataset=dataset, context=context, method=method, mask=mask, metric=metric, estimate=mean, low=lo, high=hi, groups=n, test_wells=len(own), evaluable_wells=sum((r.get(metric) not in ['', None] for r in own))))
    e.write(out / 'SUMMARY.csv', summary)
    frozen = json.loads((out / 'FROZEN.json').read_text())
    paired = []
    for dataset in ['TNBC', 'BBBC013']:
        baseline = frozen[dataset]['strongest_baseline']
        selected = frozen[dataset]['selected']
        for mask in sorted({r['mask'] for r in rows if r['dataset'] == dataset}):
            for candidate in [selected, 'MMSL', 'no_lattice', 'no_mask', 'no_prototypes', 'no_consistency', 'no_uncertainty']:
                index = {r['well']: r for r in rows if r['dataset'] == dataset and r['method'] == baseline and (r['mask'] == mask)}
                own = [r for r in rows if r['dataset'] == dataset and r['method'] == candidate and (r['mask'] == mask)]
                for metric in METRICS:
                    differences = [dict(group=r['group'], difference=float(r[metric]) - float(index[r['well']][metric])) for r in own if r[metric] and index[r['well']][metric]]
                    mean, lo, hi, n = interval(differences, 'difference')
                    paired.append(dict(dataset=dataset, method=candidate, baseline=baseline, mask=mask, metric=metric, difference=mean, low=lo, high=hi, groups=n, wells=len(differences)))
    e.write(out / 'PAIRED_COMPARISONS.csv', paired)
    contrasts = []
    for dataset in ['TNBC', 'BBBC013']:
        for ablation in ['no_mask', 'no_prototypes', 'no_consistency', 'no_lattice', 'no_uncertainty']:
            for mask in sorted({r['mask'] for r in rows if r['dataset'] == dataset}):
                index = {r['well']: r for r in rows if r['dataset'] == dataset and r['method'] == 'MMSL' and (r['mask'] == mask)}
                own = [r for r in rows if r['dataset'] == dataset and r['method'] == ablation and (r['mask'] == mask)]
                for metric in ['AP', 'AUROC', 'prevalence_error', 'Brier']:
                    diff = [dict(group=r['group'], difference=float(index[r['well']][metric]) - float(r[metric])) for r in own if r[metric] and index[r['well']][metric]]
                    mean, lo, hi, n = interval(diff, 'difference')
                    contrasts.append(dict(dataset=dataset, ablation=ablation, mask=mask, metric=metric, full_minus_ablation=mean, low=lo, high=hi, groups=n, wells=len(diff)))
    e.write(out / 'ABLATION_COMPARISONS.csv', contrasts)
    rare = read(out / 'RARE_STATE_RESULTS.csv')
    rare_summary = []
    for dataset in ['TNBC', 'BBBC013']:
        for method in sorted({r['method'] for r in rare if r['dataset'] == dataset}):
            for fraction in [0.01, 0.05, 0.1, 0.25]:
                for mask in sorted({r['mask'] for r in rare if r['dataset'] == dataset}):
                    own = [r for r in rare if r['dataset'] == dataset and r['method'] == method and (float(r['fraction']) == fraction) and (r['mask'] == mask) and (r['status'] == 'EXECUTED')]
                    for metric in ['AP', 'AUROC', 'prevalence_error']:
                        mean, lo, hi, n = interval(own, metric)
                        rare_summary.append(dict(dataset=dataset, method=method, fraction=fraction, mask=mask, metric=metric, estimate=mean, low=lo, high=hi, groups=n, wells=len({r['well'] for r in own}), cells_min=min([int(r['cells']) for r in own], default=0), cells_max=max([int(r['cells']) for r in own], default=0), technical_repeats=2))
    e.write(out / 'RARE_STATE_SUMMARY.csv', rare_summary)
    shifts = []
    for file, axes in [('LABEL_SCARCITY_RESULTS.csv', ['axis', 'fraction']), ('CELL_LINE_TRANSFER.csv', ['source_line', 'target_line'])]:
        source = read(out / file)
        tuples = sorted({tuple((r.get(k, '') for k in axes)) + (r.get('method', ''), r.get('mask', '')) for r in source if r.get('status', 'EXECUTED') == 'EXECUTED'})
        for values in tuples:
            own = [r for r in source if tuple((r.get(k, '') for k in axes)) + (r.get('method', ''), r.get('mask', '')) == values]
            for metric in ['AP', 'AUROC', 'prevalence_error', 'Brier']:
                mean, lo, hi, n = interval(own, metric)
                shifts.append(dict(file=file, **dict(zip(axes + ['method', 'mask'], values)), metric=metric, estimate=mean, low=lo, high=hi, groups=n, wells=len(own)))
    e.write(out / 'SHIFT_SUMMARY.csv', shifts)
    selective = read(out / 'SELECTIVE_PREDICTION.csv')
    risk = []
    for dataset in ['TNBC', 'BBBC013']:
        for method in sorted({r['method'] for r in selective if r['dataset'] == dataset}):
            for coverage in sorted({r['coverage'] for r in selective}):
                own = [r for r in selective if r['dataset'] == dataset and r['method'] == method and (r['coverage'] == coverage)]
                mean, lo, hi, n = interval(own, 'Brier_risk')
                risk.append(dict(dataset=dataset, method=method, coverage=coverage, estimate=mean, low=lo, high=hi, groups=n, wells=len(own)))
    e.write(out / 'RISK_COVERAGE.csv', risk)
    return summary

def get(summary, dataset, method, metric, context='pooled', mask=None):
    full = '11' if dataset == 'TNBC' else '1'
    return next((r for r in summary if r['dataset'] == dataset and r['context'] == context and (r['method'] == method) and (r['metric'] == metric) and (r['mask'] == (mask or full))))

def figures(out, summary):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 140})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    methods = ['logistic', 'random_forest', 'marker_regression', 'masked_MLP', 'MMSL', 'count_tail']
    for ax, metric in zip(axes, ['AP', 'AUROC']):
        for j, context in enumerate(['MDA231', 'MDA468']):
            for i, method in enumerate(methods):
                r = get(summary, 'TNBC', method, metric, context)
                y = r['estimate']
                ax.errorbar(i + (j - 0.5) * 0.2, y, yerr=[[max(0, y - r['low'])], [max(0, r['high'] - y)]], fmt='o', capsize=3, color=['#247ba0', '#de9320'][j], label=context if i == 0 else None)
        ax.set(xticks=range(len(methods)), xticklabels=['Logistic', 'RF', 'Marker GB', 'Masked MLP', 'MMSL', 'Count tail'], ylabel=metric, ylim=(0, 1), title='Held-out TNBC marker-high recognition')
        ax.tick_params(axis='x', rotation=25)
        ax.legend(fontsize=8)
    fig.savefig(out / 'tnbc_method_comparison.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method in ['random_forest', 'marker_regression', 'masked_MLP', 'MMSL', 'count_tail']:
        values = [get(summary, 'TNBC', method, 'AP', mask=m) for m in ['11', '10', '01']]
        axes[0].plot(range(3), [r['estimate'] for r in values], 'o-', label=method, color=COLORS[method])
        values = [get(summary, 'TNBC', method, 'prevalence_error', mask=m) for m in ['11', '10', '01']]
        axes[1].plot(range(3), [r['estimate'] for r in values], 'o-', label=method, color=COLORS[method])
    for ax, label in zip(axes, ['AP', 'Well prevalence MAE']):
        ax.set(xticks=range(3), xticklabels=['DAPI + tubulin', 'DAPI only', 'Tubulin only'], ylabel=label)
        ax.legend(fontsize=8)
    fig.savefig(out / 'missing_channel_robustness.png')
    plt.close(fig)
    rare = read(out / 'RARE_STATE_SUMMARY.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, dataset in zip(axes, ['TNBC', 'BBBC013']):
        for method in ['marker_regression', 'MMSL', 'count_tail'] if dataset == 'TNBC' else ['linear', 'MMSL', 'ratio_tail']:
            own = sorted([r for r in rare if r['dataset'] == dataset and r['method'] == method and (r['metric'] == 'AP') and (r['mask'] == ('11' if dataset == 'TNBC' else '1'))], key=lambda r: float(r['fraction']))
            ax.plot([float(r['fraction']) * 100 for r in own], [float(r['estimate']) if r['estimate'] else np.nan for r in own], 'o-', label=method, color=COLORS[method])
        ax.plot([1, 25], [0.01, 0.25], ':', color='gray')
        ax.set(xlabel='Constructed marker-state prevalence (%)', ylabel='AP', title=dataset, xticks=[1, 5, 10, 25])
        ax.legend(fontsize=8)
    fig.savefig(out / 'rare_state_robustness.png')
    plt.close(fig)
    rows = read(out / 'TEST_WELL_RESULTS.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, method in zip(axes, ['marker_regression', 'count_tail']):
        for context, color in [('MDA231', '#247ba0'), ('MDA468', '#de9320')]:
            own = [r for r in rows if r['dataset'] == 'TNBC' and r['method'] == method and (r['mask'] == '11') and (r['context'] == context)]
            ax.scatter([float(r['prevalence']) for r in own], [float(r['prevalence_estimate']) for r in own], label=context, color=color, alpha=0.75)
        ax.plot([0, 0.25], [0, 0.25], ':', color='gray')
        ax.set(xlabel='Observed marker-high fraction', ylabel='Estimated fraction', title=method, xlim=(0, 0.25), ylim=(0, 0.25))
        ax.legend()
    fig.savefig(out / 'tnbc_well_prevalence.png')
    plt.close(fig)
    risk = read(out / 'RISK_COVERAGE.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, dataset in zip(axes, ['TNBC', 'BBBC013']):
        for method in ['marker_regression', 'MMSL', 'count_tail'] if dataset == 'TNBC' else ['linear', 'MMSL', 'ratio_tail']:
            own = sorted([r for r in risk if r['dataset'] == dataset and r['method'] == method], key=lambda r: float(r['coverage']))
            ax.plot([float(r['coverage']) for r in own], [float(r['estimate']) for r in own], 'o-', label=method, color=COLORS[method])
        ax.set(xlabel='Cell coverage within held-out well', ylabel='Brier risk', title=dataset)
        ax.legend(fontsize=8)
    fig.savefig(out / 'selective_prediction.png')
    plt.close(fig)
    frame = get(summary, 'BBBC013', 'linear', 'AP')
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    methods = ['linear', 'marker_regression', 'MMSL', 'ratio_tail']
    for j, metric in enumerate(['AP', 'AUROC']):
        values = [get(summary, 'BBBC013', m, metric) for m in methods]
        y = np.array([r['estimate'] for r in values])
        ax.bar(np.arange(4) + (j - 0.5) * 0.3, y, width=0.3, label=metric, yerr=[y - np.array([r['low'] for r in values]), np.array([r['high'] for r in values]) - y], capsize=3)
    ax.set(xticks=range(4), xticklabels=methods, ylim=(0, 1), title='Independent FKHR-GFP assay: DNA-only recognition')
    ax.legend()
    fig.savefig(out / 'second_dataset.png')
    plt.close(fig)
    wells = tnbc_data(ROOT)
    model = pickle.loads((out / 'models/TNBC_count_tail.pkl').read_bytes())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    chosen = []
    for i, context in enumerate(['MDA231', 'MDA468']):
        w = sorted([w for w in wells if w['role'] == 'test' and w['context'] == context and w['label'] and (len(w['cells']) >= 25)], key=lambda w: w['id'])[0]
        with h5py.File(ROOT / w['provenance'], 'r') as f:
            position = next(iter(f['sample/0/plate/' + w['plate'] + '/experiment/' + w['id'].split(':')[1] + '/position'].values()))
            images = position['image/channel'][:, 0, 0]
        dna = images[0].astype(float)
        tubulin = images[1].astype(float)
        dna = np.clip(dna / (np.quantile(dna, 0.995) + 1e-06), 0, 1)
        tubulin = np.clip(tubulin / (np.quantile(tubulin, 0.995) + 1e-06), 0, 1)
        rgb = np.stack([tubulin * 0.6, tubulin * 0.7, dna], -1)
        p, u, var = model.predict(w['cells'], np.ones(2, dtype=np.float32), w['cutoff'])
        for j, (title, score) in enumerate([('Predicted probability', p), ('Uncertainty', u), ('Held-out γH2AX-high', w['target'])]):
            ax = axes[i, j]
            ax.imshow(rgb)
            c = ax.scatter(w['center']['x'], w['center']['y'], c=score, cmap='viridis', s=17, alpha=0.85)
            ax.set_title(w['id'] + ' | ' + title, fontsize=9)
            ax.set_axis_off()
            fig.colorbar(c, ax=ax, fraction=0.035)
        chosen.append(dict(well=w['id'], selection='first lexical treated test well >=25 cells; no prediction/marker criterion', marker_free_background='DAPI+Cy5 only'))
    fig.savefig(out / 'tnbc_marker_free_localization.png')
    plt.close(fig)
    e.write(out / 'FIGURE_WELLS.csv', chosen)
    failures = []
    for context in ['MDA231', 'MDA468']:
        own = sorted([r for r in rows if r['dataset'] == 'TNBC' and r['method'] == 'count_tail' and (r['mask'] == '11') and (r['context'] == context)], key=lambda r: -float(r['prevalence_error']))
        failures += own[:3]
    e.write(out / 'FAILURE_CASES.csv', failures)
    print('Figures saved')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/results/missing_marker')
    args = parser.parse_args()
    out = ROOT / args.output
    summary = summarize(out)
    figures(out, summary)
