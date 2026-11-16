import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/cache/matplotlib'))
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.distance import cdist, pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT / 'src'))
from features.distribution import distribution_scores
from evaluation.real_replication import weighted_wasserstein
OUT = ROOT / 'outputs/results/jump_experiment_v1'
FIG = ROOT / 'outputs/figures/jump_experiment_v1'
FIG.mkdir(parents=True, exist_ok=True)
METHODS = ['mean', 'mmd', 'sliced_w2', 'local_abundance']
NAMES = ['Mean distance squared', 'Gaussian MMD squared', 'Sliced W2 squared', 'Local abundance difference']
SEED = 123001

def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def row_number(well):
    value = 0
    for letter in well[:-2]:
        value = value * 26 + ord(letter) - 64
    return value

def layout():
    import gzip
    evidence = ROOT / 'data/manifests/jump'
    plates = {r['Metadata_Plate']: r['Metadata_Batch'] for r in read_csv(evidence / 'proposed_split.csv')}
    with gzip.open(evidence / 'metadata__well.csv.gz', 'rt') as f:
        wells = [r for r in csv.DictReader(f) if r['Metadata_Source'] == 'source_1' and r['Metadata_Plate'] in plates]
    relevant = [r for r in wells if r['Metadata_JCP2022'] in ['JCP2022_033924', 'JCP2022_037716']]
    selected = {(r['Metadata_Plate'], r['Metadata_Well']) for r in read_csv(evidence / 'proposed_split.csv')}
    counts = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for ax, (plate, batch) in zip(axes, plates.items()):
        for label, color, marker in [('JCP2022_033924', '#4477aa', 'o'), ('JCP2022_037716', '#cc3311', 's')]:
            rows = [r for r in relevant if r['Metadata_Plate'] == plate and r['Metadata_JCP2022'] == label]
            ax.scatter([int(r['Metadata_Well'][-2:]) for r in rows], [row_number(r['Metadata_Well']) for r in rows], c=color, marker=marker, s=18, label='DMSO' if label.endswith('033924') else 'AMG900')
            acquired = [r for r in rows if (plate, r['Metadata_Well']) in selected]
            ax.scatter([int(r['Metadata_Well'][-2:]) for r in acquired], [row_number(r['Metadata_Well']) for r in acquired], facecolors='none', edgecolors='black', s=35, linewidths=0.6)
            counts.append(dict(plate=plate, batch=batch, treatment='DMSO' if label.endswith('033924') else 'AMG900', all_wells=len(rows), selected_wells=len(acquired), columns=','.join(map(str, sorted({int(r['Metadata_Well'][-2:]) for r in rows}))), rows=','.join(map(str, sorted({row_number(r['Metadata_Well']) for r in rows})))))
        ax.set(xlim=(0, 49), ylim=(33, 0), xlabel='Plate column', ylabel='Plate row', title=f'{plate} / {batch}')
        ax.legend(loc='upper center', ncol=2)
    fig.suptitle('Actual plate maps: treatment is confounded with column\nBlack outlines: acquired selection; no exact column-matched DMSO')
    fig.savefig(FIG / 'plate_layout.png', dpi=180)
    plt.close(fig)
    write_csv(OUT / 'layout_counts.csv', counts)
    matching = []
    for r in relevant:
        if r['Metadata_JCP2022'] != 'JCP2022_037716':
            continue
        plate, well = (r['Metadata_Plate'], r['Metadata_Well'])
        neighbors = [a for a in relevant if a['Metadata_Plate'] == plate and a['Metadata_JCP2022'] == 'JCP2022_033924' and (row_number(a['Metadata_Well']) == row_number(well)) and (abs(int(a['Metadata_Well'][-2:]) - int(well[-2:])) == 1)]
        matching.append(dict(plate=plate, treated_well=well, adjacent_same_row_controls=';'.join((a['Metadata_Well'] for a in neighbors)), selected_adjacent_controls=';'.join((a['Metadata_Well'] for a in neighbors if (plate, a['Metadata_Well']) in selected))))
    write_csv(OUT / 'position_matching.csv', matching)
    (OUT / 'spatial_design.json').write_text(json.dumps(dict(counts=counts, exact_column_matched_controls=0, exact_physical_position_matched_controls_across_plates=0, plate_batch_one_to_one=True, treatment_column_perfect_separation=True, extra_control='UL000081 AE43 is not selected', treatment_rows=[2, 10, 18, 26], note='Every plate includes both treatments. AMG900 positions recur across plates. Row matching/adjacent-column interpolation cannot remove unobserved column-specific effects.'), indent=2))

def analyze():
    rows = [r for r in read_csv(OUT / 'wells.csv') if r['status'] == 'ok']
    metadata = {r['Metadata_Plate'] + ':' + r['Metadata_Well']: r for r in rows}
    bags = {}
    for key, r in metadata.items():
        values = np.load(OUT / 'cells' / f"{r['Metadata_Plate']}_{r['Metadata_Well']}.npz")['embeddings']
        bags[key] = values.astype(float)
    budget = min(128, min((len(v) for v in bags.values())))
    selected_indices = {}
    for key, values in bags.items():
        local_rng = np.random.default_rng(int(hashlib.sha256((key + '|analysis').encode()).hexdigest()[:8], 16))
        selected_indices[key] = local_rng.choice(len(values), budget, replace=False)
        bags[key] = values[selected_indices[key]]
    np.savez_compressed(OUT / 'analysis_cell_indices.npz', **selected_indices)
    metadata = {k: metadata[k] for k in bags}
    development = [k for k, r in metadata.items() if r['Metadata_Plate'] == 'UL001641' and r['role'] == 'reference']
    train = np.concatenate([bags[k] for k in development])
    center = train.mean(axis=0)
    scale = np.sqrt(train.var(axis=0).mean())
    bags = {k: (v - center) / scale for k, v in bags.items()}
    train = (train - center) / scale
    bandwidth = float(np.median(pdist(train[::4])))
    rng = np.random.default_rng(SEED)
    directions = rng.normal(size=(384, 64))
    directions /= np.linalg.norm(directions, axis=0)
    prototypes = train[rng.choice(len(train), 16, replace=False)]
    radius = float(np.median(np.quantile(cdist(prototypes, train), 0.1, axis=1)))
    np.savez_compressed(OUT / 'preprocessing.npz', center=center, scale=scale, directions=directions, prototypes=prototypes, radius=radius, bandwidth=bandwidth)
    (OUT / 'fit_provenance.json').write_text(json.dumps(dict(fit_wells=development, bandwidth=bandwidth, radius=radius, scale=float(scale), cell_budget=budget, analysis_wells=len(bags), excluded_under_128=[]), indent=2))

    def score(x, reference):
        return dict(mean=float(np.sum((x.mean(axis=0) - reference.mean(axis=0)) ** 2)), mmd=distribution_scores(x, reference, [bandwidth])['mmd_exact'], sliced_w2=float(np.mean((np.sort(x @ directions, axis=0) - np.sort(reference @ directions, axis=0)) ** 2)), local_abundance=float(np.max(np.abs((cdist(x, prototypes) <= radius).mean(axis=0) - (cdist(reference, prototypes) <= radius).mean(axis=0)))))

    def reference(keys):
        assert keys
        pooled = np.concatenate([bags[k] for k in sorted(keys)])
        indices = np.linspace(0, len(pooled) - 1, budget).astype(int)
        return pooled[indices]
    refs = {}
    for plate in sorted({r['Metadata_Plate'] for r in metadata.values()}):
        for side in ['left', 'right']:
            keys = [k for k, r in metadata.items() if r['Metadata_Plate'] == plate and r['side'] == side and (r['role'] == 'reference')]
            refs[plate, side] = (keys, reference(keys))
    results = []
    sensitivity = []
    for key, x in bags.items():
        r = metadata[key]
        if r['role'] == 'reference':
            continue
        plate, side = (r['Metadata_Plate'], r['side'])
        keys, ref = refs[plate, side]
        base = dict(well_id=key, plate=plate, batch=r['Metadata_Batch'], well=r['Metadata_Well'], side=side, role=r['role'], treatment=r['treatment'], cells=budget)
        results.append(base | score(x, ref))
        if r['role'] not in ['calibration']:
            variants = {'side': keys}
            for column in sorted({int(metadata[k]['Metadata_Well'][-2:]) for k in keys}):
                variants[f'column_{column}'] = [k for k in keys if int(metadata[k]['Metadata_Well'][-2:]) == column]
            variants['nearest_row'] = sorted(keys, key=lambda k: abs(row_number(metadata[k]['Metadata_Well']) - row_number(r['Metadata_Well'])))[:4]
            for variant, chosen in variants.items():
                sensitivity.append(base | dict(variant=variant, reference_wells=';'.join(chosen)) | score(x, reference(chosen)))
    for record in results:
        calibration = [r for r in results if r['plate'] == record['plate'] and r['side'] == record['side'] and (r['role'] == 'calibration')]
        for method in METHODS:
            threshold = float(np.quantile([r[method] for r in calibration], 0.95))
            record[method + '_threshold'] = threshold
            record[method + '_flag'] = int(record[method] > threshold)
    write_csv(OUT / 'well_scores.csv', results)
    write_csv(OUT / 'spatial_sensitivity.csv', sensitivity)
    row_matched = []
    for key, r in metadata.items():
        if r['treatment'] != 'AMG900':
            continue
        neighbors = [k for k, a in metadata.items() if a['Metadata_Plate'] == r['Metadata_Plate'] and a['treatment'] == 'DMSO' and (row_number(a['Metadata_Well']) == row_number(r['Metadata_Well'])) and (abs(int(a['Metadata_Well'][-2:]) - int(r['Metadata_Well'][-2:])) == 1)]
        if neighbors:
            row_matched.append(dict(plate=r['Metadata_Plate'], treated_well=key, comparison='treated_vs_adjacent_same_row', query_well=key, control_wells=';'.join(neighbors), control_count=len(neighbors), **score(bags[key], reference(neighbors))))
        if len(neighbors) == 2:
            row_matched.append(dict(plate=r['Metadata_Plate'], treated_well=key, comparison='adjacent_control_vs_control', query_well=neighbors[0], control_wells=neighbors[1], control_count=1, **score(bags[neighbors[0]], bags[neighbors[1]])))
    write_csv(OUT / 'row_matched_comparisons.csv', row_matched)
    means = np.stack([bags[k].mean(axis=0) for k in bags])
    np.savez_compressed(OUT / 'well_representations.npz', well_ids=np.array(list(bags)), embeddings=means)
    summaries = []
    rng = np.random.default_rng(SEED + 1)
    for plate in sorted({r['plate'] for r in results}):
        for method in METHODS:
            treated = [r for r in results if r['plate'] == plate and r['treatment'] == 'AMG900']
            null = [r for r in results if r['plate'] == plate and r['role'] == 'heldout_null']
            if not null:
                null = [r for r in results if r['plate'] == plate and 'held' in r['role']]
            t = np.array([r[method] for r in treated])
            n = np.array([r[method] for r in null])
            treated_mean = float(np.mean([np.mean([r[method] for r in treated if r['side'] == side]) for side in ['left', 'right']]))
            control_mean = float(np.mean([np.mean([r[method] for r in null if r['side'] == side]) for side in ['left', 'right']]))
            differences = []
            for _ in range(2000):
                sides = []
                for side in ['left', 'right']:
                    a = np.array([r[method] for r in treated if r['side'] == side])
                    b = np.array([r[method] for r in null if r['side'] == side])
                    sides.append(float(rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()))
                differences.append(np.mean(sides))
            auc = float(np.mean((t[:, None] > n[None, :]) + 0.5 * (t[:, None] == n[None, :])))
            summaries.append(dict(plate=plate, method=method, n_treated=len(t), n_heldout_controls=len(n), treated_mean=treated_mean, control_mean=control_mean, difference=treated_mean - control_mean, ci_low=float(np.quantile(differences, 0.025)), ci_high=float(np.quantile(differences, 0.975)), auc=auc, treated_flags=sum((r[method + '_flag'] for r in treated)), control_flags=sum((r[method + '_flag'] for r in null))))
    write_csv(OUT / 'baseline_comparison.csv', summaries)
    contrasts = {}
    for plate in sorted({r['plate'] for r in results}):
        contrasts[plate] = np.mean([np.mean([bags[k].mean(axis=0) for k, r in metadata.items() if r['Metadata_Plate'] == plate and r['side'] == side and (r['treatment'] == 'AMG900')], axis=0) - np.mean([bags[k].mean(axis=0) for k, r in metadata.items() if r['Metadata_Plate'] == plate and r['side'] == side and ('held' in r['role'])], axis=0) for side in ['left', 'right']], axis=0)
    consistency = []
    for p in contrasts:
        for q in contrasts:
            if p < q:
                consistency.append(dict(plate_a=p, plate_b=q, contrast_cosine=float(contrasts[p] @ contrasts[q] / (np.linalg.norm(contrasts[p]) * np.linalg.norm(contrasts[q]))), distance_a=float(np.linalg.norm(contrasts[p])), distance_b=float(np.linalg.norm(contrasts[q]))))
    write_csv(OUT / 'cross_plate_contrasts.csv', consistency)
    mixtures = []
    weighted_mixtures = []
    for donor, r in metadata.items():
        if r['treatment'] != 'AMG900':
            continue
        backgrounds = sorted((k for k, a in metadata.items() if a['Metadata_Plate'] == r['Metadata_Plate'] and a['side'] == r['side'] and ('held' in a['role'])))
        donors = sorted((k for k, a in metadata.items() if a['Metadata_Plate'] == r['Metadata_Plate'] and a['side'] == r['side'] and (a['treatment'] == 'AMG900')))
        background = backgrounds[donors.index(donor) % len(backgrounds)]
        ref = refs[r['Metadata_Plate'], r['side']][1]
        for fraction in [0, 0.01, 0.05, 0.1, 0.15, 0.25, 1]:
            count = round(budget * fraction)
            x = np.concatenate([bags[background][:budget - count], bags[donor][:count]])
            values = score(x, ref)
            calibration = [a for a in results if a['plate'] == r['Metadata_Plate'] and a['side'] == r['side'] and (a['role'] == 'calibration')]
            mixtures.append(dict(donor_well=donor, background_well=background, plate=r['Metadata_Plate'], batch=r['Metadata_Batch'], side=r['side'], nominal_fraction=fraction, actual_fraction=count / budget, treated_cells=count, total_cells=budget, **values, **{m + '_flag': int(values[m] > np.quantile([a[m] for a in calibration], 0.95)) for m in METHODS}))
            support = np.concatenate([bags[background], bags[donor]])
            mass = np.r_[np.full(budget, (1 - fraction) / budget), np.full(budget, fraction / budget)]
            kernel = np.exp(-cdist(support, support, 'sqeuclidean') / (2 * bandwidth ** 2))
            cross = np.exp(-cdist(ref, support, 'sqeuclidean') / (2 * bandwidth ** 2))
            ref_kernel = np.exp(-cdist(ref, ref, 'sqeuclidean') / (2 * bandwidth ** 2))
            values = dict(mean=float(np.sum((mass @ support - ref.mean(axis=0)) ** 2)), mmd=float(max(0, mass @ kernel @ mass + ref_kernel.mean() - 2 * cross.mean(axis=0) @ mass)), sliced_w2=weighted_wasserstein(ref @ directions, support @ directions, mass), local_abundance=float(np.max(np.abs(mass @ (cdist(support, prototypes) <= radius) - (cdist(ref, prototypes) <= radius).mean(axis=0)))))
            if fraction in [0, 1]:
                expected = score(bags[background] if fraction == 0 else bags[donor], ref)
                np.testing.assert_allclose([values[m] for m in METHODS], [expected[m] for m in METHODS], atol=1e-09, rtol=1e-09)
            weighted_mixtures.append(dict(donor_well=donor, background_well=background, plate=r['Metadata_Plate'], batch=r['Metadata_Batch'], side=r['side'], treated_mass=fraction, background_support_cells=budget, donor_support_cells=budget, construction='Exact weighted empirical mixture; not a literal integer-cell mixture', **values, **{m + '_flag': int(values[m] > np.quantile([a[m] for a in calibration], 0.95)) for m in METHODS}))
    write_csv(OUT / 'exploratory_mixtures.csv', mixtures)
    write_csv(OUT / 'weighted_exploratory_mixtures.csv', weighted_mixtures)
    mixture_summary = []
    for fraction in [0, 0.01, 0.05, 0.1, 0.15, 0.25, 1]:
        for method in METHODS:
            rr = [r for r in mixtures if r['nominal_fraction'] == fraction]
            mixture_summary.append(dict(nominal_fraction=fraction, actual_fraction=rr[0]['actual_fraction'], method=method, flags=sum((r[method + '_flag'] for r in rr)), donor_background_pairs=len(rr), unique_backgrounds=len({r['background_well'] for r in rr}), mean_score=float(np.mean([r[method] for r in rr]))))
    write_csv(OUT / 'mixture_summary.csv', mixture_summary)
    weighted_summary = []
    for fraction in [0, 0.01, 0.05, 0.1, 0.15, 0.25, 1]:
        for method in METHODS:
            rr = [r for r in weighted_mixtures if r['treated_mass'] == fraction]
            weighted_summary.append(dict(treated_mass=fraction, method=method, flags=sum((r[method + '_flag'] for r in rr)), donor_background_pairs=len(rr), unique_backgrounds=len({r['background_well'] for r in rr}), mean_score=float(np.mean([r[method] for r in rr]))))
    write_csv(OUT / 'weighted_mixture_summary.csv', weighted_summary)
    plot(results, summaries, mixture_summary, weighted_summary)

    print('Analysis completed', flush=True)

def plot(results, summaries, mixtures, weighted):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plates = sorted({r['plate'] for r in results})
    for ax, method, name in zip(axes.flat, METHODS, NAMES):
        for i, plate in enumerate(plates):
            for treatment, color, offset in [('DMSO', '#4477aa', -0.15), ('AMG900', '#cc3311', 0.15)]:
                values = [r[method] for r in results if r['plate'] == plate and r['treatment'] == treatment and (treatment == 'AMG900' or 'held' in r['role'])]
                ax.scatter(i + offset + np.linspace(-0.06, 0.06, len(values)), values, s=22, c=color, alpha=0.8, label=treatment if i == 0 else None)
        ax.set(title=name, xticks=range(3), xticklabels=plates, ylabel='Distance from side-local reference')
        ax.legend()
    fig.suptitle(f"Actual AMG900 versus held-out DMSO wells; {results[0]['cells']} cells per well\nDistances are descriptive treatment-associated effects")
    fig.savefig(FIG / 'amg900_vs_dmso.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), constrained_layout=True)
    for ax, method, name in zip(axes, METHODS, NAMES):
        rows = [r for r in summaries if r['method'] == method]
        for i, r in enumerate(rows):
            ax.errorbar(r['difference'], i, xerr=[[max(0, r['difference'] - r['ci_low'])], [max(0, r['ci_high'] - r['difference'])]], fmt='o', color='#4477aa')
        ax.axvline(0, c='gray', ls='--')
        ax.set(title=name, yticks=range(len(rows)), yticklabels=[r['plate'] for r in rows], xlabel='Treated minus held-out mean score')
    fig.suptitle('Across-plate consistency: conditional well-bootstrap 95% intervals\nThree acquisition batches; column confounding persists on every plate')
    fig.savefig(FIG / 'across_plate_robustness.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method, name in zip(METHODS, NAMES):
        rows = [r for r in mixtures if r['method'] == method and r['nominal_fraction'] < 1]
        ax.plot([100 * r['actual_fraction'] for r in rows], [r['flags'] / r['donor_background_pairs'] for r in rows], '-o', label=name)
    ax.set(xlabel='Constructed treated-cell percentage', ylabel='Empirical flagged fraction', ylim=(-0.03, 1.03), title='Exploratory mixtures: repeated donor/background wells\nThresholds are descriptive, without formal error control')
    ax.legend()
    fig.savefig(FIG / 'integer_exploratory_mixtures.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method, name in zip(METHODS, NAMES):
        rr = [r for r in weighted if r['method'] == method and r['treated_mass'] < 1]
        ax.plot([100 * r['treated_mass'] for r in rr], [r['flags'] / r['donor_background_pairs'] for r in rr], '-o', label=name)
    ax.set(xlabel='Exact treated empirical mass (%)', ylabel='Empirical flagged fraction', ylim=(-0.03, 1.03), title='Exploratory weighted mixtures of actual cell embeddings\nReused donor/background wells; no formal error control')
    ax.legend()
    fig.savefig(FIG / 'exploratory_mixtures.png', dpi=180)
    plt.close(fig)
if __name__ == '__main__':
    layout()
    analyze()
