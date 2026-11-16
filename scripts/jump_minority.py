import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/cache/matplotlib'))
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.real_replication import weighted_wasserstein
OLD = ROOT / 'outputs/results/jump_experiment_v1'
OUT = ROOT / 'outputs/results/jump_minority_signal_v1'
FIG = ROOT / 'outputs/figures/jump_minority_signal_v1'
CONFIG = ROOT / 'configs/jump_minority_signal_v1.json'
SEED = 126001
METHODS = ['mean', 'mmd', 'sliced_w2', 'local_abundance', 'region_primary', 'region_max8']
FRACTIONS = [0, 0.01, 0.05, 0.1, 0.15, 0.25, 1]

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

def priority(key):
    return hashlib.sha256(f'{SEED}|{key}'.encode()).hexdigest()

def rng_for(key):
    return np.random.default_rng(int(priority(key)[:16], 16))

def metadata():
    return {r['Metadata_Plate'] + ':' + r['Metadata_Well']: r for r in read_csv(OLD / 'wells.csv') if r['status'] == 'ok'}

def load_bags(keys, meta, parameters):
    bags = {}
    for key in keys:
        r = meta[key]
        values = np.load(OLD / 'cells' / f"{r['Metadata_Plate']}_{r['Metadata_Well']}.npz")['embeddings'].astype(float)
        bags[key] = (values - parameters['center']) / parameters['scale']
    return bags

def auc(a, b):
    a, b = (np.asarray(a), np.asarray(b))
    return float(np.mean((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])))

def interval(a, b, key):
    rng = rng_for(key)
    draws = np.mean(rng.choice(a, (2000, len(a))), axis=1) - np.mean(rng.choice(b, (2000, len(b))), axis=1)
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))

def discover():
    OUT.mkdir(exist_ok=False, parents=True)
    FIG.mkdir(exist_ok=False, parents=True)
    meta = metadata()
    treated = []
    for side in ['left', 'right']:
        keys = sorted([k for k, r in meta.items() if r['Metadata_Plate'] == 'UL001641' and r['side'] == side and (r['treatment'] == 'AMG900')], key=priority)
        treated.extend(keys[:2])
    controls = [k for k, r in meta.items() if r['Metadata_Plate'] == 'UL001641' and r['role'] == 'reference']
    evaluation = [k for k, r in meta.items() if r['treatment'] == 'AMG900' and k not in treated]
    split = dict(discovery_treated=treated, discovery_controls=controls, confirmation_treated=evaluation, confirmation_controls=[k for k, r in meta.items() if r['role'] == 'heldout_control'], config=json.loads(CONFIG.read_text()))
    assert not set(treated + controls) & set(split['confirmation_treated'] + split['confirmation_controls'])
    (OUT / 'split.json').write_text(json.dumps(split, indent=2))
    parameters = np.load(OLD / 'preprocessing.npz')
    bags = load_bags(treated + controls, meta, parameters)
    reference = np.concatenate([bags[k][rng_for(k).choice(len(bags[k]), 32, replace=False)] for k in controls])
    support = np.concatenate([bags[k][rng_for(k + 'support').choice(len(bags[k]), 32, replace=False)] for k in treated + controls])
    centers = []
    radii = []
    anchors = []
    quantiles = []
    for key in treated:
        for index in rng_for(key + 'anchors').choice(len(bags[key]), 8, replace=False):
            center = bags[key][index]
            distances = cdist(center[None], reference)[0]
            for q in [0.05, 0.1, 0.2]:
                centers.append(center)
                radii.append(np.quantile(distances, q))
                anchors.append(key)
                quantiles.append(q)
    centers = np.asarray(centers)
    radii = np.asarray(radii)
    occupancy = {k: (cdist(bags[k], centers) <= radii).mean(axis=0) for k in bags}
    control = np.stack([occupancy[k] for k in controls])
    treatment = np.stack([occupancy[k] for k in treated])
    sd = np.maximum(control.std(axis=0, ddof=1), 0.02)
    scores = (treatment.mean(axis=0) - control.mean(axis=0)) / sd
    members = cdist(support, centers) <= radii
    selected = []
    for index in np.argsort(-scores, kind='stable'):
        if scores[index] <= 0:
            continue
        if all((np.sum(members[:, index] & members[:, j]) / max(1, np.sum(members[:, index] | members[:, j])) < 0.8 for j in selected)):
            selected.append(int(index))
        if len(selected) == 8:
            break
    if not selected:
        selected = [int(np.argmax(scores))]
    primary = selected[0]
    records = []
    for i in range(len(centers)):
        records.append(dict(region=i, anchor_well=anchors[i], radius_quantile=quantiles[i], radius=float(radii[i]), discovery_treated_wells=len(treated), discovery_control_wells=len(controls), treated_occupancy=float(treatment[:, i].mean()), control_occupancy=float(control[:, i].mean()), control_sd=float(control[:, i].std(ddof=1)), control_min=float(control[:, i].min()), control_max=float(control[:, i].max()), difference=float(treatment[:, i].mean() - control[:, i].mean()), enrichment_ratio=float((treatment[:, i].mean() + 0.0001) / (control[:, i].mean() + 0.0001)), ranking_score=float(scores[i]), selected=int(i in selected), primary=int(i == primary)))
    write_csv(OUT / 'discovery_regions.csv', records)
    stability = []
    for heldout in treated:
        retained = [k for k in treated if k != heldout]
        local = (np.mean([occupancy[k] for k in retained], axis=0) - control.mean(axis=0)) / sd
        local[np.array(anchors) == heldout] = -np.inf
        index = int(np.argmax(local))
        jaccard = float(np.sum(members[:, index] & members[:, primary]) / max(1, np.sum(members[:, index] | members[:, primary])))
        stability.append(dict(heldout_discovery_well=heldout, selected_region=index, primary_membership_jaccard=jaccard, heldout_occupancy=float(occupancy[heldout][index]), control_mean=float(control[:, index].mean()), heldout_minus_control=float(occupancy[heldout][index] - control[:, index].mean())))
    write_csv(OUT / 'discovery_stability.csv', stability)
    np.savez_compressed(OUT / 'regions.npz', centers=centers, radii=radii, selected=np.array(selected), primary=primary, discovery_control_sd=sd)
    frozen = dict(primary=primary, selected=selected, discovery_treated=treated, discovery_controls=controls, confirmation_treated=evaluation, regions_sha256=hashlib.sha256((OUT / 'regions.npz').read_bytes()).hexdigest(), split_sha256=hashlib.sha256((OUT / 'split.json').read_bytes()).hexdigest(), config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(), note='Written before loading confirmation embeddings in this experiment; earlier global outcomes were already known.')
    (OUT / 'frozen_discovery.json').write_text(json.dumps(frozen, indent=2))
    print('Discovery frozen', flush=True)

class Scores:

    def __init__(self, support, reference, parameters, regions):
        self.x = support
        self.ref = reference
        self.ref_mean = reference.mean(axis=0)
        self.projection = support @ parameters['directions']
        self.ref_projection = reference @ parameters['directions']
        bandwidth = float(parameters['bandwidth'])
        self.kernel = np.exp(-cdist(support, support, 'sqeuclidean') / (2 * bandwidth ** 2))
        self.cross = np.exp(-cdist(support, reference, 'sqeuclidean') / (2 * bandwidth ** 2)).mean(axis=1)
        self.ref_kernel = float(np.exp(-cdist(reference, reference, 'sqeuclidean') / (2 * bandwidth ** 2)).mean())
        self.old_members = cdist(support, parameters['prototypes']) <= parameters['radius']
        self.old_reference = (cdist(reference, parameters['prototypes']) <= parameters['radius']).mean(axis=0)
        chosen = regions['selected'].astype(int)
        self.members = cdist(support, regions['centers'][chosen]) <= regions['radii'][chosen]
        self.reference_members = (cdist(reference, regions['centers'][chosen]) <= regions['radii'][chosen]).mean(axis=0)
        self.region_sd = regions['discovery_control_sd'][chosen]

    def evaluate(self, indices=None, mass=None):
        if indices is not None:
            x = self.x[indices]
            mean = x.mean(axis=0)
            mmd = float(self.kernel[np.ix_(indices, indices)].mean() + self.ref_kernel - 2 * self.cross[indices].mean())
            projected = self.projection[indices]
            if len(indices) == len(self.ref):
                sw = float(np.mean((np.sort(projected, axis=0) - np.sort(self.ref_projection, axis=0)) ** 2))
            else:
                sw = weighted_wasserstein(self.ref_projection, projected, np.ones(len(indices)))
            old = self.old_members[indices].mean(axis=0) - self.old_reference
            local = self.members[indices].mean(axis=0) - self.reference_members
        else:
            assert np.isclose(mass.sum(), 1) and np.all(mass >= 0)
            mean = mass @ self.x
            mmd = float(mass @ self.kernel @ mass + self.ref_kernel - 2 * mass @ self.cross)
            sw = weighted_wasserstein(self.ref_projection, self.projection, mass)
            old = mass @ self.old_members - self.old_reference
            local = mass @ self.members - self.reference_members
        return dict(mean=float(np.sum((mean - self.ref_mean) ** 2)), mmd=max(0, mmd), sliced_w2=sw, local_abundance=float(np.max(np.abs(old))), region_primary=float(local[0]), region_max8=float(np.max(local / self.region_sd)))

def evaluate():
    frozen = json.loads((OUT / 'frozen_discovery.json').read_text())
    assert hashlib.sha256((OUT / 'regions.npz').read_bytes()).hexdigest() == frozen['regions_sha256']
    meta = metadata()
    parameters = np.load(OLD / 'preprocessing.npz')
    regions = np.load(OUT / 'regions.npz')
    bags = load_bags(list(meta), meta, parameters)
    candidates = regions['centers']
    radii = regions['radii']
    selected = regions['selected'].astype(int)
    occupancy = {k: (cdist(x, candidates) <= radii).mean(axis=0) for k, x in bags.items()}
    references = {}
    shifts = {}
    for plate in sorted({r['Metadata_Plate'] for r in meta.values()}):
        for side in ['left', 'right']:
            keys = sorted([k for k, r in meta.items() if r['Metadata_Plate'] == plate and r['side'] == side and (r['role'] == 'reference')], key=priority)
            references[plate, side] = keys
    for (plate, side), keys in references.items():
        shifts[plate, side] = np.mean([bags[k].mean(axis=0) for k in keys], axis=0) - np.mean([bags[k].mean(axis=0) for k in references['UL001641', side]], axis=0)
    aligned = {k: (cdist(x - shifts[meta[k]['Metadata_Plate'], meta[k]['side']], candidates) <= radii).mean(axis=0) for k, x in bags.items()}
    summaries = []
    occupancies = []
    for k, r in meta.items():
        for region in selected:
            occupancies.append(dict(well_id=k, plate=r['Metadata_Plate'], batch=r['Metadata_Batch'], side=r['side'], role=r['role'], treatment=r['treatment'], discovery=int(k in frozen['discovery_treated'] + frozen['discovery_controls']), region=int(region), cells=len(bags[k]), occupancy=float(occupancy[k][region]), aligned_occupancy=float(aligned[k][region])))
    write_csv(OUT / 'well_region_occupancy.csv', occupancies)
    for plate in sorted({r['Metadata_Plate'] for r in meta.values()}):
        treated = [k for k in frozen['confirmation_treated'] if meta[k]['Metadata_Plate'] == plate]
        controls = [k for k, r in meta.items() if r['Metadata_Plate'] == plate and r['role'] == 'heldout_control']
        for region in selected:
            for alignment, values in [('raw', occupancy), ('reference_centered', aligned)]:
                a = np.array([values[k][region] for k in treated])
                b = np.array([values[k][region] for k in controls])
                lo, hi = interval(a, b, f'{plate}|{region}|{alignment}')
                summaries.append(dict(plate=plate, region=int(region), primary=int(region == selected[0]), alignment=alignment, treated_wells=len(a), control_wells=len(b), treated_cells=sum((len(bags[k]) for k in treated)), control_cells=sum((len(bags[k]) for k in controls)), treated_occupancy=float(a.mean()), control_occupancy=float(b.mean()), difference=float(a.mean() - b.mean()), ci_low=lo, ci_high=hi, control_sd=float(b.std(ddof=1)), enrichment_ratio=float((a.mean() + 0.0001) / (b.mean() + 0.0001)), auc=auc(a, b)))
    write_csv(OUT / 'plate_confirmation.csv', summaries)
    pairing = []
    for (plate, side), keys in references.items():
        donors = sorted([k for k in frozen['confirmation_treated'] if meta[k]['Metadata_Plate'] == plate and meta[k]['side'] == side], key=priority)
        backgrounds = sorted([k for k, r in meta.items() if r['Metadata_Plate'] == plate and r['side'] == side and (r['role'] == 'heldout_control') and (len(bags[k]) >= 100)], key=priority)
        assert len(backgrounds) >= len(donors)
        pairing.extend((dict(donor=k, background=backgrounds[i], plate=plate, side=side, donor_cells=len(bags[k]), background_cells=len(bags[backgrounds[i]])) for i, k in enumerate(donors)))
    assert len({r['background'] for r in pairing}) == len(pairing)
    write_csv(OUT / 'mixture_pairs.csv', pairing)
    native = []
    mixtures = []
    critical = []
    sampling = []
    lineage = []
    for budget in [32, 100]:
        ref_bags = {}
        for group, keys in references.items():
            per = [budget // len(keys) + (i < budget % len(keys)) for i in range(len(keys))]
            ref_bags[group] = np.concatenate([bags[k][rng_for(k + 'ref').permutation(len(bags[k]))[:n]] for k, n in zip(keys, per)])
        thresholds = {}
        for (plate, side), keys in references.items():
            group = (plate, side)
            ref = ref_bags[group]
            for k, r in meta.items():
                if (r['Metadata_Plate'], r['side']) != group or r['role'] not in ['calibration', 'heldout_control'] or len(bags[k]) < 100:
                    continue
                indices = rng_for(k + 'native').permutation(len(bags[k]))[:budget]
                values = Scores(bags[k][indices], ref, parameters, regions).evaluate(indices=np.arange(budget))
                native.append(dict(well_id=k, plate=plate, side=side, role=r['role'], budget=budget, **values))
                lineage.append(dict(well_id=k, budget=budget, purpose='native', indices=';'.join(map(str, indices))))
            calibration = [r for r in native if r['plate'] == plate and r['side'] == side and (r['budget'] == budget) and (r['role'] == 'calibration')]
            assert len(calibration) >= 2
            thresholds[group] = {m: float(np.quantile([r[m] for r in calibration], 0.95)) for m in METHODS}
        for pair in pairing:
            donor, background = (pair['donor'], pair['background'])
            group = (pair['plate'], pair['side'])
            ref = ref_bags[group]
            a = bags[donor]
            b = bags[background]
            for construction in ['weighted', 'integer']:
                repeats = 1 if construction == 'weighted' else 20
                for repeat in range(repeats):
                    da = rng_for(f'{donor}|{repeat}|donor').permutation(len(a))
                    db = rng_for(f'{background}|{repeat}|background').permutation(len(b))
                    aa = a[da[:min(budget, len(a))]]
                    bb = b[db[:budget]]
                    support = np.concatenate([bb, aa])
                    calculator = Scores(support, ref, parameters, regions)
                    for fraction in FRACTIONS:
                        count = round(budget * fraction)
                        if construction == 'integer' and count > len(aa):
                            continue
                        if construction == 'weighted':
                            mass = np.r_[np.full(len(bb), (1 - fraction) / len(bb)), np.full(len(aa), fraction / len(aa))]
                            values = calculator.evaluate(mass=mass)
                            actual = fraction
                        else:
                            indices = np.r_[np.arange(budget - count), len(bb) + np.arange(count)]
                            values = calculator.evaluate(indices=indices)
                            actual = count / budget
                        row = dict(donor=donor, background=background, plate=pair['plate'], side=pair['side'], budget=budget, construction=construction, repeat=repeat, nominal_fraction=fraction, actual_fraction=actual, donor_support=len(aa), background_support=len(bb), treated_integer_cells=count if construction == 'integer' else '', **values)
                        row.update({m + '_flag': int(values[m] > thresholds[group][m]) for m in METHODS})
                        mixtures.append(row)
                        if construction == 'weighted' and fraction == 0:
                            base = float(np.mean(calculator.members[:len(bb), 0]))
                            response = float(np.mean(calculator.members[len(bb):, 0]))
                            reference = float(calculator.reference_members[0])
                            delta = response - base
                            mass_needed = (thresholds[group]['region_primary'] + reference - base) / delta if delta > 0 else None
                            critical.append(dict(donor=donor, background=background, plate=pair['plate'], side=pair['side'], budget=budget, background_occupancy=base, donor_occupancy=response, occupancy_delta=delta, reference_occupancy=reference, threshold=thresholds[group]['region_primary'], critical_fraction=mass_needed if mass_needed is not None else '', positive_donor_difference=int(delta > 0)))
                    if repeat == 0:
                        lineage.append(dict(well_id=donor, budget=budget, purpose='mixture_donor', indices=';'.join(map(str, da[:len(aa)]))))
                        lineage.append(dict(well_id=background, budget=budget, purpose='mixture_background', indices=';'.join(map(str, db[:len(bb)]))))
        print('Budget evaluated', budget, flush=True)
    write_csv(OUT / 'native_scores.csv', native)
    write_csv(OUT / 'mixture_scores.csv', mixtures)
    write_csv(OUT / 'critical_fractions.csv', critical)
    write_csv(OUT / 'cell_selection.csv', lineage)
    summarize(native, mixtures, summaries, meta)
    diagnostics(bags, meta, parameters, regions, frozen, occupancy, aligned, shifts, native, mixtures, critical, pairing)
    report()
    print('Evaluation completed', flush=True)

def summarize(native, mixtures, confirmation, meta):
    summaries = []
    nulls = []
    thresholds = []
    for budget in [32, 100]:
        for method in METHODS:
            for plate in sorted({r['plate'] for r in native}):
                for side in ['left', 'right']:
                    cal = [r[method] for r in native if r['budget'] == budget and r['plate'] == plate and (r['side'] == side) and (r['role'] == 'calibration')]
                    for q in [0.8, 0.9, 0.95]:
                        thresholds.append(dict(budget=budget, method=method, plate=plate, side=side, quantile=q, threshold=float(np.quantile(cal, q)), calibration_wells=len(cal)))
            for q in [0.8, 0.9, 0.95]:

                def threshold(r):
                    return next((t['threshold'] for t in thresholds if t['budget'] == budget and t['method'] == method and (t['plate'] == r['plate']) and (t['side'] == r['side']) and (t['quantile'] == q)))
                null = [r for r in native if r['budget'] == budget and r['role'] == 'heldout_control']
                nulls.append(dict(budget=budget, method=method, quantile=q, flags=sum((r[method] > threshold(r) for r in null)), control_wells=len(null)))
                for construction in ['weighted', 'integer']:
                    for fraction in FRACTIONS:
                        rr = [r for r in mixtures if r['budget'] == budget and r['construction'] == construction and (r['nominal_fraction'] == fraction)]
                        if not rr:
                            continue
                        pairs = sorted({r['donor'] for r in rr})
                        rates = {k: float(np.mean([r[method] > threshold(r) for r in rr if r['donor'] == k])) for k in pairs}
                        rng = rng_for(f'{budget}|{method}|{q}|{construction}|{fraction}')
                        draws = np.zeros(2000)
                        for plate in sorted({r['plate'] for r in rr}):
                            keys = sorted({r['donor'] for r in rr if r['plate'] == plate})
                            draws += np.mean(rng.choice([rates[k] for k in keys], (2000, len(keys))), axis=1) * len(keys) / len(pairs)
                        first = [r for r in rr if r['repeat'] == 0]
                        repeats = sorted({r['repeat'] for r in rr})
                        repeat_rates = [np.mean([r[method] > threshold(r) for r in rr if r['repeat'] == repeat]) for repeat in repeats]
                        summaries.append(dict(budget=budget, method=method, quantile=q, construction=construction, nominal_fraction=fraction, actual_fraction=rr[0]['actual_fraction'], donor_wells=len(pairs), background_wells=len({r['background'] for r in rr}), first_flags=sum((r[method] > threshold(r) for r in first)), first_pairs=len(first), mean_flag_rate=float(np.mean(list(rates.values()))), ci_low=float(np.quantile(draws, 0.025)), ci_high=float(np.quantile(draws, 0.975)), cell_sampling_sd=float(np.std(repeat_rates, ddof=1)) if len(repeats) > 1 else 0, repeats=len(repeats)))
    write_csv(OUT / 'mixture_summary.csv', summaries)
    write_csv(OUT / 'control_flags.csv', nulls)
    write_csv(OUT / 'thresholds.csv', thresholds)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    occupied = read_csv(OUT / 'well_region_occupancy.csv')
    primary = int(np.load(OUT / 'regions.npz')['primary'])
    frozen = json.loads((OUT / 'frozen_discovery.json').read_text())
    for ax, plate in zip(axes, sorted({r['plate'] for r in confirmation})):
        for label, color, x in [('DMSO', '#4477aa', 0), ('AMG900', '#cc3311', 1)]:
            rr = [r for r in occupied if r['plate'] == plate and int(r['region']) == primary and (r['well_id'] not in frozen['discovery_controls'] + frozen['discovery_treated']) and (r['treatment'] == label) and (label == 'AMG900' or r['role'] == 'heldout_control')]
            ax.scatter(x + np.linspace(-0.08, 0.08, len(rr)), [float(r['occupancy']) for r in rr], color=color, label=f'{label}: {len(rr)} wells')
        ax.set(title=plate, xticks=[0, 1], xticklabels=['DMSO', 'AMG900'], ylabel='Frozen primary-region cell fraction', ylim=(-0.02, 1.02))
        ax.legend()
    fig.suptitle('Treatment-associated region enrichment on held-out wells\nRegions frozen on four development treated wells; column confounding persists')
    fig.savefig(FIG / 'minority_enrichment_reproducibility.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (budget, construction) in zip(axes.flat, [(32, 'integer'), (100, 'integer'), (32, 'weighted'), (100, 'weighted')]):
        for method in METHODS:
            rr = [r for r in summaries if r['budget'] == budget and r['construction'] == construction and (r['method'] == method) and (r['quantile'] == 0.95) and (r['nominal_fraction'] < 1)]
            ax.plot([100 * r['actual_fraction'] for r in rr], [r['mean_flag_rate'] for r in rr], '-o', label=method, ms=4)
        ax.set(title=f'{budget} cells; {construction}', xlabel='Realized treated fraction / empirical mass (%)', ylabel='Empirical flagged fraction', ylim=(-0.03, 1.03))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Matched baseline sensitivity, frozen regions and plate/side thresholds\n18 distinct donor/background pairs; integer curves average 20 cell samples')
    fig.savefig(FIG / 'weak_mixture_sensitivity.png', dpi=180)
    plt.close(fig)

def diagnostics(bags, meta, parameters, regions, frozen, occupancy, aligned, shifts, native, mixtures, critical, pairing):
    primary = int(regions['primary'])
    shift_records = []
    for group, shift in shifts.items():
        plate, side = group
        keys = [k for k, r in meta.items() if r['Metadata_Plate'] == plate and r['side'] == side and (r['role'] == 'heldout_control')]
        shift_records.append(dict(plate=plate, side=side, reference_shift_norm=float(np.linalg.norm(shift)), control_wells=len(keys), control_occupancy_mean=float(np.mean([occupancy[k][primary] for k in keys])), control_occupancy_sd=float(np.std([occupancy[k][primary] for k in keys], ddof=1)), aligned_control_occupancy_mean=float(np.mean([aligned[k][primary] for k in keys]))))
    write_csv(OUT / 'plate_shift_diagnostics.csv', shift_records)
    alignment = []
    for group, shift in shifts.items():
        plate, side = group
        cal = [k for k, r in meta.items() if r['Metadata_Plate'] == plate and r['side'] == side and (r['role'] == 'calibration') and (len(bags[k]) >= 100)]
        null = [k for k, r in meta.items() if r['Metadata_Plate'] == plate and r['side'] == side and (r['role'] == 'heldout_control') and (len(bags[k]) >= 100)]
        values = {k: float((cdist(bags[k][rng_for(k + 'native').permutation(len(bags[k]))[:100]] - shift, regions['centers'][primary:primary + 1]) <= regions['radii'][primary]).mean()) for k in cal + null}
        threshold = float(np.quantile([values[k] for k in cal], 0.95))
        pairs = [r for r in pairing if (r['plate'], r['side']) == group]
        flags = 0
        for pair in pairs:
            donor, background = (pair['donor'], pair['background'])
            a = bags[donor][rng_for(f'{donor}|0|donor').permutation(len(bags[donor]))[:100]] - shift
            b = bags[background][rng_for(f'{background}|0|background').permutation(len(bags[background]))[:100]] - shift
            pa = float((cdist(a, regions['centers'][primary:primary + 1]) <= regions['radii'][primary]).mean())
            pb = float((cdist(b, regions['centers'][primary:primary + 1]) <= regions['radii'][primary]).mean())
            flags += 0.25 * pa + 0.75 * pb > threshold
        alignment.append(dict(plate=plate, side=side, calibration_wells=len(cal), heldout_control_wells=len(null), heldout_flags=sum((values[k] > threshold for k in null)), mixture_pairs=len(pairs), weighted_25_flags=int(flags), threshold=threshold))
    write_csv(OUT / 'alignment_sensitivity.csv', alignment)
    paired = []
    summary = read_csv(OUT / 'mixture_summary.csv')
    for construction in ['integer', 'weighted']:
        for fraction in [0.01, 0.05, 0.1, 0.15, 0.25]:
            for baseline in METHODS[:4]:
                a = [r for r in mixtures if r['budget'] == 100 and r['construction'] == construction and (r['nominal_fraction'] == fraction)]
                per_pair = []
                for donor in sorted({r['donor'] for r in a}):
                    rr = [r for r in a if r['donor'] == donor]
                    per_pair.append((donor, rr[0]['plate'], float(np.mean([r['region_primary_flag'] - r[baseline + '_flag'] for r in rr]))))
                rng = rng_for(f'paired|{construction}|{fraction}|{baseline}')
                draws = np.zeros(2000)
                for plate in sorted({r[1] for r in per_pair}):
                    values = [r[2] for r in per_pair if r[1] == plate]
                    draws += np.mean(rng.choice(values, (2000, len(values))), axis=1) * len(values) / len(per_pair)
                paired.append(dict(construction=construction, fraction=fraction, local_method='region_primary', baseline=baseline, pairs=len(per_pair), flag_rate_difference=float(np.mean([r[2] for r in per_pair])), ci_low=float(np.quantile(draws, 0.025)), ci_high=float(np.quantile(draws, 0.975))))
    write_csv(OUT / 'paired_method_differences.csv', paired)
    inputs = [OLD / 'preprocessing.npz', OLD / 'wells.csv', OLD / 'encoder.json'] + list((OLD / 'cells').glob('*.npz'))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT / 'input_hashes.json').write_text(json.dumps(hashes, indent=2))
    (OUT / 'source_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG, Path(__file__)]}, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['discover', 'evaluate'])
    args = parser.parse_args()
    globals()[args.stage]()
