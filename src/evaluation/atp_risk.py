import hashlib
import numpy as np
from scipy.spatial.distance import cdist, pdist

def support_scores(train, query, alpha, neighbors=5):
    train, query = (np.asarray(train, float), np.asarray(query, float))
    if train.ndim != 2 or query.shape[1] != train.shape[1] or alpha <= 0:
        raise ValueError('Invalid support design')
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError('Nonfinite support design')
    gram = train @ train.T + alpha * np.eye(len(train))
    cross = query @ train.T
    leverage = (np.sum(query ** 2, axis=1) - np.sum(cross * np.linalg.solve(gram, cross.T).T, axis=1)) / alpha
    distances = np.sort(cdist(query, train), axis=1)
    nearest = distances[:, :min(neighbors, len(train))].mean(axis=1) / np.sqrt(train.shape[1])
    return (nearest, np.maximum(leverage, 0))

def retained_indices(scores, identifiers, coverage):
    scores = np.asarray(scores, float)
    if not 0 < coverage <= 1 or len(scores) != len(identifiers) or (not np.isfinite(scores).all()):
        raise ValueError('Invalid retention design')
    ties = [hashlib.sha256(('2301|' + key).encode()).hexdigest() for key in identifiers]
    return np.asarray(sorted(range(len(scores)), key=lambda i: (scores[i], ties[i]))[:int(np.ceil(coverage * len(scores)))])

def selective_error(errors, scores, identifiers, coverage):
    errors = np.asarray(errors, float)
    if len(errors) != len(scores) or np.any(errors < 0) or (not np.isfinite(errors).all()):
        raise ValueError('Invalid prediction errors')
    keep = retained_indices(scores, identifiers, coverage)
    return dict(retained=len(keep), actual_coverage=len(keep) / len(errors), mae_nM=float(errors[keep].mean()), retained_indices=keep.tolist())

def leave_study_out_sensitivity(values):
    if len(values) < 3 or not np.isfinite(list(values.values())).all():
        raise ValueError('Invalid study summary')
    reduced = {study: float(np.mean([value for key, value in values.items() if key != study])) for study in sorted(values)}
    return dict(full_mean=float(np.mean(list(values.values()))), omitted_study_means=reduced, range=[min(reduced.values()), max(reduced.values())], interpretation='Descriptive sensitivity over six reused technical studies, not a population CI or new validation split')

def validate_fold(entries, train, test):
    if set(train) & set(test) or not train or (not test):
        raise ValueError('Fold overlap')
    for key in ['study', 'physical_well_id']:
        if {entries[i][key] for i in train} & {entries[i][key] for i in test}:
            raise ValueError('Group overlap')
    if any((entries[i]['treatment'] == 'control' for i in list(train) + list(test))):
        raise ValueError('Control scored')

def kernel_residual_risk(training, queries, residuals, neighbors=5, bandwidth_scale=1.0, identifiers=None):
    training, queries, residuals = (np.asarray(training, float), np.asarray(queries, float), np.asarray(residuals, float))
    if training.ndim != 2 or queries.ndim != 2 or training.shape[1] != queries.shape[1] or (not len(training)):
        raise ValueError('Invalid risk geometry')
    if len(residuals) != len(training) or np.any(residuals < 0) or neighbors < 1 or (bandwidth_scale <= 0):
        raise ValueError('Invalid residual supervision')
    if not all((np.isfinite(value).all() for value in [training, queries, residuals])):
        raise ValueError('Nonfinite risk inputs')
    center, scale = (training.mean(axis=0), np.maximum(training.std(axis=0), 1e-06))
    reference, query = ((training - center) / scale, (queries - center) / scale)
    pairwise = pdist(reference)
    positive = pairwise[pairwise > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    bandwidth *= bandwidth_scale
    distances = cdist(query, reference)
    identifiers = np.asarray(identifiers if identifiers is not None else [str(i) for i in range(len(training))])
    estimates, selections, weights = ([], [], [])
    for values in distances:
        selected = np.lexsort((identifiers, values))[:min(neighbors, len(training))]
        local = values[selected]
        with np.errstate(over='ignore', invalid='ignore'):
            log_weights = -0.5 * ((local - local[0]) / bandwidth) * ((local + local[0]) / bandwidth)
            weight = np.exp(log_weights)
        weight[~np.isfinite(weight)] = 0
        if weight.sum() == 0:
            weight[0] = 1.0
        weight /= weight.sum()
        estimates.append(float(weight @ residuals[selected]))
        selections.append(selected.tolist())
        weights.append(weight.tolist())
    state = dict(center=center.tolist(), scale=scale.tolist(), bandwidth=bandwidth, neighbors=neighbors, bandwidth_scale=bandwidth_scale, selected_indices=selections, weights=weights)
    return (np.asarray(estimates), state)

def source_removal_risks(training, queries, residuals, studies, identifiers, neighbors=5, bandwidth_scale=1.0, excluded_study=None):
    training, queries, residuals = map(lambda x: np.asarray(x, float), [training, queries, residuals])
    studies, identifiers = (np.asarray(studies), np.asarray(identifiers))
    if training.ndim != 2 or queries.ndim != 2 or training.shape[1] != queries.shape[1]:
        raise ValueError('Invalid counterfactual geometry')
    if residuals.shape != (len(training),) or studies.shape != residuals.shape or identifiers.shape != residuals.shape:
        raise ValueError('Invalid counterfactual evidence')
    if len(set(identifiers)) != len(identifiers) or not len(training) or (not len(queries)):
        raise ValueError('Invalid counterfactual identifiers')
    if excluded_study is not None and excluded_study in studies:
        raise ValueError('Heldout source in evidence')
    if not all((np.isfinite(x).all() for x in [training, queries, residuals])) or np.any(residuals < 0):
        raise ValueError('Invalid counterfactual risks')
    if neighbors < 1 or bandwidth_scale <= 0 or (not np.isfinite(bandwidth_scale)):
        raise ValueError('Invalid counterfactual settings')
    columns, audit = ([], [])
    for source in sorted(set(studies)):
        keep = studies != source
        fallback = len(set(studies[keep])) < 2 or int(keep.sum()) < 2
        if fallback:
            value = float(residuals[keep].mean()) if keep.any() else float(residuals.mean())
            scores = np.full(len(queries), value)
            state = {}
        else:
            scores, state = kernel_residual_risk(training[keep], queries, residuals[keep], neighbors, bandwidth_scale, identifiers[keep])
        columns.append(scores)
        audit.append(dict(omitted_source=str(source), remaining_sources=sorted(set(studies[keep].tolist())), residual_identifiers=identifiers[keep].tolist(), fallback=fallback, fallback_rule='Remaining residual mean; original training mean only if no residual remains', scaler_center_sha256=hashlib.sha256(np.asarray(state.get('center', []), '<f8').tobytes()).hexdigest(), scaler_scale_sha256=hashlib.sha256(np.asarray(state.get('scale', []), '<f8').tobytes()).hexdigest(), bandwidth=state.get('bandwidth')))
    return (np.column_stack(columns), audit)

def joint_source_selection(risks, identifiers, retained, time_limit=60.0, absolute_tolerance=0.0001, relative_tolerance=1e-09):
    import time
    import warnings
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix, vstack
    matrix = np.asarray(risks, float)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or (not np.isfinite(matrix).all()) or np.any(matrix < 0):
        raise ValueError('Invalid source risk matrix')
    n, sources = matrix.shape
    if len(identifiers) != n or len(set(identifiers)) != n or (not all((isinstance(i, str) for i in identifiers))):
        raise ValueError('Invalid selection identifiers')
    if isinstance(retained, (bool, np.bool_)) or not isinstance(retained, (int, np.integer)) or (not 1 <= retained <= n):
        raise ValueError('Invalid retention budget')
    if not np.isfinite([time_limit, absolute_tolerance, relative_tolerance]).all() or time_limit <= 0 or min(absolute_tolerance, relative_tolerance) < 0:
        raise ValueError('Invalid solver settings')
    started = time.perf_counter()
    order = np.asarray(sorted(range(n), key=lambda i: (hashlib.sha256(('2301|' + identifiers[i]).encode()).hexdigest(), identifiers[i])))
    a = matrix[order]
    columns = sorted(range(sources), key=lambda j: tuple(a[:, j]))
    a = a[:, columns]
    scale = max(float(a.max()), 1.0)
    normalized = a / scale
    equality = csc_matrix(np.r_[np.ones(n), 0.0][None, :])
    inequalities = csc_matrix(np.column_stack([normalized.T, -retained * np.ones(sources)]))
    constraints = LinearConstraint(vstack([equality, inequalities], format='csc'), np.r_[retained, np.full(sources, -np.inf)], np.r_[retained, np.zeros(sources)])
    lower, upper = (np.zeros(n + 1), np.r_[np.ones(n), np.inf])
    integrality = np.r_[np.ones(n, dtype=int), 0]
    calls, gaps = (0, [])

    def solve(objective):
        nonlocal calls
        remaining = time_limit - (time.perf_counter() - started)
        if remaining <= 0:
            raise RuntimeError('SRSO time limit')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads', category=RuntimeWarning)
            result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints, options=dict(time_limit=remaining, mip_rel_gap=0.0, presolve=True, threads=1, mip_feasibility_tolerance=1e-09, primal_feasibility_tolerance=1e-09))
        calls += 1
        if result.status != 0 or not result.success or result.x is None:
            raise RuntimeError(f'SRSO solver status {result.status}: {result.message}')
        if not np.isfinite(result.x).all() or np.max(np.abs(result.x[:n] - np.rint(result.x[:n]))) > 1e-05:
            raise RuntimeError('Invalid solver incumbent')
        gaps.append(float(getattr(result, 'mip_gap', 0.0) or 0.0))
        return result
    if retained == n:
        selected = np.arange(n)
        optimum = float(matrix.mean(axis=0).max())
        tolerance = absolute_tolerance + relative_tolerance * max(1.0, optimum)
        status = 'analytic_full_coverage'
        primary_gap = 0.0
    else:
        primary = solve(np.r_[np.zeros(n), 1.0])
        optimum = float(primary.fun * scale)
        primary_gap = gaps[0]
        tolerance = absolute_tolerance + relative_tolerance * max(1.0, abs(optimum))
        upper[-1] = (optimum + tolerance * 0.5) / scale
        for i in range(n):
            fixed = int(lower[:i].sum())
            if fixed == retained:
                upper[i:n] = 0.0
                break
            if retained - fixed == n - i:
                lower[i:n] = 1.0
                break
            objective = np.zeros(n + 1)
            objective[i] = -1.0
            tie = solve(objective)
            lower[i] = upper[i] = float(tie.x[i] >= 0.5)
        selected = order[np.flatnonzero(lower[:n] > 0.5)]
        status = 'optimal_with_tolerance_lexicographic_tie'
    selected = np.asarray(sorted(selected.tolist()))
    actual = float(matrix[selected].mean(axis=0).max())
    if len(selected) != retained or actual > optimum + tolerance + 1e-09:
        raise RuntimeError('SRSO solution verification')
    return dict(retained_indices=selected.tolist(), objective_nM=actual, primary_optimum_nM=optimum, tie_tolerance_nM=tolerance, status=status, primary_mip_gap=primary_gap, maximum_mip_gap=max(gaps, default=0.0), solver_calls=calls, seconds=time.perf_counter() - started, exact_primary=True, approximate_fallback=False, tie_rule='Prefer inclusion in canonical SHA256(2301|well_id) order within declared primary tolerance')

def study_balanced_risk(training, queries, residuals, studies, neighbors=5, bandwidth_scale=1.0, modulation=None, uniform=False, identifiers=None):
    training, queries, residuals, studies = map(np.asarray, [training, queries, residuals, studies])
    if training.ndim != 2 or queries.ndim != 2 or queries.shape[1] != training.shape[1] or (len(residuals) != len(training)):
        raise ValueError('Invalid study geometry')
    if neighbors < 1 or bandwidth_scale <= 0 or np.any(residuals < 0) or (not all((np.isfinite(x).all() for x in [training, queries, residuals]))):
        raise ValueError('Invalid study residuals')
    if len(studies) != len(training) or len(set(studies)) == 0:
        raise ValueError('Missing source studies')
    identifiers = np.asarray(identifiers if identifiers is not None else [str(i) for i in range(len(training))])
    center, scale = (training.mean(axis=0), np.maximum(training.std(axis=0), 1e-06))
    a, b = ((training - center) / scale, (queries - center) / scale)
    pairwise = pdist(a)
    positive = pairwise[pairwise > 0]
    h = (float(np.median(positive)) if len(positive) else 1.0) * bandwidth_scale
    domains = sorted(set(studies))
    local_risk, support, global_mean, local_weights = ([], [], [], [])
    for study in domains:
        selected = np.flatnonzero(studies == study)
        distances = cdist(b, a[selected])
        inside = cdist(a[selected], a[selected])
        np.fill_diagonal(inside, np.inf)
        if len(selected) > 1:
            native = np.sort(inside, axis=1)[:, :min(neighbors, len(selected) - 1)].mean(axis=1)
            positive = native[(native > 0) & np.isfinite(native)]
            support_scale = float(np.median(positive)) if len(positive) else 1.0
        else:
            support_scale = 1.0
        risks, supports, weights = ([], [], [])
        for values in distances:
            choice = np.lexsort((identifiers[selected], values))[:min(neighbors, len(selected))]
            d = values[choice]
            weight = np.ones(len(choice)) if uniform else np.exp(-0.5 * ((d - d[0]) / h) * ((d + d[0]) / h))
            weight[~np.isfinite(weight)] = 0
            if not weight.sum():
                weight[0] = 1.0
            weight /= weight.sum()
            risks.append(float(weight @ residuals[selected[choice]]))
            supports.append(float(d.mean() / support_scale))
            row = np.zeros(len(training))
            row[selected[choice]] = weight
            weights.append(row)
        local_risk.append(risks)
        support.append(supports)
        local_weights.append(weights)
        global_mean.append(float(residuals[selected].mean()))
    local_risk, support = (np.asarray(local_risk).T, np.asarray(support).T)
    study_weights = np.full_like(local_risk, 1 / len(domains))
    gate = np.ones(len(queries))
    if modulation is not None:
        if modulation <= 0:
            raise ValueError('Invalid support temperature')
        with np.errstate(over='ignore'):
            log = np.clip(-0.5 * (support / modulation) ** 2, -745, 0)
        gate = np.exp(np.maximum(log.max(axis=1), -745))
        relative = np.exp(log - log.max(axis=1, keepdims=True))
        study_weights = relative / relative.sum(axis=1, keepdims=True)
    fallback = float(np.mean(global_mean))
    predicted = gate * np.sum(study_weights * local_risk, axis=1) + (1 - gate) * fallback
    return (predicted, dict(study_ids=domains, study_weights=study_weights.tolist(), support_ratios=support.tolist(), local_risks=local_risk.tolist(), support_gate=gate.tolist(), fallback_nM=fallback, maximum_source_share=study_weights.max(axis=1).tolist()))

def fit_study_risk_family(folds, training, queries, residuals, studies, identifiers):
    candidates = [(k, h) for k in [5, 15] for h in [0.5, 1.0, 2.0]]
    predictions, settings, diagnostics = ({}, {}, {})
    for method in ['balanced_kernel', 'domain_aware']:
        choices = []
        for k, h in candidates:
            for temperature in [0.5, 2.0] if method == 'domain_aware' else [None]:
                losses = []
                for fold in folds:
                    value, _ = study_balanced_risk(fold['training'], fold['queries'], fold['residuals'], fold['studies'], k, h, temperature, identifiers=fold['identifiers'])
                    losses.append(float(np.mean(np.abs(value - fold['validation_errors']))))
                choices.append(dict(neighbors=k, bandwidth_scale=h, modulation=temperature, validation_risk_MAE_nM=float(np.mean(losses))))
        selected = min(choices, key=lambda r: r['validation_risk_MAE_nM'])
        predictions[method], diagnostics[method] = study_balanced_risk(training, queries, residuals, studies, selected['neighbors'], selected['bandwidth_scale'], selected['modulation'], identifiers=identifiers)
        settings[method] = dict(selected=selected, grid_scores=choices)
    chosen = settings['balanced_kernel']['selected']
    predictions['balanced_uniform'], diagnostics['balanced_uniform'] = study_balanced_risk(training, queries, residuals, studies, chosen['neighbors'], chosen['bandwidth_scale'], uniform=True, identifiers=identifiers)
    predictions['unbalanced_matched'], _ = kernel_residual_risk(training, queries, residuals, chosen['neighbors'] * len(set(studies)), chosen['bandwidth_scale'], identifiers)
    return (predictions, settings, diagnostics)

def cross_fitted_residuals(labels, groups, indices, predictor):
    labels, groups, indices = (np.asarray(labels), np.asarray(groups), np.asarray(indices))
    if len(set(groups[indices])) < 3:
        raise ValueError('Insufficient residual groups')
    values = np.empty(len(indices))
    provenance = []
    for group in sorted(set(groups[indices])):
        mask = groups[indices] == group
        training, validation = (indices[~mask], indices[mask])
        estimate, state = predictor(training, validation)
        values[mask] = np.abs(estimate - labels[validation])
        provenance.append(dict(training_indices=training.tolist(), residual_indices=validation.tolist(), **state))
    return (values, provenance)

def select_risk_settings(folds, candidates=((5, 0.5), (5, 1.0), (5, 2.0), (15, 0.5), (15, 1.0), (15, 2.0))):
    if not folds:
        raise ValueError('Missing risk validation')
    scores = []
    for neighbors, multiplier in candidates:
        errors = []
        for fold in folds:
            estimate, _ = kernel_residual_risk(fold['training'], fold['queries'], fold['residuals'], neighbors, multiplier, fold.get('identifiers'))
            errors.append(float(np.mean(np.abs(estimate - fold['validation_errors']))))
        scores.append(dict(neighbors=neighbors, bandwidth_scale=multiplier, macro_risk_MAE_nM=float(np.mean(errors))))
    selected = min(range(len(scores)), key=lambda i: (scores[i]['macro_risk_MAE_nM'], i))
    return (scores[selected], scores)

def audit_residual_fold(fold, entries):
    training, test = (set(fold['training_indices']), set(fold['test_indices']))
    validate_fold(entries, list(training), list(test))

    def audit_pool(fits, allowed):
        covered = []
        for fit in fits:
            source, query = (set(fit['training_indices']), set(fit['residual_indices']))
            if source | query != allowed or source & query or set(fit['tuning_indices']) != source:
                raise ValueError('Residual fit leakage')
            validate_fold(entries, list(source), list(query))
            covered.extend(query)
        if set(covered) != allowed or len(covered) != len(allowed):
            raise ValueError('Residual coverage mismatch')
    audit_pool(fold['OOF_fits'], training)
    validation_coverage = []
    for meta in fold['risk_validation']:
        source, query = (set(meta['training_indices']), set(meta['validation_indices']))
        if source | query != training or source & query:
            raise ValueError('Risk validation leakage')
        validate_fold(entries, list(source), list(query))
        audit_pool(meta['residual_fits'], source)
        validation_coverage.extend(query)
    if set(validation_coverage) != training or len(validation_coverage) != len(training):
        raise ValueError('Risk validation coverage mismatch')
    if set(fold['final_predictor']['tuning_indices']) != training:
        raise ValueError('Final tuning leakage')
    return True

def export_risk_figure(figure, directory, name):
    for extension in ['pdf', 'png']:
        path = directory / (name + '.' + extension)
        if path.exists():
            raise FileExistsError(path)
        figure.savefig(path, dpi=300, bbox_inches='tight')
