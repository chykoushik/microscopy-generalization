import math
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from evaluation.missing_marker_data import arrays

def masks(groups):
    return [np.array([number >> j & 1 for j in range(groups)], dtype=np.float32) for number in range(1, 2 ** groups)]

def flatten(x, mask):
    m = np.broadcast_to(mask, (len(x), len(mask)))
    return np.concatenate([(x * m[:, :, None]).reshape(len(x), -1), m], axis=1)

class Network(torch.nn.Module):

    def __init__(self, groups, prototype=False, count=False, hidden=40):
        super().__init__()
        self.groups = groups
        self.prototype = prototype
        self.count = count
        if prototype:
            self.channels = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(24, 16), torch.nn.GELU()) for _ in range(groups)])
            self.encoder = torch.nn.Sequential(torch.nn.Linear(groups * 17, 32), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(32, 16))
            self.prototypes = torch.nn.Parameter(torch.randn(8, 16) * 0.1)
        else:
            self.encoder = torch.nn.Sequential(torch.nn.Linear(groups * 25, hidden), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 16), torch.nn.GELU())
            self.head = torch.nn.Linear(16, 3 if count else 1)

    def forward(self, x, mask):
        m = mask.expand(len(x), -1)
        if self.prototype:
            pieces = [self.channels[i](x[:, i]) * m[:, i:i + 1] for i in range(self.groups)]
            z = self.encoder(torch.cat(pieces + [m], dim=1))
            distance = ((z[:, None] - self.prototypes[None]) ** 2).mean(-1)
            q = torch.softmax(-distance / 0.2, dim=1)
            logit = torch.logsumexp(-distance[:, 4:] / 0.2, 1) - torch.logsumexp(-distance[:, :4] / 0.2, 1)
            return (logit, z, q)
        z = self.encoder(torch.cat([(x * m[:, :, None]).flatten(1), m], dim=1))
        return (self.head(z).squeeze(-1), z, None)

def matched_hidden(groups):
    target = sum((p.numel() for p in Network(groups, prototype=True).parameters()))
    return min(range(8, 160), key=lambda h: abs(sum((p.numel() for p in Network(groups, hidden=h).parameters())) - target))

def zinb_loss(parameters, counts):
    mean = torch.nn.functional.softplus(parameters[:, 0]) + 0.001
    shape = torch.nn.functional.softplus(parameters[:, 1]) + 0.001
    zero = parameters[:, 2]
    log_nb = torch.lgamma(counts + shape) - torch.lgamma(shape) - torch.lgamma(counts + 1) + shape * torch.log(shape / (shape + mean)) + counts * torch.log(mean / (shape + mean))
    log_zero = -torch.nn.functional.softplus(-zero)
    log_nonzero = -torch.nn.functional.softplus(zero)
    return -torch.where(counts == 0, torch.logaddexp(log_zero, log_nonzero + log_nb), log_nonzero + log_nb)

class Predictor:

    def fit(self, wells, frame, name, epochs=40, seed=719):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.frame = frame
        self.name = name
        self.groups = frame.groups
        self.fit_ids = [w['id'] for w in wells if w['role'] == 'train']
        self.seed = seed
        self.history = []
        self.calibrator = None
        data = arrays(wells, frame)
        self.prior = float(np.average(data['y'], weights=data['weights']))
        self.count = name in ['count_tail', 'ratio_tail']
        self.prototype = name in ['prototype', 'MMSL', 'no_mask', 'no_consistency', 'no_lattice']
        self.structured = name in ['MMSL', 'no_prototypes', 'no_consistency', 'no_lattice', 'masked_MLP', 'count_tail', 'ratio_tail']
        self.consistency = name in ['MMSL', 'no_mask', 'no_prototypes', 'no_lattice']
        self.lattice = name in ['MMSL', 'no_mask', 'no_consistency']
        self.epochs = epochs
        if name in ['logistic', 'linear', 'gradient_boosting', 'random_forest', 'marker_regression']:
            choices = masks(self.groups)
            x = np.concatenate([flatten(data['x'], m) for m in choices])
            y = np.tile(data['y'], len(choices))
            weight = np.tile(data['weights'], len(choices))
            weight *= len(weight) / weight.sum()
            models = dict(logistic=LogisticRegression(C=0.3, class_weight='balanced', max_iter=600, random_state=seed), linear=SGDClassifier(loss='log_loss', alpha=0.01, class_weight='balanced', max_iter=1000, tol=1e-05, random_state=seed), gradient_boosting=HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7, l2_regularization=1.0, random_state=seed), random_forest=RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=1))
            if name == 'marker_regression':
                self.count_transform = 'log1p' if wells[0]['dataset'] == 'TNBC' else 'identity'
                target = np.log1p(data['counts']) if self.count_transform == 'log1p' else data['counts']
                target = np.tile(target, len(choices))
                self.model = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=7, l2_regularization=1.0, random_state=seed).fit(x, target, sample_weight=weight)
                self.residual_scale = max(0.05, float(np.sqrt(np.average((self.model.predict(x) - target) ** 2, weights=weight))))
            else:
                self.model = models[name].fit(x, y, sample_weight=weight)
            self.parameters = None
            self.device = 'cpu'
            return self
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Network(self.groups, self.prototype, self.count, matched_hidden(self.groups)).to(self.device)
        self.parameters = sum((p.numel() for p in self.model.parameters()))
        x = torch.tensor(data['x'], device=self.device)
        y = torch.tensor(data['y'], dtype=torch.float32, device=self.device)
        weight = torch.tensor(data['weights'], dtype=torch.float32, device=self.device)
        weight /= weight.mean()
        well = torch.tensor(data['well'], device=self.device)
        context = torch.tensor(data['context'], device=self.device)
        counts = torch.tensor(data['counts'], dtype=torch.float32, device=self.device)
        ratio = float((weight * (1 - y)).sum() / max(float((weight * y).sum()), 1.0))
        balanced = name not in ['MLP', 'masked_MLP', 'count_tail']
        lossweight = weight * torch.where(y > 0, ratio if balanced else 1.0, 1.0)
        generator = np.random.default_rng(seed)
        allmasks = [torch.tensor(m, device=self.device) for m in masks(self.groups)]
        full = torch.ones(self.groups, device=self.device)
        shifts = []
        contexts = sorted({w['context'] for w in wells if w['role'] == 'train'})
        for ctx in contexts:
            controls = [frame.transform(w['cells']).mean(0) for w in wells if w['role'] in ['train', 'reference'] and w['context'] == ctx and (w.get('label', 1) == 0 or w['role'] == 'reference')]
            controls = np.array(controls)
            shifts.append(controls - controls.mean(0) if len(controls) > 1 else np.zeros((1, self.groups, 24)))
        shifts = [torch.tensor(a, device=self.device) for a in shifts]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002, weight_decay=0.01)
        q_context = torch.ones(len(contexts), device=self.device) / len(contexts)
        for epoch in range(epochs):
            losses = []
            self.model.train()
            for indices in np.array_split(generator.permutation(len(x)), max(1, math.ceil(len(x) / 256))):
                idx = torch.tensor(indices, device=self.device)
                mask = allmasks[int(generator.integers(len(allmasks)))] if self.structured else full
                logit, z, assignment = self.model(x[idx], mask)
                if self.count:
                    if name == 'ratio_tail':
                        scale = torch.nn.functional.softplus(logit[:, 1]) + 0.05
                        individual = 0.5 * ((counts[idx] - logit[:, 0]) / scale) ** 2 + torch.log(scale)
                    else:
                        individual = zinb_loss(logit, counts[idx])
                    loss = (individual * weight[idx]).mean()
                else:
                    individual = torch.nn.functional.binary_cross_entropy_with_logits(logit, y[idx], reduction='none') * lossweight[idx]
                    if name == 'group_DRO':
                        terms = []
                        for c in range(len(contexts)):
                            selected = context[idx] == c
                            terms.append(individual[selected].mean() if selected.any() else individual.mean() * 0)
                        terms = torch.stack(terms)
                        q_context = q_context * torch.exp(0.01 * terms.detach())
                        q_context /= q_context.sum()
                        loss = (q_context * terms).sum()
                    else:
                        loss = individual.mean()
                    if name == 'metric_learning':
                        norm = torch.nn.functional.normalize(z, dim=1)
                        similarity = norm @ norm.T / 0.2
                        diagonal = torch.eye(len(idx), device=self.device, dtype=torch.bool)
                        same = (y[idx, None] == y[idx][None, :]) & ~diagonal
                        logprob = similarity - torch.logsumexp(similarity.masked_fill(diagonal, -10000.0), 1, keepdim=True)
                        contrast = -(logprob * same).sum(1) / same.sum(1).clamp(min=1)
                        loss = loss + 0.1 * (contrast * lossweight[idx]).mean()
                    if self.consistency or self.lattice:
                        full_logit, _, full_assignment = self.model(x[idx], full)
                        if self.lattice and assignment is not None:
                            terms = []
                            for w in torch.unique(well[idx]):
                                for label in [0, 1]:
                                    selected = (well[idx] == w) & (y[idx] == label)
                                    if selected.any():
                                        a = assignment[selected, label * 4:(label + 1) * 4]
                                        b = full_assignment[selected, label * 4:(label + 1) * 4]
                                        a = a / a.sum(1, keepdim=True).clamp(min=1e-06)
                                        b = b / b.sum(1, keepdim=True).clamp(min=1e-06)
                                        terms.append(((a.T @ a / len(a) - b.T @ b / len(b)) ** 2).sum())
                            if terms:
                                loss = loss + 0.2 * torch.stack(terms).mean()
                        if self.consistency:
                            augmentation = torch.zeros_like(x[idx])
                            for c in range(len(contexts)):
                                selected = context[idx] == c
                                positions = torch.tensor(generator.integers(len(shifts[c]), size=int(selected.sum())), device=self.device)
                                augmentation[selected] = shifts[c][positions] * 0.5
                            shifted = self.model(x[idx] + augmentation, mask)[0]
                            loss = loss + 0.2 * ((torch.sigmoid(shifted) - torch.sigmoid(logit)) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.history.append(float(np.mean(losses)))
        self.model.cpu()
        self.device = 'cpu'
        return self

    def raw(self, x, mask, passes=8, cutoff=None):
        z = self.frame.transform(x)
        if self.parameters is None:
            if self.name == 'marker_regression':
                from scipy.stats import norm
                threshold = np.log1p(cutoff) if self.count_transform == 'log1p' else cutoff
                p = norm.sf(threshold, loc=self.model.predict(flatten(z, mask)), scale=self.residual_scale)
                p = np.clip(p, 1e-06, 1 - 1e-06)
            else:
                p = self.model.predict_proba(flatten(z, mask))[:, 1]
            return (p, np.zeros(len(p)))
        from scipy.stats import nbinom
        torch.manual_seed(self.seed + 91)
        self.model.train(passes > 1)
        predictions = []
        with torch.no_grad():
            for repeat in range(max(1, passes)):
                result = self.model(torch.tensor(z), torch.tensor(mask))[0]
                if self.count:
                    if self.name == 'ratio_tail':
                        from scipy.stats import norm
                        p = norm.sf(cutoff, loc=result[:, 0].numpy(), scale=torch.nn.functional.softplus(result[:, 1]).numpy() + 0.05)
                    else:
                        mean = torch.nn.functional.softplus(result[:, 0]).numpy() + 0.001
                        shape = torch.nn.functional.softplus(result[:, 1]).numpy() + 0.001
                        zero = torch.sigmoid(result[:, 2]).numpy()
                        p = (1 - zero) * nbinom.sf(cutoff, shape, shape / (shape + mean))
                else:
                    p = torch.sigmoid(result).numpy()
                predictions.append(np.clip(p, 1e-06, 1 - 1e-06))
        self.model.eval()
        predictions = np.array(predictions)
        return (predictions.mean(0), predictions.var(0))

    def calibrate(self, wells, passes=8):
        self.calibration_ids = [w['id'] for w in wells if w['role'] == 'calibration']
        self.calibrators = {}
        for mask in masks(self.groups):
            parts = [(w, self.raw(w['cells'], mask, passes, w['cutoff'])[0]) for w in wells if w['role'] == 'calibration']
            y = np.concatenate([w['target'] for w, p in parts])
            p = np.concatenate([p for w, p in parts])
            weight = np.concatenate([np.full(len(w['cells']), 1 / len(w['cells'])) for w, p in parts])
            weight *= len(weight) / weight.sum()
            if len(np.unique(y)) < 2:
                raise ValueError('Calibration lacks classes')
            logits = np.log(p / (1 - p))[:, None]
            self.calibrators[tuple(mask)] = LogisticRegression(C=1.0, max_iter=300, random_state=self.seed).fit(logits, y, sample_weight=weight)
        return self

    def predict(self, x, mask, cutoff, passes=8, calibrated=True):
        raw, variance = self.raw(x, mask, passes, cutoff)
        p = self.calibrators[tuple(mask)].predict_proba(np.log(raw / (1 - raw))[:, None])[:, 1] if calibrated else raw
        entropy = -p * np.log(p + 1e-12) - (1 - p) * np.log(1 - p + 1e-12)
        return (p, entropy + variance, variance)
