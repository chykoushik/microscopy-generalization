import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import json, csv, time, hashlib
import numpy as np
import torch
from torch.nn import functional as F
from evaluation.image_editing_models import SmallUNet, StateClassifier, inputs, predict
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/results/fluorescence_reconstruction'

def fit(assay, kind, seed, epochs):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = np.load(OUT / (assay + '_images.npz'))
    images = data['images'].astype(np.float32)
    n = data['nuclei']
    c = data['cytoplasm']
    state = data['state'].astype(float)
    rows = json.loads((OUT / (assay + '_cells.json')).read_text())
    train = np.array([i for i, w in enumerate(rows) if w['role'] == 'train'])
    val = np.array([i for i, w in enumerate(rows) if w['role'] == 'validation'])
    xx = inputs(images, n, c, kind, state)
    yy = images if kind == 'editor' else state if kind == 'classifier' else images[:, 1:2]
    model = StateClassifier() if kind == 'classifier' else SmallUNet(xx.shape[1], 3 if kind == 'editor' else 1)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    counts = {w['well_id']: sum((q['well_id'] == w['well_id'] for q in rows if q['role'] == 'train')) for w in rows if w['role'] == 'train'}
    weight = np.array([1 / counts[rows[i]['well_id']] for i in train])
    weight *= len(weight) / weight.sum()
    history = []
    best = float('inf')
    best_state = None
    g = np.random.default_rng(seed)
    started = time.perf_counter()
    posweight = torch.tensor(max(1, (1 - state[train].mean()) / max(state[train].mean(), 1e-05)), device=device)
    for epoch in range(epochs):
        model.train()
        losses = []
        order = g.permutation(len(train))
        for start in range(0, len(train), 32):
            local = order[start:start + 32]
            index = train[local]
            x = torch.as_tensor(xx[index], device=device)
            target = torch.as_tensor(yy[index], device=device, dtype=torch.float32)
            opt.zero_grad()
            output = model(x)
            if kind == 'classifier':
                per = F.binary_cross_entropy_with_logits(output, target, pos_weight=posweight, reduction='none')
            else:
                region = torch.as_tensor(data['cells'][index, None], device=device, dtype=torch.float32)
                per = (torch.abs(output - target) * (1 + 2 * region)).mean((1, 2, 3))
            loss = (per * torch.as_tensor(weight[local], device=device, dtype=torch.float32)).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        prediction = predict(model, xx[val])
        validation = float(np.mean(np.abs(prediction - yy[val]))) if kind != 'classifier' else float(np.mean(np.square(1 / (1 + np.exp(-prediction)) - yy[val])))
        history.append(dict(assay=assay, kind=kind, seed=seed, epoch=epoch + 1, training=float(np.mean(losses)), validation=validation))
        if validation < best:
            best = validation
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.cpu()
    path = OUT / (assay + '_' + kind + '_' + str(seed) + '.pt')
    torch.save(dict(state_dict=best_state, kind=kind, inputs=xx.shape[1], outputs=3 if kind == 'editor' else 1, seed=seed, training_ids=[rows[i]['id'] for i in train], validation_ids=[rows[i]['id'] for i in val], best_validation=best, epochs=epochs, parameters=sum((p.numel() for p in model.parameters())), seconds=time.perf_counter() - started), path)
    print('Model trained', assay, kind, flush=True)
    return history

def main():
    histories = []
    started = time.perf_counter()
    for assay in ['NF', 'GAMMA']:
        for kind, seeds, epochs in [('reconstruction', [200, 201], 24), ('dapi', [200], 24), ('structural', [200], 24)]:
            for seed in seeds:
                if (OUT / (assay + '_' + kind + '_' + str(seed) + '.pt')).exists():
                    continue
                histories += fit(assay, kind, seed, epochs)
    if histories:
        with (OUT / 'TRAINING_HISTORY.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(histories[0]))
            w.writeheader()
            w.writerows(histories)
    (OUT / 'TRAINING_RESOURCE.json').write_text(json.dumps(dict(seconds=time.perf_counter() - started, peak_vram=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'), indent=2))
if __name__ == '__main__':
    main()
