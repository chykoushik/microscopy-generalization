import hashlib
import importlib
import logging
import re
import sys
import types
import numpy as np

def load_encoder(root, config, device):
    import torch
    path = root / config['checkpoint']
    if hashlib.sha256(path.read_bytes()).hexdigest() != config['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    package = types.ModuleType('deepcellmorpho')
    package.__path__ = [str(root / config['source'])]
    sys.modules['deepcellmorpho'] = package
    logging.getLogger('dinov2').setLevel(logging.ERROR)
    module = importlib.import_module('deepcellmorpho.models.dinos.vision_transformer')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if set(checkpoint) != {'teacher'}:
        raise ValueError('Invalid pretrained checkpoint')
    teacher = checkpoint['teacher']
    if any((not k.startswith(('backbone.', 'dino_head.')) for k in teacher)):
        raise ValueError('Unknown teacher component')
    state = {}
    for key, value in teacher.items():
        if key.startswith('backbone.'):
            key = key.removeprefix('backbone.')
            key = re.sub('^blocks\\.\\d+\\.(\\d+)\\.', 'blocks.\\1.', key)
            if key in state:
                raise ValueError('Duplicate backbone parameter')
            state[key] = value
    model = module.vit_small(patch_size=16, img_size=480, init_values=1, ffn_layer='swiglu', block_chunks=0)
    result = model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError('Incomplete pretrained loading')
    record = dict(checkpoint_sha256=config['checkpoint_sha256'], checkpoint_component='teacher.backbone', excluded_projection_head=[key for key in teacher if key.startswith('dino_head.')], block_chunks=0, checkpoint_key_adaptation='Flatten four training block chunks, preserving original layer indices; strict complete backbone load', loaded_tensors=len(state), parameter_count=sum((p.numel() for p in model.parameters())), missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys), evaluation_mode=not model.training, trainable_parameters=sum((p.numel() for p in model.parameters() if p.requires_grad)), device=str(device), precision='float32', architecture=config['name'])
    return (model, record)

def infer_crops(model, crops, device, batch_size):
    import torch
    crops = np.asarray(crops, dtype=np.float32)
    if crops.ndim != 4 or crops.shape[1] != 5 or (not np.isfinite(crops).all()) or (batch_size < 1):
        raise ValueError('Invalid inference crops')
    result = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            values = model(torch.from_numpy(crops[start:start + batch_size]).to(device))
            result.append(values.float().cpu().numpy())
    return np.concatenate(result)
