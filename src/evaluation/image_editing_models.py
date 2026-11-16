import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class SmallUNet(nn.Module):

    def __init__(self, inputs, outputs, width=8):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
        self.a = block(inputs, width)
        self.b = block(width, width * 2)
        self.c = block(width * 2, width * 4)
        self.d = block(width * 6, width * 2)
        self.e = block(width * 3, width)
        self.out = nn.Conv2d(width, outputs, 1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(F.avg_pool2d(a, 2))
        c = self.c(F.avg_pool2d(b, 2))
        d = self.d(torch.cat([F.interpolate(c, scale_factor=2, mode='bilinear', align_corners=False), b], 1))
        e = self.e(torch.cat([F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False), a], 1))
        return torch.sigmoid(self.out(e))

class StateClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(4), nn.Flatten(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).flatten()

def inputs(images, nuclei, cytoplasm, kind, states=None):
    if kind == 'classifier':
        return images
    mask = np.stack([nuclei, cytoplasm], 1).astype(np.float32)
    if kind == 'editor':
        return np.concatenate([images, mask, np.broadcast_to(np.asarray(states, np.float32)[:, None, None, None], (len(images), 1, images.shape[2], images.shape[3]))], 1)
    channels = [0] if kind == 'dapi' else [2] if kind == 'structural' else [0, 2]
    return np.concatenate([images[:, channels], mask], 1)

def predict(model, x, batch=64):
    device = next(model.parameters()).device
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            output.append(model(torch.as_tensor(x[start:start + batch], device=device, dtype=torch.float32)).cpu().numpy())
    return np.concatenate(output)
