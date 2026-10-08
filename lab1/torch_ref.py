"""Еталонна реалізація в PyTorch (лише для звірки)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def torch_loss_and_grads(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray):
    """Будує еквівалентну мережу, копіює параметри NumPy-мережі, повертає (loss, grads).

    nn.Linear зберігає ваги як (out, in), тому копіюємо W.T.
    Градієнти повертаються у «NumPy-розмірностях» (in, out).
    """
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(params["W1"].T.copy()))
        net[0].bias.copy_(torch.from_numpy(params["b1"].copy()))
        net[2].weight.copy_(torch.from_numpy(params["W2"].T.copy()))
        net[2].bias.copy_(torch.from_numpy(params["b2"].copy()))

    Xt = torch.from_numpy(X.copy())
    yt = torch.from_numpy(y.copy())
    loss = nn.functional.cross_entropy(net(Xt), yt)  # reduction='mean'
    net.zero_grad()
    loss.backward()
    grads = {
        "W1": net[0].weight.grad.numpy().T.copy(),
        "b1": net[0].bias.grad.numpy().copy(),
        "W2": net[2].weight.grad.numpy().T.copy(),
        "b2": net[2].bias.grad.numpy().copy(),
    }
    return float(loss.item()), grads
