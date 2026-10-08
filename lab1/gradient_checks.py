"""Звірка з PyTorch та чисельна перевірка градієнтів."""
from __future__ import annotations

import numpy as np

from mlp_numpy import PARAM_NAMES, forward
from torch_ref import torch_loss_and_grads

TOL_TORCH = 1e-12
TOL_NUM = 1e-7
EPS = 1e-6
NUMERIC_POINTS = [("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,))]


def compare_with_torch(params, X, y, loss_np, grads_np):
    loss_t, grads_t = torch_loss_and_grads(params, X, y)
    rows = []
    d = abs(loss_np - loss_t)
    finite = bool(np.isfinite(loss_np) and np.isfinite(loss_t))
    rows.append(("Втрата", d, finite and d <= TOL_TORCH))
    for name in PARAM_NAMES:
        a, b = grads_np[name], grads_t[name]
        assert a.shape == b.shape, (name, a.shape, b.shape)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        d = float(np.max(np.abs(a - b)))
        rows.append((f"Градієнт {name}", d, finite and d <= TOL_TORCH))
    return loss_t, grads_t, rows


def numeric_check(params, X, y, grads_np):
    rows = []
    for name, idx in NUMERIC_POINTS:
        theta0 = params[name][idx].copy()
        mask0 = forward(params, X, y)[1]["Z1"] > 0
        params[name][idx] = theta0 + EPS
        l_plus, c_plus = forward(params, X, y)
        params[name][idx] = theta0 - EPS
        l_minus, c_minus = forward(params, X, y)
        params[name][idx] = theta0            # відновлення
        # збурення не має перетинати злам ReLU: маски при θ-ε, θ, θ+ε збігаються
        kink_free = bool(np.array_equal(mask0, c_plus["Z1"] > 0)
                         and np.array_equal(mask0, c_minus["Z1"] > 0))
        g_num = (l_plus - l_minus) / (2 * EPS)
        g_man = float(grads_np[name][idx])
        d = abs(g_num - g_man)
        ok = bool(np.isfinite(g_num) and np.isfinite(g_man) and d <= TOL_NUM)
        label = f"{name}[{', '.join(map(str, idx))}]"
        rows.append((label, g_man, g_num, d, ok, kink_free))
    return rows


def eps_scan(params, X, y, grads_np, name="W1", idx=(0, 0),
             eps_list=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10),
             dtype=np.float64):
    """Похибка центральної різниці залежно від eps для одного параметра.

    Обчислення виконуються в заданому dtype; еталон - ручний градієнт float64.
    Реальний крок береться з фактично збережених (округлених) значень параметра.
    """
    p = {k: v.astype(dtype) for k, v in params.items()}
    Xd = X.astype(dtype)
    g_ref = float(grads_np[name][idx])
    rows = []
    for eps in eps_list:
        old = p[name][idx]
        p[name][idx] = old + dtype(eps)
        hi = p[name][idx]
        l_plus = forward(p, Xd, y)[0]
        p[name][idx] = old - dtype(eps)
        lo = p[name][idx]
        l_minus = forward(p, Xd, y)[0]
        p[name][idx] = old
        h = (float(hi) - float(lo)) / 2          # фактичний крок
        if h == 0:                                # крок зник при округленні dtype
            rows.append((eps, None))
        else:
            rows.append((eps, abs((l_plus - l_minus) / (2 * h) - g_ref)))
    return rows
