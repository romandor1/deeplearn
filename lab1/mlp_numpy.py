"""Мережа X -> Linear(4,8) -> ReLU -> Linear(8,3) у NumPy з ручним backprop.

Автоматичне диференціювання тут не використовується.
"""
from __future__ import annotations

import numpy as np

PARAM_NAMES = ("W1", "b1", "W2", "b2")


def init_params() -> dict[str, np.ndarray]:
    """Ініціалізація: новий default_rng(0); спочатку W1 (He), потім W2 (Xavier)."""
    rng = np.random.default_rng(0)
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4), size=(4, 8))
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8 + 3)), size=(8, 3))
    return {
        "W1": W1,
        "b1": np.zeros(8),
        "W2": W2,
        "b2": np.zeros(3),
    }


def log_softmax(Z: np.ndarray) -> np.ndarray:
    """Чисельно стабільний log-softmax зі зсувом на максимум рядка."""
    shifted = Z - Z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def cross_entropy(Z2: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Крос-ентропія, усереднена за об'єктами. Повертає (втрата, log P)."""
    logp = log_softmax(Z2)
    loss = -logp[np.arange(len(y)), y].mean()
    return float(loss), logp


def forward(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray):
    """Прямий прохід. Повертає (loss, cache) зі значеннями для backward."""
    Z1 = X @ params["W1"] + params["b1"]      # (N, 8)
    A1 = np.maximum(Z1, 0.0)                  # (N, 8)
    Z2 = A1 @ params["W2"] + params["b2"]     # (N, 3)  логіти
    loss, logp = cross_entropy(Z2, y)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "logp": logp}
    return loss, cache


def loss_only(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    return forward(params, X, y)[0]


def backward(params, cache, y, buggy: bool = False) -> dict[str, np.ndarray]:
    """Ручний зворотний прохід.

    buggy=False: коректна реалізація, dZ2 = (P - Y) / N.
    buggy=True : навмисна помилка (дослід 4), dZ2 = P - Y  (без ділення на N).
    Функція втрат у forward в обох випадках однакова.
    """
    X, Z1, A1, logp = cache["X"], cache["Z1"], cache["A1"], cache["logp"]
    N = X.shape[0]

    P = np.exp(logp)                          # softmax, (N, 3)
    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1.0                  # one-hot, (N, 3)

    dZ2 = P - Y                               # (N, 3)
    if not buggy:
        dZ2 = dZ2 / N

    dW2 = A1.T @ dZ2                          # (8, 3)
    db2 = dZ2.sum(axis=0)                     # (3,)
    dA1 = dZ2 @ params["W2"].T                # (N, 8)
    dZ1 = dA1 * (Z1 > 0)                      # (N, 8)  ReLU'(0) = 0
    dW1 = X.T @ dZ1                           # (4, 8)
    db1 = dZ1.sum(axis=0)                     # (8,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def loss_and_grads(params, X, y, buggy: bool = False):
    loss, cache = forward(params, X, y)
    return loss, backward(params, cache, y, buggy=buggy)
